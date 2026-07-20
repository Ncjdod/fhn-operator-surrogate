# Batched neuron integrators as KernelAbstractions kernels: one GPU thread integrates one
# neuron.  The SAME kernel source runs on the CPU backend (used by the test suite here) and on
# CUDA (the user's GTX 1660 Ti) — the backend is chosen at runtime from the array type via
# KernelAbstractions.get_backend, so there is no separate CPU/GPU code path to keep in sync.
#
# Two integrators, both stepping a zero-order-hold (ZOH) current that is constant over each
# coarse step D = nsub*dt (this is exactly the plant map phi_D used by the control benchmark):
#   * rollout_rk4       — explicit RK4.  Fast per step but stability-capped at a tiny dt on
#                         stiff HH, so it needs many substeps.  This is the honest
#                         "well-optimized numerical solver" baseline the surrogate must beat.
#   * rollout_rosenbrock — Rosenbrock-W (2nd order, ROS2) with an analytic-via-ForwardDiff
#                         Jacobian and a d x d StaticArrays solve.  L-stable enough to take
#                         much larger steps on stiff cells, all still inside one GPU thread.
#
# State layout is a (d x N) column-major matrix: column j is neuron j.  Trajectories are
# (d x (Tc+1) x N).  Column-major + one-thread-per-column gives coalesced loads on the GPU.

using KernelAbstractions
using StaticArrays
using LinearAlgebra: I

# Allocation-free d x d Jacobian of x -> vfield(model, x, u) as an SMatrix, via a single
# forward-mode seeding with the in-house Dual (see ../ad.jl).  Returns a StaticArray with no
# heap allocation, so it is legal inside a GPU kernel.
@inline function _static_jacobian(model, x::SVector{D,T}, u) where {D,T}
    y = vfield(model, seed(x), u)
    return extract_jacobian(y)
end

@inline function _loadstate(X, j, ::Val{D}) where {D}
    return SVector{D}(ntuple(i -> @inbounds(X[i, j]), Val(D)))
end

@inline function _store_traj!(Y, x::SVector{D}, k, j) where {D}
    @inbounds for i in 1:D
        Y[i, k, j] = x[i]
    end
end

@inline function _store_col!(Y, x::SVector{D}, j) where {D}
    @inbounds for i in 1:D
        Y[i, j] = x[i]
    end
end

# ---- one explicit RK4 substep at fixed u ----------------------------------------------------
@inline function rk4_substep(model, x::SVector, u, dt)
    k1 = vfield(model, x, u)
    k2 = vfield(model, x + (dt / 2) * k1, u)
    k3 = vfield(model, x + (dt / 2) * k2, u)
    k4 = vfield(model, x + dt * k3, u)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
end

# Integrate one coarse step (nsub explicit substeps at constant u).
@inline function rk4_coarse(model, x::SVector, u, dt, nsub::Int)
    @inbounds for _ in 1:nsub
        x = rk4_substep(model, x, u, dt)
    end
    return x
end

# ---- one Rosenbrock-W (ROS2) step -----------------------------------------------------------
# (I - gamma*dt*J) k1 = f(x);   (I - gamma*dt*J) k2 = f(x+dt*k1) - 2*k1;   x' = x + dt*(3k1+k2)/2
# with gamma = 1 + 1/sqrt(2).  J is the exact Jacobian of f at x (ForwardDiff, d x d SMatrix).
# For d <= ~8 the StaticArrays solve is a couple of registers of work — cheap and GPU-safe.
@inline function ros2_step(model, x::SVector{D,T}, u, dt) where {D,T}
    gamma = T(1) + inv(sqrt(T(2)))
    f0 = vfield(model, x, u)
    J = _static_jacobian(model, x, u)
    W = SMatrix{D,D,T}(I) - (gamma * dt) * J   # (I - gamma*dt*J), the Rosenbrock system matrix
    k1 = W \ f0
    f1 = vfield(model, x + dt * k1, u)
    k2 = W \ (f1 - 2 * k1)
    return x + (dt / 2) * (3 * k1 + k2)
end

@inline function ros_coarse(model, x::SVector, u, dt, nsub::Int)
    @inbounds for _ in 1:nsub
        x = ros2_step(model, x, u, dt)
    end
    return x
end

# ---- kernels --------------------------------------------------------------------------------
@kernel function _rk4_traj_kernel!(Y, @Const(X0), @Const(U), model, dt, nsub, ::Val{D}) where {D}
    j = @index(Global)
    x = _loadstate(X0, j, Val(D))
    _store_traj!(Y, x, 1, j)
    Tc = size(U, 1)
    @inbounds for k in 1:Tc
        x = rk4_coarse(model, x, U[k, j], dt, nsub)
        _store_traj!(Y, x, k + 1, j)
    end
end

@kernel function _rk4_final_kernel!(Y, @Const(X0), @Const(U), model, dt, nsub, ::Val{D}) where {D}
    j = @index(Global)
    x = _loadstate(X0, j, Val(D))
    Tc = size(U, 1)
    @inbounds for k in 1:Tc
        x = rk4_coarse(model, x, U[k, j], dt, nsub)
    end
    _store_col!(Y, x, j)
end

@kernel function _ros_traj_kernel!(Y, @Const(X0), @Const(U), model, dt, nsub, ::Val{D}) where {D}
    j = @index(Global)
    x = _loadstate(X0, j, Val(D))
    _store_traj!(Y, x, 1, j)
    Tc = size(U, 1)
    @inbounds for k in 1:Tc
        x = ros_coarse(model, x, U[k, j], dt, nsub)
        _store_traj!(Y, x, k + 1, j)
    end
end

@kernel function _ros_final_kernel!(Y, @Const(X0), @Const(U), model, dt, nsub, ::Val{D}) where {D}
    j = @index(Global)
    x = _loadstate(X0, j, Val(D))
    Tc = size(U, 1)
    @inbounds for k in 1:Tc
        x = ros_coarse(model, x, U[k, j], dt, nsub)
    end
    _store_col!(Y, x, j)
end

# ---- host entry points ----------------------------------------------------------------------
"""
    rollout_rk4(model, X0, U, dt, nsub; trajectory=true)

Batched explicit-RK4 rollout of `model` over ZOH coarse currents.
`X0` is `(d, N)`, `U` is `(Tc, N)` (one held current per coarse step). Each coarse step of
duration `D = nsub*dt` is integrated with `nsub` RK4 substeps. Returns `(d, Tc+1, N)` if
`trajectory`, else the final `(d, N)`. Runs on whatever backend `X0` lives on (CPU/CUDA).
"""
function rollout_rk4(model::NeuronModel, X0::AbstractMatrix, U::AbstractMatrix,
                     dt::Real, nsub::Integer; trajectory::Bool=true)
    D = statedim(model)
    N = size(X0, 2)
    Tc = size(U, 1)
    @assert size(U, 2) == N "U must be (Tc, N) with the same N as X0"
    backend = get_backend(X0)
    T = eltype(X0)
    dtT = T(dt); ns = Int(nsub)
    if trajectory
        Y = similar(X0, D, Tc + 1, N)
        _rk4_traj_kernel!(backend)(Y, X0, U, model, dtT, ns, Val(D); ndrange=N)
    else
        Y = similar(X0, D, N)
        _rk4_final_kernel!(backend)(Y, X0, U, model, dtT, ns, Val(D); ndrange=N)
    end
    KernelAbstractions.synchronize(backend)
    return Y
end

"""
    rollout_rosenbrock(model, X0, U, dt, nsub; trajectory=true)

Same interface as [`rollout_rk4`](@ref) but uses the L-stable ROS2 Rosenbrock-W step, which
stays stable on stiff HH at step sizes where explicit RK4 diverges — so it reaches the same
horizon with far fewer, larger steps.
"""
function rollout_rosenbrock(model::NeuronModel, X0::AbstractMatrix, U::AbstractMatrix,
                            dt::Real, nsub::Integer; trajectory::Bool=true)
    D = statedim(model)
    N = size(X0, 2)
    Tc = size(U, 1)
    @assert size(U, 2) == N "U must be (Tc, N) with the same N as X0"
    backend = get_backend(X0)
    T = eltype(X0)
    dtT = T(dt); ns = Int(nsub)
    if trajectory
        Y = similar(X0, D, Tc + 1, N)
        _ros_traj_kernel!(backend)(Y, X0, U, model, dtT, ns, Val(D); ndrange=N)
    else
        Y = similar(X0, D, N)
        _ros_final_kernel!(backend)(Y, X0, U, model, dtT, ns, Val(D); ndrange=N)
    end
    KernelAbstractions.synchronize(backend)
    return Y
end
