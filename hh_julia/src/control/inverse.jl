# Inverse / control layer: turn any of the neuron models into a controllable plant by solving
# for the injected current that drives x -> x_target in one coarse step D.
#
# Because the current enters affinely (only dV/dt, via 1/C_m), the one-step map is
#     phi_D(x, u) ≈ f0(x) + g(x)*u,     g(x) = d phi_D / du,
# so the steering current has the SAME closed form as the surrogate's:
#     u* = <g, x_target - f0> / <g, g>       (least squares; a scalar current reaches span{g}).
# Three controllers, increasing cost / fidelity, all batched (one GPU thread per neuron):
#   * surrogate  — 1 MLP forward gives (F,G); u* in closed form.        [cheapest, amortized]
#   * lin1       — linearize the TRUE plant once about u=0 (f0,g via AD) then the same formula.
#   * gn(K)      — K Gauss-Newton iterations on the TRUE plant.          [most accurate baseline]
# lin1/gn need no training and drive the exact plant, so they are the ground-truth controllers;
# the surrogate is their amortized approximation (no stiff solves in the loop).

using KernelAbstractions
using StaticArrays

# Value f0 = phi_D(x,u) and du-sensitivity g = d phi_D/du of the coarse RK4 map, via one
# forward-mode pass (state lifted to Dual{1} carrying 0 partial, u carrying partial 1).
@inline function phi_and_sens(model, x::SVector{D,T}, u::T, dt, nsub::Int) where {D,T}
    xd = SVector{D}(ntuple(i -> Dual{1,T}(x[i], zero(SVector{1,T})), Val(D)))
    ud = Dual{1,T}(u, SVector{1,T}(one(T)))
    yd = rk4_coarse(model, xd, ud, dt, nsub)
    f0 = SVector{D,T}(ntuple(i -> value(yd[i]), Val(D)))
    g  = SVector{D,T}(ntuple(i -> partials(yd[i])[1], Val(D)))
    return f0, g
end

@inline function _closed_form_u(g::SVector, f0::SVector, tgt::SVector, ulo, uhi)
    u = sum(g .* (tgt .- f0)) / (sum(g .* g) + eltype(g)(1e-8))
    return clamp(u, ulo, uhi)
end

@kernel function _lin1_kernel!(U, @Const(X), @Const(Xtgt), model, dt, nsub, ulo, uhi, ::Val{D}) where {D}
    j = @index(Global)
    x = _loadstate(X, j, Val(D))
    tgt = _loadstate(Xtgt, j, Val(D))
    f0, g = phi_and_sens(model, x, zero(eltype(X)), dt, nsub)   # linearize about u=0
    @inbounds U[j] = _closed_form_u(g, f0, tgt, ulo, uhi)
end

@kernel function _gn_kernel!(U, @Const(X), @Const(Xtgt), @Const(Uw), model, dt, nsub, ulo, uhi, iters, ::Val{D}) where {D}
    j = @index(Global)
    x = _loadstate(X, j, Val(D))
    tgt = _loadstate(Xtgt, j, Val(D))
    u = @inbounds Uw[j]
    @inbounds for _ in 1:iters
        f0, g = phi_and_sens(model, x, u, dt, nsub)
        du = sum(g .* (tgt .- f0)) / (sum(g .* g) + eltype(X)(1e-8))
        u = clamp(u + du, ulo, uhi)
    end
    @inbounds U[j] = u
end

"Model-based one-shot controller: linearize the true plant about u=0, return steering current (N,)."
function control_lin1(model::NeuronModel, X::AbstractMatrix, Xtgt::AbstractMatrix, dt, nsub)
    D = statedim(model); N = size(X, 2); backend = get_backend(X); T = eltype(X)
    ulo, uhi = u_bounds(model)
    U = similar(X, N)
    _lin1_kernel!(backend)(U, X, Xtgt, model, T(dt), Int(nsub), T(ulo), T(uhi), Val(D); ndrange=N)
    KernelAbstractions.synchronize(backend)
    return U
end

"Model-based Gauss-Newton controller (iters steps) on the true plant; `Uw` warm-starts u (N,)."
function control_gn(model::NeuronModel, X::AbstractMatrix, Xtgt::AbstractMatrix, dt, nsub;
                    iters::Int=6, Uw=nothing)
    D = statedim(model); N = size(X, 2); backend = get_backend(X); T = eltype(X)
    ulo, uhi = u_bounds(model)
    Uw = Uw === nothing ? fill!(similar(X, N), zero(T)) : Uw
    U = similar(X, N)
    _gn_kernel!(backend)(U, X, Xtgt, Uw, model, T(dt), Int(nsub), T(ulo), T(uhi), Int(iters), Val(D); ndrange=N)
    KernelAbstractions.synchronize(backend)
    return U
end

"""
    closed_loop(plant, controller, X0, Xref, dt, nsub) -> (traj, us, track_nrmse)

Drive the TRUE `plant` to follow the reference states `Xref` (d, T, N): at each step the
`controller(X, target)` returns the injected current, which advances the plant one coarse step
via RK4. Returns the achieved trajectory (d, T+1... aligned to Xref), the applied currents, and
the standardized tracking NRMSE against `Xref`.
"""
function closed_loop(plant::NeuronModel, controller, X0::AbstractMatrix, Xref::AbstractArray{<:Any,3},
                     dt, nsub)
    d, T, N = size(Xref)
    X = copy(X0)
    traj = similar(X0, d, T, N)
    us = similar(X0, T, N)
    for k in 1:T
        tgt = Xref[:, k, :]
        u = controller(X, tgt)
        X = rollout_rk4(plant, X, reshape(u, 1, :), dt, nsub; trajectory=false)
        traj[:, k, :] .= X
        us[k, :] .= u
    end
    sd = vec(sqrt.(sum((reshape(Xref, d, :) .- (sum(reshape(Xref, d, :), dims=2) ./ (T * N))) .^ 2, dims=2) ./ (T * N))) .+ 1f-6
    track = sqrt(sum(((traj .- Xref) ./ reshape(sd, d, 1, 1)) .^ 2) / length(Xref))
    return traj, us, track
end
