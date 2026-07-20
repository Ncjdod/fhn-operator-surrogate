# Operator-learning dataset for the coarse-step flow-map: random zero-order-hold (ZOH) current
# profiles and the true trajectories they produce, sampled on the coarse grid D = stride*dt.
# Port of neuron_data.py, GPU-agnostic: the truth is produced by the batched fine-RK4 solver.
#
# Returns (X0, Uc, Y) with X0 (d, N), Uc (K, N) one held current per coarse step, and the truth
# Y (d, K+1, N).  The surrogate learns x_{k+1} = F(x_k) + G(x_k)*Uc[k].

using Random

# One random coarse ZOH current profile of a given kind, clipped to the model's range.
function _coarse_current!(u::AbstractVector, rng, kind::Symbol, lo, hi, ulo, uhi)
    K = length(u)
    if kind === :const
        fill!(u, rand(rng) * (hi - lo) + lo)
    elseif kind === :step
        a = rand(rng) * (lo + 2 - ulo) + ulo
        b = rand(rng) * (hi - lo) + lo
        k = rand(rng, max(1, K ÷ 6):max(2, 3K ÷ 5))
        @inbounds for i in 1:K; u[i] = i < k ? a : b; end
    elseif kind === :ramp
        a = rand(rng) * (lo - ulo) + ulo; b = rand(rng) * (hi - lo) + lo
        @inbounds for i in 1:K; u[i] = a + (b - a) * (i - 1) / max(1, K - 1); end
    elseif kind === :pulse
        base = rand(rng) * (lo - ulo) + ulo; hgt = rand(rng) * (hi - lo) + lo
        w = max(1, rand(rng, 2:8)); per = rand(rng, w+3:w+20)
        fill!(u, base)
        s = 1
        while s <= K
            @inbounds for i in s:min(s + w - 1, K); u[i] = hgt; end
            s += per
        end
    elseif kind === :ou
        theta = 0.05; mu = rand(rng) * (hi - lo) + lo; sig = rand(rng) * 4 + 2
        u[1] = mu
        @inbounds for i in 2:K
            u[i] = u[i-1] + theta * (mu - u[i-1]) + sig * randn(rng)
        end
    else  # :piecewise — random holds
        i = 1
        while i <= K
            seg = rand(rng, 3:20); val = rand(rng) * (hi - ulo) + ulo
            @inbounds for j in i:min(i + seg - 1, K); u[j] = val; end
            i += seg
        end
    end
    @inbounds for i in 1:K; u[i] = clamp(u[i], ulo, uhi); end
    return u
end

const _KINDS = (:const, :step, :ramp, :pulse, :ou, :piecewise)

"""
    make_dataset(model; N, K, stride, dt, seed) -> (X0, Uc, Y)

Build a coarse-grid ZOH operator dataset. `Uc` is (K, N), the truth `Y` is (d, K+1, N) produced
by fine RK4 at step `dt` (stride substeps per coarse step D = stride*dt). Arrays are `Float32`.
"""
function make_dataset(model::NeuronModel; N::Int=512, K::Int=250, stride::Int=20,
                      dt::Float64=0.02, seed::Int=101)
    rng = MersenneTwister(seed)
    d = statedim(model)
    lo, hi = firing_band(model)
    ulo, uhi = u_bounds(model)
    Uc = Matrix{Float32}(undef, K, N)
    tmp = Vector{Float64}(undef, K)
    for j in 1:N
        _coarse_current!(tmp, rng, _KINDS[(j-1) % length(_KINDS) + 1], lo, hi, ulo, uhi)
        @inbounds for k in 1:K; Uc[k, j] = tmp[k]; end
    end
    X0 = Float32.(random_init(model, rng, N))
    # integrate the true plant one coarse step at a time (stride fine RK4 substeps, constant u).
    Y = rollout_rk4(model, X0, Uc, dt, stride; trajectory=true)
    return X0, Uc, Float32.(Y)
end

"Finite-difference control sensitivity g_true = d phi_D/du on the coarse grid (for optional L_G)."
function control_sensitivity(model::NeuronModel, X0, Uc, dt, stride; delta=0.5f0)
    d, N = size(X0); K = size(Uc, 1)
    G = Array{Float32}(undef, d, K, N)
    Xk = copy(X0)
    for k in 1:K
        uk = view(Uc, k, :)
        xp = rollout_rk4(model, Xk, reshape(collect(uk) .+ delta, 1, :), dt, stride; trajectory=false)
        xm = rollout_rk4(model, Xk, reshape(collect(uk) .- delta, 1, :), dt, stride; trajectory=false)
        G[:, k, :] .= (xp .- xm) ./ (2delta)
        Xk = rollout_rk4(model, Xk, reshape(collect(uk), 1, :), dt, stride; trajectory=false)
    end
    return G
end
