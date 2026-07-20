# Control-affine flow-map surrogate:  x_{t+D} = F(x) + G(x)*u,  invertible in u in closed form.
#
# Port of flowmap_affine.py.  The injected current enters the neuron affinely and only in the
# voltage channel, so a coarse (one-big-step) flow map that is affine in a zero-order-hold
# current is structurally exact in its u-dependence and inverts in a single pass:
#
#     u* = <G(x), x_target - F(x)> / <G(x), G(x)>          (least-squares steering current)
#
# No optimizer loop -> live, per-step intervention.  F and G are MLP heads on a shared trunk;
# only u enters affinely, so forward accuracy is never traded away for invertibility.  G is
# floored on the voltage channel so <G,G> is bounded away from zero and the inverse never blows
# up.  Forward, inverse, rollout, AND the training gradients are plain dense matrix algebra, so
# the whole thing is GPU-batched (CuArray) with no code changes.
#
# Batch layout: X is (d x N), one neuron per column; a coarse current batch Uc is (K x N).

isdefined(@__MODULE__, :Dense) || include(joinpath(@__DIR__, "nn.jl"))

mutable struct AffineFlowMap{T}
    trunk::Vector{Dense}
    Fh::Dense              # F-residual head (linear)
    Gh::Dense              # G (control sensitivity) head (linear)
    mu::Vector{T}          # per-channel input mean (standardization)
    sd::Vector{T}          # per-channel input std
    v_chan::Int            # voltage channel index (1)
    g_floor::T             # absolute floor on the voltage-channel control gain
    d::Int
end

function AffineFlowMap(d::Int; hidden=(128, 128), rng, mu=zeros(d), sd=ones(d),
                       v_chan=1, g_floor=0.1, T=Float32)
    sizes = Int[d, hidden...]
    trunk = Dense[]
    for i in 1:length(sizes)-1
        push!(trunk, dense(sizes[i], sizes[i+1]; rng=rng, T=T))
    end
    Fh = dense(sizes[end], d; rng=rng, scale=0.1, T=T)
    Gh = dense(sizes[end], d; rng=rng, scale=0.05, T=T)
    return AffineFlowMap{T}(trunk, Fh, Gh, T.(mu), T.(sd), v_chan, T(g_floor), d)
end

# Ordered parameter references (for the Adam optimizer to update in place).
function params(s::AffineFlowMap)
    ps = Any[]
    for L in s.trunk; push!(ps, L.W); push!(ps, L.b); end
    push!(ps, s.Fh.W); push!(ps, s.Fh.b); push!(ps, s.Gh.W); push!(ps, s.Gh.b)
    return ps
end
zero_grads(s::AffineFlowMap) = [zero(p) for p in params(s)]

# ---- forward: drift F(x) and control sensitivity G(x) ---------------------------------------
"Return (F, G) with no cache (inference/control path)."
function FG(s::AffineFlowMap, X::AbstractMatrix)
    A0 = (X .- s.mu) ./ s.sd
    H, _ = trunk_forward(s.trunk, A0)
    F = X .+ s.sd .* (s.Fh.W * H .+ s.Fh.b)
    G = s.sd .* (s.Gh.W * H .+ s.Gh.b)
    G = _apply_floor(G, s.v_chan, s.g_floor)
    return F, G
end

# Add the voltage-channel floor without scalar-indexing a GPU array (row-mask add).
function _apply_floor(G, v_chan, g_floor)
    d = size(G, 1)
    mask = reshape(Float32[i == v_chan ? 1f0 : 0f0 for i in 1:d], d, 1)
    return G .+ g_floor .* mask
end

"One coarse step x -> F(x) + G(x).*u  (u is a length-N vector)."
function flow_step(s::AffineFlowMap, X::AbstractMatrix, u::AbstractVector)
    F, G = FG(s, X)
    return F .+ G .* reshape(u, 1, :)
end

"""
    invert(s, X, Xtgt; clip=nothing)

Closed-form least-squares steering current driving X -> Xtgt in one coarse step.
Returns (u, Xnext, r_reach) where r_reach is the unreachable residual (a scalar current can
only reach the rank-1 span of G in the d-dim target space). Fully batched.
"""
function invert(s::AffineFlowMap, X::AbstractMatrix, Xtgt::AbstractMatrix; clip=nothing)
    F, G = FG(s, X)
    num = vec(sum(G .* (Xtgt .- F), dims=1))
    den = vec(sum(G .* G, dims=1)) .+ 1f-8
    u = num ./ den
    if clip !== nothing
        u = clamp.(u, Float32(clip[1]), Float32(clip[2]))
    end
    Xnext = F .+ G .* reshape(u, 1, :)
    r = vec(sqrt.(sum((Xtgt .- Xnext) .^ 2, dims=1)))
    return u, Xnext, r
end

# ---- inference rollout over a ZOH coarse current Uc (K x N) ----------------------------------
"Roll the stepper over Uc (K x N); returns states (d, K+1, N)."
function rollout(s::AffineFlowMap, X0::AbstractMatrix, Uc::AbstractMatrix)
    d, N = size(X0)
    K = size(Uc, 1)
    Y = similar(X0, d, K + 1, N)
    Y[:, 1, :] .= X0
    X = X0
    for k in 1:K
        X = flow_step(s, X, view(Uc, k, :))
        Y[:, k+1, :] .= X
    end
    return Y
end

# ---- training: peak-weighted rollout MSE + G-conditioning keep-alive, manual BPTT ------------
# Loss(θ) = mean_{d,t,batch} w .* ((Xhat - Y)./sd)^2                      (fit)
#         + λ_cond * mean_{t,batch} relu(gfloor2 - <G,G>)                 (keep <G,G> well posed)
# with w = 1 + 3 |(Y - mean_t Y)/sd|  (up-weight the fast spike excursions), gfloor2=(1.5 g_floor)^2.
"""
    loss_and_grads(s, X0, Uc, Y; lambda_cond=0.1) -> (loss, grads)

Forward-rolls the surrogate over Uc, compares to the truth trajectory Y (d, K+1, N), and returns
the scalar loss and gradients (parallel to `params(s)`) via an explicit backward pass. Gradients
are matrix algebra only, so this trains on GPU as well as CPU.
"""
function loss_and_grads(s::AffineFlowMap, X0::AbstractMatrix, Uc::AbstractMatrix,
                        Y::AbstractArray{<:Any,3}; lambda_cond::Float64=0.1)
    d, N = size(X0)
    K = size(Uc, 1)
    sd = s.sd
    gfloor2 = (1.5f0 * s.g_floor)^2

    # peak weights from the truth (constant w.r.t. params)
    Ymean = sum(Y, dims=2) ./ (K + 1)
    W = 1f0 .+ 3f0 .* abs.((Y .- Ymean) ./ reshape(sd, d, 1, 1))
    cfit = 1f0 / (N * (K + 1) * d)

    # ---- forward, caching per-step quantities needed by backward ----
    Fs = Vector{Any}(undef, K); Gs = Vector{Any}(undef, K)
    Hs = Vector{Any}(undef, K); tcs = Vector{Any}(undef, K); Xs = Vector{Any}(undef, K)
    Xhat = similar(X0, d, K + 1, N)
    Xhat[:, 1, :] .= X0
    X = X0
    loss = 0.0
    for k in 1:K
        A0 = (X .- s.mu) ./ sd
        H, tc = trunk_forward(s.trunk, A0)
        F = X .+ sd .* (s.Fh.W * H .+ s.Fh.b)
        G = sd .* (s.Gh.W * H .+ s.Gh.b)
        G = _apply_floor(G, s.v_chan, s.g_floor)
        u = reshape(view(Uc, k, :), 1, :)
        Xn = F .+ G .* u
        Xs[k] = X; Fs[k] = F; Gs[k] = G; Hs[k] = H; tcs[k] = tc
        Xhat[:, k+1, :] .= Xn
        X = Xn
    end
    # loss value
    diff = (Xhat .- Y) ./ reshape(sd, d, 1, 1)
    loss += cfit * sum(W .* diff .^ 2)
    condvals = 0.0
    for k in 1:K
        ss = sum(Gs[k] .* Gs[k], dims=1)
        condvals += sum(max.(0f0, gfloor2 .- ss))
    end
    loss += lambda_cond * condvals / (N * K)

    # ---- backward ----
    gW = zero_grads(s)                    # parallel to params(s)
    nT = length(s.trunk)
    # helpers to index into gW for the two heads
    iFW = 2 * nT + 1; iFb = 2 * nT + 2; iGW = 2 * nT + 3; iGb = 2 * nT + 4
    gtrunkW = [gW[2i-1] for i in 1:nT]
    gtrunkb = [gW[2i] for i in 1:nT]

    gradfit(k) = (2f0 * cfit) .* view(W, :, k, :) .* (Xhat[:, k, :] .- Y[:, k, :]) ./ (sd .^ 2)
    # incoming grad on S_{K+1}
    dS = gradfit(K + 1)

    for k in K:-1:1
        F = Fs[k]; G = Gs[k]; H = Hs[k]; tc = tcs[k]
        u = reshape(view(Uc, k, :), 1, :)
        dF = dS
        dG = dS .* u
        # G-conditioning grad: d/dG of (λ/(N K)) sum relu(gfloor2 - <G,G>)
        ss = sum(G .* G, dims=1)
        active = Float32.(ss .< gfloor2)
        dG = dG .+ (Float32(lambda_cond) / (N * K)) .* (active .* (-2f0 .* G))
        # heads: F = X + sd.*(WF H + bF);  G = sd.*(WG H + bG) + floor
        dFres = sd .* dF
        dGres = sd .* dG
        gW[iFW] .+= dFres * H'
        gW[iFb] .+= vec(sum(dFres, dims=2))
        gW[iGW] .+= dGres * H'
        gW[iGb] .+= vec(sum(dGres, dims=2))
        dH = s.Fh.W' * dFres .+ s.Gh.W' * dGres
        dA0 = trunk_backward!(gtrunkW, gtrunkb, s.trunk, tc, dH)
        dXk = dF .+ dA0 ./ sd                       # skip connection (F=X+..) + standardization
        dS = dXk .+ gradfit(k)                      # add the fit-loss term sitting on S_k
    end
    return loss, gW
end
