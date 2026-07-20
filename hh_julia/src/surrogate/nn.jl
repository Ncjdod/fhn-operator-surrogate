# Tiny dense-MLP toolkit with explicit forward+backward passes and an Adam optimizer.
#
# Everything is plain dense matrix algebra (W*X .+ b, broadcast activations), so the SAME code
# runs batched on the CPU here and on a CuArray on the GPU with no changes — and, crucially, the
# backward pass is also just matrix algebra, so we can TRAIN on the GPU without any reverse-mode
# AD package (Zygote et al. pull binary artifacts we deliberately avoid; see ../ad.jl).
#
# Layout convention: a batch is a (features x N) matrix, one neuron/sample per column, matching
# the solver state layout so surrogate and simulator interoperate without transposes.

using Random

# swish / SiLU activation and its derivative (matches the JAX reference's jax.nn.swish).
@inline _sigmoid(x) = one(x) / (one(x) + exp(-x))
@inline swish(x) = x * _sigmoid(x)
@inline function dswish(x)               # d/dx [x*sigmoid(x)]
    s = _sigmoid(x)
    return s + x * s * (one(x) - s)
end

"""A dense affine layer y = W*x + b (activation applied by the caller)."""
mutable struct Dense{M<:AbstractMatrix,V<:AbstractVector}
    W::M
    b::V
end

"Glorot/Xavier uniform init; `scale` shrinks the last layer for a small-output head."
function dense(in::Int, out::Int; rng, scale::Float64=1.0, T=Float32)
    lim = T(sqrt(6.0 / (in + out)) * scale)
    W = (rand(rng, T, out, in) .* 2 .- 1) .* lim
    b = zeros(T, out)
    return Dense(W, b)
end

# A stack of trunk layers (swish) plus a final linear head.  We keep the trunk and head separate
# because the affine flow-map wires two heads (F and G) onto a shared trunk.
struct MLP
    trunk::Vector{Dense}   # each followed by swish
    head::Dense            # final linear layer, no activation
end

function mlp(sizes::Vector{Int}; rng, scale_last::Float64=1.0, T=Float32)
    layers = Dense[]
    for i in 1:length(sizes)-2
        push!(layers, dense(sizes[i], sizes[i+1]; rng=rng, T=T))
    end
    head = dense(sizes[end-1], sizes[end]; rng=rng, scale=scale_last, T=T)
    return MLP(layers, head)
end

# Forward through the trunk, caching pre-activations Z and activations A for backprop.
# Returns (H, cache) where H is the trunk output (last activation).
function trunk_forward(layers::Vector{Dense}, X)
    A = X
    Zs = Vector{typeof(X)}(undef, length(layers))
    As = Vector{typeof(X)}(undef, length(layers) + 1)
    As[1] = X
    for (i, L) in enumerate(layers)
        Z = L.W * A .+ L.b
        A = swish.(Z)
        Zs[i] = Z
        As[i+1] = A
    end
    return A, (Zs=Zs, As=As)
end

# Backprop a gradient dH (w.r.t. trunk output) back through the trunk. Accumulates parameter
# grads into gW/gb (vectors of arrays parallel to layers) and returns dX (grad w.r.t. input).
function trunk_backward!(gW, gb, layers::Vector{Dense}, cache, dH)
    dA = dH
    for i in length(layers):-1:1
        L = layers[i]
        dZ = dA .* dswish.(cache.Zs[i])
        gW[i] .+= dZ * cache.As[i]'
        gb[i] .+= vec(sum(dZ, dims=2))
        dA = L.W' * dZ
    end
    return dA
end

# ---- Adam optimizer (self-contained; no Optimisers.jl needed) --------------------------------
mutable struct Adam
    lr::Float64
    b1::Float64
    b2::Float64
    eps::Float64
    wd::Float64        # decoupled weight decay (AdamW)
    t::Int
    m::Any
    v::Any
end
Adam(; lr=2e-3, b1=0.9, b2=0.999, eps=1e-8, wd=1e-5) = Adam(lr, b1, b2, eps, wd, 0, nothing, nothing)

"Global L2 norm of a flat vector of gradient arrays."
global_norm(grads::Vector) = sqrt(sum(g -> sum(abs2, g), grads))

"Rescale grads in place so their global L2 norm is at most `maxnorm` (clip_by_global_norm)."
function clip_global_norm!(grads::Vector, maxnorm::Real)
    gn = global_norm(grads)
    if gn > maxnorm && gn > 0
        s = maxnorm / gn
        for g in grads; g .*= s; end
    end
    return gn
end

# params/grads are flat Vectors of arrays (same shapes). Updates params in place.
# `lr_scale` multiplies the base lr (used for the cosine schedule).
function adam_step!(opt::Adam, params::Vector, grads::Vector; lr_scale::Float64=1.0)
    if opt.m === nothing
        opt.m = [zero(p) for p in params]
        opt.v = [zero(p) for p in params]
    end
    opt.t += 1
    bc1 = 1 - opt.b1^opt.t
    bc2 = 1 - opt.b2^opt.t
    lr = opt.lr * lr_scale
    @inbounds for i in eachindex(params)
        g = grads[i]
        opt.m[i] .= opt.b1 .* opt.m[i] .+ (1 - opt.b1) .* g
        opt.v[i] .= opt.b2 .* opt.v[i] .+ (1 - opt.b2) .* (g .* g)
        mhat = opt.m[i] ./ bc1
        vhat = opt.v[i] ./ bc2
        params[i] .-= lr .* (mhat ./ (sqrt.(vhat) .+ opt.eps) .+ opt.wd .* params[i])
    end
    return opt
end
