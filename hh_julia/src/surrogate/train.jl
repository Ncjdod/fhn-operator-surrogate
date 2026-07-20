# Curriculum BPTT training for the control-affine flow-map (port of flowmap_affine_train.py).
#
# We grow the rollout length K across stages (1 -> a few -> many) so the stepper first learns a
# good one-step map, then long-horizon stability, exactly as in the JAX reference. Each minibatch
# samples random trajectories and a random start offset, then trains on that K-step window.
# Loss/grads come from affine.jl's hand-written BPTT, so this runs unchanged on CPU or GPU.

using Random

"""
    train!(s, Uc, Y; stages, steps, batch, lr, rng, verbose) -> loss_history

Trains the surrogate `s` in place on a coarse dataset `Uc` (Kfull, N), `Y` (d, Kfull+1, N).
`stages` is a vector of (K, fraction) pairs describing the curriculum.
"""
function train!(s::AffineFlowMap, Uc::AbstractMatrix, Y::AbstractArray{<:Any,3};
                stages=[(1, 0.15), (4, 0.15), (16, 0.20), (64, 0.25), (128, 0.25)],
                steps::Int=6000, batch::Int=64, lr::Float64=2e-3, lambda_cond::Float64=0.1,
                clip::Float64=1.0, wd::Float64=1e-5, rng=MersenneTwister(0), verbose::Bool=true)
    d, Kfull1, N = size(Y)
    Kfull = Kfull1 - 1
    opt = Adam(lr=lr, wd=wd)
    ps = params(s)
    hist = Float64[]
    i = 0
    for (Kstage, frac) in stages
        K = min(Kstage, Kfull)
        nstep = round(Int, frac * steps)
        for _ in 1:nstep
            idx = rand(rng, 1:N, batch)
            t0 = Kfull - K > 0 ? rand(rng, 1:(Kfull - K + 1)) : 1
            X0 = Y[:, t0, idx]
            Ucw = Uc[t0:t0+K-1, idx]
            Yw = Y[:, t0:t0+K, idx]
            loss, grads = loss_and_grads(s, X0, Ucw, Yw; lambda_cond=lambda_cond)
            clip_global_norm!(grads, clip)               # stabilize long-horizon BPTT
            lr_scale = 0.5 * (1 + cos(pi * i / max(1, steps)))  # cosine decay to ~0
            lr_scale = 0.01 + 0.99 * lr_scale
            adam_step!(opt, ps, grads; lr_scale=lr_scale)
            push!(hist, loss)
            if verbose && i % max(1, steps ÷ 20) == 0
                println("  step $(lpad(i,5)) K=$(lpad(K,3))  loss=$(round(loss, sigdigits=5))")
            end
            i += 1
        end
    end
    return hist
end

"Standardization stats (mean, std) per channel from a truth trajectory Y (d, T, N)."
function standardize_stats(Y::AbstractArray{<:Any,3})
    d = size(Y, 1)
    Yf = reshape(Y, d, :)
    mu = vec(sum(Yf, dims=2)) ./ size(Yf, 2)
    va = vec(sum((Yf .- mu) .^ 2, dims=2)) ./ size(Yf, 2)
    sd = sqrt.(va) .+ 1f-6
    return Float32.(mu), Float32.(sd)
end
