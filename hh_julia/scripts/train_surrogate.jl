# Train a control-affine flow-map surrogate on a Hodgkin-Huxley (or multichannel) cell and save
# it.  On the GPU pass --gpu (requires CUDA); the training arrays and BPTT then run on the device.
#
#   julia --project=hh_julia hh_julia/scripts/train_surrogate.jl [--model hh|mc] [--steps N] [--gpu]

using HHSurrogate
using StaticArrays, Random, Printf, Serialization

function parse_args(argv)
    d = Dict{String,String}(); i = 1
    while i <= length(argv)
        key = argv[i][3:end]
        if startswith(argv[i], "--") && i < length(argv) && !startswith(argv[i+1], "--")
            d[key] = argv[i+1]; i += 2
        else
            d[key] = "true"; i += 1
        end
    end
    return d
end
args = parse_args(ARGS)
modelname = get(args, "model", "hh")
nsteps    = parse(Int, get(args, "steps", "4000"))
use_gpu   = get(args, "gpu", "false") == "true"

model = modelname == "mc" ? MultiChan() : HHClassic()
d = statedim(model)
stride, dt = 20, 0.02; D = stride * dt
println("training surrogate: model=$modelname d=$d  D=$D ms  steps=$nsteps  device=$(use_gpu ? "GPU" : "CPU")")

rng = MersenneTwister(1)
X0, Uc, Y = make_dataset(model; N=512, K=150, stride=stride, dt=dt, seed=1)
mu, sd = standardize_stats(Y)

# move to device if requested
if use_gpu
    using CUDA
    Uc = CuArray(Uc); Y = CuArray(Y); mu = CuArray(mu); sd = CuArray(sd)
end

s = AffineFlowMap(d; hidden=(128,128), rng=rng, mu=Array(mu), sd=Array(sd),
                  g_floor=Float32(max(0.05, 0.3*D)))
hist = train!(s, Uc, Y; steps=nsteps, batch=64, rng=rng)

# held-out full-rollout NRMSE
Xte, Ute, Yte = make_dataset(model; N=64, K=150, stride=stride, dt=dt, seed=999)
Yhat = rollout(s, Yte[:,1,:], Ute)
nrmse = sqrt(sum(((Yhat .- Yte) ./ reshape(Array(sd), d, 1, 1)).^2) / length(Yte))
@printf("done. first-loss=%.4g last-loss=%.4g  held-out full-rollout NRMSE=%.4f\n", hist[1], hist[end], nrmse)

out = joinpath(@__DIR__, "..", "affine_$(modelname).jls")
serialize(out, (params=[Array(p) for p in params(s)], mu=Array(mu), sd=Array(sd),
                d=d, hidden=(128,128), g_floor=s.g_floor, D=D, dt=dt, stride=stride, model=modelname))
println("saved $out")
