# Forward + control benchmarks for the multineuronal setting (port of neuron_bench.py).
#
# Run:  julia --project=hh_julia hh_julia/bench/benchmarks.jl
#       julia --project=hh_julia hh_julia/bench/benchmarks.jl --gpu    # if CUDA is installed
#
# The `--gpu` flag moves every array to a CuArray; the SAME kernels then run on the GPU, so the
# numbers below become the GTX 1660 Ti figures.  On CPU the batch sizes are kept modest.

using HHSurrogate
using StaticArrays, Random, Printf

const USE_GPU = "--gpu" in ARGS
const to_dev = if USE_GPU
    @info "attempting to load CUDA…"
    using CUDA
    x -> CUDA.CuArray(x)
else
    identity
end

best_time(f, reps=8) = (f(); minimum(begin t=time(); f(); time()-t end for _ in 1:reps))

function forward_bench(model, D, dt_data, nsub_data; batches=(64,256,1024), horizon=100)
    lo, hi = firing_band(model)
    d = statedim(model)
    println("\n== FORWARD ==  model=$(nameof(typeof(model)))  D=$(round(D,digits=3)) ms  horizon=$(round(horizon*D))ms  device=$(USE_GPU ? "GPU" : "CPU")")
    @printf("  %-8s %10s %10s %10s   %s\n", "batch", "fineRK4 ms", "ROS2 ms", "sur1 ms", "sur speedup×(vs fineRK4)")
    rng = MersenneTwister(0)
    # a random-weight surrogate is fine for *timing* the 1-step forward cost
    s = AffineFlowMap(d; hidden=(128,128), rng=rng, sd=ones(Float32,d), g_floor=0.2f0)
    for B in batches
        X0 = to_dev(Float32.(random_init(model, rng, B)))
        Uc = to_dev(Float32.(clamp.(rand(rng, horizon, B).*(hi-lo).+lo, u_bounds(model)...)))
        t_fine = best_time(() -> rollout_rk4(model, X0, Uc, dt_data, nsub_data; trajectory=false))
        t_ros  = best_time(() -> rollout_rosenbrock(model, X0, Uc, D/8, 8; trajectory=false))
        t_sur  = best_time(() -> begin F,G = FG(s, X0); F .+ G end)   # 1 MLP forward
        @printf("  %-8d %10.2f %10.2f %10.4f   %.1f×\n", B, t_fine*1e3, t_ros*1e3, t_sur*1e3, t_fine/t_sur)
    end
end

function control_bench(model, D, dt_data, nsub_data; batch=128, steps=60)
    lo, hi = firing_band(model); d = statedim(model)
    println("\n== CONTROL ==  steer the TRUE plant to a reference   batch=$batch steps=$steps")
    rng = MersenneTwister(7)
    X0 = to_dev(Float32.(random_init(model, rng, batch)))
    Iref = to_dev(Float32.(clamp.(rand(rng, steps, batch).*(hi-lo).+lo, u_bounds(model)...)))
    Xref = similar(X0, d, steps, batch); X = copy(X0)
    for k in 1:steps
        X = rollout_rk4(model, X, reshape(Iref[k,:],1,:), dt_data, nsub_data; trajectory=false)
        Xref[:,k,:] .= X
    end
    controllers = (
        ("lin1  (1 stiff linearization)", (X,tgt)->control_lin1(model,X,tgt,dt_data,nsub_data), 2*nsub_data),
        ("gn(6) (6 stiff solves)",        (X,tgt)->control_gn(model,X,tgt,dt_data,nsub_data;iters=6), 2*6*nsub_data),
    )
    @printf("  %-32s %10s %14s %12s\n", "controller", "track NRMSE", "vf-evals/step", "wall ms")
    for (name, fn, vf) in controllers
        _, _, trk = closed_loop(model, fn, X0, Xref, dt_data, nsub_data)
        t = best_time(() -> fn(X0, Xref[:,1,:]), 5)
        @printf("  %-32s %10.4g %14d %12.3f\n", name, trk, vf, t*1e3)
    end
    println("  (surrogate controller = 1 MLP forward + closed form, 0 stiff solves — amortizes the above)")
end

function main()
    hh = HHClassic(); dt = 0.02; nsub = 20; D = dt*nsub
    println("HHSurrogate benchmarks — device=$(USE_GPU ? "CUDA" : "CPU")")
    forward_bench(hh, D, dt, nsub)
    control_bench(hh, D, dt, nsub)
    mc = MultiChan(); println("\n----- stiffer 7-D multichannel cell -----")
    forward_bench(mc, D, dt, nsub; batches=(64,256))
end

main()
