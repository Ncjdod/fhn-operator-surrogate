# Full comparison run: train the control-affine surrogate, then measure it against well-optimized
# numerical solvers (fine RK4, Rosenbrock-W) and reproduce the article's multineuronal forward
# model (multi-compartment cable + extracellular EI) and its differentiable inverse.  Every result
# is written to hh_julia/results/data/ as plain CSV; `make_figures.py` turns them into the figures
# and fills in results/COMPARISON.md.
#
#   julia --project=hh_julia hh_julia/scripts/run_comparison.jl [--gpu] [--quick] [--steps N]
#
# --gpu    : move batched arrays to CuArray (requires CUDA) — the GTX 1660 Ti numbers.
# --quick  : tiny config for a fast CPU smoke test of the whole pipeline.

using HHSurrogate
using StaticArrays, Random, Printf, DelimitedFiles

getflag(f) = f in ARGS
function getopt(name, default)
    i = findfirst(==("--$name"), ARGS)
    (i === nothing || i == length(ARGS)) ? default : ARGS[i+1]
end
const QUICK = getflag("--quick")
const USE_GPU = getflag("--gpu")
const NSTEPS = parse(Int, getopt("steps", QUICK ? "300" : "4000"))
const to_dev = if USE_GPU
    @eval using CUDA
    x -> Base.invokelatest(CUDA.CuArray, x)
else
    identity
end
const dev_sync = USE_GPU ? () -> Base.invokelatest(CUDA.synchronize) : () -> nothing
const DEV = USE_GPU ? "CUDA" : "CPU"
const OUT = normpath(joinpath(@__DIR__, "..", "results", getopt("out", "data")))
# The batched arrays are Float32. On the CPU a Float64 model just promotes, but inside a GPU
# kernel the resulting Dual{1,Float32} -> Dual{1,Float64} promotion in `phi_and_sens` is inferred
# dynamically and the control kernels fail to compile. Match the model to the device.
const MT = USE_GPU ? Float32 : Float64

const META = String[]
pushmeta!(k, v) = push!(META, "$k=$v")
wr(name, A; header=nothing) = open(joinpath(OUT, name), "w") do io
    header !== nothing && println(io, header)
    writedlm(io, A, ',')
end
# `time()` is the wall clock and ticks at ~1 ms on Windows, which quantizes every measurement here
# (the sub-ms surrogate step reads as exactly 0). `time_ns()` is the monotonic timer. GPU launches
# are asynchronous, so each timed region also has to be closed with a device synchronize.
best_time(f, reps=(QUICK ? 3 : 8)) =
    (f(); dev_sync();
     minimum(begin t = time_ns(); f(); dev_sync(); (time_ns() - t) / 1e9 end for _ in 1:reps))
Vrow(Y, ch, j) = collect(Y[ch, :, j])

function main()
    mkpath(OUT)
    println("== run_comparison ==  device=$DEV  quick=$QUICK  steps=$NSTEPS")
    rng = MersenneTwister(1234321)
    model = HHClassic{MT}(); d = statedim(model)
    dt = 0.02; stride = 8; D = stride * dt
    N = QUICK ? 96 : 512
    K = QUICK ? 60 : 120
    pushmeta!("device", DEV); pushmeta!("dt", dt); pushmeta!("stride", stride)
    pushmeta!("D_ms", D); pushmeta!("horizon_ms", round(K * D, digits=1))

    # 1) TRAIN ---------------------------------------------------------------------------------
    println("[1/6] training surrogate …")
    X0, Uc, Y = make_dataset(model; N=N, K=K, stride=stride, dt=dt, seed=1)
    mu, sd = standardize_stats(Y)
    s = AffineFlowMap(d; hidden=(128, 128), rng=rng, mu=Array(mu), sd=Array(sd),
                      g_floor=Float32(max(0.05, 0.3D)))
    USE_GPU && to_device!(s, to_dev)   # the weights have to sit on the same device as the batches
    ttrain = @elapsed hist = train!(s, to_dev(Uc), to_dev(Y); steps=NSTEPS, batch=64, rng=rng)
    pushmeta!("train_steps", NSTEPS); pushmeta!("train_seconds", round(ttrain, digits=1))
    @printf("      trained %d steps in %.1f s (%.1f steps/s)\n", NSTEPS, ttrain, NSTEPS / ttrain)
    wr("loss.csv", reshape(hist, :, 1); header="loss")

    # 2) ROLLOUT -------------------------------------------------------------------------------
    println("[2/6] rollout accuracy …")
    Xte, Ute, Yte = make_dataset(model; N=32, K=K, stride=stride, dt=dt, seed=999)
    Yhat = Array(rollout(s, to_dev(Yte[:, 1, :]), to_dev(Ute)))
    nrmse_full = sqrt(sum(((Yhat .- Yte) ./ reshape(Array(sd), d, 1, 1)) .^ 2) / length(Yte))
    pushmeta!("rollout_nrmse", round(nrmse_full, digits=4))
    wr("rollout_t.csv", reshape(collect(0:K) .* D, :, 1); header="t_ms")
    wr("rollout_true.csv", hcat(Vrow(Yte, 1, 1), Vrow(Yte, 1, 2), Vrow(Yte, 1, 3)); header="V1,V2,V3")
    wr("rollout_pred.csv", hcat(Vrow(Yhat, 1, 1), Vrow(Yhat, 1, 2), Vrow(Yhat, 1, 3)); header="V1,V2,V3")

    # 3) FORWARD BENCHMARK ---------------------------------------------------------------------
    println("[3/6] forward benchmark …")
    lo, hi = firing_band(model)
    # CPU wall-clock saturates past ~1e3 neurons on a few cores (a timing artifact, not compute);
    # on GPU there is no such wall — bump these up when running with --gpu to load the card.
    batches = QUICK ? (64, 256) : (USE_GPU ? (256, 1024, 4096, 16384) : (64, 256, 1024))
    rows = Vector{Vector{Float64}}()
    for B in batches
        Xb = to_dev(Float32.(random_init(model, rng, B)))
        Ub = to_dev(Float32.(clamp.(rand(rng, K, B) .* (hi - lo) .+ lo, u_bounds(model)...)))
        tfine = best_time(() -> rollout_rk4(model, Xb, Ub, dt, stride; trajectory=false))
        tros  = best_time(() -> rollout_rosenbrock(model, Xb, Ub, D / 8, 8; trajectory=false))
        # Two surrogate costs, and the difference matters: `sur1step` is ONE coarse step, which is
        # what the per-step-cost claim is about, while the solver columns integrate the whole
        # K-step horizon. Only `surRollout` (K coarse steps, same horizon) is comparable to them,
        # so the speedup column is computed from that one.
        t1    = best_time(() -> FG(s, Xb))
        troll = best_time(() -> rollout(s, Xb, Ub))
        push!(rows, [B, tfine * 1e3, tros * 1e3, t1 * 1e3, troll * 1e3, tfine / troll])
    end
    wr("forward_bench.csv", permutedims(hcat(rows...));
       header="batch,fineRK4_ms,ros2_ms,sur1step_ms,surRollout_ms,speedup")
    pushmeta!("fwd_speedup_max", round(maximum(r[6] for r in rows), digits=1))
    pushmeta!("horizon_steps", K)

    # 3b) ACCURACY-vs-COST PARETO ---------------------------------------------------------------
    # The table above times each method at *its own* accuracy, which flatters the surrogate. The
    # honest comparison is to spend the same wall clock on the classical solvers: hold the coarse
    # step D fixed and give RK4 / Rosenbrock fewer substeps per step, then measure both the error
    # against a converged reference and the time. That puts every method on one accuracy-vs-cost
    # plane, and the surrogate (zero substeps, one MLP forward per step) is just another point.
    println("[3b/6] accuracy-vs-cost pareto …")
    Bp = QUICK ? 64 : 1024
    Xp = to_dev(Float32.(random_init(model, rng, Bp)))
    Up = to_dev(Float32.(clamp.(rand(rng, K, Bp) .* (hi - lo) .+ lo, u_bounds(model)...)))
    # converged reference: 4x the substeps used to generate the training data
    ref = Array(rollout_rk4(model, Xp, Up, dt / 4, stride * 4; trajectory=true))
    sdv = reshape(Array(sd), d, 1, 1)
    perr(A) = sqrt(sum(((Array(A) .- ref) ./ sdv) .^ 2) / length(ref))
    prows = Vector{Vector{Any}}()
    for ns in (1, 2, 4, stride)
        push!(prows, ["rk4", ns, best_time(() -> rollout_rk4(model, Xp, Up, D / ns, ns; trajectory=false)) * 1e3,
                      perr(rollout_rk4(model, Xp, Up, D / ns, ns; trajectory=true))])
    end
    for ns in (1, 2, 4, 8)
        push!(prows, ["ros2", ns, best_time(() -> rollout_rosenbrock(model, Xp, Up, D / ns, ns; trajectory=false)) * 1e3,
                      perr(rollout_rosenbrock(model, Xp, Up, D / ns, ns; trajectory=true))])
    end
    push!(prows, ["surrogate", 0, best_time(() -> rollout(s, Xp, Up)) * 1e3, perr(rollout(s, Xp, Up))])
    wr("pareto.csv", permutedims(hcat(prows...)); header="method,substeps,wall_ms,nrmse")
    pushmeta!("pareto_batch", Bp)

    # 4) CONTROL -------------------------------------------------------------------------------
    println("[4/6] control comparison …")
    Nc, Tc = (QUICK ? 32 : 96), (QUICK ? 40 : 80)
    Xc0 = to_dev(Float32.(random_init(model, rng, Nc)))
    Iref = to_dev(Float32.(clamp.(rand(rng, Tc, Nc) .* (hi - lo) .+ lo, u_bounds(model)...)))
    Xref = similar(Xc0, d, Tc, Nc)
    Xcur = copy(Xc0)
    for k in 1:Tc
        Xcur = rollout_rk4(model, Xcur, reshape(Iref[k, :], 1, :), dt, stride; trajectory=false)
        Xref[:, k, :] .= Xcur
    end
    ctrls = [
        ("surrogate", (X, t) -> invert(s, X, t; clip=u_bounds(model))[1], 0),
        ("lin1",      (X, t) -> control_lin1(model, X, t, dt, stride),    2 * stride),
        ("gn6",       (X, t) -> control_gn(model, X, t, dt, stride; iters=6), 2 * 6 * stride),
    ]
    csum = Vector{Vector{Any}}()
    traces = Dict{String,Vector{Float64}}()
    for (name, fn, solves) in ctrls
        tr, us, trk = closed_loop(model, fn, Xc0, Xref, dt, stride)
        twall = best_time(() -> fn(Xc0, Xref[:, 1, :]), QUICK ? 3 : 5)
        push!(csum, [name, round(trk, digits=5), solves, round(twall * 1e3, digits=3)])
        traces[name] = Vrow(Array(tr), 1, 1)
    end
    wr("control_summary.csv", permutedims(hcat(csum...)); header="controller,track_nrmse,stiff_substeps_per_step,wall_ms")
    wr("control_trace.csv",
        hcat(collect(1:Tc) .* D, Vrow(Array(Xref), 1, 1), traces["surrogate"], traces["lin1"], traces["gn6"]);
        header="t_ms,ref_V,surrogate_V,lin1_V,gn6_V")

    # 5) CABLE + EI ----------------------------------------------------------------------------
    println("[5/6] cable + electrical image …")
    P = QUICK ? 40 : 60
    ch = HHCableChannels()
    geom = straight_axon_geometry(P; length_um=20.0, radius_um=1.0, Ra=100.0, z_um=-20.0)
    st = cable_rest(ch, P, 1; V0=-65.0); st.V[1:6, 1] .= 20.0
    dtc = 0.025; nsteps = Int(round((QUICK ? 8.0 : 12.0) / dtc))
    Vtr, Imtr = simulate_cable(ch, geom, st, k -> zeros(P, 1), dtc, nsteps)
    wr("cable_V.csv", Vtr[:, :, 1])
    elec = hex_electrode_patch(; spacing_um=30.0, cx=geom.pos[1, P ÷ 2])
    Φ = electrical_image(Imtr, geom.pos, elec)
    wr("ei_waveforms.csv", hcat(collect(0:nsteps) .* dtc, permutedims(Φ[:, :, 1])); header="t_ms,e1,e2,e3,e4,e5,e6,e7")
    f = ei_features(Φ[:, :, 1], dtc)
    wr("ei_features.csv", hcat(1:7, f.A_cap, f.A_Na, f.A_K, f.dur, f.prop); header="elec,A_cap,A_Na,A_K,dur,prop")
    tpk = [argmax(view(Vtr, i, :, 1)) for i in 15:P-5]
    vel = (geom.pos[1, P-5] - geom.pos[1, 15]) / ((tpk[end] - tpk[1]) * dtc) / 1000
    pushmeta!("cable_velocity_mps", round(vel, digits=3)); pushmeta!("cable_dt_ms", dtc)
    pushmeta!("ei_three_phase", f.A_cap[1] > 0 && f.A_Na[1] < 0 && f.A_K[1] > 0)

    # 6) DIFFERENTIABLE INVERSE ----------------------------------------------------------------
    println("[6/6] differentiable inverse …")
    base = HHClassic()
    truths = [(110.0, 30.0), (90.0, 45.0), (140.0, 25.0), (75.0, 40.0), (120.0, 36.0), (100.0, 50.0)]
    rec = Vector{Vector{Float64}}()
    for (gna, gk) in truths
        truth = HHClassic(gNa=gna, gK=gk); x0 = rest_state(truth)
        Uc2 = clamp.(rand(rng, 50) .* 20 .+ 5, u_bounds(truth)...)
        Vt = Float64[x0[1]]; xx = x0
        for k in 1:length(Uc2); xx = rk4_coarse(truth, xx, Uc2[k], dt, stride); push!(Vt, xx[1]); end
        rg, rk_, _ = fit_conductances(base, x0, Uc2, Vt; dt=dt, nsub=stride, gNa0=80.0, gK0=45.0,
                                      iters=(QUICK ? 200 : 300), lr=0.05)
        push!(rec, [gna, gk, rg, rk_])
    end
    wr("inverse_recovery.csv", permutedims(hcat(rec...)); header="true_gNa,true_gK,rec_gNa,rec_gK")
    m = HHClassic()
    Is = collect(-2.0:0.25:12.0)
    Ps = [value(spike_probability(m, I; β=0.3, nsteps=(QUICK ? 150 : 300))) for I in Is]
    wr("pi_curve.csv", hcat(Is, Ps); header="I,P")
    pushmeta!("stim_threshold", round(stimulus_threshold(m; β=0.3, nsteps=(QUICK ? 150 : 300)), digits=3))
    pushmeta!("recovery_max_gNa_err", round(maximum(abs(r[3] - r[1]) for r in rec), digits=3))

    open(joinpath(OUT, "meta.txt"), "w") do io
        for line in META; println(io, line); end
    end
    println("\nwrote results to $OUT\nnext:  python hh_julia/results/make_figures.py")
end

main()
