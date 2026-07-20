# End-to-end showcases of the two project pillars:
#   (1) multineuronal forward model  — propagate a spike along a multi-compartment cable and
#       synthesize its extracellular electrical image (EI) on a 7-electrode patch;
#   (2) inverse / control            — steer the true HH plant to a reference, recover unknown
#       channel densities, and design a neurostimulation current for a target spike probability.
#
#   julia --project=hh_julia hh_julia/scripts/demos.jl

using HHSurrogate
using StaticArrays, Random, Printf

function demo_cable_ei()
    println("\n=== (1) multi-compartment cable + extracellular EI ===")
    P = 60; ch = HHCableChannels()
    geom = straight_axon_geometry(P; length_um=20.0, radius_um=1.0, Ra=100.0, z_um=-20.0)
    st = cable_rest(ch, P, 1; V0=-65.0); st.V[1:6, 1] .= 20.0     # proximal suprathreshold init
    dt = 0.025; nsteps = Int(round(12.0/dt))
    Vtr, Imtr = simulate_cable(ch, geom, st, k->zeros(P,1), dt, nsteps)
    vel_tpk = [argmax(view(Vtr,i,:,1)) for i in 15:55]
    xs = [geom.pos[1,i] for i in 15:55]
    vel = (xs[end]-xs[1]) / ((vel_tpk[end]-vel_tpk[1])*dt) / 1000
    @printf("  spike propagated over %d compartments, distal Vpeak=%.1f mV, conduction velocity≈%.2f m/s\n",
            P, maximum(Vtr[55,:,1]), vel)
    elec = hex_electrode_patch(; spacing_um=30.0, cx=geom.pos[1, P÷2])
    Φ = electrical_image(Imtr, geom.pos, elec)
    f = ei_features(Φ[:,:,1], dt)
    @printf("  EI center electrode 3-phase: A_cap=%+.2f  A_Na=%+.2f  A_K=%+.2f µV   duration=%.2f ms\n",
            f.A_cap[1], f.A_Na[1], f.A_K[1], f.dur[1])
    @printf("  Na-peak propagation delays across 7 electrodes (ms): %s\n", string(round.(f.prop, digits=2)))
end

function demo_control()
    println("\n=== (2a) closed-loop control: steer the true HH plant to a reference ===")
    hh = HHClassic(); dt = 0.02; nsub = 20
    rng = MersenneTwister(3); N = 64; T = 80
    lo, hi = firing_band(hh)
    X0 = Float32.(random_init(hh, rng, N))
    Iref = Float32.(clamp.(rand(rng, T, N).*(hi-lo).+lo, u_bounds(hh)...))
    Xref = Array{Float32}(undef, 4, T, N); X = copy(X0)
    for k in 1:T
        X = rollout_rk4(hh, X, reshape(Iref[k,:],1,:), dt, nsub; trajectory=false)
        Xref[:,k,:] .= X
    end
    _, us_l, trk_l = closed_loop(hh, (X,t)->control_lin1(hh,X,t,dt,nsub), X0, Xref, dt, nsub)
    _, _,    trk_g = closed_loop(hh, (X,t)->control_gn(hh,X,t,dt,nsub;iters=6), X0, Xref, dt, nsub)
    @printf("  tracking NRMSE: lin1=%.4g  gn(6)=%.4g   (current recovery |u-Iref| mean lin1=%.2f)\n",
            trk_l, trk_g, sum(abs, us_l .- Iref)/length(Iref))
end

function demo_inference()
    println("\n=== (2b) differentiable inverse: recover channel densities from a trace ===")
    truth = HHClassic(gNa=110.0, gK=30.0); base = HHClassic()
    x0 = rest_state(truth); dt = 0.02; nsub = 20; rng = MersenneTwister(0)
    Uc = clamp.(rand(rng, 50).*20 .+ 5, u_bounds(truth)...)
    Vt = Float64[x0[1]]; x = x0
    for k in 1:length(Uc); x = rk4_coarse(truth, x, Uc[k], dt, nsub); push!(Vt, x[1]); end
    gNa, gK, _ = fit_conductances(base, x0, Uc, Vt; gNa0=80.0, gK0=45.0, iters=250, lr=0.05)
    @printf("  recovered gNa=%.2f (true 110)  gK=%.2f (true 30)\n", gNa, gK)

    println("\n=== (2c) neurostimulation design: current for a target spike probability ===")
    m = HHClassic(); Ith = stimulus_threshold(m; β=0.3, nsteps=250)
    @printf("  stimulus threshold (P=0.5): I≈%.2f µA/cm²\n", Ith)
    for pt in (0.1, 0.9)
        I = design_stimulus(m, pt; β=0.3, nsteps=250)
        @printf("  target spike prob %.1f  ->  design current I=%.2f µA/cm²\n", pt, I)
    end
end

demo_cable_ei()
demo_control()
demo_inference()
println("\nall demos completed.")
