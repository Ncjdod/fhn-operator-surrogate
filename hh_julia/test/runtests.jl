# HHSurrogate test suite — runs entirely on the CPU (KernelAbstractions CPU backend), no GPU
# and no binary-artifact dependencies required.  Every check that mattered during development is
# encoded here: model correctness vs an independent NumPy oracle, the in-house AD, the batched
# solvers, the surrogate's closed-form inverse and BPTT gradients, closed-loop control on the
# true plant, cable propagation + three-phase EI, and the differentiable biophysical inverse.

using HHSurrogate
using StaticArrays
using Random
using LinearAlgebra
using Test

include("fixtures.jl")

maxerr(a, b) = maximum(abs.(collect(a) .- collect(b)))

@testset "HHSurrogate" begin

    @testset "models vs NumPy oracle" begin
        hh = HHClassic()
        @test maxerr(rest_state(hh), HH_REST) < 1e-12
        for (x, u, f) in HH_PROBES
            @test maxerr(vfield(hh, SVector{4}(x...), u), f) < 1e-10
        end
        # one big coarse step == fine RK4 chain of the oracle (20 ms @ dt=0.01)
        x0 = reshape(collect(rest_state(hh)), 4, 1)
        yf = rollout_rk4(hh, x0, reshape([10.0], 1, 1), 0.01, 2000; trajectory=false)
        @test maxerr(vec(yf), HH_RK4_20MS_I10) < 1e-9

        mc = MultiChan()
        @test maxerr(rest_state(mc), MC_REST) < 1e-12
        x0m = reshape(collect(rest_state(mc)), 7, 1)
        yfm = rollout_rk4(mc, x0m, reshape([25.0], 1, 1), 0.005, 4000; trajectory=false)
        @test maxerr(vec(yfm), MC_RK4_20MS_I25) < 1e-8
    end

    @testset "RGC channels (article Fohlmeister kinetics)" begin
        rgc = RGCChannels()
        d0 = vfield(rgc, rest_state(rgc), 0.0)
        @test maximum(abs.(d0)) < 0.2                 # near-equilibrium at rest
        @test all(isfinite, rest_state(rgc))
        @test statedim(rgc) == 6
    end

    @testset "in-house AD Jacobian vs finite differences" begin
        hh = HHClassic()
        x = SVector(-20.0, 0.5, 0.3, 0.5); u = 20.0
        y = vfield(hh, seed(x), u)
        J = extract_jacobian(y)
        fd = SMatrix{4,4}(ntuple(idx -> begin
            i = ((idx - 1) % 4) + 1; k = ((idx - 1) ÷ 4) + 1
            e = 1e-6; xp = setindex(x, x[k] + e, k); xm = setindex(x, x[k] - e, k)
            (vfield(hh, xp, u)[i] - vfield(hh, xm, u)[i]) / (2e)
        end, 16))
        @test maximum(abs.(J .- fd)) < 1e-5
    end

    @testset "Rosenbrock stiff solver (subthreshold convergence)" begin
        hh = HHClassic()
        x0 = reshape(collect(rest_state(hh)), 4, 1)
        ref = rollout_rk4(hh, x0, reshape([1.5], 1, 1), 50.0 / 500000, 500000; trajectory=false)
        errs = [abs(rollout_rosenbrock(hh, x0, reshape([1.5], 1, 1), 50.0 / ns, ns; trajectory=false)[1] - ref[1])
                for ns in (250, 500, 1000)]
        @test issorted(errs; rev=true)               # error shrinks with step
        @test errs[end] < 1e-3
    end

    @testset "affine flow-map: closed-form inverse optimality" begin
        rng = MersenneTwister(1); d = 4; N = 6
        s = AffineFlowMap(d; hidden=(8, 8), rng=rng, sd=Float32[20, .2, .2, .2], g_floor=0.1f0)
        X = randn(rng, Float32, d, N); Xtgt = randn(rng, Float32, d, N) .* 10f0
        u, Xn, r = invert(s, X, Xtgt)
        F, G = FG(s, X)
        @test maximum(abs.(vec(sum(G .* (Xtgt .- Xn), dims=1)))) < 1e-4   # residual ⟂ G
    end

    @testset "affine flow-map: BPTT gradient check (Float64)" begin
        rng = MersenneTwister(1); d = 4; N = 5; K = 3
        s = AffineFlowMap(d; hidden=(8, 8), rng=rng, sd=Float64[20, .2, .2, .2], g_floor=0.1, T=Float64)
        X0 = randn(rng, d, N); Uc = randn(rng, K, N) .* 5.0
        Y = randn(rng, d, K + 1, N) .* [20, .2, .2, .2]
        _, grads = loss_and_grads(s, X0, Uc, Y)
        ps = params(s); maxrel = 0.0
        for (pi, p) in enumerate(ps), idx in 1:min(length(p), 5)
            old = p[idx]; e = 1e-5
            p[idx] = old + e; lp, _ = loss_and_grads(s, X0, Uc, Y)
            p[idx] = old - e; lm, _ = loss_and_grads(s, X0, Uc, Y); p[idx] = old
            fd = (lp - lm) / (2e)
            maxrel = max(maxrel, abs(fd - grads[pi][idx]) / (abs(fd) + abs(grads[pi][idx]) + 1e-10))
        end
        @test maxrel < 1e-4
    end

    @testset "closed-loop control on the true HH plant" begin
        hh = HHClassic(); dt = 0.02; nsub = 20
        rng = MersenneTwister(3); N = 16; T = 40
        lo, hi = firing_band(hh)
        X0 = Float32.(random_init(hh, rng, N))
        Iref = Float32.(clamp.(rand(rng, T, N) .* (hi - lo) .+ lo, u_bounds(hh)...))
        Xref = Array{Float32}(undef, 4, T, N); X = copy(X0)
        for k in 1:T
            X = rollout_rk4(hh, X, reshape(Iref[k, :], 1, :), dt, nsub; trajectory=false)
            Xref[:, k, :] .= X
        end
        _, _, trk_gn = closed_loop(hh, (X, tgt) -> control_gn(hh, X, tgt, dt, nsub; iters=6), X0, Xref, dt, nsub)
        _, _, trk_l1 = closed_loop(hh, (X, tgt) -> control_lin1(hh, X, tgt, dt, nsub), X0, Xref, dt, nsub)
        @test trk_gn < 1e-4        # Gauss-Newton inverts the true plant essentially exactly
        @test trk_l1 < 0.05        # one-shot linearization tracks well
    end

    @testset "multi-compartment cable: propagation + three-phase EI" begin
        P = 60; ch = HHCableChannels()
        geom = straight_axon_geometry(P; length_um=20.0, radius_um=1.0, Ra=100.0, z_um=-20.0)
        st = cable_rest(ch, P, 1; V0=-65.0); st.V[1:6, 1] .= 20.0
        dt = 0.025; nsteps = Int(round(12.0 / dt))
        Vtr, Imtr = simulate_cable(ch, geom, st, k -> zeros(P, 1), dt, nsteps)
        @test all(isfinite, Vtr)
        @test all(maximum(Vtr[i, :, 1]) > 0 for i in 10:55)      # spike propagates
        tpk = [argmax(view(Vtr, i, :, 1)) for i in 8:55]
        @test issorted(tpk)                                       # monotone arrival
        elec = hex_electrode_patch(; spacing_um=30.0, cx=geom.pos[1, P ÷ 2])
        Φ = electrical_image(Imtr, geom.pos, elec)
        f = ei_features(Φ[:, :, 1], dt)
        @test f.A_cap[1] > 0 && f.A_Na[1] < 0 && f.A_K[1] > 0     # capacitive/sodium/potassium
    end

    @testset "differentiable inverse: parameter recovery + stimulus design" begin
        truth = HHClassic(gNa=110.0, gK=30.0); base = HHClassic()
        x0 = rest_state(truth); dt = 0.02; nsub = 20
        rng = MersenneTwister(0)
        Uc = clamp.(rand(rng, 40) .* 20 .+ 5, u_bounds(truth)...)
        Vt = Float64[x0[1]]; x = x0
        for k in 1:length(Uc); x = rk4_coarse(truth, x, Uc[k], dt, nsub); push!(Vt, x[1]); end
        gNa, gK, hist = fit_conductances(base, x0, Uc, Vt; dt=dt, nsub=nsub, gNa0=80.0, gK0=45.0,
                                         iters=250, lr=0.05)
        @test abs(gNa - 110.0) < 1.0 && abs(gK - 30.0) < 1.0
        @test hist[end] < 1e-4

        m = HHClassic()
        Ilo = design_stimulus(m, 0.1; β=0.3, nsteps=250)
        Ihi = design_stimulus(m, 0.9; β=0.3, nsteps=250)
        @test value(spike_probability(m, Ilo; β=0.3, nsteps=250)) < 0.2
        @test value(spike_probability(m, Ihi; β=0.3, nsteps=250)) > 0.8
    end

end
