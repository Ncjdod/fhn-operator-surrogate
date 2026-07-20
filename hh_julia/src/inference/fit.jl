# Differentiable biophysical inverse problem (reference article §3): recover Hodgkin-Huxley
# parameters (here Na/K channel densities) by gradient descent through the simulator, and design
# neurostimulation via a differentiable spike-probability relaxation of the (otherwise binary)
# stimulus threshold.
#
# Gradients come from the in-house forward-mode Dual (../ad.jl): the parameters we invert are a
# handful of scalars (conductances, and by extension radii/positions), so forward mode over the
# parameter dimension is exact and cheap — the article's stance, with JAXLEY replaced by our own
# differentiable integrator.  Everything is validated by recovering known ground-truth values.

using StaticArrays

# Build a Dual-typed HHClassic whose gNa,gK carry the two seed partials; all other fields are
# constants (zero partial).  Differentiating any output then yields d/d(gNa,gK) directly.
function _hh_dual(base::HHClassic, gNa::TD, gK::TD) where {TD}
    z(x) = TD(x)
    return HHClassic{TD}(z(base.Cm), gNa, gK, z(base.gL), z(base.ENa), z(base.EK), z(base.EL),
                         z(base.u_lo), z(base.u_hi), z(base.fire_lo), z(base.fire_hi), z(base.V0))
end

"""
    fit_conductances(base, x0, Uc, Vtarget; dt, nsub, gNa0, gK0, iters, lr) -> (gNa, gK, history)

Recover (gNa, gK) by minimizing the squared error between the simulated voltage trace and
`Vtarget` (length K+1) via forward-mode-AD gradient descent through the RK4 integrator. `base`
supplies the fixed parameters and the state dimension. Returns the fitted conductances and the
loss history. Reparameterized in log-space so conductances stay positive.
"""
function fit_conductances(base::HHClassic, x0::SVector, Uc::AbstractVector, Vtarget::AbstractVector;
                          dt=0.02, nsub=20, gNa0=100.0, gK0=30.0, iters=200, lr=0.05)
    D = length(x0)
    logθ = SVector(log(gNa0), log(gK0))     # optimize in log-space (positivity)
    hist = Float64[]
    # Adam state
    m = zero(SVector{2,Float64}); v = zero(SVector{2,Float64}); b1=0.9; b2=0.999; eps=1e-8
    for it in 1:iters
        TD = Dual{2,Float64}
        gNa = exp(TD(logθ[1], SVector(1.0, 0.0)))
        gK  = exp(TD(logθ[2], SVector(0.0, 1.0)))
        model = _hh_dual(base, gNa, gK)
        x = SVector{D,TD}(ntuple(i -> TD(x0[i]), Val(D)))
        loss = TD(0.0)
        for k in 1:length(Uc)
            x = rk4_coarse(model, x, TD(Uc[k]), dt, nsub)
            r = x[1] - Vtarget[k+1]
            loss = loss + r * r
        end
        loss = loss / length(Uc)
        g = SVector(partials(loss)[1], partials(loss)[2])   # d loss / d logθ
        # Adam update on logθ
        m = b1 .* m .+ (1-b1) .* g
        v = b2 .* v .+ (1-b2) .* (g .* g)
        mh = m ./ (1-b1^it); vh = v ./ (1-b2^it)
        logθ = logθ .- lr .* mh ./ (sqrt.(vh) .+ eps)
        push!(hist, value(loss))
    end
    return exp(logθ[1]), exp(logθ[2]), hist
end

# ---- differentiable neurostimulation (article §2.4 / §3.1, Appendix D) -----------------------
# Soft peak depolarization over a constant-current simulation (smooth max via log-sum-exp).
@inline function _soft_peak_V(model, x0::SVector{D,T}, I, dt, nsub, nsteps; τ=1.0) where {D,T}
    x = x0
    # running log-sum-exp of V/τ (numerically stabilized incrementally)
    mx = x[1] / τ
    acc = one(T)
    for _ in 1:nsteps
        x = rk4_coarse(model, x, I, dt, nsub)
        z = x[1] / τ
        if z > mx
            acc = acc * exp(mx - z) + one(T)
            mx = z
        else
            acc = acc + exp(z - mx)
        end
    end
    return τ * (mx + log(acc))
end

"""
    spike_probability(model, I; Vthresh, β, dt, nsub, nsteps) -> p ∈ (0,1)

Differentiable relaxation of the binary spike/no-spike outcome for a constant single-electrode
current `I`: a sigmoid of how far the soft peak membrane voltage rises above `Vthresh`. Smooth in
both `I` and the model parameters, so it supports gradient-based threshold matching / stimulus
design (the article's P_stim).
"""
function spike_probability(model::NeuronModel, I::Real; Vthresh=0.0, β=0.3, dt=0.02, nsub=20,
                           nsteps=300)
    x0 = rest_state(model)
    D = length(x0)
    Iv = float(I)
    T = typeof(Iv)
    xx = SVector{D,T}(ntuple(i -> convert(T, x0[i]), Val(D)))
    vp = _soft_peak_V(model, xx, Iv, dt, nsub, nsteps)
    return 1 / (1 + exp(-β * (vp - Vthresh)))
end

# Robust monotone root-find of spike_probability(I) - p_target over the model's current range.
# HH spikes are all-or-none, so P(I) is a near-step; bisection is robust where Newton overshoots.
# (spike_probability stays AD-differentiable in the PARAMETERS, which is what the inference loss
# needs — the threshold *current* itself is found here by bisection.)
function _bisect_current(model, p_target; Vthresh, β, dt, nsub, nsteps, iters=45)
    lo, hi = u_bounds(model)
    plo = value(spike_probability(model, lo; Vthresh=Vthresh, β=β, dt=dt, nsub=nsub, nsteps=nsteps))
    phi = value(spike_probability(model, hi; Vthresh=Vthresh, β=β, dt=dt, nsub=nsub, nsteps=nsteps))
    (p_target <= plo) && return float(lo)
    (p_target >= phi) && return float(hi)
    for _ in 1:iters
        mid = 0.5 * (lo + hi)
        pm = value(spike_probability(model, mid; Vthresh=Vthresh, β=β, dt=dt, nsub=nsub, nsteps=nsteps))
        pm < p_target ? (lo = mid) : (hi = mid)
    end
    return 0.5 * (lo + hi)
end

"""
    stimulus_threshold(model; ...) -> I_threshold

The stimulus amplitude at which `spike_probability == 0.5` (the differentiable stimulus threshold
used as an inference feature and, inverted, as a neurostimulation design target).
"""
stimulus_threshold(model::NeuronModel; Vthresh=0.0, β=0.3, dt=0.02, nsub=20, nsteps=300) =
    _bisect_current(model, 0.5; Vthresh=Vthresh, β=β, dt=dt, nsub=nsub, nsteps=nsteps)

"""
    design_stimulus(model, p_target; ...) -> I

Inverse of the stimulation forward model: the constant current whose spike probability equals
`p_target` (e.g. 0.9 to reliably elicit a spike, 0.1 to stay subthreshold). This is the
neurostimulation "control" use — pick the desired response, get the current that produces it.
"""
design_stimulus(model::NeuronModel, p_target::Real; Vthresh=0.0, β=0.3, dt=0.02, nsub=20,
                nsteps=300) =
    _bisect_current(model, p_target; Vthresh=Vthresh, β=β, dt=dt, nsub=nsub, nsteps=nsteps)
