# Conductance-based neuron models, control-affine in the injected current.
#
# Every model here writes  dx/dt = f(x) + b*u  with the injected current u entering ONLY
# the voltage channel through b = e_1 / C_m.  That exact affine-in-u structure is what makes
# the closed-form control inverse (see ../control) well posed, and it matches the stimulus
# term I_stim in the Hodgkin-Huxley membrane equation (Lotlikar et al. 2026, Eq. 1/17).
#
# States are SVectors so a vector-field evaluation is fully stack-allocated and inlines into
# the batched RK4 / Rosenbrock kernels with no heap traffic — this is what lets one GPU
# thread integrate one neuron. Parameters live in immutable structs whose fields are plain
# scalars, so ForwardDiff.Dual flows through them and the whole simulator is differentiable
# w.r.t. conductances/geometry (the inverse-problem requirement).

using StaticArrays

abstract type NeuronModel end

"State dimension d."
@inline statedim(m::NeuronModel) = length(rest_state(m))

"Injected-current bounds (u_lo, u_hi) used by data generation and control saturation."
@inline u_bounds(m::NeuronModel) = (m.u_lo, m.u_hi)

"Firing band of injected currents that produce tonic spiking (for stimulus sampling)."
@inline firing_band(m::NeuronModel) = (m.fire_lo, m.fire_hi)

"Index of the voltage channel (always 1 here) and the affine input gain b_V = 1/C_m."
@inline voltage_channel(::NeuronModel) = 1
@inline input_gain(m::NeuronModel) = inv(m.Cm)

# x/(exp(x)-1) with the removable singularity at x=0 handled (-> 1 - x/2).  Written so it is
# type-stable under ForwardDiff.Dual and never divides by ~0 near the HH rate singularities.
@inline function safe_exprel(x::T) where {T}
    return abs(x) < T(1e-4) ? one(T) - x / 2 : x / expm1(x)
end

# ---------------------------------------------------------------------------------------------
# Classic Hodgkin-Huxley (1952 squid axon).  Direct port of hh_model.py.  State x=(V,m,h,n).
# ---------------------------------------------------------------------------------------------
Base.@kwdef struct HHClassic{T} <: NeuronModel
    Cm::T = 1.0
    gNa::T = 120.0
    gK::T = 36.0
    gL::T = 0.3
    ENa::T = 50.0
    EK::T = -77.0
    EL::T = -54.387
    u_lo::T = -5.0
    u_hi::T = 60.0
    fire_lo::T = 2.0
    fire_hi::T = 35.0
    V0::T = -65.0
end

@inline _am_hh(V) = safe_exprel(-(V + 40) / 10)
@inline _bm_hh(V) = 4 * exp(-(V + 65) / 18)
@inline _ah_hh(V) = 7 // 100 * exp(-(V + 65) / 20)
@inline _bh_hh(V) = inv(1 + exp(-(V + 35) / 10))
@inline _an_hh(V) = 1 // 10 * safe_exprel(-(V + 55) / 10)
@inline _bn_hh(V) = 125 // 1000 * exp(-(V + 65) / 80)

@inline function vfield(p::HHClassic, x::SVector{4}, u)
    V, m, h, n = x[1], x[2], x[3], x[4]
    iNa = p.gNa * m^3 * h * (V - p.ENa)
    iK  = p.gK * n^4 * (V - p.EK)
    iL  = p.gL * (V - p.EL)
    dV = (u - iNa - iK - iL) / p.Cm
    dm = _am_hh(V) * (1 - m) - _bm_hh(V) * m
    dh = _ah_hh(V) * (1 - h) - _bh_hh(V) * h
    dn = _an_hh(V) * (1 - n) - _bn_hh(V) * n
    return SVector(dV, dm, dh, dn)
end

@inline function gate_inf(p::HHClassic, V)
    m = _am_hh(V) / (_am_hh(V) + _bm_hh(V))
    h = _ah_hh(V) / (_ah_hh(V) + _bh_hh(V))
    n = _an_hh(V) / (_an_hh(V) + _bn_hh(V))
    return SVector(m, h, n)
end

function rest_state(p::HHClassic{T}) where {T}
    g = gate_inf(p, p.V0)
    return SVector{4,T}(p.V0, g[1], g[2], g[3])
end

# ---------------------------------------------------------------------------------------------
# Multi-channel fast-spiking cell (7-D).  Direct port of multichan_model.py.
# HH (Na, Kd, leak) + I_M (slow non-inactivating K, spike-frequency adaptation)
# + I_A (transient A-type K).  A kinetic speed-up PHI makes it genuinely stiff.
# State x = (V, m, h, n, p, a, b).
# ---------------------------------------------------------------------------------------------
Base.@kwdef struct MultiChan{T} <: NeuronModel
    Cm::T = 1.0
    gNa::T = 120.0
    gK::T = 36.0
    gL::T = 0.3
    gM::T = 1.0
    gA::T = 20.0
    ENa::T = 50.0
    EK::T = -77.0
    EL::T = -54.387
    phi::T = 1.5
    u_lo::T = -5.0
    u_hi::T = 40.0
    fire_lo::T = 18.0
    fire_hi::T = 32.0
    V0::T = -70.0
end

@inline _p_inf(V) = inv(1 + exp(-(V + 35) / 10))
@inline _p_tau(V) = 100 / (33 // 10 * exp((V + 35) / 20) + exp(-(V + 35) / 20))
@inline _a_inf(V) = inv(1 + exp(-(V + 50) / 20))
@inline _a_tau(V) = 1 // 2 + 3 // 2 * inv(1 + exp((V + 40) / 10))
@inline _b_inf(V) = inv(1 + exp((V + 80) / 6))
@inline _b_tau(V) = 8 + 12 * inv(1 + exp((V + 55) / 10))

@inline function vfield(pp::MultiChan, x::SVector{7}, u)
    V, m, h, n, p, a, b = x[1], x[2], x[3], x[4], x[5], x[6], x[7]
    iNa = pp.gNa * m^3 * h * (V - pp.ENa)
    iK  = pp.gK * n^4 * (V - pp.EK)
    iL  = pp.gL * (V - pp.EL)
    iM  = pp.gM * p * (V - pp.EK)
    iA  = pp.gA * a^3 * b * (V - pp.EK)
    dV = (u - iNa - iK - iL - iM - iA) / pp.Cm
    dm = pp.phi * (_am_hh(V) * (1 - m) - _bm_hh(V) * m)
    dh = pp.phi * (_ah_hh(V) * (1 - h) - _bh_hh(V) * h)
    dn = pp.phi * (_an_hh(V) * (1 - n) - _bn_hh(V) * n)
    dp = (_p_inf(V) - p) / _p_tau(V)
    da = pp.phi * (_a_inf(V) - a) / _a_tau(V)
    db = (_b_inf(V) - b) / _b_tau(V)
    return SVector(dV, dm, dh, dn, dp, da, db)
end

function rest_state(pp::MultiChan{T}) where {T}
    V = pp.V0
    m = _am_hh(V) / (_am_hh(V) + _bm_hh(V))
    h = _ah_hh(V) / (_ah_hh(V) + _bh_hh(V))
    n = _an_hh(V) / (_an_hh(V) + _bn_hh(V))
    return SVector{7,T}(V, m, h, n, _p_inf(V), _a_inf(V), _b_inf(V))
end

# ---------------------------------------------------------------------------------------------
# Retinal ganglion cell channels (Fohlmeister-Miller / Kish et al. 2023), as used by the
# reference article (Lotlikar et al. 2026, Appendix L, Tables 4-5).  Na + Kd + Ca + K,Ca + pas.
# State x = (V, m, h, n, c, Ca_i).  gbar_* are per-unit-area maximal conductances (S/cm^2);
# the multi-compartment cable (../models/cable.jl) supplies per-compartment gbar and surface
# area, so this single-compartment version is the point kinetics reused at every segment.
#
# Rate constants are Table 5 (35 C).  Calcium reversal E_Ca is a Nernst potential that tracks
# the intracellular calcium pool Ca_i (Table 4); a standard first-order pool balances Ca influx
# against extrusion.  These calcium details are the one place the paper defers to its cited
# sources rather than printing every constant, so the pool parameters below are the widely used
# Fohlmeister-Miller values and are collected here as named fields for transparency/tuning.
# ---------------------------------------------------------------------------------------------
# Conductances are stored in mS/cm^2 to keep the same (V in mV, t in ms, Cm in uF/cm^2,
# current density in uA/cm^2) unit system as the classic-HH port, so every solver/kernel is
# unit-consistent across models.  The article/Kish Tables report gbar in S/cm^2; multiply by
# 1000 to get these fields (e.g. SOCB gNa 0.25 S/cm^2 -> 250 mS/cm^2, leak 1e-4 -> 0.1).
Base.@kwdef struct RGCChannels{T} <: NeuronModel
    Cm::T = 1.0            # uF/cm^2
    gNa::T = 250.0         # mS/cm^2  (= 0.25 S/cm^2, SOCB baseline, Table 6/8)
    gK::T = 150.0          # = 0.15 S/cm^2
    gCa::T = 0.75          # = 7.5e-4 S/cm^2
    gKCa::T = 0.17         # = 1.7e-4 S/cm^2
    gpas::T = 0.1          # = 1e-4 S/cm^2
    ENa::T = 60.60         # mV (Table 4)
    EK::T = -101.34
    Epas::T = -64.58
    Ca_e::T = 1.8          # mM external Ca (Fohlmeister-Miller)
    Ca_rest::T = 1e-4      # mM resting internal Ca
    Ca_tau::T = 50.0       # ms extrusion time constant
    Ca_gain::T = 1e-4      # mM per (uA/cm^2) per ms: lumped influx factor (I_Ca -> dCa_i)
    Ca_diss::T = 1e-3      # mM half-activation of the Ca-gated K conductance (Hill n=2)
    RT_2F::T = 13.35       # mV, (R*T)/(2F) at 35 C -> E_Ca = RT_2F*ln(Ca_e/Ca_i)
    u_lo::T = -5.0
    u_hi::T = 60.0
    fire_lo::T = 2.0
    fire_hi::T = 30.0
    V0::T = -64.58
end

# Fohlmeister rate functions (Table 5).  safe_exprel handles the /(exp(-0.1(V+x))-1) singularities.
@inline _am_rgc(V) = 2725 // 1000 * safe_exprel(-(V + 35) / 10)          # -2.725(V+35)/(e^{-0.1(V+35)}-1)
@inline _bm_rgc(V) = 9083 // 100 * exp(-(V + 60) / 20)
@inline _ah_rgc(V) = 1817 // 1000 * exp(-(V + 52) / 20)
@inline _bh_rgc(V) = 2725 // 100 * inv(1 + exp(-(V + 22) / 10))
@inline _an_rgc(V) = 9575 // 100000 * safe_exprel(-(V + 37) / 10)        # -0.09575(V+37)/(e^{-0.1(V+37)}-1)
@inline _bn_rgc(V) = 1915 // 1000 * exp(-(V + 47) / 80)
@inline _ac_rgc(V) = 1362 // 1000 * safe_exprel(-(V + 13) / 10)
@inline _bc_rgc(V) = 4541 // 100 * exp(-(V + 38) / 18)

@inline function _E_Ca(p::RGCChannels, Ca_i)
    # Guard the log against a non-positive pool under Dual arithmetic.
    ci = max(Ca_i, typeof(Ca_i)(1e-6))
    return p.RT_2F * log(p.Ca_e / ci)
end

@inline function vfield(p::RGCChannels, x::SVector{6}, u)
    V, m, h, n, c, Ca = x[1], x[2], x[3], x[4], x[5], x[6]
    ECa = _E_Ca(p, Ca)
    iNa = p.gNa * m^3 * h * (V - p.ENa)
    iK  = p.gK * n^4 * (V - p.EK)
    iCa = p.gCa * c^3 * (V - ECa)
    # Calcium-gated K conductance: near-off at rest (low Ca_i), activated by the Ca that spikes
    # admit -> spike-frequency adaptation.  The paper's Eq.18 writes gbar_K,Ca (V-EK) with the
    # Ca-dependence folded into the effective conductance; we make it explicit (Fohlmeister-Miller).
    wca = Ca^2 / (Ca^2 + p.Ca_diss^2)
    iKCa = p.gKCa * wca * (V - p.EK)
    ipas = p.gpas * (V - p.Epas)
    dV = (u - iNa - iK - iCa - iKCa - ipas) / p.Cm
    dm = _am_rgc(V) * (1 - m) - _bm_rgc(V) * m
    dh = _ah_rgc(V) * (1 - h) - _bh_rgc(V) * h
    dn = _an_rgc(V) * (1 - n) - _bn_rgc(V) * n
    dc = _ac_rgc(V) * (1 - c) - _bc_rgc(V) * c
    # Calcium pool: influx driven by I_Ca (lumped Ca_gain absorbs Faraday/shell-depth geometry)
    # minus first-order extrusion toward Ca_rest.  Ca_i feeds back only through E_Ca (Nernst),
    # so the exact influx scale weakly affects spiking; it is kept as a named field for tuning.
    dCa = -iCa * p.Ca_gain - (Ca - p.Ca_rest) / p.Ca_tau
    return SVector(dV, dm, dh, dn, dc, dCa)
end

function rest_state(p::RGCChannels{T}) where {T}
    V = p.V0
    m = _am_rgc(V) / (_am_rgc(V) + _bm_rgc(V))
    h = _ah_rgc(V) / (_ah_rgc(V) + _bh_rgc(V))
    n = _an_rgc(V) / (_an_rgc(V) + _bn_rgc(V))
    c = _ac_rgc(V) / (_ac_rgc(V) + _bc_rgc(V))
    return SVector{6,T}(V, m, h, n, c, p.Ca_rest)
end

# ---------------------------------------------------------------------------------------------
# Batch initial-condition sampler shared by data generation and benchmarks.  Returns a
# (d x N) column-major matrix: column j is neuron j's state, which is exactly the layout the
# batched kernels index (one thread per column).
# ---------------------------------------------------------------------------------------------
function random_init(m::NeuronModel, rng, N::Integer; vjit=15.0, gjit=0.03)
    d = statedim(m)
    X = Matrix{Float64}(undef, d, N)
    x0 = rest_state(m)
    for j in 1:N
        V = x0[1] + (rand(rng) - 0.5) * vjit
        # Re-evaluate steady-state gates at the jittered V so the sample sits near the manifold.
        xs = _init_at_voltage(m, V, rng, gjit)
        @inbounds for i in 1:d
            X[i, j] = xs[i]
        end
    end
    return X
end

@inline function _init_at_voltage(m::HHClassic, V, rng, gjit)
    g = gate_inf(m, V)
    return SVector(V, clamp(g[1] + (rand(rng) - .5) * gjit, 0, 1),
                      clamp(g[2] + (rand(rng) - .5) * gjit, 0, 1),
                      clamp(g[3] + (rand(rng) - .5) * gjit, 0, 1))
end
@inline function _init_at_voltage(m::MultiChan, V, rng, gjit)
    x = rest_state(m)  # cheap; then jitter gates
    return SVector(V, clamp(x[2] + (rand(rng)-.5)*gjit,0,1), clamp(x[3]+(rand(rng)-.5)*gjit,0,1),
                      clamp(x[4]+(rand(rng)-.5)*gjit,0,1), clamp(x[5]+(rand(rng)-.5)*gjit,0,1),
                      clamp(x[6]+(rand(rng)-.5)*gjit,0,1), clamp(x[7]+(rand(rng)-.5)*gjit,0,1))
end
@inline function _init_at_voltage(m::RGCChannels, V, rng, gjit)
    x = rest_state(m)
    return SVector(V, clamp(x[2]+(rand(rng)-.5)*gjit,0,1), clamp(x[3]+(rand(rng)-.5)*gjit,0,1),
                      clamp(x[4]+(rand(rng)-.5)*gjit,0,1), clamp(x[5]+(rand(rng)-.5)*gjit,0,1), x[6])
end
