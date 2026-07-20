# Multi-compartment Hodgkin-Huxley cable — the multineuronal forward model of the reference
# article (Lotlikar et al. 2026, Eq. 1/17).  Each neuron is an unbranched chain of P cylindrical
# compartments coupled by axial conductances:
#
#     S_i C_m dV_i/dt = I_stim_i - S_i J_ion_i(V_i, gates_i) + Σ_{j~i} G_ax_ij (V_j - V_i)
#
# Dividing by S_i turns the axial + stimulus terms into an effective per-area drive
#     u_i = (I_stim_i + Σ_j G_ax_ij (V_j - V_i)) / S_i,
# so each compartment reuses the SAME validated single-compartment channel kinetics (dV/dt =
# (u_i - J_ion)/C_m plus the gate ODEs).  The whole RHS is written in array form over a
# (P × N) grid — P compartments, N neurons — so it is one broadcast/kernel and runs batched on
# CPU or CUDA with no per-compartment Python-style loop.
#
# Geometry (positions, radii, lengths) follows the article's straight-axon and 5-region RGC
# setups (Appendix I/L); the resulting membrane currents feed the line-source extracellular
# model in ../inference/extracellular.jl to synthesize electrical images (EIs).

using StaticArrays

# Axial resistivity (Ω·cm) and the unit factor that makes G_ax (in the code's µA/mV/µS system)
# consistent with V in mV, currents in µA, areas in cm².  RA is the one knob that sets
# conduction velocity; the default gives an unmyelinated-axon-like velocity.
const RA_DEFAULT = 100.0     # Ω·cm

"""
    CableGeometry

Per-compartment geometry of one neuron's cable, shared across the batch (all N neurons use the
same morphology here; per-neuron parameters live in `CableParams`). Lengths/radii in µm,
positions in µm (3 × P). `S` is surface area in cm², `gax` the P-1 inter-compartment axial
conductances (µS-equivalent in the code's unit system).
"""
struct CableGeometry
    pos::Matrix{Float64}    # 3 × P compartment centers (µm)
    radius::Vector{Float64} # P radii (µm)
    length::Vector{Float64} # P lengths (µm)
    S::Vector{Float64}      # P surface areas (cm²)
    gax::Vector{Float64}    # (P-1) axial conductances between i and i+1
end

# Convert µm-geometry to surface areas (cm²) and adjacent-compartment axial conductances.
function _build_geometry(pos, radius, length_, Ra)
    P = size(pos, 2)
    um2cm = 1e-4
    S = [2π * radius[i] * length_[i] * um2cm^2 for i in 1:P]         # cm²
    gax = zeros(P - 1)
    for i in 1:P-1
        # series axial conductance of two half-compartments: g = 1 / (R_i/2 + R_{i+1}/2),
        # R_k = Ra * l_k / (π r_k²).  Convert to the code's current/voltage units (×1e3: mV·µS→µA).
        Rk(k) = Ra * (length_[k] * um2cm) / (π * (radius[k] * um2cm)^2)  # Ω
        g = 1.0 / (0.5 * Rk(i) + 0.5 * Rk(i + 1))                        # S
        gax[i] = g * 1e3                                                 # → µA per mV (µS)
    end
    return CableGeometry(Matrix{Float64}(pos), collect(radius), collect(length_), S, gax)
end

"""
    straight_axon_geometry(P; length_um=..., radius_um=..., Ra=RA_DEFAULT)

A straight axon of P compartments along +x (the article's controlled straight-axon setup,
Appendix I). Returns a `CableGeometry`.
"""
function straight_axon_geometry(P::Int; length_um=10.0, radius_um=0.5, Ra=RA_DEFAULT, z_um=-10.0)
    pos = zeros(3, P)
    for i in 1:P
        pos[1, i] = (i - 0.5) * length_um
        pos[3, i] = z_um
    end
    _build_geometry(pos, fill(radius_um, P), fill(length_um, P), Ra)
end

# ---- channel currents & gate derivatives in array form --------------------------------------
# Classic HH channels (robust propagation) — the default cable channel set.  V, gates are (P×N).
struct HHCableChannels
    Cm::Float64
    gNa::Float64
    gK::Float64
    gL::Float64
    ENa::Float64
    EK::Float64
    EL::Float64
end
HHCableChannels(; Cm=1.0, gNa=120.0, gK=36.0, gL=0.3, ENa=50.0, EK=-77.0, EL=-54.387) =
    HHCableChannels(Cm, gNa, gK, gL, ENa, EK, EL)

nstate(::HHCableChannels) = 4    # V,m,h,n

# Ionic current density J_ion (µA/cm²) elementwise on (P×N) arrays.
@inline function _cable_Jion(ch::HHCableChannels, V, m, h, n)
    return ch.gNa .* m.^3 .* h .* (V .- ch.ENa) .+ ch.gK .* n.^4 .* (V .- ch.EK) .+ ch.gL .* (V .- ch.EL)
end

# Exponential-Euler gate update (unconditionally stable): x <- x_inf + (x - x_inf) exp(-dt/τ).
@inline function _gate_expeuler(α, β, x, dt)
    xinf = α ./ (α .+ β)
    τ = 1 ./ (α .+ β)
    return xinf .+ (x .- xinf) .* exp.(-dt ./ τ)
end

# Axial current into each compartment: Σ_j gax_ij (V_j - V_i).  V is (P×N); coupling along P.
function _axial_current(V, gax)
    P = size(V, 1)
    ax = zero(V)
    @inbounds for i in 1:P
        acc = @view ax[i, :]
        if i > 1
            acc .+= gax[i-1] .* (view(V, i-1, :) .- view(V, i, :))
        end
        if i < P
            acc .+= gax[i] .* (view(V, i+1, :) .- view(V, i, :))
        end
    end
    return ax
end

"""
    CableState

Batched cable state: each field is (P × N). Built by [`cable_rest`](@ref) or user init.
"""
mutable struct CableState
    V::Matrix{Float64}
    m::Matrix{Float64}
    h::Matrix{Float64}
    n::Matrix{Float64}
end

function cable_rest(ch::HHCableChannels, P::Int, N::Int; V0=-65.0)
    m = _am_hh(V0) / (_am_hh(V0) + _bm_hh(V0))
    h = _ah_hh(V0) / (_ah_hh(V0) + _bh_hh(V0))
    n = _an_hh(V0) / (_an_hh(V0) + _bn_hh(V0))
    return CableState(fill(V0, P, N), fill(m, P, N), fill(h, P, N), fill(n, P, N))
end

# Precomputed constant tridiagonal for the implicit axial solve  (I - c*A) V^{n+1} = rhs,
# where c_i = dt/(S_i C_m) and A is the axial coupling operator.  Geometry & dt are fixed, so
# the matrix is built once and reused every step.
struct AxialTridiag
    lower::Vector{Float64}   # sub-diagonal  M[i,i-1], i=2..P
    diag::Vector{Float64}    # diagonal      M[i,i]
    upper::Vector{Float64}   # super-diagonal M[i,i+1], i=1..P-1
    c::Vector{Float64}       # per-compartment dt/(S_i C_m)
end

function _build_tridiag(ch::HHCableChannels, geom::CableGeometry, dt::Float64)
    P = length(geom.S)
    c = [dt / (geom.S[i] * ch.Cm) for i in 1:P]
    lower = zeros(P); upper = zeros(P); diag = ones(P)
    @inbounds for i in 1:P
        gl = i > 1 ? geom.gax[i-1] : 0.0
        gr = i < P ? geom.gax[i]   : 0.0
        diag[i] = 1.0 + c[i] * (gl + gr)
        if i > 1; lower[i] = -c[i] * gl; end
        if i < P; upper[i] = -c[i] * gr; end
    end
    return AxialTridiag(lower, diag, upper, c)
end

# Batched Thomas algorithm: solve the (constant) tridiagonal for every column of rhs (P×N).
function _thomas_solve!(Vnew, tri::AxialTridiag, rhs)
    P, N = size(rhs)
    cp = Vector{Float64}(undef, P)
    @inbounds for j in 1:N
        cp[1] = tri.upper[1] / tri.diag[1]
        Vnew[1, j] = rhs[1, j] / tri.diag[1]
        for i in 2:P
            denom = tri.diag[i] - tri.lower[i] * cp[i-1]
            cp[i] = tri.upper[i] / denom
            Vnew[i, j] = (rhs[i, j] - tri.lower[i] * Vnew[i-1, j]) / denom
        end
        for i in P-1:-1:1
            Vnew[i, j] -= cp[i] * Vnew[i+1, j]
        end
    end
    return Vnew
end

"""
    simulate_cable(ch, geom, st0, Istim_fn, dt, nsteps) -> (Vtrace, Im_trace)

IMEX-integrate the cable for `nsteps` at step `dt`: the stiff linear axial diffusion is solved
implicitly (backward Euler, tridiagonal Thomas solve — unconditionally stable), the ionic
channels are treated explicitly, and the gates use exponential Euler. `Istim_fn(k)` returns the
(P×N) stimulus at step k. Returns V (P, nsteps+1, N) and the membrane-current trace I_m used by
the EI line-source model.
"""
function simulate_cable(ch::HHCableChannels, geom::CableGeometry, st0::CableState,
                        Istim_fn, dt::Float64, nsteps::Int)
    P, N = size(st0.V)
    tri = _build_tridiag(ch, geom, dt)
    Scol = reshape(geom.S, :, 1)
    Vtr = Array{Float64}(undef, P, nsteps + 1, N)
    Imtr = Array{Float64}(undef, P, nsteps + 1, N)
    st = deepcopy(st0)
    Vtr[:, 1, :] .= st.V
    Imtr[:, 1, :] .= Istim_fn(1) .+ _axial_current(st.V, geom.gax)
    Vnew = similar(st.V)
    for k in 1:nsteps
        I1 = Istim_fn(k)
        Jion = _cable_Jion(ch, st.V, st.m, st.h, st.n)
        # implicit axial + explicit channels:  (I - c A) V^{n+1} = V^n + c (I_stim - S J_ion)
        rhs = st.V .+ tri.c .* (I1 .- Scol .* Jion)
        _thomas_solve!(Vnew, tri, rhs)
        # gates by exponential Euler at V^n
        m = _gate_expeuler(_am_hh.(st.V), _bm_hh.(st.V), st.m, dt)
        h = _gate_expeuler(_ah_hh.(st.V), _bh_hh.(st.V), st.h, dt)
        n = _gate_expeuler(_an_hh.(st.V), _bn_hh.(st.V), st.n, dt)
        st = CableState(copy(Vnew), m, h, n)
        Vtr[:, k+1, :] .= st.V
        Imtr[:, k+1, :] .= I1 .+ _axial_current(st.V, geom.gax)   # membrane current for the EI
    end
    return Vtr, Imtr
end
