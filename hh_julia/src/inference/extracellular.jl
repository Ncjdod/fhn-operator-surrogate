# Extracellular forward model: the electrical image (EI).  Line-source approximation of the
# reference article (Lotlikar et al. 2026, Eq. 4, after Gold et al. 2006):
#
#     Φ_m(t) ≈ 1/(4πσ) Σ_i  S_i (J_ion_i(t) + C_m dV_i/dt) / ||x_i - x_m||
#            = 1/(4πσ) Σ_i  I_m,i(t) / ||x_i - x_m||
#
# where I_m,i is the per-compartment membrane current already returned by the cable simulator.
# So the EI is a single (M × P) distance-weighted matrix applied to the membrane-current trace —
# a batched matmul over time, GPU-friendly.  The characteristic three peaks (capacitive, sodium,
# potassium) and the sodium-peak propagation delay across electrodes are the differentiable
# features used to invert biophysical parameters (see fit.jl).

using LinearAlgebra

const SIGMA_EXTRA = 0.03    # extracellular conductivity S/cm (Gold et al.); Φ comes out in µV.

"""
    hex_electrode_patch(; spacing_um=30.0, z_um=0.0, center=(x,y)) -> (3 × 7) positions

The article's M=7 hexagonal electrode patch (30 µm pitch): one center electrode plus a ring of 6.
"""
function hex_electrode_patch(; spacing_um=30.0, z_um=0.0, cx=0.0, cy=0.0)
    pos = zeros(3, 7)
    pos[1, 1] = cx; pos[2, 1] = cy; pos[3, 1] = z_um
    for k in 1:6
        θ = (k - 1) * (π / 3)
        pos[1, k+1] = cx + spacing_um * cos(θ)
        pos[2, k+1] = cy + spacing_um * sin(θ)
        pos[3, k+1] = z_um
    end
    return pos
end

# (M × P) inverse-distance weights W[m,i] = 1/(4πσ ||x_i - x_m||), distances µm → cm.
function _leadfield(comp_pos::AbstractMatrix, elec_pos::AbstractMatrix; sigma=SIGMA_EXTRA)
    P = size(comp_pos, 2); M = size(elec_pos, 2)
    W = Matrix{Float64}(undef, M, P)
    um2cm = 1e-4
    @inbounds for m in 1:M, i in 1:P
        dx = (comp_pos[1, i] - elec_pos[1, m]) * um2cm
        dy = (comp_pos[2, i] - elec_pos[2, m]) * um2cm
        dz = (comp_pos[3, i] - elec_pos[3, m]) * um2cm
        d = sqrt(dx^2 + dy^2 + dz^2) + um2cm      # + one grid unit to avoid r→0 singularity
        W[m, i] = 1.0 / (4π * sigma * d)
    end
    return W
end

"""
    electrical_image(Im_trace, comp_pos, elec_pos; sigma) -> Φ (M, T, N)

Line-source EI for a membrane-current trace `Im_trace` (P, T, N). Returns per-electrode
extracellular potentials (µV). One `(M×P)` leadfield contracted against P per time/neuron.
"""
function electrical_image(Im_trace::AbstractArray{<:Any,3}, comp_pos, elec_pos; sigma=SIGMA_EXTRA)
    P, T, N = size(Im_trace)
    W = _leadfield(comp_pos, elec_pos; sigma=sigma)     # (M × P)
    M = size(W, 1)
    Φ = Array{Float64}(undef, M, T, N)
    @inbounds for j in 1:N
        Φ[:, :, j] .= W * view(Im_trace, :, :, j)       # (M×P)*(P×T) = (M×T)
    end
    return Φ
end

# ---- differentiable EI features (article §3.1, Appendix E) -----------------------------------
# Soft argmax over time via a temperature-β softmax so peak *timing* varies smoothly with θ.
@inline function _soft_argmax(w::AbstractVector, t::AbstractVector; β=5.0)
    a = softmax_weights(w .* β)
    return sum(a .* t)
end
function softmax_weights(z::AbstractVector)
    zmax = maximum(z)
    e = exp.(z .- zmax)
    return e ./ sum(e)
end

"""
    ei_features(Φ, dt) -> NamedTuple

Per-electrode differentiable EI features for one neuron's EI Φ (M × T):
  * `A_cap`, `A_Na`, `A_K` — capacitive (early positive), sodium (large negative), and potassium
    (late positive) peak amplitudes;
  * `t_Na` — soft sodium-peak time per electrode (ms);
  * `dur`  — spike duration = time from Na trough to K peak (ms);
  * `prop` — sodium-peak propagation delay relative to the strongest electrode (ms).
These are the smooth-in-θ features the article fits against, implemented with soft argmin/argmax.
"""
function ei_features(Φ::AbstractMatrix, dt::Float64; β=5.0)
    M, T = size(Φ)
    t = collect(0:T-1) .* dt
    A_Na = Vector{Float64}(undef, M); A_K = Vector{Float64}(undef, M); A_cap = Vector{Float64}(undef, M)
    t_Na = Vector{Float64}(undef, M); dur = Vector{Float64}(undef, M)
    for m in 1:M
        v = @view Φ[m, :]
        # sodium peak = most negative excursion (soft); use -v for argmax
        tNa = _soft_argmax(-v, t; β=β)
        kNa = clamp(round(Int, tNa / dt) + 1, 1, T)
        A_Na[m] = minimum(v)
        # capacitive = max positive before the Na trough; K = max positive after
        A_cap[m] = kNa > 1 ? maximum(view(v, 1:kNa)) : v[1]
        A_K[m] = kNa < T ? maximum(view(v, kNa:T)) : v[T]
        tK = kNa < T ? _soft_argmax(view(v, kNa:T), view(t, kNa:T); β=β) : t[T]
        t_Na[m] = tNa
        dur[m] = tK - tNa
    end
    # propagation delay: Na-peak time relative to the electrode with the largest |A_Na|
    ref = argmax(abs.(A_Na))
    prop = t_Na .- t_Na[ref]
    return (A_cap=A_cap, A_Na=A_Na, A_K=A_K, t_Na=t_Na, dur=dur, prop=prop, ref=ref)
end
