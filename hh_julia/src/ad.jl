# Minimal forward-mode automatic differentiation — a self-contained dual number carrying a
# value plus an SVector of N partials.  Zero external dependencies (so the package loads fast
# and installs with no binary artifacts), and `isbits` by construction, so a Dual is legal
# inside a GPU kernel exactly like a Float.  It is <: Real, so it flows transparently through
# the generic model code (`vfield`), StaticArrays algebra, and the solvers.
#
# We use it for two things:
#   * the d x d Jacobian of the vector field (Rosenbrock solver, N = state dim d),
#   * gradients of a scalar loss w.r.t. a handful of biophysical parameters (inverse problem).
# Both are small-N forward mode, which is the right regime here; heavy neural-net training uses
# reverse mode (Zygote) instead — see ../surrogate/train.jl.

using StaticArrays

struct Dual{N,T<:Real} <: Real
    v::T
    p::SVector{N,T}
end

@inline Dual{N,T}(x::Real) where {N,T} = Dual{N,T}(T(x), zero(SVector{N,T}))
@inline Dual{N}(v::T, p::SVector{N,T}) where {N,T} = Dual{N,T}(v, p)

@inline value(d::Dual) = d.v
@inline value(x::Real) = x
@inline partials(d::Dual) = d.p

# Promotion / conversion so mixed Dual+Real expressions Just Work.
Base.convert(::Type{Dual{N,T}}, x::Real) where {N,T} = Dual{N,T}(x)
Base.convert(::Type{Dual{N,T}}, d::Dual{N}) where {N,T} = Dual{N,T}(T(d.v), SVector{N,T}(d.p))
Base.promote_rule(::Type{Dual{N,T}}, ::Type{S}) where {N,T,S<:Real} = Dual{N,promote_type(T,S)}
Base.promote_rule(::Type{Dual{N,T}}, ::Type{Dual{N,S}}) where {N,T,S} = Dual{N,promote_type(T,S)}

@inline Base.zero(::Type{Dual{N,T}}) where {N,T} = Dual{N,T}(zero(T), zero(SVector{N,T}))
@inline Base.one(::Type{Dual{N,T}}) where {N,T} = Dual{N,T}(one(T), zero(SVector{N,T}))
@inline Base.zero(d::Dual{N,T}) where {N,T} = zero(Dual{N,T})
@inline Base.one(d::Dual{N,T}) where {N,T} = one(Dual{N,T})
@inline Base.oneunit(::Type{Dual{N,T}}) where {N,T} = one(Dual{N,T})
@inline Base.float(d::Dual) = d

# --- arithmetic (each rule: value op, then the chain rule on the partials) -------------------
@inline Base.:+(a::Dual{N}, b::Dual{N}) where {N} = Dual{N}(a.v + b.v, a.p + b.p)
@inline Base.:-(a::Dual{N}, b::Dual{N}) where {N} = Dual{N}(a.v - b.v, a.p - b.p)
@inline Base.:-(a::Dual{N}) where {N} = Dual{N}(-a.v, -a.p)
@inline Base.:*(a::Dual{N}, b::Dual{N}) where {N} = Dual{N}(a.v * b.v, a.v * b.p + b.v * a.p)
@inline function Base.:/(a::Dual{N}, b::Dual{N}) where {N}
    inv_b = inv(b.v)
    q = a.v * inv_b
    return Dual{N}(q, (a.p - q * b.p) * inv_b)
end

@inline Base.inv(a::Dual{N,T}) where {N,T} = Dual{N}(inv(a.v), (-inv(a.v)^2) * a.p)
@inline Base.exp(a::Dual{N}) where {N} = (e = exp(a.v); Dual{N}(e, e * a.p))
@inline Base.expm1(a::Dual{N}) where {N} = Dual{N}(expm1(a.v), exp(a.v) * a.p)
@inline Base.log(a::Dual{N}) where {N} = Dual{N}(log(a.v), a.p / a.v)
@inline Base.sqrt(a::Dual{N}) where {N} = (s = sqrt(a.v); Dual{N}(s, a.p / (2 * s)))
@inline Base.tanh(a::Dual{N}) where {N} = (t = tanh(a.v); Dual{N}(t, (1 - t^2) * a.p))
@inline Base.abs(a::Dual{N}) where {N} = a.v < 0 ? -a : a

@inline function Base.:^(a::Dual{N,T}, n::Integer) where {N,T}
    # d/dx x^n = n x^(n-1); handled for small integer powers used by the models (m^3, n^4, ...).
    return Dual{N}(a.v^n, (n * a.v^(n - 1)) * a.p)
end
@inline function Base.:^(a::Dual{N,T}, b::Real) where {N,T}
    y = a.v^b
    return Dual{N}(y, (b * a.v^(b - 1)) * a.p)
end
# Disambiguate the literal_pow path (x^2, x^3, ...) so it hits the integer rule above.
@inline Base.literal_pow(::typeof(^), a::Dual, ::Val{p}) where {p} = a^p

# --- comparisons act on the value (branch selection is not differentiated) -------------------
for op in (:<, :<=, :>, :>=, :(==))
    @eval @inline Base.$op(a::Dual, b::Dual) = $op(a.v, b.v)
    @eval @inline Base.$op(a::Dual, b::Real) = $op(a.v, b)
    @eval @inline Base.$op(a::Real, b::Dual) = $op(a, b.v)
end
@inline Base.isless(a::Dual, b::Dual) = isless(a.v, b.v)
@inline Base.isfinite(a::Dual) = isfinite(a.v)
@inline Base.isnan(a::Dual) = isnan(a.v)
@inline Base.max(a::Dual, b::Dual) = a.v < b.v ? b : a
@inline Base.min(a::Dual, b::Dual) = a.v < b.v ? a : b
@inline Base.max(a::Dual{N,T}, b::Real) where {N,T} = max(a, Dual{N,T}(b))
@inline Base.max(a::Real, b::Dual{N,T}) where {N,T} = max(Dual{N,T}(a), b)
@inline Base.min(a::Dual{N,T}, b::Real) where {N,T} = min(a, Dual{N,T}(b))
@inline Base.clamp(a::Dual{N,T}, lo::Real, hi::Real) where {N,T} = min(max(a, lo), hi)

# --- seeding helpers -------------------------------------------------------------------------
"Seed an SVector x with unit partials (x[i] carries e_i) to differentiate w.r.t. all of x."
@inline function seed(x::SVector{N,T}) where {N,T}
    return SVector{N}(ntuple(i ->
        Dual{N,T}(x[i], SVector{N,T}(ntuple(k -> (k == i ? one(T) : zero(T)), Val(N)))), Val(N)))
end

"Extract the (M x N) Jacobian from an SVector{M} of Dual{N}."
@inline function extract_jacobian(y::SVector{M,Dual{N,T}}) where {M,N,T}
    return SMatrix{M,N,T}(ntuple(idx -> begin
        i = ((idx - 1) % M) + 1     # row (output component), column-major linear index
        k = ((idx - 1) ÷ M) + 1     # col (input component)
        partials(y[i])[k]
    end, Val(M * N)))
end
