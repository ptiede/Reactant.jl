#==============================================================================
NUFFT options. Defaults follow FINUFFT (sigma=2, w from eps via the
log10 heuristic). `chunk_size` and `bin_dims` control the bin-sorted
chunked-scatter strategy.
==============================================================================#

"""
    NUFFTOptions{T}

Configuration knobs that don't depend on the runtime point distribution.

# Fields
- `eps::T`                — target tolerance (default `1e-6` for `Float32`).
- `sigma::T`              — oversampling factor (default 2.0; 1.25 supported).
- `nspread::Int`          — kernel half-width `w`. If `< 0`, picked from `eps`.
- `chunk_size::Int`       — points per `@trace for` chunk in execute. Default 65536.
- `bin_dims::NTuple{D,Int}` or `NTuple{0,Int}` — bin width per dim used
  to define the sort order; `()` means "auto" (heuristic per dim).
- `sort::Symbol`          — `:auto`, `:always`, or `:never`. For type-1 we always
  sort regardless (mirrors cuFINUFFT). For type-2, `:auto` ⇒ sort.
"""
Base.@kwdef struct NUFFTOptions{T<:Real}
    eps::T = T(1.0e-6)
    sigma::T = T(2)
    nspread::Int = -1                  # < 0 ⇒ derive from eps
    chunk_size::Int = 65536
    bin_dims::Tuple = ()               # () ⇒ auto
    sort::Symbol = :auto
end

# Convenience constructor with auto-promotion.
function NUFFTOptions(::Type{T};
    eps::Real=1.0e-6,
    sigma::Real=2,
    nspread::Integer=-1,
    chunk_size::Integer=65536,
    bin_dims::Tuple=(),
    sort::Symbol=:auto,
) where {T<:Real}
    return NUFFTOptions{T}(
        T(eps), T(sigma), Int(nspread), Int(chunk_size), bin_dims, sort
    )
end

# Per-FINUFFT setup_spreader: w = ceil(log10(1/eps)) + (sigma==2 ? 1 : 2),
# capped at 16, floored at 2.
function _nspread_for_eps(::Type{T}, eps::Real, sigma::Real) where {T<:Real}
    e = max(T(eps), eps_tolerance_floor(T))
    base = ceil(Int, log10(T(1) / e))
    extra = sigma >= 1.99 ? 1 : 2
    return clamp(base + extra, 2, 16)
end

# Smallest tolerance we target without losing all accuracy.
eps_tolerance_floor(::Type{Float32}) = 1.0f-7
eps_tolerance_floor(::Type{Float64}) = 1.0e-13
eps_tolerance_floor(::Type{T}) where {T<:Real} = eps(T)
