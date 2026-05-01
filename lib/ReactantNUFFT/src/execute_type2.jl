#==============================================================================
Type-2 execute: zero-pad+deconvolve → iFFT → interpolate.

Mirror of execute_type1. The only non-Base call is `@opcall gather` for the
batched stencil read; everything else is plain Julia broadcasting / slicing.
==============================================================================#

# ---------- corner embed + post-FFT shift -----------------------------------
#
# fk_deconv has the centered-mode layout produced by `_central_view`:
# along each spatial dim, the first `half = nmodes[d] ÷ 2` entries correspond
# to negative modes and the remaining `n_pos = nmodes[d] - half` to
# non-negative modes.
#
# The *centered* embed places those into the (ngrid_d,) array as
#   F_n = [non-neg modes; zeros; neg modes]
# i.e. F_n = circshift(corner_pad(fk_deconv), -half) along each spatial dim.
# Naively this requires 2^D pairs of dynamic_slice / dynamic_update_slice.
#
# We use the FFT shift theorem to avoid the corner shuffle. Computing
#   G = corner_pad(fk_deconv)        (zeros at the high end of each spatial axis)
#   FW = fft_or_bfft(G, 1:D)
# gives the same result as `fft_or_bfft(F_n, 1:D)` *up to* a per-axis
# linear-phase factor on the spatial-domain output. We apply that phase
# post-FFT in `_post_fft_shift_t2` — the math is in the docstring there.

function _corner_embed(fk_deconv, nmodes::NTuple{D,Int}, ngrid::NTuple{D,Int}) where {D}
    if all(nmodes .== ngrid)
        return fk_deconv
    end
    CT = Reactant.unwrapped_eltype(eltype(fk_deconv))
    R = D + 1
    fk_t = Reactant.promote_to(Reactant.TracedRArray{CT,R}, fk_deconv)
    low      = Int64[ntuple(_ -> 0, Val(D))..., 0]
    high     = Int64[ntuple(d -> ngrid[d] - nmodes[d], Val(D))..., 0]
    interior = Int64[ntuple(_ -> 0, Val(D))..., 0]
    zero_val = Reactant.promote_to(Reactant.TracedRNumber{CT}, zero(CT))
    return Reactant.Ops.@opcall pad(fk_t, zero_val; low=low, high=high, interior=interior)
end

# Multiply `fw` (post-FFT spatial output, shape (ngrid..., ntrans)) by the
# separable phase factor that converts an FFT-of-corner-padded result into
# the FFT-of-centered-embedded result.
#
# Derivation. With F_n = circshift(G, -half_d) along axis d, the shift
# theorem gives
#   fft (iflag=-1):  fft(F_n)[n]  = fft(G)[n]  * exp(+2πi * half_d * n_d / ngrid_d)
#   bfft (iflag=+1): bfft(F_n)[n] = bfft(G)[n] * exp(-2πi * half_d * n_d / ngrid_d)
# i.e. phase[n_d] = exp(-iflag * 2πi * half_d * n_d / ngrid_d). The full
# multi-D factor is the outer product of per-axis phases — applied as D
# separable broadcasts so the (ngrid..., 1) outer product is never
# materialized as a constant.
#
# Skip when nmodes == ngrid (no embed → no shift, matching the old contract).
function _post_fft_shift_t2(fw, plan::NUFFTPlan{T,D}) where {T,D}
    if all(plan.nmodes .== plan.ngrid)
        return fw
    end
    CT = Complex{T}
    R = D + 1
    for d in 1:D
        half_d = plan.nmodes[d] ÷ 2
        half_d == 0 && continue
        ngrid_d = plan.ngrid[d]
        ang = (-T(plan.iflag) * 2 * T(pi) * T(half_d) / T(ngrid_d)) .* collect(T, 0:(ngrid_d - 1))
        v_flat = CT.(cos.(ang) .+ im .* sin.(ang))
        shape = ntuple(i -> i == d ? ngrid_d : 1, R)
        v = reshape(v_flat, shape...)
        fw = fw .* v
    end
    return fw
end

# ---------- batched gather for stencil reads --------------------------------

function _gather_spatial(
    fw::Reactant.AnyTracedRArray{CT,Dp1},
    gather_idx::AbstractMatrix,
    ::Val{D},
    ntrans::Int,
) where {CT,Dp1,D}
    @assert Dp1 == D + 1
    res = Reactant.Ops.@opcall gather(
        fw,
        Reactant.promote_to(Reactant.TracedRArray{Int,2}, gather_idx);
        offset_dims=Int64[2],                                # ntrans dim of result
        collapsed_slice_dims=collect(Int64, 1:D),            # spatial dims of fw
        operand_batching_dims=Int64[],
        start_indices_batching_dims=Int64[],
        start_index_map=collect(Int64, 1:D),
        index_vector_dim=Int64(2),
        slice_sizes=Int64[ntuple(_ -> 1, D)..., ntrans],
    )
    return res                                                # (Nupd, ntrans)
end

# ---------- main entry point ------------------------------------------------

"""
    execute_type2(prep, fk) -> c

Type-2 NUFFT: uniform → nonuniform.
- `fk::AbstractArray{<:Complex}` of shape `nmodes...` or `(nmodes..., ntrans)`.
- Returns `c` of shape `(M,)` or `(M, ntrans)` matching the input rank.

Designed to be called inside `Reactant.@jit`.
"""
function execute_type2(prep::NUFFTSetPts{T,D}, fk::AbstractArray) where {T,D}
    plan = prep.plan
    @assert nufft_type(plan) == 2 "Plan was not built for type-2"
    @assert size(fk)[1:D] == plan.nmodes "fk shape mismatch with plan.nmodes"

    squeeze_out = ndims(fk) == D
    fk_full = squeeze_out ? reshape(fk, plan.nmodes..., 1) : fk
    ntrans = size(fk_full, D + 1)

    return _execute_type2_impl(prep, fk_full, ntrans, squeeze_out)
end

function _execute_type2_impl(
    prep::NUFFTSetPts{T,D}, fk_full::AbstractArray, ntrans::Int, squeeze_out::Bool
) where {T,D}
    CT = Complex{T}
    plan = prep.plan
    w = plan.nspread
    ngrid = plan.ngrid
    M_pad = prep.M_pad
    nchunks = prep.nchunks
    cs = prep.chunk_size

    # 1. Deconvolve by separable phi_hat product.
    phih = _phi_hat_tensor(plan, Val(D + 1))
    fk_dec = fk_full ./ phih

    # 2. Embed into oversampled grid (corner pad — single XLA op).
    fw_hat = _corner_embed(fk_dec, plan.nmodes, ngrid)

    # 3. (Inverse-)FFT (same sign convention as type-1).
    fw = plan.iflag < 0 ?
         AbstractFFTs.fft(fw_hat, 1:D) :
         AbstractFFTs.bfft(fw_hat, 1:D)

    # 3a. Apply FFT-shift phase to recover centered-mode semantics
    #     (compensates for corner_embed not doing the circshift).
    fw = _post_fft_shift_t2(fw, plan)

    # 4. Single-shot gather. (Chunked variant kept for reference but currently
    #    fails to propagate writes in some trace patterns; the diagnostic
    #    showed gather is not the biggest bottleneck so single-shot stays.)
    M = prep.M
    coefs = Reactant.promote_to(Reactant.TracedRArray{T,2}, plan.horner_coefs)
    offsets_row = reshape(collect(0:(w - 1)), 1, w)
    wpd = ntuple(d -> _per_dim_weights(coefs, prep.frac_sorted[d]), Val(D))
    idx_per_dim = ntuple(d -> _per_dim_indices(prep.base_sorted[d], ngrid[d], offsets_row), Val(D))
    weights = _outer_product_weights(wpd)
    gather_idx = _stack_scatter_indices(idx_per_dim)
    vals = _gather_spatial(fw, gather_idx, Val(D), ntrans)
    vals = reshape(vals, M_pad, w^D, ntrans)
    c_sorted = dropdims(sum(vals .* reshape(weights, M_pad, w^D, 1); dims=2); dims=2)

    # 5. Inverse permutation, returning length M (un-padded).
    c = c_sorted[prep.invperm, :]
    return squeeze_out ? dropdims(c; dims=2) : c
end
