#==============================================================================
Type-1 execute: spread → FFT → deconvolve + crop.

Spreading is chunked: each chunk processes `chunk_size` bin-sorted points,
materializing only a (chunk_size, w^D, ntrans) intermediate before the
scatter-add. This bounds peak memory and gives XLA scatter-add bin-localized
targets (less atomic contention on GPU, better cache reuse on CPU).

The body is plain Julia for everything except the scatter, which calls
`@opcall scatter` with custom `dimension_numbers` so the (Nf..., ntrans)
layout works without permuting the FFT axes.
==============================================================================#

# ---------- helpers ---------------------------------------------------------

# Per-chunk Horner weights for one dim:
#   t = 2*frac - 1 ∈ [-1, 1], shape (M,)
#   weights = horner_eval(coefs, t) :: (M, w)
@inline function _per_dim_weights(coefs::AbstractMatrix{T}, frac::AbstractVector) where {T}
    return horner_eval(coefs, T(2) .* frac .- T(1))
end

# Outer product of per-dim weight matrices (M, w_d) for d in 1..D.
# Returns (M, prod(w_d)) — for our case w_d = w for all d, so (M, w^D).
@inline function _outer_product_weights(wpd::NTuple{1,AbstractMatrix})
    return wpd[1]                                              # (M, w)
end
@inline function _outer_product_weights(wpd::NTuple{2,AbstractMatrix})
    M, w = size(wpd[1])
    return reshape(reshape(wpd[1], M, w, 1) .* reshape(wpd[2], M, 1, w), M, w * w)
end
@inline function _outer_product_weights(wpd::NTuple{3,AbstractMatrix})
    M, w = size(wpd[1])
    a = reshape(wpd[1], M, w, 1, 1)
    b = reshape(wpd[2], M, 1, w, 1)
    c = reshape(wpd[3], M, 1, 1, w)
    return reshape(a .* b .* c, M, w * w * w)
end

# Per-dim local stencil indices (M, w), 1-based Julia indices into the fw grid.
# `base` is 0-based physical position of the leftmost stencil cell (may be
# negative or > ngrid; periodic wrap is applied here). `offsets_row` is a
# precomputed (1, w) Int row, hoisted outside any `@trace for` to keep `w`
# from being lifted to a traced value.
@inline function _per_dim_indices(base::AbstractVector, ngrid::Integer, offsets_row::AbstractMatrix)
    M = length(base)
    return mod.(reshape(base, M, 1) .+ offsets_row, ngrid) .+ 1
end

# Stack per-dim (M, w) index matrices into a single (M*w^D, D) Int matrix
# whose row r=(j*w^D + linidx) gives the multi-D index for one update.
@inline function _stack_scatter_indices(idxs::NTuple{1,AbstractMatrix})
    M, w = size(idxs[1])
    return reshape(idxs[1], M * w, 1)
end
@inline function _stack_scatter_indices(idxs::NTuple{2,AbstractMatrix})
    M, w = size(idxs[1])
    a = repeat(reshape(idxs[1], M, w, 1), 1, 1, w)             # (M, w, w)
    b = repeat(reshape(idxs[2], M, 1, w), 1, w, 1)             # (M, w, w)
    return hcat(reshape(a, M * w * w, 1), reshape(b, M * w * w, 1))
end
@inline function _stack_scatter_indices(idxs::NTuple{3,AbstractMatrix})
    M, w = size(idxs[1])
    a = repeat(reshape(idxs[1], M, w, 1, 1), 1, 1, w, w)
    b = repeat(reshape(idxs[2], M, 1, w, 1), 1, w, 1, w)
    c = repeat(reshape(idxs[3], M, 1, 1, w), 1, w, w, 1)
    N = M * w * w * w
    return hcat(reshape(a, N, 1), reshape(b, N, 1), reshape(c, N, 1))
end

# ---------- the scatter call ------------------------------------------------
#
# fw layout: (Nf_1, ..., Nf_D, ntrans). Updates: (Nupdates, ntrans). Indices:
# (Nupdates, D). The window dim of updates is the trailing ntrans axis.
function _scatter_add_spatial!(
    fw::Reactant.AnyTracedRArray{CT,Dp1},
    scatter_idx::AbstractMatrix,
    updates::AbstractMatrix,
    ::Val{D},
) where {CT,Dp1,D}
    @assert Dp1 == D + 1
    # Note on hints: `indices_are_sorted=true` was tried (post-bin-sort indices
    # are clustered but not strictly lex-sorted). It's a no-op on XLA-CPU and
    # ~7× SLOWER on XLA-GPU at 2D M=10⁶ N=1024² (XLA-GPU picks a sequential
    # code path when the contract is asserted). `unique_indices=false` is the
    # truthful value (multiple updates target the same fw cell) and matches
    # the default, so we don't pass it explicitly.
    res = Reactant.Ops.@opcall scatter(
        +,
        [fw],
        Reactant.promote_to(Reactant.TracedRArray{Int,2}, scatter_idx),
        [Reactant.promote_to(Reactant.TracedRArray{CT,2}, updates)];
        update_window_dims=Int64[2],                       # ntrans axis of updates
        inserted_window_dims=collect(Int64, 1:D),          # spatial dims of fw
        input_batching_dims=Int64[],
        scatter_indices_batching_dims=Int64[],
        scatter_dims_to_operand_dims=collect(Int64, 1:D),  # idx cols → spatial dims
        index_vector_dim=Int64(2),
    )
    return only(res)
end

# ---------- central-mode crop (with periodic wrap) --------------------------
#
# Read the 2^D corners of fw_hat (positive modes from front, negative modes
# from back of the oversampled grid) and place them into a pre-allocated
# central buffer in the centered-mode layout `[neg_half; non_neg_half]` along
# each dim. Avoids the materialized intermediates that the previous
# cat-along-each-dim approach introduced (~5 ms / call at 2D 1024², ~22 ms
# at 4096²).

function _central_view(fw_hat, nmodes::NTuple{D,Int}, ngrid::NTuple{D,Int}) where {D}
    if all(nmodes .== ngrid)
        return fw_hat
    end
    R = D + 1
    @assert ndims(fw_hat) == R
    CT = Reactant.unwrapped_eltype(eltype(fw_hat))
    sz_out = (nmodes..., size(fw_hat, R))
    out = Reactant.Ops.@opcall fill(zero(CT), Int64[sz_out...])
    halves = ntuple(d -> nmodes[d] ÷ 2, Val(D))
    n_poss = ntuple(d -> nmodes[d] - halves[d], Val(D))
    for sign_bits in 0:(1 << D - 1)
        is_neg = ntuple(d -> ((sign_bits >> (d - 1)) & 1) == 1, Val(D))
        valid = true
        for d in 1:D
            if is_neg[d] && halves[d] == 0
                valid = false; break
            end
        end
        valid || continue
        src_starts = ntuple(d -> is_neg[d] ? ngrid[d] - halves[d] + 1 : 1, Val(D))
        src_lens   = ntuple(d -> is_neg[d] ? halves[d] : n_poss[d], Val(D))
        dst_starts = ntuple(d -> is_neg[d] ? 1 : halves[d] + 1, Val(D))
        slab = _slice_nd_t1(fw_hat, src_starts, src_lens, R)
        out = _update_nd_t1(out, slab, dst_starts, R)
    end
    return out
end

@inline function _slice_nd_t1(x, starts::NTuple{D,Int}, lens::NTuple{D,Int}, R::Int) where {D}
    T = Reactant.unwrapped_eltype(eltype(x))
    return Reactant.Ops.@opcall dynamic_slice(
        Reactant.promote_to(Reactant.TracedRArray{T,R}, x),
        Any[starts..., 1],
        Int64[lens..., size(x, R)],
    )
end

@inline function _update_nd_t1(operand, update, starts::NTuple{D,Int}, R::Int) where {D}
    T = Reactant.unwrapped_eltype(eltype(operand))
    return Reactant.Ops.@opcall dynamic_update_slice(
        Reactant.promote_to(Reactant.TracedRArray{T,R}, operand),
        Reactant.promote_to(Reactant.TracedRArray{T,R}, update),
        Any[starts..., 1],
    )
end

# ---------- phi_hat outer-product for deconvolution -------------------------
#
# Build a (nmodes_1, ..., nmodes_D, 1) tensor whose entries are the product of
# the per-dim phi_hat values, ready to broadcast-divide fw_central.
function _phi_hat_tensor(plan::NUFFTPlan{T,D}, ::Val{ntrans_axis}) where {T,D,ntrans_axis}
    # ntrans_axis is the dim index where the (singleton) ntrans dim should sit
    # in the result. We always put it as the last dim so dim D+1 is size 1.
    out = nothing
    for d in 1:D
        shape = ntuple(i -> i == d ? plan.nmodes[d] : 1, D + 1)
        v = reshape(plan.phi_hat[d], shape...)
        out = isnothing(out) ? v : out .* v
    end
    return out
end

# ---------- main entry point ------------------------------------------------

"""
    execute_type1(prep, c) -> fk

Type-1 NUFFT: nonuniform → uniform.
- `c::AbstractArray{<:Complex}` of shape `(M,)` or `(M, ntrans)`.
- Returns `fk` of shape `nmodes...` (when input was `(M,)`) or
  `(nmodes..., ntrans)` (when input was `(M, ntrans)`).

Designed to be called inside `Reactant.@jit`.
"""
function execute_type1(prep::NUFFTSetPts{T,D}, c::AbstractArray) where {T,D}
    plan = prep.plan
    @assert nufft_type(plan) == 1 "Plan was not built for type-1"
    @assert size(c, 1) == prep.M "Strength count mismatch with prepared points"

    squeeze_out = ndims(c) == 1
    cmat = squeeze_out ? reshape(c, prep.M, 1) : c
    ntrans = size(cmat, 2)

    return _execute_type1_impl(prep, cmat, ntrans, squeeze_out)
end

function _execute_type1_impl(
    prep::NUFFTSetPts{T,D}, cmat::AbstractMatrix, ntrans::Int, squeeze_out::Bool
) where {T,D}
    CT = Complex{T}
    plan = prep.plan
    w = plan.nspread
    ngrid = plan.ngrid
    M_pad = prep.M_pad
    nchunks = prep.nchunks
    cs = prep.chunk_size

    # 1. Chunked spread: bin-sorted points → fw via repeated scatter-add.
    #    The bin-sort permutation and the pad-mask are applied per-chunk
    #    inside `_spread_chunks` rather than via an upfront materialized
    #    `c_sorted = cmat[perm, :] .* mask` — keeping the gather in the same
    #    op group as the contribution build lets XLA fuse them and avoids
    #    a (M_pad, ntrans) intermediate that costs ~25–35 % of e2e on the
    #    M=10⁶ T1 rows. (See PROFILE.md "Largest gaps" → B2.)
    #
    #    When `FW_REPLICAS[] > 1` the chunks are round-robin'd across R
    #    independent `fw` shards, then reduce-summed before FFT (B3). Each
    #    shard sees ~M/R writes with ~contention/R-way collisions, so
    #    fp32 atomic-add throughput recovers near-linearly until contention
    #    drops to ~4–8 way. R=1 uses the original unsharded scatter.
    coefs = Reactant.promote_to(Reactant.TracedRArray{T,2}, plan.horner_coefs)
    offsets_row = reshape(collect(0:(w - 1)), 1, w)         # static (1, w) Int row
    R = FW_REPLICAS[]
    if SORT_MERGE_SPREAD[]
        # Sort-merge variant is unsharded for now; it's gated off by default.
        fw = Reactant.Ops.@opcall fill(zero(CT), Int64[ngrid..., ntrans])
        fw = _spread_chunks_sortmerge(
            fw, cmat, prep.perm, prep.mask,
            prep.base_sorted, prep.frac_sorted, coefs,
            ngrid, w, ntrans, nchunks, cs, offsets_row, Val(D),
        )
    elseif R <= 1 || nchunks <= 1
        fw = Reactant.Ops.@opcall fill(zero(CT), Int64[ngrid..., ntrans])
        fw = _spread_chunks(
            fw, cmat, prep.perm, prep.mask,
            prep.base_sorted, prep.frac_sorted, coefs,
            ngrid, w, ntrans, nchunks, cs, offsets_row, Val(D),
        )
    else
        fw = _spread_chunks_replicated(
            cmat, prep.perm, prep.mask,
            prep.base_sorted, prep.frac_sorted, coefs,
            ngrid, w, ntrans, nchunks, cs, offsets_row, Val(D),
            Val(min(R, nchunks)),
        )
    end

    # 4. FFT (sign per iflag).
    fw_hat = plan.iflag < 0 ?
             AbstractFFTs.fft(fw, 1:D) :
             AbstractFFTs.bfft(fw, 1:D)

    # 5. Crop central modes (with periodic wrap).
    fw_central = _central_view(fw_hat, plan.nmodes, ngrid)

    # 6. Deconvolve by separable phi_hat product.
    phih = _phi_hat_tensor(plan, Val(D + 1))
    fk = fw_central ./ phih

    return squeeze_out ? dropdims(fk; dims=D + 1) : fk
end

# --- chunked spread ---------------------------------------------------------
#
# Each iteration computes Horner weights, multi-D stencil indices, and a
# contribution tensor of shape (chunk_size, w^D, ntrans), then scatter-adds
# it into `fw`. With bin-sorted ordering, each chunk's scatter targets are
# spatially localized.
function _spread_chunks(
    fw, cmat, perm, mask, base_full, frac_full, coefs,
    ngrid::NTuple{ND,Int}, w::Int, ntrans::Int,
    nchunks::Int, cs::Int, offsets_row::AbstractMatrix, dimval::Val{ND},
) where {ND}
    nd = ND
    wD = w^nd
    # Static unroll — for our target VLBI sweep nchunks ≤ ~16 so the IR stays
    # bounded. Larger M (10^7) would benefit from a `@trace for` loop, but
    # that path runs into loop-carried-state issues with our scatter pattern.
    for k in 1:nchunks
        j0 = (k - 1) * cs + 1          # 1-based start (Reactant subtracts 1 internally)
        bs = ntuple(d -> _slice1(base_full[d], j0, cs), dimval)
        fr = ntuple(d -> _slice1(frac_full[d], j0, cs), dimval)
        cc = _gather_chunk_strengths(cmat, perm, mask, j0, cs, ntrans)

        wpd   = ntuple(d -> _per_dim_weights(coefs, fr[d]), dimval)
        idxpd = ntuple(d -> _per_dim_indices(bs[d], ngrid[d], offsets_row), dimval)
        weights = _outer_product_weights(wpd)               # (cs, wD)
        scatter_idx = _stack_scatter_indices(idxpd)         # (cs*wD, nd)

        contrib = reshape(weights, cs, wD, 1) .* reshape(cc, cs, 1, ntrans)
        contrib_flat = reshape(contrib, cs * wD, ntrans)

        fw = _scatter_add_spatial!(fw, scatter_idx, contrib_flat, dimval)
    end
    return fw
end

# Per-chunk strength gather + mask. Replaces the upfront
# `c_sorted = cmat[perm, :] .* mask` materialization (B2).
@inline function _gather_chunk_strengths(cmat, perm, mask, j0, cs::Int, ntrans::Int)
    perm_slice = _slice1(perm, j0, cs)         # (cs,) Int — bin-sorted source rows
    mask_slice = _slice1(mask, j0, cs)         # (cs,)     — pad-row mask
    cc_un = cmat[perm_slice, :]                 # (cs, ntrans) — gather original strengths
    return cc_un .* reshape(mask_slice, cs, 1)
end

# B3 — replicated `fw` scatter. Allocates R independent `(ngrid..., ntrans)`
# shards and round-robins chunks across them, then pairwise-sums the shards.
# Each shard sees only ~nchunks/R chunks of writes, so its atomic-add path
# faces ~R× lower per-cell contention than the unsharded variant. R=1 falls
# back to the original `_spread_chunks` path (skipped by the caller).
function _spread_chunks_replicated(
    cmat, perm, mask, base_full, frac_full, coefs,
    ngrid::NTuple{ND,Int}, w::Int, ntrans::Int,
    nchunks::Int, cs::Int, offsets_row::AbstractMatrix, dimval::Val{ND}, ::Val{R},
) where {ND, R}
    @assert R >= 2 "replicated scatter requires R >= 2"
    nd = ND
    wD = w^nd
    T = Reactant.unwrapped_eltype(eltype(coefs))
    CT = Complex{T}
    fw_shards = ntuple(_ -> Reactant.Ops.@opcall(fill(zero(CT), Int64[ngrid..., ntrans])), Val(R))
    for k in 1:nchunks
        r = mod1(k, R)
        j0 = (k - 1) * cs + 1
        bs = ntuple(d -> _slice1(base_full[d], j0, cs), dimval)
        fr = ntuple(d -> _slice1(frac_full[d], j0, cs), dimval)
        cc = _gather_chunk_strengths(cmat, perm, mask, j0, cs, ntrans)

        wpd   = ntuple(d -> _per_dim_weights(coefs, fr[d]), dimval)
        idxpd = ntuple(d -> _per_dim_indices(bs[d], ngrid[d], offsets_row), dimval)
        weights = _outer_product_weights(wpd)
        scatter_idx = _stack_scatter_indices(idxpd)

        contrib = reshape(weights, cs, wD, 1) .* reshape(cc, cs, 1, ntrans)
        contrib_flat = reshape(contrib, cs * wD, ntrans)

        new_shard = _scatter_add_spatial!(fw_shards[r], scatter_idx, contrib_flat, dimval)
        fw_shards = Base.setindex(fw_shards, new_shard, r)
    end
    return _pairwise_sum_shards(fw_shards)
end

# Pairwise reduction tree for the R shards. Pairwise (rather than left-fold)
# so XLA can fuse the additions into log2(R) parallel streams.
@inline _pairwise_sum_shards(t::NTuple{1}) = t[1]
@inline _pairwise_sum_shards(t::NTuple{2}) = t[1] .+ t[2]
@inline function _pairwise_sum_shards(t::NTuple{N}) where {N}
    h = N ÷ 2
    a = _pairwise_sum_shards(ntuple(i -> t[i], h))
    b = _pairwise_sum_shards(ntuple(i -> t[h + i], N - h))
    return a .+ b
end

# Module-level toggle for the sort-merge spread variant. When true, each
# chunk's (idx, contrib) pairs are sorted by linear destination index and
# scattered with `indices_are_sorted=true`. Trades a per-chunk Int sort for
# atomic-free contiguous accumulation. Default off; turn on once the
# heuristic in `_execute_type1_impl` confirms a win for the workload.
const SORT_MERGE_SPREAD = Ref(false)

# Number of replicated `fw` shards used by the spread (B3). When R > 1,
# chunks are round-robin'd across R shards which are reduce-summed before
# FFT. Hypothesis was that fp32 atomic-add throughput on a hot cell-set
# would recover near-linearly as R rises (cuFINUFFT's SM-method exploits
# this with shared-memory tiles); measurements showed otherwise. On the
# 2D T1 large-M loss rows, R=2/4/8 are all flat-to-worse than R=1, by
# 0–17 % depending on grid size:
#
# | row                              | R=1      | R=4      | R=8      |
# |----------------------------------|---------:|---------:|---------:|
# | D=2 M=10⁵ N=256²  T1             |  0.66 ms |  1.27 ms |  1.26 ms |
# | D=2 M=10⁶ N=256²  T1             |  6.53 ms |  6.55 ms |  6.57 ms |
# | D=2 M=10⁶ N=1024² T1             |  6.82 ms |  7.04 ms |  7.37 ms |
# | D=2 M=10⁶ N=2048² T1             | 14.13 ms | 15.37 ms | 16.53 ms |
# | D=3 M=10⁵ T1                     | 27.11 ms | 27.37 ms | 27.37 ms |
#
# Conclusion: the actual scatter bottleneck is DRAM bandwidth on the
# atomic read-modify-write cycle, not atomic-conflict throughput. R×
# replicas multiply the bandwidth need without buying back anything,
# so R>1 is net-negative. Default kept at R=1 (no replication); the
# implementation stays in place in case a future workload (heavier
# contention, smaller grids that fit in L2) finds it useful.
const FW_REPLICAS = Ref(1)

# Sort-merge variant of `_spread_chunks`. Per chunk:
#   1. Compute per-dim stencil indices `idxpd[d] :: (cs, w)`.
#   2. Compute the flat linear index per (point, k_1, .., k_D) update via
#      `lin = i_1 + (i_2 - 1)*ngrid_1 + (i_3 - 1)*ngrid_1*ngrid_2 + ...`,
#      reshaped to a single (cs * w^D,) tensor.
#   3. `sortperm` the linear index and gather both the index tensor and
#      the corresponding contribution slab by that perm.
#   4. Reshape `fw` to `(prod(ngrid), ntrans)` and call a 1D scatter with
#      `indices_are_sorted=true`. XLA-GPU is expected to switch to a
#      contiguous-accumulate code path here (the same path that hurt us
#      with weakly-clustered indices is finally a fast path with strict
#      ordering).
function _spread_chunks_sortmerge(
    fw, cmat, perm, mask, base_full, frac_full, coefs,
    ngrid::NTuple{ND,Int}, w::Int, ntrans::Int,
    nchunks::Int, cs::Int, offsets_row::AbstractMatrix, dimval::Val{ND},
) where {ND}
    nd = ND
    wD = w^nd
    Ntotal = prod(ngrid)
    CT = Reactant.unwrapped_eltype(eltype(fw))
    fw_flat = reshape(fw, Ntotal, ntrans)
    Nupd = cs * wD
    for k in 1:nchunks
        j0 = (k - 1) * cs + 1
        bs = ntuple(d -> _slice1(base_full[d], j0, cs), dimval)
        fr = ntuple(d -> _slice1(frac_full[d], j0, cs), dimval)
        cc = _gather_chunk_strengths(cmat, perm, mask, j0, cs, ntrans)

        wpd   = ntuple(d -> _per_dim_weights(coefs, fr[d]), dimval)
        idxpd = ntuple(d -> _per_dim_indices(bs[d], ngrid[d], offsets_row), dimval)
        weights = _outer_product_weights(wpd)               # (cs, wD)
        contrib = reshape(weights, cs, wD, 1) .* reshape(cc, cs, 1, ntrans)
        contrib_flat = reshape(contrib, Nupd, ntrans)       # (cs*wD, ntrans)

        # Build flat linear destination index of shape (cs*wD,).
        lin_idx_flat = _flat_linear_index(idxpd, ngrid, cs, wD, dimval)

        # Sort by destination index. Two gathers (one for each tensor); the
        # `Ops.sort` overload that takes multiple co-sortable arrays would
        # save a gather but isn't trivially exposed.
        perm = sortperm(lin_idx_flat)
        idx_sorted     = lin_idx_flat[perm]
        contrib_sorted = contrib_flat[perm, :]

        # Scatter into the flattened fw with indices_are_sorted=true.
        idx_col = reshape(idx_sorted, Nupd, 1)
        fw_flat = _scatter_add_1d_sorted!(fw_flat, idx_col, contrib_sorted)
    end
    return reshape(fw_flat, ngrid..., ntrans)
end

# Compute (cs * w^D,) flat linear index from per-dim (cs, w) indices.
@inline function _flat_linear_index(
    idxpd::NTuple{1,AbstractMatrix}, ::NTuple{1,Int}, cs::Int, wD::Int, ::Val{1},
)
    return reshape(idxpd[1], cs * wD)
end
@inline function _flat_linear_index(
    idxpd::NTuple{2,AbstractMatrix}, ngrid::NTuple{2,Int}, cs::Int, wD::Int, ::Val{2},
)
    s1 = ngrid[1]
    a = reshape(idxpd[1], cs, w_of(idxpd), 1)
    b = reshape(idxpd[2], cs, 1, w_of(idxpd))
    lin = a .+ (b .- 1) .* s1
    return reshape(lin, cs * wD)
end
@inline function _flat_linear_index(
    idxpd::NTuple{3,AbstractMatrix}, ngrid::NTuple{3,Int}, cs::Int, wD::Int, ::Val{3},
)
    s1 = ngrid[1]
    s2 = ngrid[1] * ngrid[2]
    w_ = w_of(idxpd)
    a = reshape(idxpd[1], cs, w_, 1, 1)
    b = reshape(idxpd[2], cs, 1, w_, 1)
    c = reshape(idxpd[3], cs, 1, 1, w_)
    lin = a .+ (b .- 1) .* s1 .+ (c .- 1) .* s2
    return reshape(lin, cs * wD)
end
@inline w_of(idxpd::NTuple{D,AbstractMatrix}) where {D} = size(idxpd[1], 2)

# 1D scatter-add into fw_flat (Ntotal, ntrans), indices column-vector
# (Nupd, 1) into the spatial dim, ntrans is window. Sets
# `indices_are_sorted=true` to invoke XLA's sorted-scatter code path.
function _scatter_add_1d_sorted!(
    fw_flat,
    scatter_idx::AbstractMatrix,
    updates::AbstractMatrix,
)
    CT = Reactant.unwrapped_eltype(eltype(fw_flat))
    fw_flat_t = Reactant.promote_to(Reactant.TracedRArray{CT,2}, fw_flat)
    res = Reactant.Ops.@opcall scatter(
        +,
        [fw_flat_t],
        Reactant.promote_to(Reactant.TracedRArray{Int,2}, scatter_idx),
        [Reactant.promote_to(Reactant.TracedRArray{CT,2}, updates)];
        update_window_dims=Int64[2],          # ntrans axis of updates
        inserted_window_dims=Int64[1],        # spatial dim of fw_flat
        input_batching_dims=Int64[],
        scatter_indices_batching_dims=Int64[],
        scatter_dims_to_operand_dims=Int64[1],
        index_vector_dim=Int64(2),
        indices_are_sorted=true,
    )
    return only(res)
end

# Static-length 1D dynamic_slice: returns operand[j0+1 : j0+cs] (1-based).
@inline function _slice1(operand, j0, cs::Int)
    T = Reactant.unwrapped_eltype(eltype(operand))
    return Reactant.Ops.@opcall dynamic_slice(
        Reactant.promote_to(Reactant.TracedRArray{T,1}, operand),
        Any[j0],
        Int64[cs],
    )
end

# 2D dynamic_slice along dim 1, full along dim 2.
@inline function _slice2(operand, j0, cs::Int, ntrans::Int)
    T = Reactant.unwrapped_eltype(eltype(operand))
    return Reactant.Ops.@opcall dynamic_slice(
        Reactant.promote_to(Reactant.TracedRArray{T,2}, operand),
        Any[j0, 1],            # 1-based (Reactant subtracts 1 internally)
        Int64[cs, ntrans],
    )
end
