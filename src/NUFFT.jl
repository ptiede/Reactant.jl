const NUFFT_METHOD_NUPTSDRIVEN = :nuptsdriven
const NUFFT_METHOD_SUBPROBLEM = :subproblem
const NUFFT_METHOD_OUTPUTDRIVEN = :outputdriven

"""
    NUFFTOptions{T}

NUFFT planning and execution options.
"""
Base.@kwdef struct NUFFTOptions{T<:Number}
    eps::T = T(1.0e-6)
    upsampfac::T = T(2)
    nspread::Int = 4
    method::Symbol = NUFFT_METHOD_OUTPUTDRIVEN
    sort_points::Bool = true
    binsize::NTuple{3,Int} = (16, 16, 16)
    maxsubprobsize::Int = 1024
    np::Int = 128
end

function NUFFTOptions(
    ::Type{T};
    eps::Real=1.0e-6,
    upsampfac::Real=2,
    nspread::Integer=4,
    method::Union{Integer,Symbol}=NUFFT_METHOD_OUTPUTDRIVEN,
    sort_points::Bool=true,
    binsize::Union{Integer,NTuple{3,<:Integer}}=(16, 16, 16),
    maxsubprobsize::Integer=1024,
    np::Integer=128,
) where {T<:Number}
    m = _normalize_nufft_method(method)
    b = binsize isa Integer ? (Int(binsize), Int(binsize), Int(binsize)) :
        (Int(binsize[1]), Int(binsize[2]), Int(binsize[3]))
    @assert nspread > 0 "nspread must be positive"
    @assert all(>(0), b) "binsize entries must be positive"
    @assert maxsubprobsize > 0 "maxsubprobsize must be positive"
    @assert np > 0 "np must be positive"
    return NUFFTOptions{T}(
        T(eps),
        T(upsampfac),
        Int(nspread),
        m,
        sort_points,
        b,
        Int(maxsubprobsize),
        Int(np),
    )
end

struct NUFFTPlan{T<:Number,D,O<:NUFFTOptions{T}}
    nufft_type::Int
    nmodes::NTuple{D,Int}
    ngrid::NTuple{D,Int}
    iflag::Int
    opts::O
end

struct PreparedNUFFTPlan{P,S,PS,I,B,N,PB}
    plan::P
    points::S
    points_scaled::PS
    idxnupts::I
    sortidx::I
    binsize::B
    nbins::N
    pointbins::PB
end

"""
    make_nufft_plan(::Type{T}, nufft_type, nmodes; iflag=-1, kwargs...)

Create a FINUFFT-style NUFFT plan.
"""
function make_nufft_plan(
    ::Type{T},
    nufft_type::Integer,
    nmodes::NTuple{D,<:Integer};
    iflag::Integer=-1,
    opts::NUFFTOptions{T}=NUFFTOptions(T),
    kwargs...,
) where {T<:Number,D}
    o = isempty(kwargs) ? opts : NUFFTOptions(T; kwargs...)
    type_i = Int(nufft_type)
    @assert 1 <= type_i <= 3 "NUFFT type must be 1, 2, or 3"
    modes = ntuple(i -> Int(nmodes[i]), D)
    @assert all(>(0), modes) "All mode sizes must be positive"
    ngrid = _nufft_oversampled_grid(modes, o.upsampfac, o.nspread)
    return NUFFTPlan{T,D,typeof(o)}(type_i, modes, ngrid, iflag >= 0 ? 1 : -1, o)
end

make_nufft_plan(::Type{T}, nufft_type::Integer, nmode::Integer; kwargs...) where {T} =
    make_nufft_plan(T, nufft_type, (Int(nmode),); kwargs...)

"""
    set_nufft_points(plan, points)
    set_nufft_points(plan, x, y, ...)

Prepare point metadata for a previously created NUFFT plan.
"""
function set_nufft_points(plan::NUFFTPlan{T,D}, points::NTuple{D,<:AbstractVector}) where {T,D}
    _validate_points(points)
    M = length(points[1])

    binsize = ntuple(d -> max(1, plan.opts.binsize[d]), D)
    nbins = ntuple(d -> max(1, cld(plan.ngrid[d], binsize[d])), D)

    points_scaled = ntuple(
        d -> _nufft_wrap_2pi.(points[d]) .* (T(plan.ngrid[d]) / (T(2) * T(pi))),
        D,
    )
    pointbins = _nufft_pointbins(points_scaled, binsize, nbins)

    do_sort = plan.opts.sort_points && !_nufft_points_traced(points)
    idxnupts = if do_sort
        sortperm(pointbins)
    else
        Base.OneTo(M)
    end
    sortidx = if do_sort
        invperm(idxnupts)
    else
        idxnupts
    end

    return PreparedNUFFTPlan(plan, points, points_scaled, idxnupts, sortidx, binsize, nbins, pointbins)
end

set_nufft_points(plan::NUFFTPlan, x::AbstractVector) = set_nufft_points(plan, (x,))
set_nufft_points(plan::NUFFTPlan{T,D}, x::AbstractVector, xs::AbstractVector...) where {T,D} =
    set_nufft_points(plan, tuple(x, xs...))
set_nufft_points!(plan::NUFFTPlan, points...) = set_nufft_points(plan, points...)

"""
    execute_nufft(prepared_plan, data)

Execute a prepared NUFFT plan.
"""
function execute_nufft(prep::PreparedNUFFTPlan, data::AbstractArray)
    plan = prep.plan
    if plan.nufft_type == 1
        return _execute_type1(prep, data)
    elseif plan.nufft_type == 2
        return _execute_type2(prep, data)
    elseif plan.nufft_type == 3
        error("Type-3 NUFFT is not implemented yet")
    else
        error("Unsupported NUFFT type $(plan.nufft_type)")
    end
end

const RComplex = Union{Complex, TracedRNumber{<:Complex}}

"""
    nufft_type1(points, c, nmodes; kwargs...)

Type-1 NUFFT convenience API.
"""
function nufft_type1(
    points::NTuple{D,<:AbstractVector},
    c::AbstractVector{<:RComplex},
    nmodes::NTuple{D,<:Integer};
    iflag::Integer=-1,
    T::Union{Nothing,Type{<:Number}}=nothing,
    kwargs...,
) where {D}
    _validate_points(points)
    @assert length(c) == length(points[1]) "Strength count must match number of points"

    Treal = _nufft_real_type(T, c)
    plan = make_nufft_plan(Treal, 1, ntuple(i -> Int(nmodes[i]), D); iflag, kwargs...)
    prep = set_nufft_points(plan, points)
    return execute_nufft(prep, c)
end

nufft_type1(
    x::AbstractVector,
    c::AbstractVector,
    nmode::Integer;
    kwargs...,
) = nufft_type1((x,), c, (Int(nmode),); kwargs...)

"""
    nufft_type2(points, fk; kwargs...)

Type-2 NUFFT convenience API.
"""
function nufft_type2(
    points::NTuple{D,<:AbstractVector},
    fk::AbstractArray{<:RComplex,D};
    iflag::Integer=-1,
    T::Union{Nothing,Type{<:Number}}=nothing,
    kwargs...,
) where {D}
    _validate_points(points)

    Treal = _nufft_real_type(T, fk)
    nmodes = ntuple(d -> size(fk, d), D)
    plan = make_nufft_plan(Treal, 2, nmodes; iflag, kwargs...)
    prep = set_nufft_points(plan, points)
    return execute_nufft(prep, fk)
end

nufft_type2(x::AbstractVector, fk::AbstractVector; kwargs...) = nufft_type2((x,), fk; kwargs...)

# FINUFFT-style aliases.
const nufft_makeplan = make_nufft_plan
const nufft_execute = execute_nufft
const nufft_setpts = set_nufft_points
const nufft_setpts! = set_nufft_points!

function _execute_type1(prep::PreparedNUFFTPlan, c::AbstractArray)
    plan = prep.plan
    @assert ndims(c) == 1 "Type-1 strengths must be a vector"
    @assert length(c) == length(prep.points[1]) "Strength count must match number of points"
    return _execute_outputdriven_type1(prep, c)
end

function _execute_type2(prep::PreparedNUFFTPlan, fk::AbstractArray)
    plan = prep.plan
    @assert size(fk) == plan.nmodes "Mode array shape must match plan.nmodes"
    return _execute_outputdriven_type2(prep, fk)
end

function _execute_outputdriven_type1(prep::PreparedNUFFTPlan, c::AbstractVector)
    plan = prep.plan
    T = typeof(float(plan.opts.eps))
    grid = _nufft_spread_outputdriven(prep.points_scaled, c, plan.ngrid, plan.opts, T)
    grid_hat = _nufft_fft_with_iflag(grid, plan.iflag)
    return _nufft_extract_modes(grid_hat, plan.nmodes, plan.ngrid, plan.opts, T)
end

function _execute_outputdriven_type2(prep::PreparedNUFFTPlan, fk::AbstractArray)
    plan = prep.plan
    T = typeof(float(plan.opts.eps))
    grid_hat = _nufft_embed_modes(fk, plan.nmodes, plan.ngrid)
    grid = _nufft_fft_with_iflag(grid_hat, plan.iflag)
    return _nufft_interp_outputdriven(prep.points_scaled, grid, plan.ngrid, plan.opts, T)
end

function _nufft_spread_outputdriven(
    points_scaled::Tuple{Vararg{<:AbstractVector}},
    c::AbstractVector{<:RComplex},
    ngrid::Tuple{Vararg{Int}},
    opts::NUFFTOptions,
    realT::Type{<:Number},
)
    D = length(points_scaled)
    CT = unwrapped_eltype(c)
    nspread = opts.nspread
    np = opts.np
    ncombos = nspread^D
    grid_flat = similar(c, CT, (prod(ngrid),))
    beta = _nufft_kernel_beta(realT, nspread, float(opts.upsampfac))
    kernel_lut = _nufft_kernel_lut(realT, nspread, beta, np)
    bases = ntuple(d -> _nufft_floor_int64(points_scaled[d]), D)

    if D == 1
        @allowscalar @trace for ci in Base.OneTo(ncombos)
            off1 = _nufft_combo_offset(ci, nspread, 1)
            lin, w = _nufft_dim_stencil_offset(
                points_scaled[1], bases[1], ngrid[1], off1, nspread, kernel_lut, np, realT
            )
            updates = _nufft_vec_mul(c, w)
            _nufft_scatter_add_flat!(grid_flat, lin, updates)
        end
    elseif D == 2
        @allowscalar @trace for ci in Base.OneTo(ncombos)
            off1 = _nufft_combo_offset(ci, nspread, 1)
            lin, w = _nufft_dim_stencil_offset(
                points_scaled[1], bases[1], ngrid[1], off1, nspread, kernel_lut, np, realT
            )
            off2 = _nufft_combo_offset(ci, nspread, 2)
            idx2, wt2 = _nufft_dim_stencil_offset(
                points_scaled[2], bases[2], ngrid[2], off2, nspread, kernel_lut, np, realT
            )
            lin = _nufft_lin_update(lin, idx2, ngrid[1])
            w = _nufft_vec_mul(w, wt2)
            updates = _nufft_vec_mul(c, w)
            _nufft_scatter_add_flat!(grid_flat, lin, updates)
        end
    else
        @assert D == 3
        @allowscalar @trace for ci in Base.OneTo(ncombos)
            off1 = _nufft_combo_offset(ci, nspread, 1)
            lin, w = _nufft_dim_stencil_offset(
                points_scaled[1], bases[1], ngrid[1], off1, nspread, kernel_lut, np, realT
            )
            lin_stride = promote_to(TracedRNumber{Int}, ngrid[1])
            off2 = _nufft_combo_offset(ci, nspread, 2)
            idx2, wt2 = _nufft_dim_stencil_offset(
                points_scaled[2], bases[2], ngrid[2], off2, nspread, kernel_lut, np, realT
            )
            lin = _nufft_lin_update(lin, idx2, lin_stride)
            w = _nufft_vec_mul(w, wt2)
            lin_stride *= ngrid[2]
            off3 = _nufft_combo_offset(ci, nspread, 3)
            idx3, wt3 = _nufft_dim_stencil_offset(
                points_scaled[3], bases[3], ngrid[3], off3, nspread, kernel_lut, np, realT
            )
            lin = _nufft_lin_update(lin, idx3, lin_stride)
            w = _nufft_vec_mul(w, wt3)
            updates = _nufft_vec_mul(c, w)
            _nufft_scatter_add_flat!(grid_flat, lin, updates)
        end
    end
    return reshape(grid_flat, ngrid)
end

function _nufft_interp_outputdriven(
    points_scaled::Tuple{Vararg{<:AbstractVector}},
    grid::AbstractArray,
    ngrid::Tuple{Vararg{Int}},
    opts::NUFFTOptions,
    realT::Type{<:Number},
)
    D = length(points_scaled)
    nspread = opts.nspread
    np = opts.np
    ncombos = nspread^D
    grid_flat = copy(vec(grid))
    CT = unwrapped_eltype(grid)
    beta = _nufft_kernel_beta(realT, nspread, float(opts.upsampfac))
    kernel_lut = _nufft_kernel_lut(realT, nspread, beta, np)
    bases = ntuple(d -> _nufft_floor_int64(points_scaled[d]), D)

    out = similar(grid, CT, (length(points_scaled[1]),))
    if D == 1
        @allowscalar @trace for ci in Base.OneTo(ncombos)
            off1 = _nufft_combo_offset(ci, nspread, 1)
            lin, w = _nufft_dim_stencil_offset(
                points_scaled[1], bases[1], ngrid[1], off1, nspread, kernel_lut, np, realT
            )
            _nufft_vec_axpy!(out, gather_flat(grid_flat, lin), w)
        end
    elseif D == 2
        @allowscalar @trace for ci in Base.OneTo(ncombos)
            off1 = _nufft_combo_offset(ci, nspread, 1)
            lin, w = _nufft_dim_stencil_offset(
                points_scaled[1], bases[1], ngrid[1], off1, nspread, kernel_lut, np, realT
            )
            off2 = _nufft_combo_offset(ci, nspread, 2)
            idx2, wt2 = _nufft_dim_stencil_offset(
                points_scaled[2], bases[2], ngrid[2], off2, nspread, kernel_lut, np, realT
            )
            lin = _nufft_lin_update(lin, idx2, ngrid[1])
            w = _nufft_vec_mul(w, wt2)
            _nufft_vec_axpy!(out, gather_flat(grid_flat, lin), w)
        end
    else
        @assert D == 3
        @allowscalar @trace for ci in Base.OneTo(ncombos)
            off1 = _nufft_combo_offset(ci, nspread, 1)
            lin, w = _nufft_dim_stencil_offset(
                points_scaled[1], bases[1], ngrid[1], off1, nspread, kernel_lut, np, realT
            )
            lin_stride = promote_to(TracedRNumber{Int}, ngrid[1])
            off2 = _nufft_combo_offset(ci, nspread, 2)
            idx2, wt2 = _nufft_dim_stencil_offset(
                points_scaled[2], bases[2], ngrid[2], off2, nspread, kernel_lut, np, realT
            )
            lin = _nufft_lin_update(lin, idx2, lin_stride)
            w = _nufft_vec_mul(w, wt2)
            lin_stride *= ngrid[2]
            off3 = _nufft_combo_offset(ci, nspread, 3)
            idx3, wt3 = _nufft_dim_stencil_offset(
                points_scaled[3], bases[3], ngrid[3], off3, nspread, kernel_lut, np, realT
            )
            lin = _nufft_lin_update(lin, idx3, lin_stride)
            w = _nufft_vec_mul(w, wt3)
            _nufft_vec_axpy!(out, gather_flat(grid_flat, lin), w)
        end
    end
    return out
end

function _nufft_lin_update(lin::AbstractVector, idx_d::AbstractArray, stride::Number)
    return vec(lin) .+ (vec(idx_d) .- 1) .* stride
end

function _nufft_vec_mul(a::AbstractVector, b::AbstractArray)
    return vec(a) .* vec(b)
end

function _nufft_vec_axpy!(out::AbstractVector, g::AbstractArray, w::AbstractArray)
    copyto!(out, vec(out) .+ vec(g) .* vec(w))
    return out
end

function _nufft_dim_stencil_offset(
    x::AbstractVector,
    base::AbstractVector,
    ng::Number,
    offset::Number,
    nspread::Number,
    kernel_lut,
    np::Number,
    ::Type{T},
) where {T<:Number}
    ngT = _nufft_convert_scalar(T, ng)
    idx = mod.(base .+ offset, ng) .+ 1
    x_idx = idx .- 1
    dist = abs.(x .- x_idx)
    dist = min.(dist, ngT .- dist)
    wt = _nufft_kernel_value(dist, nspread, kernel_lut, np, T)
    return idx, wt
end

@inline function _nufft_combo_offset(ci, nspread::Int, dim::Int)
    q = ci - 1
    @inbounds for _ in 1:(dim - 1)
        q = fld(q, nspread)
    end
    return mod(q, nspread) - fld(nspread, 2)
end

function _nufft_mode_linear_indices_traced(
    nmodes::NTuple{D,Int}, ngrid::NTuple{D,Int}
) where {D}
    shape = collect(Int, nmodes)
    iota1 = @opcall iota(Int, shape; iota_dimension=1)
    m1 = iota1 .- Int(fld(nmodes[1], 2))
    lin = mod.(m1, Int(ngrid[1])) .+ 1
    stride = ngrid[1]
    for d in 2:D
        iotad = @opcall iota(Int, shape; iota_dimension=d)
        md = iotad .- Int(fld(nmodes[d], 2))
        idxd = mod.(md, Int(ngrid[d])) .+ 1
        lin = lin .+ (idxd .- 1) .* stride
        stride *= ngrid[d]
    end
    return lin
end


function _nufft_extract_modes(
    grid_hat::AnyTracedRArray,
    nmodes::NTuple{D,Int},
    ngrid::NTuple{D,Int},
    opts::NUFFTOptions,
    ::Type{T},
) where {D,T<:Number}
    lin = vec(_nufft_mode_linear_indices_traced(nmodes, ngrid))
    vals = gather_flat(vec(grid_hat), lin)
    vals = reshape(vals, nmodes)
    return vals ./ denom
end

function _nufft_embed_modes(
    fk::AnyTracedRArray, nmodes::NTuple{D,Int}, ngrid::NTuple{D,Int}
) where {D}
    CT = unwrapped_eltype(fk)
    grid_flat = similar(fk, CT, (prod(ngrid),))
    lin = vec(_nufft_mode_linear_indices_traced(nmodes, ngrid))
    _nufft_scatter_set_flat!(grid_flat, lin, vec(fk))
    return reshape(grid_flat, ngrid)
end

function _nufft_fft_with_iflag(grid::AnyTracedRArray{T,D}, iflag::Integer) where {T,D}
    arr = TracedUtils.materialize_traced_array(grid)
    fft_lengths = Int[size(arr, d) for d in D:-1:1]
    if iflag < 0
        return @opcall fft(arr; type="FFT", length=fft_lengths)
    end
    return @opcall(fft(arr; type="IFFT", length=fft_lengths)) .* prod(size(arr))
end

function _nufft_fft_with_iflag(grid::AbstractArray, iflag::Integer)
    error(
        "Output-driven NUFFT execution requires traced arrays. Call via `@jit` for Reactant-native execution.",
    )
end

function _nufft_scatter_add_flat!(dest::AnyTracedRArray{T,1}, lin::AbstractArray, updates::AbstractArray) where {T}
    y = vec(dest)
    idx = vec(lin)
    upd = promote_to(TracedRArray{T,1}, vec(updates))
    scattered = @opcall scatter(
        +,
        [y],
        idx,
        [upd];
        update_window_dims=Int[],
        inserted_window_dims=Int[1],
        input_batching_dims=Int[],
        scatter_indices_batching_dims=Int[],
        scatter_dims_to_operand_dims=Int[1],
        index_vector_dim=Int(2),
    )
    copyto!(dest, only(scattered))
    return dest
end

function _nufft_scatter_set_flat!(dest::AnyTracedRArray{T,1}, lin::AbstractArray, updates::AbstractArray) where {T}
    dest[vec(lin)] = vec(updates)
    return dest
end

gather_flat(src, inds) = src[vec(inds)]



function _nufft_pointbins(
    points_scaled::NTuple{D,<:AbstractVector},
    binsize::NTuple{D,Int},
    nbins::NTuple{D,Int},
) where {D}
    bins = _nufft_dim_bins(points_scaled[1], binsize[1], nbins[1])
    stride = nbins[1]
    for d in 2:D
        bd = _nufft_dim_bins(points_scaled[d], binsize[d], nbins[d])
        bins = bins .+ (bd .- 1) .* stride
        stride *= nbins[d]
    end
    return bins
end

function _nufft_dim_bins(x::AbstractVector, binsize::Int, nbins::Int)
    return clamp.(_nufft_floor_int64(x ./ binsize) .+ 1, 1, nbins)
end

function _nufft_floor_int64(x::AnyTracedRArray)
    return floor.(Int, x)
end

_nufft_floor_int64(x::AbstractArray) = floor.(Int, x)

function _nufft_kernel_beta(::Type{T}, nspread::Int, upsampfac) where {T<:Number}
    # FINUFFT-inspired ES-kernel beta scaling for upsampfac ~= 2.
    gamma = T(2.30) * (T(2) / max(T(1), T(upsampfac)))
    return gamma * T(nspread)
end

function _nufft_kernel_lut(::Type{T}, nspread::Int, beta, np::Int) where {T<:Number}
    lut = _nufft_kernel_lut_values(T, nspread, beta, np)
    return promote_to(TracedRArray{T,1}, lut)
end

function _nufft_kernel_lut_values(::Type{T}, nspread::Int, beta, np::Int) where {T<:Number}
    t = collect(range(zero(T), one(T); length=np + 1))
    inside = max.(zero(T), one(T) .- (t .* t))
    return exp.(beta .* (sqrt.(inside) .- one(T)))
end

function _nufft_kernel_value(dist, nspread::Number, kernel_lut, np::Number, ::Type{T}) where {T<:Number}
    halfwidth = _nufft_convert_scalar(T, nspread) / T(2)
    t = dist ./ halfwidth
    tclip = clamp.(t, zero(T), one(T))
    npT = _nufft_convert_scalar(T, np)
    u = tclip .* npT
    i0 = clamp.(_nufft_floor_int64(u) .+ 1, 1, np + 1)
    i1 = clamp.(i0 .+ 1, 1, np + 1)
    frac = u .- (i0 .- 1)

    v0 = _nufft_kernel_lookup(kernel_lut, i0)
    v1 = _nufft_kernel_lookup(kernel_lut, i1)
    vals = (one(T) .- frac) .* v0 .+ frac .* v1
    return ifelse.(t .<= one(T), vals, zero(T))
end

_nufft_convert_scalar(::Type{T}, x::Number) where {T<:Number} = T(x)
_nufft_convert_scalar(::Type{T}, x::TracedRNumber) where {T<:Number} = promote_to(TracedRNumber{T}, x)

function _nufft_kernel_lookup(kernel_lut::AnyTracedRArray{T,1}, idx::AbstractArray) where {T}
    vals = gather_flat(kernel_lut, idx)
    return reshape(vals, size(idx))
end

function _nufft_kernel_lookup(kernel_lut::AbstractVector, idx::AbstractArray)
    return reshape(kernel_lut[vec(idx)], size(idx))
end

function _nufft_direct_type1(
    points::NTuple{D,<:AbstractVector},
    c::AbstractVector{<:RComplex},
    nmodes::NTuple{D,Int},
    iflag::Integer,
) where {D}
    return _nufft_direct_type1(points, c, nmodes, iflag, float(real(unwrapped_eltype(c))))
end

function _nufft_direct_type1(
    points::NTuple{D,<:AbstractVector},
    c::AbstractVector,
    nmodes::NTuple{D,Int},
    iflag::Integer,
    ::Type{T},
) where {D,T<:Number}
    phase = _nufft_phase_tensor(points, nmodes, iflag, T)
    kernel = cis.(phase)
    cview = reshape(c, (length(c), ntuple(_ -> 1, D)...))
    return dropdims(sum(cview .* kernel; dims=1); dims=1)
end

function _nufft_direct_type2(
    points::NTuple{D,<:AbstractVector},
    fk::AbstractArray{<:RComplex},
    iflag::Integer,
) where {D}
    return _nufft_direct_type2(points, fk, iflag, float(real(unwrapped_eltype(fk))))
end

function _nufft_direct_type2(
    points::NTuple{D,<:AbstractVector},
    fk::AbstractArray{<:Any,D},
    iflag::Integer,
    ::Type{T},
) where {D,T<:Number}
    nmodes = ntuple(d -> size(fk, d), D)
    phase = _nufft_phase_tensor(points, nmodes, iflag, T)
    kernel = cis.(phase)
    fkview = reshape(fk, (1, size(fk)...))
    rdims = ntuple(i -> i + 1, D)
    return dropdims(sum(fkview .* kernel; dims=rdims); dims=rdims)
end

function _nufft_phase_tensor(
    points::NTuple{D,<:AbstractVector},
    nmodes::NTuple{D,Int},
    iflag::Integer,
    ::Type{T},
) where {D,T<:Number}
    _validate_points(points)
    M = length(points[1])
    phase = similar(points[1], T, (M, nmodes...))
    ones_tail = ntuple(_ -> 1, D)
    for d in 1:D
        x = reshape(_nufft_wrap_2pi.(points[d]), (M, ones_tail...))
        md = reshape(
            _nufft_mode_axis(nmodes[d], T), ntuple(i -> i == d + 1 ? nmodes[d] : 1, D + 1)
        )
        phase = phase .+ x .* md
    end
    return T(iflag) .* phase
end

function _nufft_mode_axis_int(n::Int)
    return collect(-fld(n, 2):(cld(n, 2) - 1))
end

_nufft_mode_axis(n::Int, ::Type{T}) where {T} = T.(_nufft_mode_axis_int(n))

function _nufft_oversampled_grid(nmodes::NTuple{D,Int}, upsampfac, nspread) where {D}
    return ntuple(
        d -> max(nmodes[d], Int(ceil(float(upsampfac) * nmodes[d])), 2 * nspread + 1),
        D,
    )
end

@inline _nufft_wrap_2pi(x) = mod(x, 2 * float(pi))

function _validate_points(points::NTuple{D,<:AbstractVector}) where {D}
    @assert D >= 1 "Need at least one coordinate dimension"
    @assert D <= 3 "This NUFFT milestone supports dimensions 1, 2, and 3"
    M = length(points[1])
    for d in 2:D
        @assert length(points[d]) == M "All coordinate vectors must have the same length"
    end
    return nothing
end

function _normalize_nufft_method(method::Symbol)
    if !(
        method in (
            NUFFT_METHOD_NUPTSDRIVEN,
            NUFFT_METHOD_SUBPROBLEM,
            NUFFT_METHOD_OUTPUTDRIVEN,
        )
    )
        error(
            "Unknown NUFFT method $method. Supported methods: :nuptsdriven, :subproblem, :outputdriven.",
        )
    end
    return method
end

function _normalize_nufft_method(method::Integer)
    if method == 1
        return NUFFT_METHOD_NUPTSDRIVEN
    elseif method == 2
        return NUFFT_METHOD_SUBPROBLEM
    elseif method == 3
        return NUFFT_METHOD_OUTPUTDRIVEN
    end
    error("Unknown NUFFT method id $method. Supported ids are 1, 2, and 3.")
end

function _nufft_real_type(
    Tkw::Union{Nothing,Type{<:Number}}, arr::AbstractArray
)
    if Tkw !== nothing
        return Tkw
    end
    return float(real(unwrapped_eltype(arr)))
end
