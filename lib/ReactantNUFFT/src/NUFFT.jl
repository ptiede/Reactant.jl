"""
    AbstractNUFFTMethod

Marker abstract type for selecting a NUFFT algorithm in `NUFFTOptions.method`.

To define your own method tag, create a subtype such as

```julia
struct MyMethod <: AbstractNUFFTMethod end
```

and pass an instance with `method=MyMethod()`.

A custom method should implement:

```julia
execute_type1(method::MyMethod, prep::PreparedNUFFTPlan, c::AbstractVector)
execute_type2(method::MyMethod, prep::PreparedNUFFTPlan, fk::AbstractArray)
```

The public `execute_nufft` entry point dispatches through `prep.plan.opts.method`,
so these overloads are the main extension hook for new execution strategies.
"""
abstract type AbstractNUFFTMethod end

"""
    AbstractNUFFTKernel

Abstract type for NUFFT spreading/interpolation kernels used by `NUFFTOptions.kernel`.

To define a custom kernel, create a subtype such as

```julia
struct MyKernel <: AbstractNUFFTKernel end
```

and pass an instance with `kernel=MyKernel()`.

A custom kernel only needs to implement:

```julia
kernel_profile(kernel::MyKernel, t, opts::NUFFTOptions)
```

Here `t` is the normalized distance from the grid point, clipped to `[0, 1]`.
The NUFFT code automatically zeros the kernel outside its support, so custom
kernels only need to describe the profile on that interval.

For performance-sensitive kernels, you can optionally overload `prepare_kernel`
and `kernel_weights` to precompute lookup tables or use another specialized
evaluation strategy, but most custom kernels should not need that.
"""
abstract type AbstractNUFFTKernel end

function kernel_profile end

struct NUPtsDriven <: AbstractNUFFTMethod end
struct SubProb <: AbstractNUFFTMethod end
struct OutputDriven <: AbstractNUFFTMethod end
struct ExpSemicircleKernel <: AbstractNUFFTKernel end

struct PreparedExpSemicircleKernel{A}
    table::A
    nspread::Int
    np::Int
end

"""
    NUFFTOptions{T}

NUFFT planning and execution options.
"""
Base.@kwdef struct NUFFTOptions{T<:Number,M<:AbstractNUFFTMethod,K<:AbstractNUFFTKernel}
    eps::T = T(1.0e-6)
    upsampfac::T = T(2)
    nspread::Int = 4
    method::M = OutputDriven()
    kernel::K = ExpSemicircleKernel()
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
    method::M=OutputDriven(),
    kernel::K=ExpSemicircleKernel(),
    sort_points::Bool=true,
    binsize::Union{Integer,NTuple{3,<:Integer}}=(16, 16, 16),
    maxsubprobsize::Integer=1024,
    np::Integer=128,
) where {T<:Number,M<:AbstractNUFFTMethod,K<:AbstractNUFFTKernel}
    b = binsize isa Integer ? (Int(binsize), Int(binsize), Int(binsize)) :
        (Int(binsize[1]), Int(binsize[2]), Int(binsize[3]))
    @assert nspread > 0 "nspread must be positive"
    @assert all(>(0), b) "binsize entries must be positive"
    @assert maxsubprobsize > 0 "maxsubprobsize must be positive"
    @assert np > 0 "np must be positive"
    return NUFFTOptions{T,M,K}(
        T(eps),
        T(upsampfac),
        Int(nspread),
        method,
        kernel,
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
    plan_nufft(::Type{T}, nufft_type, nmodes; iflag=-1, kwargs...)

Create a NUFFT plan.
"""
function plan_nufft(
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
    ngrid = oversampled_grid(modes, o.upsampfac, o.nspread)
    return NUFFTPlan{T,D,typeof(o)}(type_i, modes, ngrid, iflag >= 0 ? 1 : -1, o)
end

plan_nufft(::Type{T}, nufft_type::Integer, nmode::Integer; kwargs...) where {T} =
    plan_nufft(T, nufft_type, (Int(nmode),); kwargs...)

function plan_nufft(
    points::NTuple{D,<:AbstractVector},
    nufft_type::Integer,
    nmodes::NTuple{D,<:Integer};
    kwargs...,
) where {D}
    T = float(real(unwrapped_eltype(points[1])))
    return set_nufft_points(plan_nufft(T, nufft_type, nmodes; kwargs...), points)
end

"""
    set_nufft_points(plan, points)
    set_nufft_points(plan, x, y, ...)

Prepare point metadata for a previously created NUFFT plan.
"""
function set_nufft_points(plan::NUFFTPlan{T,D}, points::NTuple{D,<:AbstractVector}) where {T,D}
    validate_points(points)
    M = length(points[1])

    binsize = ntuple(d -> max(1, plan.opts.binsize[d]), Val(D))
    nbins = ntuple(d -> max(1, cld(plan.ngrid[d], binsize[d])), Val(D))

    points_scaled = ntuple(
        d -> mod.(points[d], 2π) .* (plan.ngrid[d] / (T(2π))),
        Val(D),
    )
    pointbins = point_bins(points_scaled, binsize, nbins)

    do_sort = plan.opts.sort_points && !any(p -> p isa AnyTracedRArray, points)
    idxnupts = if do_sort
        sortperm(pointbins)
    else
        1:M
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
    set_nufft_points(plan, (x, xs...))
set_nufft_points!(plan::NUFFTPlan, points...) = set_nufft_points(plan, points...)

"""
    execute_nufft(prepared_plan, data)

Execute a prepared NUFFT plan.
"""
function execute_nufft(prep::PreparedNUFFTPlan, data::AbstractArray)
    plan = prep.plan
    if plan.nufft_type == 1
        return execute_type1(prep, data)
    elseif plan.nufft_type == 2
        return execute_type2(prep, data)
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
    nmodes::Dims{D};
    iflag::Integer=-1,
    kwargs...,
) where {D}
    @assert length(c) == length(points[1]) "Strength count must match number of points"

    prep = plan_nufft(points, 1, nmodes; iflag, kwargs...)
    return execute_nufft(prep, c)
end

nufft_type1(
    x::AbstractVector,
    c::AbstractVector,
    nmode::Integer;
    kwargs...,
) = nufft_type1((x,), c, (nmode,); kwargs...)

"""
    nufft_type2(points, fk; kwargs...)

Type-2 NUFFT convenience API.
"""
function nufft_type2(
    points::NTuple{D,<:AbstractVector},
    fk::AbstractArray{<:RComplex,D};
    iflag::Integer=-1,
    kwargs...,
) where {D}
    nmodes = size(fk)
    prep = plan_nufft(points, 2, nmodes; iflag, kwargs...)
    return execute_nufft(prep, fk)
end

nufft_type2(x::AbstractVector, fk::AbstractVector; kwargs...) = nufft_type2((x,), fk; kwargs...)

function execute_type1(prep::PreparedNUFFTPlan, c::AbstractArray)
    @assert ndims(c) == 1 "Type-1 strengths must be a vector"
    @assert length(c) == length(prep.points[1]) "Strength count must match number of points"
    return execute_type1(prep.plan.opts.method, prep, c)
end

function execute_type2(prep::PreparedNUFFTPlan, fk::AbstractArray)
    plan = prep.plan
    @assert size(fk) == plan.nmodes "Mode array shape must match plan.nmodes"
    return execute_type2(prep.plan.opts.method, prep, fk)
end

function execute_type1(::OutputDriven, prep::PreparedNUFFTPlan, c::AbstractVector)
    plan = prep.plan
    T = typeof(plan.opts.eps)
    grid = spread_outputdriven(prep.points_scaled, c, plan.ngrid, plan.opts, T)
    grid_hat = fft_with_iflag(grid, plan.iflag)
    return extract_modes(grid_hat, plan.nmodes, plan.ngrid)
end

execute_type1(::NUPtsDriven, prep::PreparedNUFFTPlan, c::AbstractVector) =
    execute_type1(OutputDriven(), prep, c)

execute_type1(::SubProb, prep::PreparedNUFFTPlan, c::AbstractVector) =
    execute_type1(OutputDriven(), prep, c)

function execute_type2(::OutputDriven, prep::PreparedNUFFTPlan, fk::AbstractArray)
    plan = prep.plan
    grid_hat = embed_modes(fk, plan.nmodes, plan.ngrid)
    grid = fft_with_iflag(grid_hat, plan.iflag)
    return interp_outputdriven(prep.points_scaled, grid, plan.ngrid, plan.opts)
end

execute_type2(::NUPtsDriven, prep::PreparedNUFFTPlan, fk::AbstractArray) =
    execute_type2(OutputDriven(), prep, fk)

execute_type2(::SubProb, prep::PreparedNUFFTPlan, fk::AbstractArray) =
    execute_type2(OutputDriven(), prep, fk)

@inline function stencil_contribution(
    stencil_id,
    points_scaled::Tuple{Vararg{<:AbstractVector,D}},
    bases::Tuple{Vararg{<:AbstractVector,D}},
    ngrid::NTuple{D,Int},
    opts::NUFFTOptions,
    prepared_kernel,
    realT::Type{<:Number},
) where {D}
    nspread = opts.nspread
    offset = stencil_offset(stencil_id, nspread, 1)
    lin, wt = dim_stencil(
        points_scaled[1], bases[1], ngrid[1], offset, opts, prepared_kernel, realT
    )
    stride = ngrid[1]
    @inbounds for d in 2:D
        offset = stencil_offset(stencil_id, nspread, d)
        idx_d, wt_d = dim_stencil(
            points_scaled[d], bases[d], ngrid[d], offset, opts, prepared_kernel, realT
        )
        lin = lin .+ (idx_d .- 1) .* stride
        wt = wt .* wt_d
        d < D && (stride *= ngrid[d])
    end
    return lin, wt
end

function spread_outputdriven(
    points_scaled::Tuple{Vararg{<:AbstractVector,D}},
    c::AbstractVector{<:RComplex},
    ngrid::NTuple{D,Int},
    opts::NUFFTOptions,
    realT::Type{<:Number},
) where {D}
    CT = unwrapped_eltype(c)
    ncombos = opts.nspread^D
    grid_flat = similar(c, CT, (prod(ngrid),))
    prepared_kernel = prepare_kernel(opts.kernel, realT, opts)
    bases = ntuple(d -> floor.(Int, points_scaled[d]), Val(D))

    @allowscalar @trace for stencil_id in 1:ncombos
        lin, wt = stencil_contribution(
            stencil_id, points_scaled, bases, ngrid, opts, prepared_kernel, realT
        )
        scatter_add_flat!(grid_flat, lin, c .* wt)
    end
    return reshape(grid_flat, ngrid)
end

function interp_outputdriven(
    points_scaled::Tuple{Vararg{<:AbstractVector,D}},
    grid::AbstractArray,
    ngrid::NTuple{D,Int},
    opts::NUFFTOptions,
) where {D}
    realT = typeof(opts.eps)
    ncombos = opts.nspread^D
    grid_flat = vec(grid)
    CT = unwrapped_eltype(grid)
    prepared_kernel = prepare_kernel(opts.kernel, realT, opts)
    bases = ntuple(d -> floor.(Int, points_scaled[d]), Val(D))

    out = similar(grid, CT, (length(points_scaled[1]),))
    @allowscalar @trace for stencil_id in 1:ncombos
        lin, wt = stencil_contribution(
            stencil_id, points_scaled, bases, ngrid, opts, prepared_kernel, realT
        )
        copyto!(out, out .+ grid_flat[lin] .* wt)
    end
    return out
end

function dim_stencil(
    x::AbstractVector,
    base::AbstractVector,
    ng::Number,
    offset::Number,
    opts::NUFFTOptions,
    prepared_kernel,
    ::Type{T},
) where {T<:Number}
    ngT = convert_scalar(T, ng)
    idx = mod.(base .+ offset, ng) .+ 1
    dist = abs.(x .- (idx .- 1))
    dist = min.(dist, ngT .- dist)
    wt = kernel_weights(prepared_kernel, dist, opts)
    return idx, wt
end

@inline function stencil_offset(stencil_id, nspread::Int, dim::Int)
    q = stencil_id - 1
    for _ in 1:(dim - 1)
        q = fld(q, nspread)
    end
    return mod(q, nspread) - fld(nspread, 2)
end

function mode_linear_indices(
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

function extract_modes(
    grid_hat::AnyTracedRArray, nmodes::NTuple{D,Int}, ngrid::NTuple{D,Int}
) where {D}
    lin = vec(mode_linear_indices(nmodes, ngrid))
    return reshape(vec(grid_hat)[lin], nmodes)
end

function embed_modes(
    fk::AnyTracedRArray, nmodes::NTuple{D,Int}, ngrid::NTuple{D,Int}
) where {D}
    CT = unwrapped_eltype(fk)
    grid_flat = similar(fk, CT, (prod(ngrid),))
    lin = vec(mode_linear_indices(nmodes, ngrid))
    grid_flat[lin] = vec(fk)
    return reshape(grid_flat, ngrid)
end

function fft_with_iflag(grid::AnyTracedRArray{T,D}, iflag::Integer) where {T,D}
    dims = ntuple(i -> D - i + 1, Val(D))
    if iflag < 0
        return AbstractFFTs.fft(grid, dims)
    end
    return AbstractFFTs.bfft(grid, dims)
end

function scatter_add_flat!(
    dest::AnyTracedRArray{T,1},
    lin::AbstractVector,
    updates::AbstractVector,
) where {T}
    y = dest
    idx = lin
    upd = promote_to(TracedRArray{T,1}, updates)
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

function point_bins(
    points_scaled::NTuple{D,<:AbstractVector},
    binsize::NTuple{D,Int},
    nbins::NTuple{D,Int},
) where {D}
    bins = dim_bins(points_scaled[1], binsize[1], nbins[1])
    stride = nbins[1]
    for d in 2:D
        bd = dim_bins(points_scaled[d], binsize[d], nbins[d])
        bins = bins .+ (bd .- 1) .* stride
        stride *= nbins[d]
    end
    return bins
end

function dim_bins(x::AbstractVector, binsize::Int, nbins::Int)
    return clamp.(floor.(Int, x ./ binsize) .+ 1, 1, nbins)
end

function expsemicircle_beta(::Type{T}, nspread::Int, upsampfac) where {T<:Number}
    # FINUFFT-inspired ES-kernel beta scaling for upsampfac ~= 2.
    gamma = T(2.30) * (T(2) / max(T(1), T(upsampfac)))
    return gamma * T(nspread)
end

prepare_kernel(kernel::AbstractNUFFTKernel, ::Type{<:Number}, ::NUFFTOptions) = kernel

function kernel_weights(kernel::AbstractNUFFTKernel, dist, opts::NUFFTOptions)
    T = float(real(unwrapped_eltype(dist)))
    halfwidth = convert_scalar(T, opts.nspread) / T(2)
    t = dist ./ halfwidth
    vals = kernel_profile(kernel, clamp.(t, zero(T), one(T)), opts)
    return ifelse.(t .<= one(T), vals, zero(T))
end

function prepare_kernel(::ExpSemicircleKernel, ::Type{T}, opts::NUFFTOptions) where {T<:Number}
    beta = expsemicircle_beta(T, opts.nspread, float(opts.upsampfac))
    table = promote_to(TracedRArray{T,1}, expsemicircle_samples(T, beta, opts.np))
    return PreparedExpSemicircleKernel(table, opts.nspread, opts.np)
end

function expsemicircle_samples(::Type{T}, beta, np::Int) where {T<:Number}
    t = collect(range(zero(T), one(T); length=np + 1))
    inside = max.(zero(T), one(T) .- (t .* t))
    return exp.(beta .* (sqrt.(inside) .- one(T)))
end

function kernel_profile(::ExpSemicircleKernel, t, opts::NUFFTOptions)
    T = float(real(unwrapped_eltype(t)))
    beta = expsemicircle_beta(T, opts.nspread, float(opts.upsampfac))
    inside = max.(zero(T), one(T) .- (t .* t))
    return exp.(beta .* (sqrt.(inside) .- one(T)))
end

function kernel_weights(kernel::PreparedExpSemicircleKernel, dist, ::NUFFTOptions)
    T = float(real(unwrapped_eltype(dist)))
    halfwidth = convert_scalar(T, kernel.nspread) / T(2)
    t = dist ./ halfwidth
    tclip = clamp.(t, zero(T), one(T))
    npT = convert_scalar(T, kernel.np)
    u = tclip .* npT
    i0 = clamp.(floor.(Int, u) .+ 1, 1, kernel.np + 1)
    i1 = clamp.(i0 .+ 1, 1, kernel.np + 1)
    frac = u .- (i0 .- 1)

    v0 = sample_kernel_table(kernel.table, i0)
    v1 = sample_kernel_table(kernel.table, i1)
    vals = (one(T) .- frac) .* v0 .+ frac .* v1
    return ifelse.(t .<= one(T), vals, zero(T))
end

convert_scalar(::Type{T}, x::Number) where {T<:Number} = T(x)
convert_scalar(::Type{T}, x::TracedRNumber) where {T<:Number} = promote_to(TracedRNumber{T}, x)

function sample_kernel_table(kernel_data::AbstractVector, idx::AbstractArray)
    return reshape(kernel_data[vec(idx)], size(idx))
end

function direct_type1(
    points::NTuple{D,<:AbstractVector},
    c::AbstractVector{<:RComplex},
    nmodes::NTuple{D,Int},
    iflag::Integer,
) where {D}
    phase = phase_tensor(points, nmodes, iflag)
    kernel = cis.(phase)
    cview = reshape(c, (length(c), ntuple(_ -> 1, D)...))
    return dropdims(sum(cview .* kernel; dims=1); dims=1)
end

function direct_type2(
    points::NTuple{D,<:AbstractVector},
    fk::AbstractArray{<:RComplex},
    iflag::Integer,
) where {D}
    nmodes = ntuple(d -> size(fk, d), D)
    phase = phase_tensor(points, nmodes, iflag)
    kernel = cis.(phase)
    fkview = reshape(fk, (1, size(fk)...))
    rdims = ntuple(i -> i + 1, D)
    return dropdims(sum(fkview .* kernel; dims=rdims); dims=rdims)
end

function phase_tensor(
    points::NTuple{D,<:AbstractVector},
    nmodes::NTuple{D,Int},
    iflag::Integer,
) where {D}
    validate_points(points)
    realT = float(real(unwrapped_eltype(points[1])))
    M = length(points[1])
    phase = similar(points[1], realT, (M, nmodes...))
    ones_tail = ntuple(_ -> 1, Val(D))
    for d in 1:D
        x = reshape(mod.(points[d], 2 * float(pi)), (M, ones_tail...))
        md = reshape(
            realT.(centered_mode_axis(nmodes[d])),
            ntuple(i -> i == d + 1 ? nmodes[d] : 1, Val(D + 1)),
        )
        phase = phase .+ x .* md
    end
    return realT(iflag) .* phase
end

function centered_mode_axis(n::Int)
    return -fld(n, 2):(cld(n, 2) - 1)
end

function oversampled_grid(nmodes::NTuple{D,Int}, upsampfac, nspread) where {D}
    return ntuple(
        d -> max(nmodes[d], Int(ceil(float(upsampfac) * nmodes[d])), 2 * nspread + 1),
        Val(D),
    )
end

function validate_points(points::NTuple{D,<:AbstractVector}) where {D}
    @assert D >= 1 "Need at least one coordinate dimension"
    @assert D <= 3 "NUFFT currently supports dimensions 1, 2, and 3"
    M = length(points[1])
    for d in 2:D
        @assert length(points[d]) == M "All coordinate vectors must have the same length"
    end
    return nothing
end
