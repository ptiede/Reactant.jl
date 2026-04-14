"""
    AbstractNUFFTMethod

Marker abstract type for selecting a NUFFT algorithm in `NUFFTOptions.method`.

To define your own method tag, create a subtype such as

```julia
struct MyMethod <: AbstractNUFFTMethod end
```

and pass an instance with `method=MyMethod()`.

A custom method will usually implement:

```julia
spread_to_grid(method::MyMethod, prep::PreparedNUFFTPlan, c::AbstractVector, kernel)
interp_from_grid(method::MyMethod, prep::PreparedNUFFTPlan, grid::AbstractArray, kernel)
```

The shared `execute_type1`/`execute_type2` pipeline handles FFTs and mode
embedding/extraction around those hooks. For full control, a custom method may
still overload `execute_type1` or `execute_type2` directly instead.

Built-in method tags are:

- `OutputDriven()`: traced stencil loop over output contributions
- `NUPtsDriven()`: unsorted point-driven batched spreading/interpolation
- `AutoMethod()`: lightweight heuristic selector between the built-in methods
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
kernel_profile(kernel::MyKernel, t)
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

"""
    method_kernel(method, prep)

Optional hook for preparing method-specific kernel data before spreading or
interpolation. The default implementation prepares `prep.plan.opts.kernel`.
"""
function method_kernel end

"""
    spread_to_grid(method, prep, c, kernel)

Hook used by the shared type-1 execution pipeline. It should return the
oversampled grid before the FFT step.
"""
function spread_to_grid end

"""
    interp_from_grid(method, prep, grid, kernel)

Hook used by the shared type-2 execution pipeline. It should interpolate values
from the post-FFT oversampled grid back to the nonuniform points.
"""
function interp_from_grid end

"""Unsorted point-driven NUFFT execution strategy."""
struct NUPtsDriven <: AbstractNUFFTMethod end
"""Output-driven NUFFT execution strategy."""
struct OutputDriven <: AbstractNUFFTMethod end
"""Heuristic selector between the built-in execution strategies."""
struct AutoMethod <: AbstractNUFFTMethod end
struct ExpSemicircleKernel <: AbstractNUFFTKernel end

struct PreparedExpSemicircleKernel{A,N,P}
    table::A
    nspread::N
    np::P
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
    np::Int = 128
end

function NUFFTOptions(
    ::Type{T};
    eps::Real=1.0e-6,
    upsampfac::Real=2,
    nspread::Integer=4,
    method::M=OutputDriven(),
    kernel::K=ExpSemicircleKernel(),
    np::Integer=128,
) where {T<:Number,M<:AbstractNUFFTMethod,K<:AbstractNUFFTKernel}
    @assert nspread > 0 "nspread must be positive"
    @assert np > 0 "np must be positive"
    return NUFFTOptions{T,M,K}(
        T(eps),
        T(upsampfac),
        Int(nspread),
        method,
        kernel,
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

struct PreparedNUFFTPlan{P,S,PS,B}
    plan::P
    points::S
    points_scaled::PS
    bases::B
end

function merge_options(opts::NUFFTOptions{T}; kwargs...) where {T}
    isempty(kwargs) && return opts
    return NUFFTOptions(T;
        eps=get(kwargs, :eps, opts.eps),
        upsampfac=get(kwargs, :upsampfac, opts.upsampfac),
        nspread=get(kwargs, :nspread, opts.nspread),
        method=get(kwargs, :method, opts.method),
        kernel=get(kwargs, :kernel, opts.kernel),
        np=get(kwargs, :np, opts.np),
    )
end

"""
    plan_nufft(::Type{T}, nufft_type, nmodes; iflag=-1, kwargs...)

Create a NUFFT plan.
"""
function plan_nufft(
    ::Type{T},
    nufft_type::Integer,
    nmodes::NTuple{D,Integer};
    iflag::Integer=-1,
    opts::NUFFTOptions{T}=NUFFTOptions(T),
    kwargs...,
) where {T<:Number,D}
    o = merge_options(opts; kwargs...)
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
    points::NTuple{D,AbstractVector},
    nufft_type::Integer,
    nmodes::NTuple{D,Integer};
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
function set_nufft_points(plan::NUFFTPlan{T,D}, points::NTuple{D,AbstractVector}) where {T,D}
    validate_points(points)
    period = T(2 * float(pi))
    points_scaled = ntuple(
        d -> mod.(points[d], period) .* (plan.ngrid[d] / period),
        Val(D),
    )
    bases = ntuple(d -> floor.(Int, points_scaled[d]), Val(D))
    return PreparedNUFFTPlan(plan, points, points_scaled, bases)
end

set_nufft_points(plan::NUFFTPlan, x::AbstractVector) = set_nufft_points(plan, (x,))
set_nufft_points(plan::NUFFTPlan{T,D}, x::AbstractVector, xs::AbstractVector...) where {T,D} =
    set_nufft_points(plan, (x, xs...))

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
    points::NTuple{D,AbstractVector},
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
    points::NTuple{D,AbstractVector},
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

real_type(prep::PreparedNUFFTPlan) = typeof(prep.plan.opts.eps)

function resolved_method(method::AbstractNUFFTMethod, nufft_type::Integer, ::PreparedNUFFTPlan)
    if nufft_type in (1, 2)
        return method
    end
    error("Unsupported NUFFT type $nufft_type")
end

function resolved_method(::AutoMethod, nufft_type::Integer, prep::PreparedNUFFTPlan)
    D = length(prep.plan.nmodes)
    if nufft_type == 1
        return D == 2 ? OutputDriven() : NUPtsDriven()
    elseif nufft_type == 2
        return OutputDriven()
    end
    error("Unsupported NUFFT type $nufft_type")
end

resolve_auto_method(nufft_type::Integer, prep::PreparedNUFFTPlan) =
    resolved_method(AutoMethod(), nufft_type, prep)

function method_kernel(method::AbstractNUFFTMethod, prep::PreparedNUFFTPlan)
    plan = prep.plan
    return prepare_kernel(plan.opts.kernel, real_type(prep), plan.opts)
end

function execute_type1(method::AbstractNUFFTMethod, prep::PreparedNUFFTPlan, c::AbstractVector)
    resolved = resolved_method(method, 1, prep)
    plan = prep.plan
    grid = spread_to_grid(resolved, prep, c, method_kernel(resolved, prep))
    grid_hat = fft_with_iflag(grid, plan.iflag)
    return extract_modes(grid_hat, plan.nmodes, plan.ngrid)
end

function execute_type2(method::AbstractNUFFTMethod, prep::PreparedNUFFTPlan, fk::AbstractArray)
    resolved = resolved_method(method, 2, prep)
    plan = prep.plan
    grid_hat = embed_modes(fk, plan.nmodes, plan.ngrid)
    grid = fft_with_iflag(grid_hat, plan.iflag)
    return interp_from_grid(resolved, prep, grid, method_kernel(resolved, prep))
end

function spread_to_grid(
    ::OutputDriven, prep::PreparedNUFFTPlan, c::AbstractVector, prepared_kernel
)
    plan = prep.plan
    return spread_outputdriven(
        prep.points_scaled,
        prep.bases,
        c,
        plan.ngrid,
        plan.opts.nspread,
        prepared_kernel,
        real_type(prep),
    )
end

function interp_from_grid(
    ::OutputDriven, prep::PreparedNUFFTPlan, grid::AbstractArray, prepared_kernel
)
    plan = prep.plan
    return interp_outputdriven(
        prep.points_scaled,
        prep.bases,
        grid,
        plan.ngrid,
        plan.opts.nspread,
        prepared_kernel,
        real_type(prep),
    )
end

function spread_to_grid(
    ::NUPtsDriven, prep::PreparedNUFFTPlan, c::AbstractVector, prepared_kernel
)
    plan = prep.plan
    return spread_nuptsdriven(
        prep.points_scaled,
        prep.bases,
        c,
        plan.ngrid,
        plan.opts.nspread,
        prepared_kernel,
        real_type(prep),
    )
end

function interp_from_grid(
    ::NUPtsDriven, prep::PreparedNUFFTPlan, grid::AbstractArray, prepared_kernel
)
    plan = prep.plan
    return interp_nuptsdriven(
        prep.points_scaled,
        prep.bases,
        grid,
        plan.ngrid,
        plan.opts.nspread,
        prepared_kernel,
        real_type(prep),
    )
end

@inline function stencil_offset_index(stencil_id, nspread::Number, dim::Int)
    q = stencil_id - 1
    for _ in 1:(dim - 1)
        q = fld(q, nspread)
    end
    return mod(q, nspread) + 1
end

function dim_stencil_cache(
    x::AbstractVector,
    base::AbstractVector,
    ng::Number,
    nspread::Number,
    prepared_kernel,
    ::Type{T},
) where {T<:Number}
    centered = fld(nspread, 2)
    return ntuple(
        i -> dim_stencil(
            x,
            base,
            ng,
            i - centered - 1,
            nspread,
            prepared_kernel,
            T,
        ),
        nspread,
    )
end

function outputdriven_stencil_cache(
    points_scaled::NTuple{D,AbstractVector},
    bases::NTuple{D,AbstractVector},
    ngrid::NTuple{D,Number},
    nspread::Number,
    prepared_kernel,
    realT::Type{<:Number},
) where {D}
    return ntuple(
        d -> dim_stencil_cache(
            points_scaled[d], bases[d], ngrid[d], nspread, prepared_kernel, realT
        ),
        D,
    )
end

@inline function stencil_contribution(
    stencil_id::Integer,
    cache,
    ngrid::NTuple{D,Number},
    nspread::Number,
) where {D}
    offset_i = stencil_offset_index(stencil_id, nspread, 1)
    lin, wt = cache[1][offset_i]
    stride = ngrid[1]
    @inbounds for d in 2:D
        offset_i = stencil_offset_index(stencil_id, nspread, d)
        idx_d, wt_d = cache[d][offset_i]
        lin = lin .+ (idx_d .- 1) .* stride
        wt = wt .* wt_d
        d < D && (stride *= ngrid[d])
    end
    return lin, wt
end

function spread_outputdriven(
    points_scaled::NTuple{D,AbstractVector},
    bases::NTuple{D,AbstractVector},
    c::AbstractVector{<:RComplex},
    ngrid::NTuple{D,Int},
    nspread::Int,
    prepared_kernel,
    realT::Type{<:Number},
) where {D}
    CT = unwrapped_eltype(c)
    ncombos = nspread^D
    grid_flat = similar(c, CT, (prod(ngrid),))
    fill!(grid_flat, zero(CT))
    cache = outputdriven_stencil_cache(
        points_scaled, bases, ngrid, nspread, prepared_kernel, realT
    )

    @allowscalar for stencil_id in 1:ncombos
        lin, wt = stencil_contribution(stencil_id, cache, ngrid, nspread)
        scatter_add_flat!(grid_flat, lin, c .* wt)
    end
    return reshape(grid_flat, ngrid)
end

function stencil_offset_ids(stencil_ids::AbstractUnitRange{<:Integer}, nspread::Int, ::Val{D}) where {D}
    return ntuple(
        d -> [stencil_offset_index(stencil_id, nspread, d) for stencil_id in stencil_ids],
        Val(D),
    )
end

stencil_offset_ids(nspread::Int, ::Val{D}) where {D} =
    stencil_offset_ids(1:(nspread^D), nspread, Val(D))

function cache_columns(dim_cache)
    idx = hcat((entry[1] for entry in dim_cache)...)
    wt = hcat((entry[2] for entry in dim_cache)...)
    return idx, wt
end

function point_stencil_tables(
    cache,
    ngrid::NTuple{D,Number},
    offset_ids::NTuple{D,AbstractVector},
) where {D}
    idx1, wt1 = cache_columns(cache[1])
    lin = idx1[:, offset_ids[1]]
    wt = wt1[:, offset_ids[1]]
    stride = ngrid[1]

    @inbounds for d in 2:D
        idxd, wtd = cache_columns(cache[d])
        idxd = idxd[:, offset_ids[d]]
        wtd = wtd[:, offset_ids[d]]
        lin = lin .+ (idxd .- 1) .* stride
        wt = wt .* wtd
        d < D && (stride *= ngrid[d])
    end

    return lin, wt
end

point_stencil_tables(cache, ngrid::NTuple{D,Number}, nspread::Int) where {D} =
    point_stencil_tables(cache, ngrid, stencil_offset_ids(nspread, Val(D)))

function spread_nuptsdriven(
    points_scaled::NTuple{D,AbstractVector},
    bases::NTuple{D,AbstractVector},
    c::AbstractVector{<:RComplex},
    ngrid::NTuple{D,Int},
    nspread::Int,
    prepared_kernel,
    realT::Type{<:Number},
) where {D}
    CT = unwrapped_eltype(c)
    grid_flat = similar(c, CT, (prod(ngrid),))
    fill!(grid_flat, zero(CT))
    ncombos = nspread^D
    cache = outputdriven_stencil_cache(
        points_scaled, bases, ngrid, nspread, prepared_kernel, realT
    )
    if should_stream_nuptsdriven_spread(Val(D), length(c), ncombos, ngrid)
        chunk_size = nupts_combo_chunk_size(nspread, Val(D))
        @allowscalar for combo_start in 1:chunk_size:ncombos
            combo_stop = combo_start + chunk_size - 1
            offset_ids = stencil_offset_ids(combo_start:combo_stop, nspread, Val(D))
            lin, wt = point_stencil_tables(cache, ngrid, offset_ids)
            updates = reshape(c, (length(c), 1)) .* wt
            scatter_add_flat!(grid_flat, vec(lin), vec(updates))
        end
    else
        lin, wt = point_stencil_tables(cache, ngrid, nspread)
        updates = reshape(c, (length(c), 1)) .* wt
        scatter_add_flat!(grid_flat, vec(lin), vec(updates))
    end

    return reshape(grid_flat, ngrid)
end

function interp_outputdriven(
    points_scaled::NTuple{D,AbstractVector},
    bases::NTuple{D,AbstractVector},
    grid::AbstractArray,
    ngrid::NTuple{D,Int},
    nspread::Int,
    prepared_kernel,
    realT::Type{<:Number},
) where {D}
    ncombos = nspread^D
    CT = unwrapped_eltype(grid)
    grid_flat = promote_to(TracedRArray{CT,1}, vec(grid))
    cache = outputdriven_stencil_cache(
        points_scaled, bases, ngrid, nspread, prepared_kernel, realT
    )

    out = similar(grid, CT, (length(points_scaled[1]),))
    fill!(out, zero(CT))
    @allowscalar for stencil_id in 1:ncombos
        lin, wt = stencil_contribution(stencil_id, cache, ngrid, nspread)
        vals = promote_to(TracedRArray{CT,1}, grid_flat[lin])
        copyto!(out, out .+ vals .* wt)
    end
    return out
end

function interp_nuptsdriven(
    points_scaled::NTuple{D,AbstractVector},
    bases::NTuple{D,AbstractVector},
    grid::AbstractArray,
    ngrid::NTuple{D,Int},
    nspread::Int,
    prepared_kernel,
    realT::Type{<:Number},
) where {D}
    CT = unwrapped_eltype(grid)
    grid_flat = promote_to(TracedRArray{CT,1}, vec(grid))
    cache = outputdriven_stencil_cache(
        points_scaled, bases, ngrid, nspread, prepared_kernel, realT
    )
    lin, wt = point_stencil_tables(cache, ngrid, nspread)
    vals = promote_to(TracedRArray{CT,1}, grid_flat[vec(lin)])
    vals = reshape(vals, size(lin))
    return vec(dropdims(sum(vals .* wt; dims=2); dims=2))
end

function dim_stencil(
    x::AbstractVector,
    base::AbstractVector,
    ng::Number,
    offset::Number,
    nspread::Number,
    prepared_kernel,
    ::Type{T},
) where {T<:Number}
    ngT = ng * one(T)
    idx = mod.(base .+ offset, ng) .+ 1
    dist = abs.(x .- (idx .- 1))
    dist = min.(dist, ngT .- dist)
    wt = kernel_weights(prepared_kernel, dist, nspread)
    return idx, wt
end

@inline function stencil_offset(stencil_id, nspread::Number, dim::Int)
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
        lin += (idxd .- 1) .* stride
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

function extract_modes(::AbstractArray, ::NTuple{D,Int}, ::NTuple{D,Int}) where {D}
    error(
        "Output-driven NUFFT execution requires traced arrays. Call via `@jit` for Reactant-native execution.",
    )
end

function embed_modes(
    fk::AnyTracedRArray, nmodes::NTuple{D,Int}, ngrid::NTuple{D,Int}
) where {D}
    CT = unwrapped_eltype(fk)
    grid_flat = similar(fk, CT, (prod(ngrid),))
    fill!(grid_flat, zero(CT))
    lin = vec(mode_linear_indices(nmodes, ngrid))
    scatter_add_flat!(grid_flat, lin, vec(fk))
    return reshape(grid_flat, ngrid)
end

function embed_modes(::AbstractArray, ::NTuple{D,Int}, ::NTuple{D,Int}) where {D}
    error(
        "Output-driven NUFFT execution requires traced arrays. Call via `@jit` for Reactant-native execution.",
    )
end

function fft_with_iflag(grid::AnyTracedRArray{T,D}, iflag::Integer) where {T,D}
    if iflag < 0
        return AbstractFFTs.fft(grid)
    end
    return AbstractFFTs.bfft(grid)
end

function scatter_add_flat!(
    dest::AnyTracedRArray{T,1},
    lin::AbstractVector,
    updates::AbstractVector,
) where {T}
    y = dest
    IT = unwrapped_eltype(lin)
    idx = promote_to(TracedRArray{IT,1}, lin)
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

should_stream_nuptsdriven_spread(::Val{D}, M::Integer, ncombos::Integer, ngrid::NTuple{D,Int}) where {D} =
    D == 3 && M * ncombos >= 8 * prod(ngrid)

nupts_combo_chunk_size(nspread::Int, ::Val{3}) = nspread^2
nupts_combo_chunk_size(nspread::Int, ::Val{D}) where {D} = nspread^D

function expsemicircle_beta(::Type{T}, nspread::Int, upsampfac) where {T<:Number}
    # FINUFFT-inspired ES-kernel beta scaling for upsampfac ~= 2.
    gamma = T(2.30) * (T(2) / max(T(1), T(upsampfac)))
    return gamma * T(nspread)
end

prepare_kernel(kernel::AbstractNUFFTKernel, ::Type{<:Number}, ::NUFFTOptions) = kernel

function kernel_weights(kernel::AbstractNUFFTKernel, dist, nspread::Number)
    T = real(eltype(dist))
    halfwidth = nspread / T(2)
    t = dist ./ halfwidth
    vals = kernel_profile(kernel, clamp.(t, zero(T), one(T)))
    return ifelse.(t .<= one(T), vals, zero(T))
end

function kernel_weight(
    kernel::AbstractNUFFTKernel, dist, nspread::Number, ::Type{T}
) where {T<:Number}
    halfwidth = nspread / T(2)
    t = dist / halfwidth
    val = kernel_profile(kernel, clamp(t, zero(T), one(T)))
    return ifelse(t <= one(T), val, zero(T))
end

function prepare_kernel(::ExpSemicircleKernel, ::Type{T}, opts::NUFFTOptions) where {T<:Number}
    beta = expsemicircle_beta(T, opts.nspread, opts.upsampfac)
    table = expsemicircle_samples(T, beta, opts.np)
    return PreparedExpSemicircleKernel(table, opts.nspread, opts.np)
end

function expsemicircle_samples(::Type{T}, beta, np::Int) where {T<:Number}
    t = promote_to(TracedRArray{T,1}, range(zero(T), one(T); length=np + 1))
    inside = max.(zero(T), one(T) .- (t .* t))
    return exp.(beta .* (sqrt.(inside) .- one(T)))
end

function kernel_weights(kernel::PreparedExpSemicircleKernel, dist, ::Number)
    T = real(eltype(dist))
    halfwidth = kernel.nspread / T(2)
    t = dist ./ halfwidth
    tclip = clamp.(t, zero(T), one(T))
    npT = kernel.np * one(T)
    u = tclip .* npT
    i0 = clamp.(floor.(Int, u) .+ 1, 1, kernel.np + 1)
    i1 = clamp.(i0 .+ 1, 1, kernel.np + 1)
    frac = u .- (i0 .- 1)

    v0 = kernel.table[i0]
    v1 = kernel.table[i1]
    vals = (one(T) .- frac) .* v0 .+ frac .* v1
    return ifelse.(t .<= one(T), vals, zero(T))
end

function kernel_weight(
    kernel::PreparedExpSemicircleKernel, dist, ::Number, ::Type{T}
) where {T<:Number}
    halfwidth = kernel.nspread / T(2)
    t = dist / halfwidth
    tclip = clamp(t, zero(T), one(T))
    npT = kernel.np * one(T)
    u = tclip * npT
    i0 = clamp(floor(Int, u) + 1, 1, kernel.np + 1)
    i1 = clamp(i0 + 1, 1, kernel.np + 1)
    frac = u - (i0 - 1)

    v0 = kernel.table[i0]
    v1 = kernel.table[i1]
    val = (one(T) - frac) * v0 + frac * v1
    return ifelse(t <= one(T), val, zero(T))
end

function direct_type1(
    points::NTuple{D,AbstractVector},
    c::AbstractVector{<:RComplex},
    nmodes::NTuple{D,Int},
    iflag::Integer,
) where {D}
    phase = phase_tensor(points, nmodes, iflag)
    kernel = cis.(phase)
    cview = reshape(c, (length(c), ntuple(_ -> 1, Val(D))...))
    return dropdims(sum(cview .* kernel; dims=1); dims=1)
end

function direct_type2(
    points::NTuple{D,AbstractVector},
    fk::AbstractArray{<:RComplex},
    iflag::Integer,
) where {D}
    nmodes = ntuple(d -> size(fk, d), D)
    phase = phase_tensor(points, nmodes, iflag)
    kernel = cis.(phase)
    fkview = reshape(fk, (1, size(fk)...))
    rdims = ntuple(i -> i + 1, Val(D))
    return dropdims(sum(fkview .* kernel; dims=rdims); dims=rdims)
end

function phase_tensor(
    points::NTuple{D,AbstractVector},
    nmodes::NTuple{D,Int},
    iflag::Integer,
) where {D}
    validate_points(points)
    realT = real(eltype(points[1]))
    period = realT(2π)
    M = length(points[1])
    phase = similar(points[1], realT, (M, nmodes...))
    fill!(phase, zero(realT))
    ones_tail = ntuple(_ -> 1, Val(D))
    for d in 1:D
        x = reshape(mod.(points[d], period), (M, ones_tail...))
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
        d -> max(nmodes[d], ceil(Int, upsampfac * nmodes[d]), 2 * nspread + 1),
        Val(D),
    )
end

function validate_points(points::NTuple{D,AbstractVector}) where {D}
    @assert D >= 1 "Need at least one coordinate dimension"
    @assert D <= 3 "NUFFT currently supports dimensions 1, 2, and 3"
    M = length(points[1])
    for d in 2:D
        @assert length(points[d]) == M "All coordinate vectors must have the same length"
    end
    return nothing
end
