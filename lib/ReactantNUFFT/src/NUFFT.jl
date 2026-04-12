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

struct NUPtsDriven <: AbstractNUFFTMethod end
struct SubProb <: AbstractNUFFTMethod end
struct OutputDriven <: AbstractNUFFTMethod end
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

struct PreparedNUFFTPlan{P,S,PS}
    plan::P
    points::S
    points_scaled::PS
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
    nmodes::NTuple{D,<:Integer};
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
    period = T(2 * float(pi))
    points_scaled = ntuple(
        d -> mod.(points[d], period) .* (plan.ngrid[d] / period),
        Val(D),
    )
    return PreparedNUFFTPlan(plan, points, points_scaled)
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

function execute_type2(::OutputDriven, prep::PreparedNUFFTPlan, fk::AbstractArray)
    plan = prep.plan
    grid_hat = embed_modes(fk, plan.nmodes, plan.ngrid)
    grid = fft_with_iflag(grid_hat, plan.iflag)
    return interp_outputdriven(prep.points_scaled, grid, plan.ngrid, plan.opts)
end

@inline function stencil_contribution(
    stencil_id,
    points_scaled::Tuple{Vararg{<:AbstractVector,D}},
    bases::Tuple{Vararg{<:AbstractVector,D}},
    ngrid::NTuple{D,<:Number},
    nspread::Number,
    prepared_kernel,
    realT::Type{<:Number},
) where {D}
    offset = stencil_offset(stencil_id, nspread, 1)
    lin, wt = dim_stencil(
        points_scaled[1], bases[1], ngrid[1], offset, nspread, prepared_kernel, realT
    )
    stride = ngrid[1]
    @inbounds for d in 2:D
        offset = stencil_offset(stencil_id, nspread, d)
        idx_d, wt_d = dim_stencil(
            points_scaled[d], bases[d], ngrid[d], offset, nspread, prepared_kernel, realT
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
    nspread = opts.nspread
    grid_flat = similar(c, CT, (prod(ngrid),))
    fill!(grid_flat, zero(CT))
    prepared_kernel = prepare_kernel(opts.kernel, realT, opts)
    bases = ntuple(d -> floor.(Int, points_scaled[d]), Val(D))

    @allowscalar @trace for stencil_id in 1:ncombos
        lin, wt = stencil_contribution(
            stencil_id, points_scaled, bases, ngrid, nspread, prepared_kernel, realT
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
    nspread = opts.nspread
    CT = unwrapped_eltype(grid)
    grid_flat = promote_to(TracedRArray{CT,1}, vec(grid))
    prepared_kernel = prepare_kernel(opts.kernel, realT, opts)
    bases = ntuple(d -> floor.(Int, points_scaled[d]), Val(D))

    out = similar(grid, CT, (length(points_scaled[1]),))
    fill!(out, zero(CT))
    @allowscalar @trace for stencil_id in 1:ncombos
        lin, wt = stencil_contribution(
            stencil_id, points_scaled, bases, ngrid, nspread, prepared_kernel, realT
        )
        vals = promote_to(TracedRArray{CT,1}, grid_flat[lin])
        copyto!(out, out .+ vals .* wt)
    end
    return out
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

    v0 = sample_kernel_table(kernel.table, i0)
    v1 = sample_kernel_table(kernel.table, i1)
    vals = (one(T) .- frac) .* v0 .+ frac .* v1
    return ifelse.(t .<= one(T), vals, zero(T))
end

function sample_kernel_table(kernel_data::AbstractVector, idx::AbstractArray)
    return kernel_data[idx]
end

function direct_type1(
    points::NTuple{D,<:AbstractVector},
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
    points::NTuple{D,<:AbstractVector},
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
    points::NTuple{D,<:AbstractVector},
    nmodes::NTuple{D,Int},
    iflag::Integer,
) where {D}
    validate_points(points)
    realT = real(eltype(points[1]))
    period = realT(2 * float(pi))
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

function validate_points(points::NTuple{D,<:AbstractVector}) where {D}
    @assert D >= 1 "Need at least one coordinate dimension"
    @assert D <= 3 "NUFFT currently supports dimensions 1, 2, and 3"
    M = length(points[1])
    for d in 2:D
        @assert length(points[d]) == M "All coordinate vectors must have the same length"
    end
    return nothing
end
