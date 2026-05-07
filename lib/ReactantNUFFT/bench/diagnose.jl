#!/usr/bin/env julia
# ReactantNUFFT performance diagnostic harness.
#
# Builds per-stage compiled thunks (sync=true), times each in isolation
# against FINUFFT/cuFINUFFT baselines, and writes a Markdown report at
# `lib/ReactantNUFFT/bench/PROFILE.md`.
#
# Usage:
#   julia --project=lib/ReactantNUFFT/bench lib/ReactantNUFFT/bench/diagnose.jl \
#       [--backend=cpu|gpu] [--threads=N] [--quick] [--no-hlo] [--no-xprof]
#
# Flags:
#   --backend=cpu|gpu : Reactant backend (default: cpu)
#   --threads=N       : FINUFFT thread count (default: Sys.CPU_THREADS).
#                       The harness always also runs a 1-thread row.
#   --quick           : tiny sweep (one size per dim) for fast iteration
#   --no-hlo          : skip @code_xla dumps
#   --no-xprof        : skip XProf traces (still does sync=true wall-clock)

using Dates
using Printf
using Random
using Statistics

using Reactant
using ReactantNUFFT
using AbstractFFTs

# ----------------------------- arg parsing ----------------------------------

args = Dict{String,Any}(
    "backend" => "cpu",
    "threads" => Sys.CPU_THREADS,
    "quick" => false,
    "no-hlo" => false,
    "no-xprof" => false,
)
for a in ARGS
    if a == "--quick"
        args["quick"] = true
    elseif a == "--no-hlo"
        args["no-hlo"] = true
    elseif a == "--no-xprof"
        args["no-xprof"] = true
    elseif startswith(a, "--backend=")
        args["backend"] = split(a, "=")[2]
    elseif startswith(a, "--threads=")
        args["threads"] = parse(Int, split(a, "=")[2])
    else
        error("unknown flag: $a")
    end
end

const BACKEND     = String(args["backend"])
const THREADS     = Int(args["threads"])
const QUICK       = args["quick"]
const DUMP_HLO    = !args["no-hlo"]
const DUMP_XPROF  = !args["no-xprof"]

Reactant.set_default_backend(BACKEND)

const HAS_GPU = try
    BACKEND == "gpu" && contains(string(Reactant.devices()[1]), "CUDA")
catch
    false
end

# Lazy-load FINUFFT (CPU) and CUDA (for cuFINUFFT) — both optional.
const HAS_FINUFFT = try
    @eval using FINUFFT
    true
catch err
    @warn "FINUFFT unavailable: $err"
    false
end

const HAS_CUFINUFFT = if HAS_GPU && HAS_FINUFFT
    try
        @eval using CUDA
        true
    catch err
        @warn "CUDA unavailable for cuFINUFFT: $err"
        false
    end
else
    false
end

# ----------------------------- size sweep -----------------------------------

# (D, M, N::NTuple) — VLBI-scale 1D + 2D, plus 3D smoke.
const FULL_SWEEP = vcat(
    [(1, M, (N,))             for M in (10_000, 100_000, 1_000_000) for N in (1024, 4096, 16384)],
    # 2D: span the M/cells regime — small grids (M >> cells, dot_general
    # bandwidth-bound on gather output) through large grids (FFT-bound).
    [(2, M, (N, N))            for M in (10_000, 100_000, 1_000_000) for N in (128, 256, 1024)],
    [(2, 1_000_000, (2048, 2048))],
    [(3, M, (128, 128, 128))   for M in (100_000,)],
)
const QUICK_SWEEP = [
    (1, 10_000, (1024,)),
    (2, 10_000, (256, 256)),
]

const SWEEP = QUICK ? QUICK_SWEEP : FULL_SWEEP

# Representative sizes for HLO dumps (small, share the op pattern).
const HLO_REPS = [
    (1, 1024,  (1024,)),
    (2, 4096,  (256, 256)),
]

# ----------------------------- timing helpers -------------------------------

# Run `f` repeatedly and return the 10th-percentile wall time (close to
# uncontested GPU time, robust to occasional desktop / X-server stalls).
# Defaults: 30 warmup calls + 51 timed samples — enough for sub-ms ops on
# a shared GPU. Override per-call via kwargs if you need fewer for very
# expensive workloads. Caller must ensure `f` syncs.
function timed_median(f::F; nrep::Int=51, warmup::Int=30) where {F}
    for _ in 1:warmup
        f()
    end
    samples = Float64[]
    for _ in 1:nrep
        t0 = time_ns()
        f()
        push!(samples, (time_ns() - t0) * 1e-9)
    end
    sort!(samples)
    return samples[max(1, round(Int, 0.1 * nrep))]
end

# ----------------------------- ReactantNUFFT internals ----------------------
#
# We import private helpers so the harness can compile each stage in
# isolation. These mirror execute_type1.jl / execute_type2.jl but stop early.

const RN = ReactantNUFFT

function stage_t1_perm(prep, c)
    cmat = ndims(c) == 1 ? reshape(c, prep.M, 1) : c
    return cmat[prep.perm, :] .* reshape(prep.mask, prep.M_pad, 1)
end

function stage_t1_spread(prep, c)
    plan = prep.plan
    T = eltype(plan)
    D = RN.ndims_(prep)
    w = plan.nspread
    ngrid = plan.ngrid
    nchunks = prep.nchunks
    cs = prep.chunk_size
    cmat = ndims(c) == 1 ? reshape(c, prep.M, 1) : c
    ntrans = size(cmat, 2)

    coefs = Reactant.promote_to(Reactant.TracedRArray{T,2}, plan.horner_coefs)
    offsets_row = reshape(collect(0:(w - 1)), 1, w)
    fw_re = similar(cmat, T, ngrid..., ntrans)
    fw_im = similar(cmat, T, ngrid..., ntrans)
    fw_re, fw_im = RN._spread_chunks(
        fw_re, fw_im, cmat, prep.perm, prep.mask,
        prep.base_sorted, prep.frac_sorted, coefs,
        ngrid, w, ntrans, nchunks, cs, offsets_row, Val(D),
    )
    return complex.(fw_re, fw_im)
end

function stage_t1_fft(prep, fw)
    D = RN.ndims_(prep)
    iflag = prep.plan.iflag
    return iflag < 0 ? AbstractFFTs.fft(fw, 1:D) : AbstractFFTs.bfft(fw, 1:D)
end

function stage_t1_crop_deconv(prep, fw_hat)
    plan = prep.plan
    D = RN.ndims_(prep)
    fw_central = RN._central_view(fw_hat, plan.nmodes, plan.ngrid)
    phih = RN._phi_hat_tensor(plan)
    return fw_central ./ phih
end

function stage_t1_full(prep, c)
    return RN.execute_nufft(prep, c)
end

# Type-2 stages

function stage_t2_deconv_pad(prep, fk)
    plan = prep.plan
    D = RN.ndims_(prep)
    fk_full = ndims(fk) == D ? reshape(fk, plan.nmodes..., 1) : fk
    phih = RN._phi_hat_tensor(plan)
    return RN._corner_embed(fk_full ./ phih, plan.nmodes, plan.ngrid)
end

function stage_t2_ifft(prep, fw_hat)
    D = RN.ndims_(prep)
    iflag = prep.plan.iflag
    return iflag < 0 ? AbstractFFTs.fft(fw_hat, 1:D) : AbstractFFTs.bfft(fw_hat, 1:D)
end

function stage_t2_gather(prep, fw)
    plan = prep.plan
    T = eltype(plan)
    D = RN.ndims_(prep)
    M_pad = prep.M_pad
    w = plan.nspread
    ngrid = plan.ngrid
    ntrans = size(fw, D + 1)

    coefs = Reactant.promote_to(Reactant.TracedRArray{T,2}, plan.horner_coefs)
    offsets_row = reshape(collect(0:(w - 1)), 1, w)
    wpd = ntuple(d -> RN._per_dim_weights(coefs, prep.frac_sorted[d]), Val(D))
    idx_per_dim = ntuple(d -> RN._per_dim_indices(prep.base_sorted[d], ngrid[d], offsets_row), Val(D))
    weights = RN._outer_product_weights(wpd)
    gather_idx = RN._stack_scatter_indices(idx_per_dim)
    vals = RN._gather_spatial(fw, gather_idx, Val(D), ntrans)
    vals = reshape(vals, M_pad, w^D, ntrans)
    return dropdims(sum(vals .* reshape(weights, M_pad, w^D, 1); dims=2); dims=2)
end

function stage_t2_invperm(prep, c_sorted)
    return c_sorted[prep.invperm, :]
end

function stage_t2_full(prep, fk)
    return RN.execute_nufft(prep, fk)
end

# Helper to compile one stage with sync=true and return a callable thunk +
# its output.
macro compile_sync(call)
    quote
        Reactant.@compile sync = true $(esc(call))
    end
end

# ----------------------------- one-row diagnostic ---------------------------

function diagnose_row(io, D::Int, M::Int, nmodes::NTuple{Dn,Int};
    T::Type=Float32, eps::Real=1e-6, ntrans::Int=1, iflag::Int=-1,
) where {Dn}
    @assert Dn == D
    Random.seed!(1234)
    pts = ntuple(_ -> T.(2π .* rand(M)), D)
    pts_ra = ntuple(d -> Reactant.to_rarray(pts[d]), D)

    plan_t1 = plan_nufft(T, 1, nmodes; iflag, eps)
    plan_t2 = plan_nufft(T, 2, nmodes; iflag, eps)

    # ------------------- setpts: compile once, time only execution -------
    f_setpts_t1 = Reactant.@compile sync = true set_nufft_points(plan_t1, pts_ra)
    f_setpts_t2 = Reactant.@compile sync = true set_nufft_points(plan_t2, pts_ra)
    t_setpts1 = timed_median(() -> f_setpts_t1(plan_t1, pts_ra); nrep=5)

    prep_t1 = f_setpts_t1(plan_t1, pts_ra)
    prep_t2 = f_setpts_t2(plan_t2, pts_ra)

    # ------------------- type-1 stages ------------------------------------
    c_arr = Complex{T}.(randn(Complex{T}, M, ntrans))
    c_ra = Reactant.to_rarray(c_arr)

    f_perm   = Reactant.@compile sync = true stage_t1_perm(prep_t1, c_ra)
    c_sorted = f_perm(prep_t1, c_ra)
    f_spread = Reactant.@compile sync = true stage_t1_spread(prep_t1, c_ra)
    fw       = f_spread(prep_t1, c_ra)
    f_fft1   = Reactant.@compile sync = true stage_t1_fft(prep_t1, fw)
    fw_hat   = f_fft1(prep_t1, fw)
    f_dconv  = Reactant.@compile sync = true stage_t1_crop_deconv(prep_t1, fw_hat)
    f_full1  = Reactant.@compile sync = true stage_t1_full(prep_t1, c_ra)

    t_perm   = timed_median(() -> f_perm(prep_t1, c_ra))
    t_spread = timed_median(() -> f_spread(prep_t1, c_ra))
    t_fft1   = timed_median(() -> f_fft1(prep_t1, fw))
    t_dconv  = timed_median(() -> f_dconv(prep_t1, fw_hat))
    t_full1  = timed_median(() -> f_full1(prep_t1, c_ra))

    # ------------------- type-2 stages ------------------------------------
    fk_arr = Complex{T}.(randn(Complex{T}, nmodes..., ntrans))
    fk_ra  = Reactant.to_rarray(fk_arr)

    f_pad   = Reactant.@compile sync = true stage_t2_deconv_pad(prep_t2, fk_ra)
    fw_hat2 = f_pad(prep_t2, fk_ra)
    f_ifft2 = Reactant.@compile sync = true stage_t2_ifft(prep_t2, fw_hat2)
    fw2     = f_ifft2(prep_t2, fw_hat2)
    f_gath  = Reactant.@compile sync = true stage_t2_gather(prep_t2, fw2)
    cs2     = f_gath(prep_t2, fw2)
    f_inv   = Reactant.@compile sync = true stage_t2_invperm(prep_t2, cs2)
    f_full2 = Reactant.@compile sync = true stage_t2_full(prep_t2, fk_ra)

    t_pad    = timed_median(() -> f_pad(prep_t2, fk_ra))
    t_ifft   = timed_median(() -> f_ifft2(prep_t2, fw_hat2))
    t_gather = timed_median(() -> f_gath(prep_t2, fw2))
    t_inv    = timed_median(() -> f_inv(prep_t2, cs2))
    t_full2  = timed_median(() -> f_full2(prep_t2, fk_ra))

    # ------------------- FINUFFT baselines --------------------------------
    finufft_t1_1, finufft_t1_n, finufft_t2_1, finufft_t2_n = NaN, NaN, NaN, NaN
    if HAS_FINUFFT
        c_host  = ndims(c_arr) == 2 && size(c_arr, 2) == 1 ? vec(c_arr) : c_arr
        fk_host = ntrans == 1 ? reshape(fk_arr, nmodes...) : fk_arr
        finufft_t1_1 = timed_median(() -> finufft_type1(pts, c_host, nmodes; iflag, eps, nthreads=1))
        finufft_t1_n = timed_median(() -> finufft_type1(pts, c_host, nmodes; iflag, eps, nthreads=THREADS))
        finufft_t2_1 = timed_median(() -> finufft_type2(pts, fk_host;        iflag, eps, nthreads=1))
        finufft_t2_n = timed_median(() -> finufft_type2(pts, fk_host;        iflag, eps, nthreads=THREADS))
    end

    cufinufft_t1, cufinufft_t2 = NaN, NaN
    if HAS_CUFINUFFT
        # Time only `exec` for cuFINUFFT — plan+setpts are amortized
        # outside the timing loop, matching what we do for Reactant
        # (`prep` is built outside `timed_median`; only `execute_nufft` is
        # timed). Earlier versions of this harness rebuilt the plan +
        # setpts every call and biased *against* cuFINUFFT.
        c_host  = ndims(c_arr) == 2 && size(c_arr, 2) == 1 ? vec(c_arr) : c_arr
        fk_host = ntrans == 1 ? reshape(fk_arr, nmodes...) : fk_arr
        try
            cufinufft_t1 = cufinufft_exec_only(1, pts, c_host, nmodes; iflag, eps)
            cufinufft_t2 = cufinufft_exec_only(2, pts, fk_host, nmodes; iflag, eps)
        catch err
            @warn "cuFINUFFT call failed at D=$D M=$M N=$nmodes: $err"
        end
    end

    write_row!(io, D, M, nmodes, T, ntrans;
        t_setpts1=t_setpts1,
        t_perm=t_perm, t_spread=t_spread, t_fft1=t_fft1, t_dconv=t_dconv, t_full1=t_full1,
        t_pad=t_pad, t_ifft=t_ifft, t_gather=t_gather, t_inv=t_inv, t_full2=t_full2,
        finufft_t1_1=finufft_t1_1, finufft_t1_n=finufft_t1_n,
        finufft_t2_1=finufft_t2_1, finufft_t2_n=finufft_t2_n,
        cufinufft_t1=cufinufft_t1, cufinufft_t2=cufinufft_t2,
    )
end

function write_row!(io, D, M, nmodes, T, ntrans; kwargs...)
    function ms(x)
        return isnan(x) ? "—" : @sprintf("%.3f ms", 1000 * x)
    end
    function ratio(a, b)
        (isnan(a) || isnan(b) || b == 0) && return "—"
        return @sprintf("%.2f×", a / b)
    end

    println(io)
    println(io, "### D=$D, M=$(M), N=$(join(nmodes, '×')), T=$T, ntrans=$ntrans")
    println(io)
    println(io, "**setpts** (Reactant compiled+sync, median over 5):  $(ms(kwargs[:t_setpts1]))")
    println(io)
    println(io, "**Type-1**")
    println(io, "")
    println(io, "| Stage              | Reactant   | FINUFFT 1-thr | FINUFFT $(THREADS)-thr | cuFINUFFT |")
    println(io, "|--------------------|------------|---------------|-------------|-----------|")
    println(io, "| `c[perm, :]`       | $(ms(kwargs[:t_perm]))   |               |             |           |")
    println(io, "| spread (scatter)   | $(ms(kwargs[:t_spread])) |               |             |           |")
    println(io, "| FFT                | $(ms(kwargs[:t_fft1]))   |               |             |           |")
    println(io, "| crop+deconv        | $(ms(kwargs[:t_dconv]))  |               |             |           |")
    println(io, "| **end-to-end**     | $(ms(kwargs[:t_full1])) | $(ms(kwargs[:finufft_t1_1])) | $(ms(kwargs[:finufft_t1_n])) | $(ms(kwargs[:cufinufft_t1])) |")
    println(io, "| ratio vs FINUFFT-1 | $(ratio(kwargs[:t_full1], kwargs[:finufft_t1_1])) | – | – | – |")
    println(io, "| ratio vs FINUFFT-N | $(ratio(kwargs[:t_full1], kwargs[:finufft_t1_n])) | – | – | – |")
    println(io, "| ratio vs cuFINUFFT | $(ratio(kwargs[:t_full1], kwargs[:cufinufft_t1])) | – | – | – |")
    println(io, "")
    println(io, "**Type-2**")
    println(io, "")
    println(io, "| Stage              | Reactant   | FINUFFT 1-thr | FINUFFT $(THREADS)-thr | cuFINUFFT |")
    println(io, "|--------------------|------------|---------------|-------------|-----------|")
    println(io, "| pad+deconv         | $(ms(kwargs[:t_pad]))    |               |             |           |")
    println(io, "| iFFT               | $(ms(kwargs[:t_ifft]))   |               |             |           |")
    println(io, "| gather+sum         | $(ms(kwargs[:t_gather])) |               |             |           |")
    println(io, "| invperm            | $(ms(kwargs[:t_inv]))    |               |             |           |")
    println(io, "| **end-to-end**     | $(ms(kwargs[:t_full2])) | $(ms(kwargs[:finufft_t2_1])) | $(ms(kwargs[:finufft_t2_n])) | $(ms(kwargs[:cufinufft_t2])) |")
    println(io, "| ratio vs FINUFFT-1 | $(ratio(kwargs[:t_full2], kwargs[:finufft_t2_1])) | – | – | – |")
    println(io, "| ratio vs FINUFFT-N | $(ratio(kwargs[:t_full2], kwargs[:finufft_t2_n])) | – | – | – |")
    println(io, "| ratio vs cuFINUFFT | $(ratio(kwargs[:t_full2], kwargs[:cufinufft_t2])) | – | – | – |")
    flush(io)
end

# ----------------------------- FINUFFT wrappers -----------------------------

function finufft_type1(points::NTuple{1,<:AbstractVector}, c, nmodes; iflag=-1, eps=1e-6, nthreads=1)
    return FINUFFT.nufft1d1(points[1], c, iflag, eps, nmodes[1]; nthreads)
end
function finufft_type1(points::NTuple{2,<:AbstractVector}, c, nmodes; iflag=-1, eps=1e-6, nthreads=1)
    return FINUFFT.nufft2d1(points[1], points[2], c, iflag, eps, nmodes[1], nmodes[2]; nthreads)
end
function finufft_type1(points::NTuple{3,<:AbstractVector}, c, nmodes; iflag=-1, eps=1e-6, nthreads=1)
    return FINUFFT.nufft3d1(
        points[1], points[2], points[3], c, iflag, eps,
        nmodes[1], nmodes[2], nmodes[3]; nthreads,
    )
end
function finufft_type2(points::NTuple{1,<:AbstractVector}, fk; iflag=-1, eps=1e-6, nthreads=1)
    return FINUFFT.nufft1d2(points[1], iflag, eps, fk; nthreads)
end
function finufft_type2(points::NTuple{2,<:AbstractVector}, fk; iflag=-1, eps=1e-6, nthreads=1)
    return FINUFFT.nufft2d2(points[1], points[2], iflag, eps, fk; nthreads)
end
function finufft_type2(points::NTuple{3,<:AbstractVector}, fk; iflag=-1, eps=1e-6, nthreads=1)
    return FINUFFT.nufft3d2(points[1], points[2], points[3], iflag, eps, fk; nthreads)
end

# Stub cuFINUFFT wrappers — populated via FINUFFT plan API when CUDA is loaded.
function cufinufft_type1(points::NTuple{D,<:AbstractVector}, c, nmodes; iflag=-1, eps=1e-6) where {D}
    HAS_CUFINUFFT || return NaN
    plan = FINUFFT.cufinufft_makeplan(1, collect(Int64, nmodes), iflag, 1, eps;
        dtype=eltype(c) === ComplexF32 ? Float32 : Float64,
    )
    pts_dev = ntuple(d -> CUDA.CuArray(points[d]), D)
    if D == 1
        FINUFFT.cufinufft_setpts!(plan, pts_dev[1])
    elseif D == 2
        FINUFFT.cufinufft_setpts!(plan, pts_dev[1], pts_dev[2])
    else
        FINUFFT.cufinufft_setpts!(plan, pts_dev[1], pts_dev[2], pts_dev[3])
    end
    c_dev  = CUDA.CuArray(c)
    fk_dev = CUDA.zeros(eltype(c), nmodes...)
    FINUFFT.cufinufft_exec!(plan, c_dev, fk_dev)
    CUDA.device_synchronize()
    FINUFFT.cufinufft_destroy!(plan)
    return Array(fk_dev)
end
function cufinufft_type2(points::NTuple{D,<:AbstractVector}, fk; iflag=-1, eps=1e-6) where {D}
    HAS_CUFINUFFT || return NaN
    M = length(points[1])
    plan = FINUFFT.cufinufft_makeplan(2, collect(Int64, size(fk)), iflag, 1, eps;
        dtype=eltype(fk) === ComplexF32 ? Float32 : Float64,
    )
    pts_dev = ntuple(d -> CUDA.CuArray(points[d]), D)
    if D == 1
        FINUFFT.cufinufft_setpts!(plan, pts_dev[1])
    elseif D == 2
        FINUFFT.cufinufft_setpts!(plan, pts_dev[1], pts_dev[2])
    else
        FINUFFT.cufinufft_setpts!(plan, pts_dev[1], pts_dev[2], pts_dev[3])
    end
    fk_dev = CUDA.CuArray(fk)
    c_dev  = CUDA.zeros(eltype(fk), M)
    FINUFFT.cufinufft_exec!(plan, fk_dev, c_dev)
    CUDA.device_synchronize()
    FINUFFT.cufinufft_destroy!(plan)
    return Array(c_dev)
end

# Plan+setpts amortized; time only `exec`. Returns median seconds.
function cufinufft_exec_only(K::Int, points::NTuple{D,<:AbstractVector}, data, nmodes;
    iflag::Integer=-1, eps::Real=1e-6,
) where {D}
    HAS_CUFINUFFT || return NaN
    M = length(points[1])
    Tdtype = eltype(data) === ComplexF32 ? Float32 : Float64
    nmodes_v = collect(Int64, K == 1 ? nmodes : size(data))
    plan = FINUFFT.cufinufft_makeplan(K, nmodes_v, Int(iflag), 1, eps; dtype=Tdtype)
    pts_dev = ntuple(d -> CUDA.CuArray(points[d]), D)
    if D == 1
        FINUFFT.cufinufft_setpts!(plan, pts_dev[1])
    elseif D == 2
        FINUFFT.cufinufft_setpts!(plan, pts_dev[1], pts_dev[2])
    else
        FINUFFT.cufinufft_setpts!(plan, pts_dev[1], pts_dev[2], pts_dev[3])
    end
    # Pool of distinct inputs (defeat cuFINUFFT short-circuit on identical
    # buffer pointers) + device_synchronize (cuFINUFFT uses its own stream).
    n_pool = 16
    in_shape = K == 1 ? (M,) : nmodes
    out_shape = K == 1 ? nmodes : (M,)
    pool = [CUDA.CuArray(eltype(data).(randn(eltype(data), in_shape...))) for _ in 1:n_pool]
    out_dev = CUDA.zeros(eltype(data), out_shape...)
    try
        for k in 1:6
            FINUFFT.cufinufft_exec!(plan, pool[mod1(k, n_pool)], out_dev)
        end
        CUDA.device_synchronize()
        nrep = 11
        ts = Float64[]
        for k in 1:nrep
            t0 = time_ns()
            FINUFFT.cufinufft_exec!(plan, pool[mod1(k, n_pool)], out_dev)
            CUDA.device_synchronize()
            push!(ts, (time_ns() - t0) * 1e-9)
        end
        return median(ts)
    finally
        FINUFFT.cufinufft_destroy!(plan)
    end
end

# ----------------------------- HLO dumps ------------------------------------

function dump_hlo(D::Int, M::Int, nmodes::NTuple{Dn,Int}; T::Type=Float32, eps=1e-6) where {Dn}
    @assert Dn == D
    Random.seed!(0)
    pts = ntuple(_ -> T.(2π .* rand(M)), D)
    pts_ra = ntuple(d -> Reactant.to_rarray(pts[d]), D)
    c_ra  = Reactant.to_rarray(Complex{T}.(randn(Complex{T}, M)))
    fk_ra = Reactant.to_rarray(Complex{T}.(randn(Complex{T}, nmodes...)))

    plan1 = plan_nufft(T, 1, nmodes; eps)
    prep1 = set_nufft_points(plan1, pts_ra)
    plan2 = plan_nufft(T, 2, nmodes; eps)
    prep2 = set_nufft_points(plan2, pts_ra)

    tag1 = "t1_D$(D)_M$(M)_N$(join(nmodes, 'x'))"
    tag2 = "t2_D$(D)_M$(M)_N$(join(nmodes, 'x'))"
    here = @__DIR__
    open(joinpath(here, "hlo", "$(tag1).mlir"), "w") do io
        show(io, Reactant.@code_xla execute_nufft(prep1, c_ra))
    end
    open(joinpath(here, "hlo", "$(tag2).mlir"), "w") do io
        show(io, Reactant.@code_xla execute_nufft(prep2, fk_ra))
    end
end

# ----------------------------- main -----------------------------------------

function main()
    here = @__DIR__
    report_path = joinpath(here, "PROFILE.md")

    open(report_path, "w") do io
        ncpu = Sys.CPU_THREADS
        println(io, "# ReactantNUFFT performance profile")
        println(io)
        println(io, "Generated: $(Dates.now())")
        println(io)
        println(io, "- Reactant backend: **$BACKEND**")
        println(io, "- HAS_GPU: $HAS_GPU")
        println(io, "- Julia threads: $(Threads.nthreads())")
        println(io, "- CPU.threads: $ncpu")
        println(io, "- FINUFFT threads compared: 1 and $THREADS")
        println(io, "- HAS_FINUFFT: $HAS_FINUFFT")
        println(io, "- HAS_CUFINUFFT: $HAS_CUFINUFFT")
        println(io, "- Sweep: $(QUICK ? "QUICK" : "FULL")")
        println(io)
        println(io, "All Reactant timings use `@compile sync=true` thunks; p10 over 51 timed calls after 30 warmup calls (robust to desktop hiccups on a shared GPU).")
        println(io)
        println(io, "## Per-row stage breakdown")

        for (D, M, N) in SWEEP
            try
                println("Diagnosing D=$D M=$M N=$N ...")
                diagnose_row(io, D, M, N)
                flush(io)
            catch err
                @warn "Skipping D=$D M=$M N=$N: $err"
                println(io, "\n### D=$D, M=$M, N=$(join(N, '×')) — ERROR: $err\n")
            end
        end

        if DUMP_HLO
            println(io)
            println(io, "## HLO dumps")
            println(io)
            println(io, "Post-optimization XLA HLO captured for:")
            for (D, M, N) in HLO_REPS
                tag = "D$(D)_M$(M)_N$(join(N, 'x'))"
                try
                    println("Dumping HLO for $tag ...")
                    dump_hlo(D, M, N)
                    println(io, "- `bench/hlo/t1_$(tag).mlir`")
                    println(io, "- `bench/hlo/t2_$(tag).mlir`")
                catch err
                    @warn "HLO dump $tag failed: $err"
                    println(io, "- `$tag` — ERROR: $err")
                end
            end
        end

        println(io)
        println(io, "## Largest gaps")
        println(io)
        println(io, "_To be filled in by hand after inspecting the table above._")
        println(io)
    end
    println("Report written: $report_path")
end

main()
