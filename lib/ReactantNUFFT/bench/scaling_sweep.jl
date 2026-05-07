#!/usr/bin/env julia
# Scaling sweep: Reactant vs FINUFFT (1-thr, N-thr) vs cuFINUFFT.
# Times execute_nufft / cufinufft_exec / finufft_exec across M for fixed
# representative N at each D, for both T1 and T2. Plan + setpts are built
# outside the timer (matches what Reactant amortizes via @compile).
#
# Output: /tmp/scaling.csv  (D, M, N, K, lib, time_ms)
#
# Usage:
#   julia --project=lib/ReactantNUFFT/bench lib/ReactantNUFFT/bench/scaling_sweep.jl

using Printf
using Random
using Statistics

using Reactant
using ReactantNUFFT

# ---- args ------------------------------------------------------------------
backend = "gpu"
dtype = "f32"
out_path = "/tmp/scaling.csv"
for a in ARGS
    if startswith(a, "--backend=")
        global backend = String(split(a, "=")[2])
    elseif startswith(a, "--dtype=")
        global dtype = lowercase(String(split(a, "=")[2]))
    elseif startswith(a, "--out=")
        global out_path = String(split(a, "=")[2])
    end
end
Reactant.set_default_backend(backend)
const DTYPE = dtype

const HAS_GPU = try
    backend == "gpu" && contains(string(Reactant.devices()[1]), "CUDA")
catch
    false
end

const HAS_FINUFFT = try
    @eval using FINUFFT
    true
catch
    false
end

const HAS_CUFINUFFT = if HAS_GPU && HAS_FINUFFT
    try
        @eval using CUDA
        true
    catch
        false
    end
else
    false
end

const T = DTYPE == "f64" ? Float64 : Float32
const EPS_TARGET = T === Float32 ? 1e-6 : 1e-9
const NTRANS = 1
const FT_NTHR = max(1, min(Sys.CPU_THREADS, 16))   # cap N-thr at 16 to dodge oversub warnings

# Sweep configuration.
# Per dimension, fix N; sweep M.
const SWEEP = [
    (1, [10_000, 100_000, 1_000_000, 10_000_000], (4096,)),
    # Two 2D rows: small grid (gather/dot_general bandwidth-bound regime
    # where M ≫ cells) and large grid (FFT-dominated regime).
    (2, [10_000, 100_000, 1_000_000],             (128, 128)),
    (2, [10_000, 100_000, 1_000_000],             (1024, 1024)),
    (3, [10_000, 100_000, 1_000_000],             (128, 128, 128)),
]

# ---- timing helpers --------------------------------------------------------
function timed_p10(f::F; nrep::Int=51, warmup::Int=30) where {F}
    for _ in 1:warmup; f(); end
    samples = Float64[]
    for _ in 1:nrep
        t0 = time_ns()
        f()
        push!(samples, (time_ns() - t0) * 1e-9)
    end
    sort!(samples)
    return samples[max(1, round(Int, 0.10 * nrep))]
end

# ---- Reactant timing -------------------------------------------------------
function reactant_time(D::Int, M::Int, nmodes, K::Int)
    Random.seed!(0xCAFE + M + D + K)
    pts = ntuple(_ -> T.(2π .* rand(M)), D)
    pts_ra = ntuple(d -> Reactant.to_rarray(pts[d]), D)
    plan = plan_nufft(T, K, nmodes; iflag=-1, eps=EPS_TARGET)
    prep = Reactant.@jit set_nufft_points(plan, pts_ra)
    pool_size = 8
    if K == 1
        pool = [Reactant.to_rarray(Complex{T}.(randn(Complex{T}, M, NTRANS))) for _ in 1:pool_size]
        f = Reactant.@compile sync = true execute_nufft(prep, pool[1])
        return timed_p10(() -> f(prep, pool[mod1(rand(1:pool_size), pool_size)]))
    else
        pool = [Reactant.to_rarray(Complex{T}.(randn(Complex{T}, nmodes..., NTRANS))) for _ in 1:pool_size]
        f = Reactant.@compile sync = true execute_nufft(prep, pool[1])
        return timed_p10(() -> f(prep, pool[mod1(rand(1:pool_size), pool_size)]))
    end
end

# ---- CPU FINUFFT timing ----------------------------------------------------
function finufft_time(D::Int, M::Int, nmodes, K::Int, nthreads::Int)
    HAS_FINUFFT || return NaN
    Random.seed!(0xCAFE + M + D + K)
    pts = ntuple(_ -> T.(2π .* rand(M)), D)
    in_shape_M  = NTRANS == 1 ? (M,) : (M, NTRANS)
    in_shape_fk = NTRANS == 1 ? nmodes : (nmodes..., NTRANS)
    plan = FINUFFT.finufft_makeplan(K, collect(Int64, nmodes), -1, NTRANS, EPS_TARGET;
        dtype=T, nthreads=nthreads,
    )
    try
        if D == 1
            FINUFFT.finufft_setpts!(plan, pts[1])
        elseif D == 2
            FINUFFT.finufft_setpts!(plan, pts[1], pts[2])
        else
            FINUFFT.finufft_setpts!(plan, pts[1], pts[2], pts[3])
        end
        # I/O buffers
        if K == 1
            n_pool = 8
            pool = [Complex{T}.(randn(Complex{T}, in_shape_M...)) for _ in 1:n_pool]
            outp = zeros(Complex{T}, in_shape_fk...)
            for k in 1:6; FINUFFT.finufft_exec!(plan, pool[mod1(k, n_pool)], outp); end
            return timed_p10(() ->
                FINUFFT.finufft_exec!(plan, pool[rand(1:n_pool)], outp); nrep=21, warmup=4)
        else
            n_pool = 8
            pool = [Complex{T}.(randn(Complex{T}, in_shape_fk...)) for _ in 1:n_pool]
            outp = zeros(Complex{T}, in_shape_M...)
            for k in 1:6; FINUFFT.finufft_exec!(plan, pool[mod1(k, n_pool)], outp); end
            return timed_p10(() ->
                FINUFFT.finufft_exec!(plan, pool[rand(1:n_pool)], outp); nrep=21, warmup=4)
        end
    finally
        FINUFFT.finufft_destroy!(plan)
    end
end

# ---- cuFINUFFT timing ------------------------------------------------------
function cufinufft_time(D::Int, M::Int, nmodes, K::Int)
    HAS_CUFINUFFT || return NaN
    Random.seed!(0xCAFE + M + D + K)
    pts = ntuple(_ -> T.(2π .* rand(M)), D)
    pts_dev = ntuple(d -> CUDA.CuArray(pts[d]), D)
    in_shape_M  = NTRANS == 1 ? (M,) : (M, NTRANS)
    in_shape_fk = NTRANS == 1 ? nmodes : (nmodes..., NTRANS)
    plan = FINUFFT.cufinufft_makeplan(K, collect(Int64, nmodes), -1, NTRANS, EPS_TARGET;
        dtype=T,
    )
    try
        if D == 1
            FINUFFT.cufinufft_setpts!(plan, pts_dev[1])
        elseif D == 2
            FINUFFT.cufinufft_setpts!(plan, pts_dev[1], pts_dev[2])
        else
            FINUFFT.cufinufft_setpts!(plan, pts_dev[1], pts_dev[2], pts_dev[3])
        end
        n_pool = 8
        if K == 1
            pool = [CUDA.CuArray(Complex{T}.(randn(Complex{T}, in_shape_M...))) for _ in 1:n_pool]
            outp = CUDA.zeros(Complex{T}, in_shape_fk...)
        else
            pool = [CUDA.CuArray(Complex{T}.(randn(Complex{T}, in_shape_fk...))) for _ in 1:n_pool]
            outp = CUDA.zeros(Complex{T}, in_shape_M...)
        end
        for k in 1:6; FINUFFT.cufinufft_exec!(plan, pool[mod1(k, n_pool)], outp); end
        CUDA.device_synchronize()
        nrep = 21
        ts = Float64[]
        for k in 1:nrep
            t0 = time_ns()
            FINUFFT.cufinufft_exec!(plan, pool[rand(1:n_pool)], outp)
            CUDA.device_synchronize()
            push!(ts, (time_ns() - t0) * 1e-9)
        end
        sort!(ts)
        return ts[max(1, round(Int, 0.10 * nrep))]
    finally
        FINUFFT.cufinufft_destroy!(plan)
    end
end

# ---- driver ----------------------------------------------------------------
const OUT = out_path
open(OUT, "w") do io
    println(io, "D,M,N,K,lib,time_ms,dtype")
    flush(io)
    for (D, Ms, nmodes) in SWEEP
        Nlabel = D == 1 ? string(nmodes[1]) :
                 D == 2 ? string(nmodes[1], "x", nmodes[2]) :
                          string(nmodes[1], "x", nmodes[2], "x", nmodes[3])
        for M in Ms, K in (1, 2)
            tag = "D=$D M=$M N=$Nlabel T$K"
            tR = try reactant_time(D, M, nmodes, K) catch e; @warn "Reactant $tag: $e"; NaN end
            tC = try cufinufft_time(D, M, nmodes, K) catch e; @warn "cuFINUFFT $tag: $e"; NaN end
            tF1 = try finufft_time(D, M, nmodes, K, 1) catch e; @warn "FINUFFT-1 $tag: $e"; NaN end
            tFN = try finufft_time(D, M, nmodes, K, FT_NTHR) catch e; @warn "FINUFFT-N $tag: $e"; NaN end
            for (lib, t) in (("Reactant", tR), ("cuFINUFFT", tC), ("FINUFFT-1", tF1), ("FINUFFT-$FT_NTHR", tFN))
                println(io, "$D,$M,$Nlabel,$K,$lib,$(1000*t),$DTYPE")
            end
            flush(io)
            @printf("%s  R=%.3f cuF=%.3f F1=%.3f F%d=%.3f ms\n",
                tag, 1000*tR, 1000*tC, 1000*tF1, FT_NTHR, 1000*tFN)
        end
    end
end
println("Wrote $OUT")
