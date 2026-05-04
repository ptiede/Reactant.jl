#!/usr/bin/env julia
# D=2 scaling sweep: Reactant T1/T2 vs cuFINUFFT, varying both M (NU points)
# and N (grid edge length). Writes /tmp/d2_scaling.csv with one row per
# (M, N, type, lib) measurement. Render with d2_scaling_plot.jl.
#
# Usage:
#   julia --project=lib/ReactantNUFFT/bench lib/ReactantNUFFT/bench/d2_scaling_sweep.jl

ENV["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"

using Printf
using Random
using Statistics

using Reactant
using ReactantNUFFT
using CUDA
using FINUFFT

Reactant.set_default_backend("gpu")

const T          = Float32
const EPS_TARGET = 1.0e-6
const NTRANS     = 1

const Ms = [10_000, 30_000, 100_000, 300_000, 1_000_000]
const Ns = [64, 128, 256, 512, 1024]
const OUT = "/tmp/d2_scaling.csv"

# ---- p10 wall-clock --------------------------------------------------------
# More warmup + reps than the original to tame bimodal sub-millisecond
# distributions where cold-cache iters dominate the early samples.
function timed_p10(f; nrep::Int=151, warmup::Int=100)
    for _ in 1:warmup; f(); end
    samples = Float64[]
    for _ in 1:nrep
        t0 = time_ns(); f(); push!(samples, (time_ns() - t0) * 1e-9)
    end
    sort!(samples)
    return samples[max(1, round(Int, 0.10 * nrep))]
end

# ---- Reactant timing -------------------------------------------------------
function reactant_time(K::Int, M::Int, nmodes)
    Random.seed!(0xCAFE + M + K + nmodes[1])
    pts = ntuple(_ -> T.(2π .* rand(M)), 2)
    pts_ra = ntuple(d -> Reactant.to_rarray(pts[d]), 2)
    plan = plan_nufft(T, K, nmodes; iflag=-1, eps=EPS_TARGET)
    prep = Reactant.@jit set_nufft_points(plan, pts_ra)
    in_pool, out_ra = if K == 1
        ([Reactant.to_rarray(Complex{T}.(randn(Complex{T}, M, NTRANS))) for _ in 1:8],
         Reactant.to_rarray(zeros(Complex{T}, nmodes..., NTRANS)))
    else
        ([Reactant.to_rarray(Complex{T}.(randn(Complex{T}, nmodes..., NTRANS))) for _ in 1:8],
         Reactant.to_rarray(zeros(Complex{T}, M, NTRANS)))
    end
    f = Reactant.@compile sync=true execute_nufft!(out_ra, prep, in_pool[1])
    return timed_p10(() -> f(out_ra, prep, in_pool[mod1(rand(1:8), 8)]))
end

# ---- cuFINUFFT timing ------------------------------------------------------
function cufinufft_time(K::Int, M::Int, nmodes)
    Random.seed!(0xCAFE + M + K + nmodes[1])
    pts = ntuple(_ -> T.(2π .* rand(M)), 2)
    pts_dev = ntuple(d -> CUDA.CuArray(pts[d]), 2)
    in_shape_M  = NTRANS == 1 ? (M,) : (M, NTRANS)
    in_shape_fk = NTRANS == 1 ? nmodes : (nmodes..., NTRANS)
    plan = FINUFFT.cufinufft_makeplan(K, collect(Int64, nmodes), -1, NTRANS, EPS_TARGET; dtype=Float32)
    try
        FINUFFT.cufinufft_setpts!(plan, pts_dev[1], pts_dev[2])
        n_pool = 8
        pool = K == 1 ?
            [CUDA.CuArray(Complex{T}.(randn(Complex{T}, in_shape_M...)))  for _ in 1:n_pool] :
            [CUDA.CuArray(Complex{T}.(randn(Complex{T}, in_shape_fk...))) for _ in 1:n_pool]
        outp = K == 1 ?
            CUDA.zeros(Complex{T}, in_shape_fk...) :
            CUDA.zeros(Complex{T}, in_shape_M...)
        for k in 1:100; FINUFFT.cufinufft_exec!(plan, pool[mod1(k, n_pool)], outp); end
        CUDA.device_synchronize()
        nrep = 151
        ts = Float64[]
        for _ in 1:nrep
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
open(OUT, "w") do io
    println(io, "K,M,N,lib,time_ms")
    flush(io)
    for K in (1, 2), N in Ns, M in Ms
        nmodes = (N, N)
        tag = "T$K  M=$M  N=$N×$N"
        tR = try reactant_time(K, M, nmodes) catch e; @warn "Reactant $tag: $e"; NaN end
        tC = try cufinufft_time(K, M, nmodes) catch e; @warn "cuFINUFFT $tag: $e"; NaN end
        for (lib, t) in (("Reactant", tR), ("cuFINUFFT", tC))
            println(io, "$K,$M,$N,$lib,$(1000*t)")
        end
        flush(io)
        ratio = tR / tC
        @printf("%s  R=%.3f cuF=%.3f  ratio=%.2fx\n", tag, 1000*tR, 1000*tC, ratio)
    end
end
println("Wrote $OUT")
