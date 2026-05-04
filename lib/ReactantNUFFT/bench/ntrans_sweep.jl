#!/usr/bin/env julia
# A3 — ntrans (batched-strengths) sweep.
#
# For the cuFINUFFT-leading workloads, time Reactant and cuFINUFFT at
# ntrans ∈ {1, 4, 8}. Comrade-style Bayesian inference does many transforms
# with shared NU points but different strength vectors per call (or per
# gradient eval), so the per-call fixed costs (FFT, embed/crop, sort)
# amortize across ntrans while the scatter scales linearly. Hypothesis:
# Reactant pulls further ahead at ntrans ≥ 4.
#
# Usage:
#   julia --project=lib/ReactantNUFFT/bench lib/ReactantNUFFT/bench/ntrans_sweep.jl --backend=gpu

using Printf
using Random
using Statistics

using Reactant
using ReactantNUFFT

# ---- args ------------------------------------------------------------------

backend = "gpu"
for a in ARGS
    if startswith(a, "--backend=")
        global backend = String(split(a, "=")[2])
    end
end
Reactant.set_default_backend(backend)

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

const T = Float32

# ---- timing helper ---------------------------------------------------------

function timed_median(f::F; nrep::Int=11, warmup::Int=2) where {F}
    for _ in 1:warmup
        f()
    end
    s = Float64[]
    for _ in 1:nrep
        t0 = time_ns()
        f()
        push!(s, (time_ns() - t0) * 1e-9)
    end
    return median(s)
end

# ---- workloads -------------------------------------------------------------

# Each row: (D, M, nmodes, K). Mix of cuFINUFFT-leading + already-winning.
const ROWS = [
    (1, 1_000_000, (1024,),       1),
    (1, 1_000_000, (1024,),       2),
    (2, 1_000_000, (256, 256),    1),
    (2, 1_000_000, (1024, 1024),  1),
    (2, 1_000_000, (2048, 2048),  1),
    (2, 1_000_000, (1024, 1024),  2),
    (3, 100_000,   (128, 128, 128), 1),
]

const NTRANS_LIST = (1, 4, 8)

# ---- Reactant timing -------------------------------------------------------

function reactant_time(D::Int, M::Int, nmodes, K::Int, ntr::Int)
    Random.seed!(0xCAFE + M + ntr)
    pts = ntuple(_ -> T.(2π .* rand(M)), D)
    pts_ra = ntuple(d -> Reactant.to_rarray(pts[d]), D)
    plan = plan_nufft(T, K, nmodes; iflag=-1, eps=1e-6)
    prep = Reactant.@jit set_nufft_points(plan, pts_ra)
    if K == 1
        c_arr = Complex{T}.(randn(Complex{T}, M, ntr))
        c_ra = Reactant.to_rarray(c_arr)
        f = Reactant.@compile sync = true execute_nufft(prep, c_ra)
        return timed_median(() -> f(prep, c_ra))
    else
        fk_arr = Complex{T}.(randn(Complex{T}, nmodes..., ntr))
        fk_ra = Reactant.to_rarray(fk_arr)
        f = Reactant.@compile sync = true execute_nufft(prep, fk_ra)
        return timed_median(() -> f(prep, fk_ra))
    end
end

# ---- cuFINUFFT timing ------------------------------------------------------

function cufinufft_time(D::Int, M::Int, nmodes, K::Int, ntr::Int)
    HAS_CUFINUFFT || return NaN
    Random.seed!(0xCAFE + M + ntr)
    pts = ntuple(_ -> T.(2π .* rand(M)), D)

    # Match cuFINUFFT's array-shape convention exactly: ntr==1 ⇒ bare
    # (M,)/(nmodes...,), ntr>1 ⇒ trailing ntr dim.
    in_shape_M  = ntr == 1 ? (M,) : (M, ntr)
    in_shape_fk = ntr == 1 ? nmodes : (nmodes..., ntr)
    if K == 1
        c_arr = Complex{T}.(randn(Complex{T}, in_shape_M...))
        inp  = CUDA.CuArray(c_arr)
        outp = CUDA.zeros(Complex{T}, in_shape_fk...)
    else
        fk_arr = Complex{T}.(randn(Complex{T}, in_shape_fk...))
        inp  = CUDA.CuArray(fk_arr)
        outp = CUDA.zeros(Complex{T}, in_shape_M...)
    end
    pts_dev = ntuple(d -> CUDA.CuArray(pts[d]), D)

    # Build plan + setpts ONCE outside the timing loop — matches what we do
    # for Reactant (prep built outside timed_median; only `execute_nufft`
    # timed). Vary input each iteration (cuFINUFFT short-circuits identical
    # buffers) and use `device_synchronize` (cuFINUFFT launches into a
    # stream the default `synchronize` doesn't wait on, which made earlier
    # numbers look impossibly fast at ~8 µs).
    plan = FINUFFT.cufinufft_makeplan(K, collect(Int64, nmodes), -1, ntr, 1.0e-6;
        dtype=Float32,
    )
    try
        if D == 1
            FINUFFT.cufinufft_setpts!(plan, pts_dev[1])
        elseif D == 2
            FINUFFT.cufinufft_setpts!(plan, pts_dev[1], pts_dev[2])
        else
            FINUFFT.cufinufft_setpts!(plan, pts_dev[1], pts_dev[2], pts_dev[3])
        end
        # Pool of distinct inputs to defeat short-circuit caching.
        n_pool = 16
        pool = if K == 1
            [CUDA.CuArray(Complex{T}.(randn(Complex{T}, in_shape_M...))) for _ in 1:n_pool]
        else
            [CUDA.CuArray(Complex{T}.(randn(Complex{T}, in_shape_fk...))) for _ in 1:n_pool]
        end
        # Generous warmup — first ~5 calls have launcher overhead.
        for k in 1:6
            FINUFFT.cufinufft_exec!(plan, pool[mod1(k, n_pool)], outp)
        end
        CUDA.device_synchronize()
        nrep = 11
        ts = Float64[]
        for k in 1:nrep
            t0 = time_ns()
            FINUFFT.cufinufft_exec!(plan, pool[mod1(k, n_pool)], outp)
            CUDA.device_synchronize()
            push!(ts, (time_ns() - t0) * 1e-9)
        end
        return median(ts)
    finally
        FINUFFT.cufinufft_destroy!(plan)
    end
end

# ---- driver ----------------------------------------------------------------

println("# A3 ntrans sweep   backend=$(backend)")
println()
@printf("%-30s  %-10s  ", "workload", "type")
for ntr in NTRANS_LIST
    @printf("%-22s  ", "ntr=$ntr [R / cF / R÷cF]")
end
println()
println(repeat("-", 30 + 12 + 22 * length(NTRANS_LIST)))

for (D, M, nmodes, K) in ROWS
    label = "$(D)D M=$(M) N=$(join(nmodes, 'x'))"
    @printf("%-30s  %-10s  ", label, "type-$K")
    for ntr in NTRANS_LIST
        tR = try
            reactant_time(D, M, nmodes, K, ntr)
        catch err
            @warn "Reactant $label T$K ntr=$ntr failed: $err"
            NaN
        end
        tC = try
            cufinufft_time(D, M, nmodes, K, ntr)
        catch err
            @warn "cuFINUFFT $label T$K ntr=$ntr failed: $err"
            NaN
        end
        ratio = isnan(tR) || isnan(tC) ? NaN : tR / tC
        if isnan(tR) || isnan(tC)
            @printf("%-22s  ", "—")
        else
            @printf("%5.1f / %5.1f / %4.2f×    ", 1000 * tR, 1000 * tC, ratio)
        end
    end
    println()
    flush(stdout)
end
