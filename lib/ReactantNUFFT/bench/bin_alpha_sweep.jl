#!/usr/bin/env julia
# A1 — bin_dims `α·w` multiplier sweep.
#
# For the 4 GPU rows where cuFINUFFT still leads us, sweep `α ∈ {1.0, 1.5,
# 2.0, 3.0}` (default 2.0 for 2D, 1.5 for 3D, 4.0 for 1D) and record the
# end-to-end execute time. Pick the winning α per (D, problem_size).
#
# Usage:
#   julia --project=lib/ReactantNUFFT/bench lib/ReactantNUFFT/bench/bin_alpha_sweep.jl --backend=gpu
#
# Flags:
#   --backend=cpu|gpu (default cpu)
#   --quick           (run a smaller subset for fast iteration)

using Printf
using Random
using Statistics

using Reactant
using ReactantNUFFT

# ---- args ------------------------------------------------------------------

backend = "cpu"
quick = false
for a in ARGS
    if startswith(a, "--backend=")
        global backend = String(split(a, "=")[2])
    elseif a == "--quick"
        global quick = true
    else
        error("unknown flag: $a")
    end
end
Reactant.set_default_backend(backend)

const T = Float32

# ---- timing helper ---------------------------------------------------------

function timed_median(f::F; nrep::Int=11, warmup::Int=2) where {F}
    for _ in 1:warmup
        f()
    end
    samples = Float64[]
    for _ in 1:nrep
        t0 = time_ns()
        f()
        push!(samples, (time_ns() - t0) * 1e-9)
    end
    return median(samples)
end

# ---- sweep -----------------------------------------------------------------

# Workloads: (D, M, nmodes, K)  — focus on GPU-cuFINUFFT-leading cases
const WORKLOADS = quick ? [
    (2, 100_000, (1024, 1024), 2),     # type-2 5.3 ms vs cuFINUFFT 1.3 ms
    (2, 1_000_000, (1024, 1024), 1),   # type-1 7.0 ms vs cuFINUFFT 3.3 ms
] : [
    (2, 1_000_000, (256, 256),   1),   # type-1 6.9 ms vs cuFINUFFT 2.1 ms (3.21×)
    (2, 1_000_000, (1024, 1024), 1),   # type-1 7.0 ms vs cuFINUFFT 3.3 ms (2.14×)
    (2, 1_000_000, (2048, 2048), 1),   # type-1 15.2 ms vs cuFINUFFT 16.4 ms (parity)
    (3, 100_000,   (128, 128, 128), 1),# type-1 26.7 ms vs cuFINUFFT 13.8 ms (1.93×)
    (2, 100_000,   (1024, 1024), 2),   # type-2 5.3 ms vs cuFINUFFT 1.3 ms (4.23×)
    # baselines for sanity (Reactant already wins, expect α=2 ≈ best)
    (1, 1_000_000, (1024,),      1),   # type-1 already wins
    (1, 1_000_000, (16384,),     1),   # type-1 already wins
]

const ALPHAS = (1.0, 1.5, 2.0, 3.0, 4.0)

function run_one(D, M, nmodes, K, alpha)
    Random.seed!(0xC0FFEE)
    pts = ntuple(_ -> T.(2π .* rand(M)), D)
    pts_ra = ntuple(d -> Reactant.to_rarray(pts[d]), D)
    AUTO_BIN_ALPHA_save = ReactantNUFFT.AUTO_BIN_ALPHA[]
    try
        # New AUTO_BIN_ALPHA layout: tuple-of-tuples ((K=1,D=1..3), (K=2,D=1..3))
        per_K = AUTO_BIN_ALPHA_save[K]
        new_K = ntuple(d -> d == D ? Float64(alpha) : per_K[d], 3)
        new_outer = ntuple(k -> k == K ? new_K : AUTO_BIN_ALPHA_save[k], 2)
        ReactantNUFFT.AUTO_BIN_ALPHA[] = new_outer
        plan = plan_nufft(T, K, nmodes; iflag=-1, eps=1e-6)
        prep = Reactant.@jit set_nufft_points(plan, pts_ra)
        if K == 1
            c_ra = Reactant.to_rarray(ComplexF32.(randn(ComplexF32, M)))
            f = Reactant.@compile sync = true execute_nufft(prep, c_ra)
            t = timed_median(() -> f(prep, c_ra))
        else
            fk_ra = Reactant.to_rarray(ComplexF32.(randn(ComplexF32, nmodes...)))
            f = Reactant.@compile sync = true execute_nufft(prep, fk_ra)
            t = timed_median(() -> f(prep, fk_ra))
        end
        return t, plan.bin_dims[1]
    finally
        ReactantNUFFT.AUTO_BIN_ALPHA[] = AUTO_BIN_ALPHA_save
    end
end

# ---- driver ----------------------------------------------------------------

println("# A1 bin_dims α·w sweep   backend=$(backend)")
println()
@printf("%-28s  %-7s  %-7s  %-9s  %-9s  %-9s  %-9s  %-9s  %s\n",
    "workload", "type", "winner", "α=1.0", "α=1.5", "α=2.0", "α=3.0", "α=4.0", "bin@best")
println(repeat("-", 130))

for (D, M, nmodes, K) in WORKLOADS
    label = "$(D)D M=$(M) N=$(join(nmodes, 'x')) T$(K)"
    times = Dict{Float64,Float64}()
    bins  = Dict{Float64,Int}()
    for α in ALPHAS
        try
            t, b = run_one(D, M, nmodes, K, α)
            times[α] = t
            bins[α] = b
        catch err
            @warn "α=$α $label failed: $err"
            times[α] = NaN
            bins[α] = -1
        end
    end
    best_α = argmin(α -> times[α], collect(ALPHAS))
    cells = [haskey(times, α) ? @sprintf("%.3f", 1000 * times[α]) : "—" for α in ALPHAS]
    bin_at_best = get(bins, best_α, -1)
    @printf("%-28s  %-7s  α=%-5.1f  %-9s  %-9s  %-9s  %-9s  %-9s  %d\n",
        label, "type-$K", best_α, cells..., bin_at_best)
    flush(stdout)
end
