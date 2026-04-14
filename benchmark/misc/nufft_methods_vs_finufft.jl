using JSON
using Printf
using Random: Random
using Reactant

include("nufft.jl")

const STANDARD_CASES = (
    (1, 256, (64,)),
    (1, 1024, (128,)),
    (2, 192, (32, 16)),
    (2, 640, (64, 32)),
    (3, 96, (8, 10, 20)),
    (3, 192, (10, 16, 32)),
)

const STANDARD_METHODS = (
    ("outputdriven", ReactantNUFFT.OutputDriven()),
    ("nuptsdriven", ReactantNUFFT.NUPtsDriven()),
    ("automethod", ReactantNUFFT.AutoMethod()),
)

function compile_and_measure(fn, args)
    GC.gc(true)
    timed = @timed Reactant.compile(
        fn, args; compile_options=reactant_single_thread_compile_options(; sync=true)
    )
    return timed.value, timed.time, timed.bytes
end

function benchmark_cpu(fn, args; seconds=0.35, samples=8)
    fn(args...)
    bench = @b fn($(args)...) seconds = seconds evals = 1 samples = samples
    return bench.time
end

function benchmark_compiled(compiled_fn, args; seconds=0.35, samples=8)
    compiled_fn(args...)
    bench = @b compiled_fn($(args)...) seconds = seconds evals = 1 samples = samples
    return bench.time
end

function run_nufft_method_vs_finufft_benchmark()
    Random.seed!(1234)
    T = Float32
    results = Dict{String,Any}[]

    for (D, M, nmodes) in STANDARD_CASES
        points = ntuple(_ -> rand(T, M) .* T(2 * pi), D)
        points_ra = to_rpoints(points)
        c = randn(Complex{T}, M)
        c_ra = Reactant.to_rarray(c)
        fk = randn(Complex{T}, nmodes...)
        fk_ra = Reactant.to_rarray(fk)

        finufft_t1_s = benchmark_cpu(finufft_type1, (points, c, nmodes))
        finufft_t2_s = benchmark_cpu(finufft_type2, (points, fk))

        push!(
            results,
            Dict(
                "case" => "type1",
                "dimension" => D,
                "m" => M,
                "nmodes" => collect(nmodes),
                "implementation" => "finufft_1thread",
                "compile_time_s" => nothing,
                "compile_alloc_bytes" => nothing,
                "steady_runtime_s" => finufft_t1_s,
            ),
        )
        push!(
            results,
            Dict(
                "case" => "type2",
                "dimension" => D,
                "m" => M,
                "nmodes" => collect(nmodes),
                "implementation" => "finufft_1thread",
                "compile_time_s" => nothing,
                "compile_alloc_bytes" => nothing,
                "steady_runtime_s" => finufft_t2_s,
            ),
        )

        @printf(
            "baseline D=%d M=%d N=%s finufft_type1=%.6gs finufft_type2=%.6gs\n",
            D,
            M,
            join(nmodes, "x"),
            finufft_t1_s,
            finufft_t2_s,
        )
        flush(stdout)

        for (method_name, method) in STANDARD_METHODS
            compiled_t1, compile_t1_s, compile_t1_bytes = compile_and_measure(
                reactant_type1, (points_ra, c_ra, nmodes, method)
            )
            steady_t1_s = benchmark_compiled(compiled_t1, (points_ra, c_ra, nmodes, method))
            push!(
                results,
                Dict(
                    "case" => "type1",
                    "dimension" => D,
                    "m" => M,
                    "nmodes" => collect(nmodes),
                    "implementation" => method_name,
                    "compile_time_s" => compile_t1_s,
                    "compile_alloc_bytes" => compile_t1_bytes,
                    "steady_runtime_s" => steady_t1_s,
                ),
            )
            @printf(
                "type1 D=%d M=%d N=%s impl=%s compile=%.4fs steady=%.6gs\n",
                D,
                M,
                join(nmodes, "x"),
                method_name,
                compile_t1_s,
                steady_t1_s,
            )
            flush(stdout)

            compiled_t2, compile_t2_s, compile_t2_bytes = compile_and_measure(
                reactant_type2, (points_ra, fk_ra, method)
            )
            steady_t2_s = benchmark_compiled(compiled_t2, (points_ra, fk_ra, method))
            push!(
                results,
                Dict(
                    "case" => "type2",
                    "dimension" => D,
                    "m" => M,
                    "nmodes" => collect(nmodes),
                    "implementation" => method_name,
                    "compile_time_s" => compile_t2_s,
                    "compile_alloc_bytes" => compile_t2_bytes,
                    "steady_runtime_s" => steady_t2_s,
                ),
            )
            @printf(
                "type2 D=%d M=%d N=%s impl=%s compile=%.4fs steady=%.6gs\n",
                D,
                M,
                join(nmodes, "x"),
                method_name,
                compile_t2_s,
                steady_t2_s,
            )
            flush(stdout)
        end
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    Reactant.set_default_backend("cpu")
    results = run_nufft_method_vs_finufft_benchmark()
    path = joinpath(@__DIR__, "results", "misc_nufft_methods_vs_finufft_CPU.json")
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.json(io, results; pretty=true)
    end
    @info "Saved method-vs-finufft benchmark results to $path"
end
