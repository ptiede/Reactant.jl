using Reactant
using ReactantNUFFT
using FINUFFT
using Random: Random

include("common.jl")

to_rpoints(points::NTuple{D,<:AbstractVector}) where {D} =
    ntuple(d -> Reactant.to_rarray(points[d]), D)

function benchmark_nmodes(D::Int, N::Int)
    if D == 1
        return (N,)
    elseif D == 2
        return (N, max(8, N ÷ 2))
    end
    return (max(8, N ÷ 3), max(8, N ÷ 2), N)
end

reactant_type1(points, c, nmodes, method=ReactantNUFFT.OutputDriven()) =
    ReactantNUFFT.nufft_type1(points, c, nmodes; method)
reactant_type2(points, fk, method=ReactantNUFFT.OutputDriven()) =
    ReactantNUFFT.nufft_type2(points, fk; method)

function finufft_type1(points::NTuple{1,<:AbstractVector}, c, nmodes; iflag=-1, eps=1e-6)
    return FINUFFT.nufft1d1(points[1], c, iflag, eps, nmodes[1]; nthreads=1)
end

function finufft_type1(points::NTuple{2,<:AbstractVector}, c, nmodes; iflag=-1, eps=1e-6)
    return FINUFFT.nufft2d1(
        points[1], points[2], c, iflag, eps, nmodes[1], nmodes[2]; nthreads=1
    )
end

function finufft_type1(points::NTuple{3,<:AbstractVector}, c, nmodes; iflag=-1, eps=1e-6)
    return FINUFFT.nufft3d1(
        points[1],
        points[2],
        points[3],
        c,
        iflag,
        eps,
        nmodes[1],
        nmodes[2],
        nmodes[3];
        nthreads=1,
    )
end

function finufft_type2(points::NTuple{1,<:AbstractVector}, fk; iflag=-1, eps=1e-6)
    return FINUFFT.nufft1d2(points[1], iflag, eps, fk; nthreads=1)
end

function finufft_type2(points::NTuple{2,<:AbstractVector}, fk; iflag=-1, eps=1e-6)
    return FINUFFT.nufft2d2(points[1], points[2], iflag, eps, fk; nthreads=1)
end

function finufft_type2(points::NTuple{3,<:AbstractVector}, fk; iflag=-1, eps=1e-6)
    return FINUFFT.nufft3d2(points[1], points[2], points[3], iflag, eps, fk; nthreads=1)
end

function run_nufft_benchmark!(results, backend)
    Random.seed!(1234)
    T = Float32

    sweeps = Dict(1 => [(256, 64), (1024, 128)], 2 => [(192, 32), (640, 64)], 3 => [(96, 20), (192, 32)])

    configs = [BenchmarkConfiguration("Default"; compile_options=reactant_single_thread_compile_options(), nrepeat=10)]

    for D in 1:3
        for (M, Nbase) in sweeps[D]
            nmodes = benchmark_nmodes(D, Nbase)
            points = ntuple(_ -> rand(T, M) .* T(2 * pi), D)
            points_ra = to_rpoints(points)
            c = randn(Complex{T}, M)
            c_ra = Reactant.to_rarray(c)
            fk = randn(Complex{T}, nmodes...)
            fk_ra = Reactant.to_rarray(fk)

            tag = "nufft/$(D)d/M=$(M)/N=$(join(nmodes, 'x'))"

            run_benchmark!(
                results,
                backend,
                "$(tag)/type1_finufft_cpu",
                finufft_type1,
                (points, c, nmodes),
                ();
                configs=BenchmarkConfiguration[],
                benchmark_seconds=2.0,
                benchmark_samples=40,
            )

            run_benchmark!(
                results,
                backend,
                "$(tag)/type1_reactant_outputdriven",
                reactant_type1,
                (points, c, nmodes),
                (points_ra, c_ra, nmodes);
                skip_cpu=true,
                configs,
                benchmark_seconds=2.0,
                benchmark_samples=40,
                reactant_measurement=:compiled_walltime,
            )

            run_benchmark!(
                results,
                backend,
                "$(tag)/type2_finufft_cpu",
                finufft_type2,
                (points, fk),
                ();
                configs=BenchmarkConfiguration[],
                benchmark_seconds=2.0,
                benchmark_samples=40,
            )

            run_benchmark!(
                results,
                backend,
                "$(tag)/type2_reactant_outputdriven",
                reactant_type2,
                (points, fk),
                (points_ra, fk_ra);
                skip_cpu=true,
                configs,
                benchmark_seconds=2.0,
                benchmark_samples=40,
                reactant_measurement=:compiled_walltime,
            )
        end
    end

    return nothing
end

function run_nufft_method_benchmark!(results, backend)
    Random.seed!(1234)
    T = Float32

    sweeps = Dict(
        1 => [(256, 64), (1024, 128)],
        2 => [(192, 32), (640, 64)],
        3 => [(96, 20), (192, 32)],
    )

    configs = [BenchmarkConfiguration("Default"; compile_options=reactant_single_thread_compile_options(), nrepeat=10)]
    methods = (
        ("outputdriven", ReactantNUFFT.OutputDriven()),
        ("nuptsdriven", ReactantNUFFT.NUPtsDriven()),
        ("automethod", ReactantNUFFT.AutoMethod()),
    )

    for D in 1:3
        for (M, Nbase) in sweeps[D]
            nmodes = benchmark_nmodes(D, Nbase)
            points = ntuple(_ -> rand(T, M) .* T(2 * pi), D)
            points_ra = to_rpoints(points)
            c = randn(Complex{T}, M)
            c_ra = Reactant.to_rarray(c)
            fk = randn(Complex{T}, nmodes...)
            fk_ra = Reactant.to_rarray(fk)

            tag = "nufft_methods/$(D)d/M=$(M)/N=$(join(nmodes, 'x'))"

            for (method_name, method) in methods
                run_benchmark!(
                    results,
                    backend,
                    "$(tag)/type1_reactant_$(method_name)",
                    reactant_type1,
                    (points, c, nmodes, method),
                    (points_ra, c_ra, nmodes, method);
                    skip_cpu=true,
                    configs,
                    benchmark_seconds=2.0,
                    benchmark_samples=40,
                    reactant_measurement=:compiled_walltime,
                )

                run_benchmark!(
                    results,
                    backend,
                    "$(tag)/type2_reactant_$(method_name)",
                    reactant_type2,
                    (points, fk, method),
                    (points_ra, fk_ra, method);
                    skip_cpu=true,
                    configs,
                    benchmark_seconds=2.0,
                    benchmark_samples=40,
                    reactant_measurement=:compiled_walltime,
                )
            end
        end
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    Reactant.set_default_backend("cpu")
    backend = "CPU"
    results = Dict{String,Dict{String,Float64}}()
    run_nufft_benchmark!(results, backend)
    save_results(results, joinpath(@__DIR__, "results"), "misc_nufft", backend)
    pretty_print_results(results, "misc-nufft", backend)
end
