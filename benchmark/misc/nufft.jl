using Reactant
using Random: Random

include("common.jl")

centered_mode(i::Integer, n::Integer) = i - div(n, 2) - 1

function direct_type1(points::NTuple{D,<:AbstractVector}, c, nmodes; iflag=-1) where {D}
    T = float(real(eltype(c)))
    out = zeros(eltype(c), nmodes)
    for I in CartesianIndices(out)
        mode = ntuple(d -> centered_mode(I[d], nmodes[d]), D)
        acc = zero(eltype(c))
        for j in eachindex(c)
            phase = zero(T)
            for d in 1:D
                phase += T(mode[d]) * points[d][j]
            end
            acc += c[j] * cis(T(iflag) * phase)
        end
        out[I] = acc
    end
    return out
end

function direct_type2(points::NTuple{D,<:AbstractVector}, fk; iflag=-1) where {D}
    T = float(real(eltype(fk)))
    out = zeros(eltype(fk), length(points[1]))
    nmodes = size(fk)
    for j in eachindex(out)
        acc = zero(eltype(fk))
        for I in CartesianIndices(fk)
            mode = ntuple(d -> centered_mode(I[d], nmodes[d]), D)
            phase = zero(T)
            for d in 1:D
                phase += T(mode[d]) * points[d][j]
            end
            acc += fk[I] * cis(T(iflag) * phase)
        end
        out[j] = acc
    end
    return out
end

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

reactant_type1(points, c, nmodes) =
    Reactant.nufft_type1(points, c, nmodes; nspread=4, np=128, method=:outputdriven)
reactant_type2(points, fk) =
    Reactant.nufft_type2(points, fk; nspread=4, np=128, method=:outputdriven)

function run_nufft_benchmark!(results, backend)
    Random.seed!(1234)
    T = Float32

    sweeps = Dict(1 => [(256, 64), (1024, 128)], 2 => [(192, 32), (640, 64)], 3 => [(96, 20), (192, 32)])

    configs = [BenchmarkConfiguration("Default"; compile_options=Reactant.CompileOptions(), nrepeat=10)]

    for D in 1:3
        for (M, Nbase) in sweeps[D]
            nmodes = benchmark_nmodes(D, Nbase)
            points = ntuple(_ -> rand(T, M) .* T(2 * pi), D)
            points_ra = to_rpoints(points)
            c = randn(Complex{T}, M)
            c_ra = Reactant.to_rarray(c)
            fk_ref = direct_type1(points, c, nmodes)
            fk_ref_ra = Reactant.to_rarray(fk_ref)

            tag = "nufft/$(D)d/M=$(M)/N=$(join(nmodes, 'x'))"

            run_benchmark!(
                results,
                backend,
                "$(tag)/type1_direct_dense",
                direct_type1,
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
                configs,
                benchmark_seconds=2.0,
                benchmark_samples=40,
            )

            run_benchmark!(
                results,
                backend,
                "$(tag)/type2_direct_dense",
                direct_type2,
                (points, fk_ref),
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
                (points, fk_ref),
                (points_ra, fk_ref_ra);
                configs,
                benchmark_seconds=2.0,
                benchmark_samples=40,
            )
        end
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    backend = get_backend()
    results = Dict{String,Dict{String,Float64}}()
    run_nufft_benchmark!(results, backend)
    pretty_print_results(results, "misc-nufft", backend)
end
