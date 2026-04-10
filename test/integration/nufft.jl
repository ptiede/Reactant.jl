using Reactant
using Test
using Random

const RunningOnCUDA = try
    contains(string(Reactant.devices()[1]), "CUDA")
catch
    false
end

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

@testset "NUFFT Lifecycle and Method Normalization" begin
    T = Float32
    plan_int = Reactant.nufft_makeplan(T, 1, (16,); method=3, nspread=6)
    @test plan_int.opts.method == Reactant.NUFFT_METHOD_OUTPUTDRIVEN

    plan_sym = Reactant.nufft_makeplan(T, 1, (16,); method=:subproblem, nspread=6)
    @test plan_sym.opts.method == Reactant.NUFFT_METHOD_SUBPROBLEM

    M = 20
    x = rand(T, M) .* T(2 * pi)
    c = randn(Complex{T}, M)
    x_ra = Reactant.to_rarray(x)
    c_ra = Reactant.to_rarray(c)

    prep = Reactant.nufft_setpts(plan_int, x_ra)
    fk_lifecycle = @jit Reactant.nufft_execute(prep, c_ra)
    fk_wrapper = @jit Reactant.nufft_type1(
        (x_ra,), c_ra, (16,); method=:outputdriven, nspread=6
    )
    @test Array(fk_lifecycle) ≈ Array(fk_wrapper)

    plan_t2 = Reactant.nufft_makeplan(T, 2, (16,); method=1, nspread=6)
    prep_t2 = Reactant.nufft_setpts(plan_t2, x_ra)
    c_lifecycle = @jit Reactant.nufft_execute(prep_t2, fk_lifecycle)
    c_wrapper = @jit Reactant.nufft_type2(
        (x_ra,), fk_lifecycle; method=:nuptsdriven, nspread=6
    )
    @test Array(c_lifecycle) ≈ Array(c_wrapper)

    @test_throws ErrorException Reactant.nufft_type1(
        (x,), c, (16,); method=:outputdriven, nspread=6
    )
    @test_throws ErrorException Reactant.nufft_type2(
        (x,), randn(Complex{T}, 16); method=:outputdriven, nspread=6
    )
end

@testset "NUFFT Correctness 1D/2D/3D" begin
    Random.seed!(1234)
    T = Float32

    cases = ((24, (20,)), (20, (12, 10)), (16, (8, 6, 10)))

    for (M, nmodes) in cases
        D = length(nmodes)
        points = ntuple(_ -> rand(T, M) .* T(2 * pi), D)
        c = randn(Complex{T}, M)

        fk_ref = direct_type1(points, c, nmodes; iflag=-1)
        c_ref = direct_type2(points, fk_ref; iflag=-1)

        points_ra = to_rpoints(points)
        c_ra = Reactant.to_rarray(c)
        fk_ref_ra = Reactant.to_rarray(fk_ref)

        fk_ra = @jit Reactant.nufft_type1(
            points_ra, c_ra, nmodes; nspread=8, method=3, iflag=-1
        )
        c_ra_est = @jit Reactant.nufft_type2(
            points_ra, fk_ref_ra; nspread=8, method=:subproblem, iflag=-1
        )

        tol_t1 = D == 1 ? 1.2 : D == 2 ? 2.5 : 4.5
        tol_t2 = D == 1 ? 2.0 : D == 2 ? 4.0 : 7.0
        @test isapprox(Array(fk_ra), fk_ref; atol=tol_t1, rtol=tol_t1)
        @test isapprox(Array(c_ra_est), c_ref; atol=tol_t2, rtol=tol_t2)
    end
end

@testset "NUFFT Dispatch Regression (@jit)" begin
    T = Float32
    M = 16
    N = 12
    x = rand(T, M) .* T(2 * pi)
    c = randn(Complex{T}, M)
    fk = randn(Complex{T}, N)

    x_ra = Reactant.to_rarray(x)
    c_ra = Reactant.to_rarray(c)
    fk_ra = Reactant.to_rarray(fk)

    y1 = @jit Reactant.nufft_type1((x_ra,), c_ra, (N,); method=3, nspread=8)
    y2 = @jit Reactant.nufft_type1((x_ra,), c_ra, (N,); method=:outputdriven, nspread=8)
    z1 = @jit Reactant.nufft_type2((x_ra,), fk_ra; method=1, nspread=8)
    z2 = @jit Reactant.nufft_type2((x_ra,), fk_ra; method=:nuptsdriven, nspread=8)

    @test size(y1) == (N,)
    @test size(y2) == (N,)
    @test size(z1) == (M,)
    @test size(z2) == (M,)
end

@testset "NUFFT HLO Sanity" begin
    T = Float32
    M = 12
    N = 10
    x = rand(T, M) .* T(2 * pi)
    c = randn(Complex{T}, M)
    fk = randn(Complex{T}, N)

    x_ra = Reactant.to_rarray(x)
    c_ra = Reactant.to_rarray(c)
    fk_ra = Reactant.to_rarray(fk)

    hlo_t1 = repr(@code_hlo Reactant.nufft_type1((x_ra,), c_ra, (N,); nspread=6, method=3))
    hlo_t2 = repr(@code_hlo Reactant.nufft_type2((x_ra,), fk_ra; nspread=6, method=:outputdriven))

    @test occursin("stablehlo.scatter", hlo_t1) || occursin("stablehlo.dynamic_update_slice", hlo_t1)
    @test !occursin("julia_callback", hlo_t1)

    @test occursin("stablehlo.gather", hlo_t2)
    @test !occursin("julia_callback", hlo_t2)
end

if RunningOnCUDA
    @testset "NUFFT CUDA @jit Coverage" begin
        T = Float32
        M = 14
        nmodes = (10, 8)

        x = rand(T, M) .* T(2 * pi)
        y = rand(T, M) .* T(2 * pi)
        c = randn(Complex{T}, M)

        fk_ref = direct_type1((x, y), c, nmodes)

        x_ra = Reactant.to_rarray(x)
        y_ra = Reactant.to_rarray(y)
        c_ra = Reactant.to_rarray(c)

        fk_ra = @jit Reactant.nufft_type1(
            (x_ra, y_ra), c_ra, nmodes; nspread=8, method=:subproblem
        )
        @test isapprox(Array(fk_ra), fk_ref; atol=3.0, rtol=3.0)
    end
end
