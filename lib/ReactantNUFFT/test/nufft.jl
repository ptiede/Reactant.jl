using FFTW
using Reactant
using ReactantNUFFT
using Test
using Random

const NUFFT = ReactantNUFFT

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

struct WrappedMethod <: NUFFT.AbstractNUFFTMethod end

NUFFT.spread_to_grid(::WrappedMethod, prep::NUFFT.PreparedNUFFTPlan, c::AbstractVector, kernel) =
    NUFFT.spread_to_grid(NUFFT.NUPtsDriven(), prep, c, kernel)

NUFFT.interp_from_grid(
    ::WrappedMethod, prep::NUFFT.PreparedNUFFTPlan, grid::AbstractArray, kernel
) = NUFFT.interp_from_grid(NUFFT.OutputDriven(), prep, grid, kernel)

@testset "NUFFT Lifecycle and Method Types" begin
    T = Float32
    plan_default = NUFFT.plan_nufft(T, 1, (16,); method=NUFFT.OutputDriven(), nspread=6)
    @test plan_default.opts.method isa NUFFT.OutputDriven
    @test plan_default.opts.kernel isa NUFFT.ExpSemicircleKernel

    plan_nupts = NUFFT.plan_nufft(T, 1, (16,); method=NUFFT.NUPtsDriven(), nspread=6)
    @test plan_nupts.opts.method isa NUFFT.NUPtsDriven

    plan_auto = NUFFT.plan_nufft(T, 1, (16,); method=NUFFT.AutoMethod(), nspread=6)
    @test plan_auto.opts.method isa NUFFT.AutoMethod

    plan_wrapped = NUFFT.plan_nufft(T, 1, (16,); method=WrappedMethod(), nspread=6)
    @test plan_wrapped.opts.method isa WrappedMethod

    opts = NUFFT.NUFFTOptions(T; method=NUFFT.OutputDriven(), nspread=8, np=64)
    plan_merged = NUFFT.plan_nufft(T, 1, (16,); opts, eps=T(1.0e-5))
    @test plan_merged.opts.method isa NUFFT.OutputDriven
    @test plan_merged.opts.nspread == 8
    @test plan_merged.opts.np == 64
    @test plan_merged.opts.eps == T(1.0e-5)

    M = 20
    x = rand(T, M) .* T(2 * pi)
    c = randn(Complex{T}, M)
    x_ra = Reactant.to_rarray(x)
    c_ra = Reactant.to_rarray(c)

    prep = NUFFT.set_nufft_points(plan_default, x_ra)
    fk_lifecycle = @jit NUFFT.execute_nufft(prep, c_ra)
    fk_wrapper = @jit NUFFT.nufft_type1(
        (x_ra,), c_ra, (16,); method=NUFFT.OutputDriven(), nspread=6
    )
    @test Array(fk_lifecycle) ≈ Array(fk_wrapper)

    plan_t2 = NUFFT.plan_nufft(T, 2, (16,); method=NUFFT.OutputDriven(), nspread=6)
    prep_t2 = NUFFT.set_nufft_points(plan_t2, x_ra)
    c_lifecycle = @jit NUFFT.execute_nufft(prep_t2, fk_lifecycle)
    c_wrapper = @jit NUFFT.nufft_type2(
        (x_ra,), fk_lifecycle; method=NUFFT.OutputDriven(), nspread=6
    )
    @test Array(c_lifecycle) ≈ Array(c_wrapper)

    fk_nupts = @jit NUFFT.nufft_type1(
        (x_ra,), c_ra, (16,); method=NUFFT.NUPtsDriven(), nspread=6
    )
    fk_auto = @jit NUFFT.nufft_type1(
        (x_ra,), c_ra, (16,); method=NUFFT.AutoMethod(), nspread=6
    )
    fk_wrapped = @jit NUFFT.nufft_type1(
        (x_ra,), c_ra, (16,); method=WrappedMethod(), nspread=6
    )
    c_nupts = @jit NUFFT.nufft_type2(
        (x_ra,), fk_lifecycle; method=NUFFT.NUPtsDriven(), nspread=6
    )
    c_auto = @jit NUFFT.nufft_type2(
        (x_ra,), fk_lifecycle; method=NUFFT.AutoMethod(), nspread=6
    )
    c_wrapped = @jit NUFFT.nufft_type2(
        (x_ra,), fk_lifecycle; method=WrappedMethod(), nspread=6
    )
    @test Array(fk_nupts) ≈ Array(fk_wrapper)
    @test Array(fk_auto) ≈ Array(fk_wrapper)
    @test Array(fk_wrapped) ≈ Array(fk_wrapper)
    @test Array(c_nupts) ≈ Array(c_wrapper)
    @test Array(c_auto) ≈ Array(c_wrapper)
    @test Array(c_wrapped) ≈ Array(c_wrapper)

    @test NUFFT.resolve_auto_method(1, prep) isa NUFFT.NUPtsDriven
    @test NUFFT.resolve_auto_method(2, prep) isa NUFFT.OutputDriven

    y = rand(T, M) .* T(2 * pi)
    y_ra = Reactant.to_rarray(y)
    prep_2d = NUFFT.set_nufft_points(
        NUFFT.plan_nufft(T, 2, (12, 10); method=NUFFT.AutoMethod(), nspread=6),
        (x_ra, y_ra),
    )
    @test NUFFT.resolve_auto_method(1, prep_2d) isa NUFFT.OutputDriven
    @test NUFFT.resolve_auto_method(2, prep_2d) isa NUFFT.OutputDriven

    z = rand(T, M) .* T(2 * pi)
    z_ra = Reactant.to_rarray(z)
    prep_3d = NUFFT.set_nufft_points(
        NUFFT.plan_nufft(T, 2, (12, 10, 8); method=NUFFT.AutoMethod(), nspread=6),
        (x_ra, y_ra, z_ra),
    )
    @test NUFFT.resolve_auto_method(1, prep_3d) isa NUFFT.NUPtsDriven
    @test NUFFT.resolve_auto_method(2, prep_3d) isa NUFFT.OutputDriven

    @test_throws ErrorException NUFFT.nufft_type1(
        (x,), c, (16,); method=NUFFT.OutputDriven(), nspread=6
    )
    @test_throws ErrorException NUFFT.nufft_type2(
        (x,), randn(Complex{T}, 16); method=NUFFT.OutputDriven(), nspread=6
    )
end

@testset "NUFFT Correctness 1D/2D/3D" begin
    Random.seed!(1234)
    T = Float32

    cases = ((24, (20,)), (20, (12, 10)), (16, (8, 6, 10)))
    methods = (
        NUFFT.OutputDriven(),
        NUFFT.NUPtsDriven(),
        NUFFT.AutoMethod(),
    )

    for method in methods
        for (M, nmodes) in cases
            D = length(nmodes)
            points = ntuple(_ -> rand(T, M) .* T(2 * pi), D)
            c = randn(Complex{T}, M)

            fk_ref = direct_type1(points, c, nmodes; iflag=-1)
            c_ref = direct_type2(points, fk_ref; iflag=-1)

            points_ra = to_rpoints(points)
            c_ra = Reactant.to_rarray(c)
            fk_ref_ra = Reactant.to_rarray(fk_ref)

            fk_ra = @jit NUFFT.nufft_type1(
                points_ra, c_ra, nmodes; nspread=8, method, iflag=-1
            )
            c_ra_est = @jit NUFFT.nufft_type2(
                points_ra, fk_ref_ra; nspread=8, method, iflag=-1
            )

            tol_t1 = D == 1 ? 1.2 : D == 2 ? 2.5 : 4.5
            tol_t2 = D == 1 ? 2.0 : D == 2 ? 4.0 : 7.0
            @test isapprox(Array(fk_ra), fk_ref; atol=tol_t1, rtol=tol_t1)
            @test isapprox(Array(c_ra_est), c_ref; atol=tol_t2, rtol=tol_t2)
        end
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

    methods = (
        NUFFT.OutputDriven(),
        NUFFT.NUPtsDriven(),
        NUFFT.AutoMethod(),
    )

    for method in methods
        y1 = @jit NUFFT.nufft_type1((x_ra,), c_ra, (N,); method, nspread=8)
        y2 = @jit NUFFT.nufft_type1((x_ra,), c_ra, (N,); method, nspread=8)
        z1 = @jit NUFFT.nufft_type2((x_ra,), fk_ra; method, nspread=8)
        z2 = @jit NUFFT.nufft_type2((x_ra,), fk_ra; method, nspread=8)

        @test size(y1) == (N,)
        @test size(y2) == (N,)
        @test size(z1) == (M,)
        @test size(z2) == (M,)
    end
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

    hlo_output_t1 = repr(
        @code_hlo NUFFT.nufft_type1(
            (x_ra,), c_ra, (N,); nspread=6, method=NUFFT.OutputDriven()
        )
    )
    hlo_nupts_t2 = repr(
        @code_hlo NUFFT.nufft_type2(
            (x_ra,), fk_ra; nspread=6, method=NUFFT.NUPtsDriven()
        )
    )
    hlo_auto_t2 = repr(
        @code_hlo NUFFT.nufft_type2(
            (x_ra,), fk_ra; nspread=6, method=NUFFT.AutoMethod()
        )
    )

    @test occursin("stablehlo.scatter", hlo_output_t1) || occursin("stablehlo.dynamic_update_slice", hlo_output_t1)
    @test !occursin("julia_callback", hlo_output_t1)

    @test occursin("stablehlo.gather", hlo_nupts_t2)
    @test !occursin("julia_callback", hlo_nupts_t2)
    @test occursin("stablehlo.gather", hlo_auto_t2)
    @test !occursin("julia_callback", hlo_auto_t2)
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

        fk_ra = @jit NUFFT.nufft_type1(
            (x_ra, y_ra), c_ra, nmodes; nspread=8, method=NUFFT.OutputDriven()
        )
        @test isapprox(Array(fk_ra), fk_ref; atol=3.0, rtol=3.0)
    end
end
