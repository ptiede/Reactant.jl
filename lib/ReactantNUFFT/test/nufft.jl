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

# --- helpers ----------------------------------------------------------------

# Brute-force reference using the package's own implementation (which lives
# in non-traced Julia).
brute_type1 = ReactantNUFFT.direct_type1
brute_type2 = ReactantNUFFT.direct_type2

# Tolerance scaled to the requested eps + a margin for kernel/quadrature error
# and accumulated Float32 round-off. The bound is loose enough for F32 at
# eps=1e-6 in 3D while still catching gross algorithmic mistakes.
tolerance(eps_target, D, T=Float32) = max(eps_target * 500 * D, T === Float32 ? 5e-4 : 1e-9)

# --- Type-1 correctness -----------------------------------------------------

@testset "Type-1 correctness D=$D T=$T eps=$eps_target iflag=$iflag" for
    D in 1:3,
    T in (Float32, Float64),
    eps_target in (1e-3, 1e-6),
    iflag in (-1, +1)

    if T === Float32 && eps_target < 1e-6
        continue
    end

    Random.seed!(0xCAFE + D + Int(round(-log10(eps_target))))
    M = 32
    nmodes = D == 1 ? (24,) : D == 2 ? (12, 10) : (8, 6, 10)
    pts = ntuple(_ -> T.(2π .* rand(M)), D)
    c = Complex{T}.(randn(Complex{T}, M))

    fk_ref = brute_type1(pts, c, nmodes; iflag=iflag)
    pts_ra = ntuple(d -> Reactant.to_rarray(pts[d]), D)
    c_ra = Reactant.to_rarray(c)
    fk = Reactant.@jit nufft_type1(pts_ra, c_ra, nmodes; iflag=iflag, eps=eps_target)

    @test size(fk) == nmodes
    rel_err = maximum(abs.(Array(fk) .- fk_ref)) / maximum(abs.(fk_ref))
    @test rel_err < tolerance(eps_target, D, T)
end

# --- Type-2 correctness -----------------------------------------------------

@testset "Type-2 correctness D=$D T=$T eps=$eps_target iflag=$iflag" for
    D in 1:3,
    T in (Float32, Float64),
    eps_target in (1e-3, 1e-6),
    iflag in (-1, +1)

    if T === Float32 && eps_target < 1e-6
        continue
    end

    Random.seed!(0xBEAD + D + Int(round(-log10(eps_target))))
    M = 32
    nmodes = D == 1 ? (24,) : D == 2 ? (12, 10) : (8, 6, 10)
    pts = ntuple(_ -> T.(2π .* rand(M)), D)
    fk = Complex{T}.(randn(Complex{T}, nmodes...))

    c_ref = brute_type2(pts, fk; iflag=iflag)
    pts_ra = ntuple(d -> Reactant.to_rarray(pts[d]), D)
    fk_ra = Reactant.to_rarray(fk)
    c = Reactant.@jit nufft_type2(pts_ra, fk_ra; iflag=iflag, eps=eps_target)

    @test size(c) == (M,)
    rel_err = maximum(abs.(Array(c) .- c_ref)) / maximum(abs.(c_ref))
    @test rel_err < tolerance(eps_target, D, T)
end

# --- Batched (ntrans>1) -----------------------------------------------------

@testset "Type-1 batched ntrans D=$D ntrans=$ntrans" for D in 1:3, ntrans in (1, 4)
    Random.seed!(0xF00D + D + ntrans)
    T = Float32
    M = 24
    nmodes = D == 1 ? (16,) : D == 2 ? (8, 8) : (6, 5, 4)
    pts = ntuple(_ -> T.(2π .* rand(M)), D)
    C = Complex{T}.(randn(Complex{T}, M, ntrans))

    pts_ra = ntuple(d -> Reactant.to_rarray(pts[d]), D)
    C_ra = Reactant.to_rarray(C)
    FK = Reactant.@jit nufft_type1(pts_ra, C_ra, nmodes; iflag=-1, eps=1e-6)
    @test size(FK) == (nmodes..., ntrans)

    for t in 1:ntrans
        fk_ref = brute_type1(pts, C[:, t], nmodes; iflag=-1)
        slice = ntuple(d -> Colon(), D)
        rel_err = maximum(abs.(Array(FK)[slice..., t] .- fk_ref)) / maximum(abs.(fk_ref))
        @test rel_err < tolerance(1e-6, D, T)
    end
end

@testset "Type-2 batched ntrans D=$D ntrans=$ntrans" for D in 1:3, ntrans in (1, 4)
    Random.seed!(0xB10B + D + ntrans)
    T = Float32
    M = 24
    nmodes = D == 1 ? (16,) : D == 2 ? (8, 8) : (6, 5, 4)
    pts = ntuple(_ -> T.(2π .* rand(M)), D)
    FK = Complex{T}.(randn(Complex{T}, nmodes..., ntrans))

    pts_ra = ntuple(d -> Reactant.to_rarray(pts[d]), D)
    FK_ra = Reactant.to_rarray(FK)
    C = Reactant.@jit nufft_type2(pts_ra, FK_ra; iflag=-1, eps=1e-6)
    @test size(C) == (M, ntrans)

    for t in 1:ntrans
        slice = ntuple(d -> Colon(), D)
        c_ref = brute_type2(pts, FK[slice..., t]; iflag=-1)
        rel_err = maximum(abs.(Array(C)[:, t] .- c_ref)) / maximum(abs.(c_ref))
        @test rel_err < tolerance(1e-6, D, T)
    end
end

# --- Plan + setpts + execute reuse ------------------------------------------

@testset "Plan reuse: same points, multiple strengths" begin
    T = Float32
    M = 16
    nmodes = (8, 8)
    pts = ntuple(_ -> T.(2π .* rand(M)), 2)
    pts_ra = ntuple(d -> Reactant.to_rarray(pts[d]), 2)

    plan = plan_nufft(T, 1, nmodes; iflag=-1, eps=1e-6)
    prep = Reactant.@jit set_nufft_points(plan, pts_ra)
    @test prep isa NUFFTSetPts{T,2}

    for trial in 1:3
        c = Complex{T}.(randn(Complex{T}, M))
        c_ra = Reactant.to_rarray(c)
        fk = Reactant.@jit execute_nufft(prep, c_ra)
        fk_ref = brute_type1(pts, c, nmodes; iflag=-1)
        rel_err = maximum(abs.(Array(fk) .- fk_ref)) / maximum(abs.(fk_ref))
        @test rel_err < 5e-4
    end
end

# --- HLO sanity -------------------------------------------------------------

@testset "Compiled IR contains scatter / gather / fft" begin
    T = Float32
    M = 8
    nmodes = (16,)
    pts = ntuple(_ -> T.(2π .* rand(M)), 1)
    c = Complex{T}.(randn(Complex{T}, M))
    pts_ra = ntuple(d -> Reactant.to_rarray(pts[d]), 1)
    c_ra = Reactant.to_rarray(c)

    plan = plan_nufft(T, 1, nmodes; eps=1e-6)
    prep = Reactant.@jit set_nufft_points(plan, pts_ra)
    hlo_t1 = sprint(io -> show(io, Reactant.@code_hlo execute_nufft(prep, c_ra)))
    @test occursin("stablehlo.scatter", hlo_t1)
    @test occursin("stablehlo.fft", hlo_t1)

    fk = Complex{T}.(randn(Complex{T}, nmodes...))
    fk_ra = Reactant.to_rarray(fk)
    plan2 = plan_nufft(T, 2, nmodes; eps=1e-6)
    prep2 = Reactant.@jit set_nufft_points(plan2, pts_ra)
    hlo_t2 = sprint(io -> show(io, Reactant.@code_hlo execute_nufft(prep2, fk_ra)))
    @test occursin("stablehlo.gather", hlo_t2)
    @test occursin("stablehlo.fft", hlo_t2)
end
