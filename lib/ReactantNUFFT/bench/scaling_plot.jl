#!/usr/bin/env julia
# Render scaling plots from /tmp/scaling.csv (produced by scaling_sweep.jl).
# Output: /tmp/scaling_T1.png and /tmp/scaling_T2.png — log-log time-vs-M
# with one panel per dimension D∈{1,2,3} and one curve per library.
#
# Usage:
#   julia --project=@Plotting lib/ReactantNUFFT/bench/scaling_plot.jl

using DelimitedFiles
using Printf

using CairoMakie

const CSV_PATH = "/tmp/scaling.csv"
@assert isfile(CSV_PATH) "Run scaling_sweep.jl first"

raw, hdr = readdlm(CSV_PATH, ','; header=true)
hdr = vec(hdr)
@assert hdr == ["D","M","N","K","lib","time_ms"] "header mismatch: $hdr"

# Index columns
icol(name) = findfirst(==(name), hdr)
const iD, iM, iN, iK, iLIB, iT = icol("D"), icol("M"), icol("N"), icol("K"), icol("lib"), icol("time_ms")

rows = [(Int(raw[r, iD]), Int(raw[r, iM]), String(raw[r, iN]),
         Int(raw[r, iK]), String(raw[r, iLIB]), Float64(raw[r, iT]))
        for r in 1:size(raw, 1)]

LIBS = unique([r[5] for r in rows])
LIB_ORDER = filter(x -> x in LIBS, ["FINUFFT-1", "FINUFFT-16", "FINUFFT-32", "cuFINUFFT", "Reactant"])
COLORS = Dict("Reactant" => :crimson, "cuFINUFFT" => :royalblue,
              "FINUFFT-1" => :darkorange, "FINUFFT-16" => :seagreen,
              "FINUFFT-32" => :seagreen)
MARKERS = Dict("Reactant" => :circle, "cuFINUFFT" => :rect,
               "FINUFFT-1" => :utriangle, "FINUFFT-16" => :diamond,
               "FINUFFT-32" => :diamond)

function curve_for(D, K, lib)
    pts = [(r[2], r[6]) for r in rows if r[1] == D && r[4] == K && r[5] == lib && isfinite(r[6]) && r[6] > 0]
    sort!(pts; by=first)
    return [p[1] for p in pts], [p[2] for p in pts]
end

function n_label(D)
    rs = [r for r in rows if r[1] == D]
    isempty(rs) && return ""
    return rs[1][3]
end

function render_panel!(ax, D, K)
    for lib in LIB_ORDER
        xs, ys = curve_for(D, K, lib)
        isempty(xs) && continue
        scatterlines!(ax, xs, ys;
            color=COLORS[lib], marker=MARKERS[lib], markersize=10, linewidth=2.0,
            label=lib)
    end
    ax.xscale = log10
    ax.yscale = log10
    ax.xlabel = "M (NU points)"
    ax.ylabel = "exec time (ms)"
    ax.title = "D=$D, N=$(n_label(D))"
end

function make_figure(K::Int, title::String, out::String)
    fig = Figure(size=(1200, 420))
    Label(fig[0, 1:3], title; fontsize=18, font=:bold)
    axes = [Axis(fig[1, d]) for d in 1:3]
    for (i, D) in enumerate(1:3); render_panel!(axes[i], D, K); end
    # Single legend on the right.
    elems = [LineElement(color=COLORS[lib], linewidth=2) for lib in LIB_ORDER]
    Legend(fig[1, 4], elems, LIB_ORDER; framevisible=false)
    save(out, fig)
    println("Wrote $out")
end

make_figure(1, "Type-1 NUFFT scaling (Float32, eps=1e-6, ntrans=1)", "/tmp/scaling_T1.png")
make_figure(2, "Type-2 NUFFT scaling (Float32, eps=1e-6, ntrans=1)", "/tmp/scaling_T2.png")
