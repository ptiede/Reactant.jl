#!/usr/bin/env julia
# Render D=2 scaling plot from /tmp/d2_scaling.csv (produced by
# d2_scaling_sweep.jl). Output: /tmp/d2_scaling.png — 2 panels (T1, T2),
# each log-log time-vs-M, one solid (Reactant) + one dashed (cuFINUFFT)
# curve per N.
#
# Usage:
#   julia --project=@Plotting lib/ReactantNUFFT/bench/d2_scaling_plot.jl

using DelimitedFiles
using Printf
using CairoMakie

const CSV_PATH = "/tmp/d2_scaling.csv"
@assert isfile(CSV_PATH) "Run d2_scaling_sweep.jl first"

raw, hdr = readdlm(CSV_PATH, ','; header=true)
hdr = vec(hdr)
@assert hdr == ["K","M","N","lib","time_ms"] "header mismatch: $hdr"

const rows = [(Int(raw[r, 1]), Int(raw[r, 2]), Int(raw[r, 3]),
               String(raw[r, 4]), Float64(raw[r, 5]))
              for r in 1:size(raw, 1)]

const Ns_present = sort(unique(r[3] for r in rows))
const N_COLORS = let c = cgrad(:viridis, length(Ns_present); categorical=true)
    Dict(N => c[i] for (i, N) in enumerate(Ns_present))
end
const LINESTYLE = Dict("Reactant" => :solid, "cuFINUFFT" => :dash)

function curve(K, N, lib)
    pts = [(r[2], r[5]) for r in rows
           if r[1] == K && r[3] == N && r[4] == lib && isfinite(r[5]) && r[5] > 0]
    sort!(pts; by=first)
    return [p[1] for p in pts], [p[2] for p in pts]
end

function render_panel!(ax, K)
    for N in Ns_present, lib in ("Reactant", "cuFINUFFT")
        xs, ys = curve(K, N, lib)
        isempty(xs) && continue
        scatterlines!(ax, xs, ys;
            color=N_COLORS[N], linestyle=LINESTYLE[lib],
            marker=lib == "Reactant" ? :circle : :rect,
            markersize=8, linewidth=2.0,
            label="$lib N=$N",
        )
    end
    ax.xscale  = log10
    ax.yscale  = log10
    ax.xlabel  = "M (NU points)"
    ax.ylabel  = "exec time (ms)"
    ax.title   = "Type-$K (D=2)"
    ax.xtickformat = xs -> [@sprintf("10^%d", round(Int, log10(x))) for x in xs]
end

fig = Figure(size=(1300, 520))
Label(fig[0, 1:3], "D=2 NUFFT scaling: Reactant (solid) vs cuFINUFFT (dashed)  —  Float32, eps=1e-6, ntrans=1";
      fontsize=16, font=:bold)
ax_t1 = Axis(fig[1, 1])
ax_t2 = Axis(fig[1, 2])
render_panel!(ax_t1, 1)
render_panel!(ax_t2, 2)

# Combined legend on the right: one line element per N (color), and a
# (solid/dashed) entry per library.
n_elems = [LineElement(color=N_COLORS[N], linewidth=3) for N in Ns_present]
n_labels = ["N=$N" for N in Ns_present]
lib_elems = [
    LineElement(color=:black, linewidth=2, linestyle=:solid),
    LineElement(color=:black, linewidth=2, linestyle=:dash),
]
lib_labels = ["Reactant", "cuFINUFFT"]
Legend(fig[1, 3], [n_elems, lib_elems], [n_labels, lib_labels], ["grid (N×N)", "library"];
       framevisible=false)

save("/tmp/d2_scaling.png", fig)
println("Wrote /tmp/d2_scaling.png")
