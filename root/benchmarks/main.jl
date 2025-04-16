using CUDA
using Printf
using Plots

include("../algorithms/hillis_steele.jl")
include("../algorithms/blelloch.jl")
include("utils.jl")

(sizes, 
 times_seq_hs_cpu_rand, times_seq_bel_cpu_rand, times_hs_gpu_rand, times_bel_gpu_opt_rand,
 times_seq_hs_cpu_sparse, times_seq_bel_cpu_sparse, times_hs_gpu_sparse, times_bel_gpu_opt_sparse,
 times_seq_hs_cpu_skewed, times_seq_bel_cpu_skewed, times_hs_gpu_skewed, times_bel_gpu_opt_skewed,
 times_seq_hs_cpu_sorted, times_seq_bel_cpu_sorted, times_hs_gpu_sorted, times_bel_gpu_opt_sorted,
 times_seq_hs_cpu_alt, times_seq_bel_cpu_alt, times_hs_gpu_alt, times_bel_gpu_opt_alt,
 times_seq_hs_cpu_real, times_seq_bel_cpu_real, times_hs_gpu_real, times_bel_gpu_opt_real,
 times_seq_hs_cpu_mnist, times_seq_bel_cpu_mnist, times_hs_gpu_mnist, times_bel_gpu_opt_mnist) = run_benchmarks()

x_ticks = [2^i for i in 10:4:26]
x_tick_labels = ["2^{$i}" for i in 10:4:26]

y_ticks = [10.0^i for i in -5:0]
y_tick_labels = ["10 μs", "100 μs", "1 ms", "10 ms", "100 ms", "1 s"]

distributions = ["rand", "sparse", "skewed", "sorted", "alt", "real", "mnist"]

for (dist_idx, dist) in enumerate(distributions)
    local times_seq_hs_cpu = eval(Symbol("times_seq_hs_cpu_$dist"))
    local times_seq_bel_cpu = eval(Symbol("times_seq_bel_cpu_$dist"))
    local times_hs_gpu = eval(Symbol("times_hs_gpu_$dist"))
    local times_bel_gpu_opt = eval(Symbol("times_bel_gpu_opt_$dist"))

    # All Implementations Performance
    local p1 = plot(sizes, times_seq_hs_cpu, label="Hillis & Steele (CPU)", 
              marker=:square, linewidth=2, xscale=:log2, yscale=:log10, 
              xticks=(x_ticks, x_tick_labels), yticks=(y_ticks, y_tick_labels),
              legend=:topleft, size=(800, 600), fontsize=10)
    plot!(p1, sizes, times_seq_bel_cpu, label="Blelloch (CPU)", marker=:diamond, linewidth=2)
    plot!(p1, sizes, times_hs_gpu, label="Hillis & Steele (GPU)", marker=:cross, linewidth=2)
    plot!(p1, sizes, times_bel_gpu_opt, label="Blelloch (GPU)", marker=:circle, linewidth=2)
    xlabel!("Array Size")
    ylabel!("Time")
    title!("Performance: $dist Data")
    savefig(p1, "plots/$(dist)_all_implementations.png")
    println("Saved as '$(dist)_all_implementations.png'")

    # Speedup CPU vs GPU
    local speedup_hs = times_seq_hs_cpu ./ times_hs_gpu
    local speedup_bel = times_seq_bel_cpu ./ times_bel_gpu_opt
    local max_speedup = ceil(maximum([speedup_hs; speedup_bel]))
    local y_ticks_speedup = 0:1:max_speedup
    local p2 = plot(sizes, speedup_hs, label="Hillis & Steele (CPU/GPU)", 
              marker=:circle, linewidth=2, xscale=:log2, 
              xticks=(x_ticks, x_tick_labels), yticks=y_ticks_speedup,
              legend=:topleft, size=(800, 600), fontsize=10)
    plot!(p2, sizes, speedup_bel, label="Blelloch (CPU/GPU)", marker=:diamond, linewidth=2)
    xlabel!("Array Size")
    ylabel!("Speedup")
    title!("Speedup: CPU vs GPU ($dist)")
    savefig(p2, "plots/$(dist)_speedup_cpu_vs_gpu.png")
    println("Saved as '$(dist)_speedup_cpu_vs_gpu.png'")

    # GPU vs GPU
    local speedup_gpu_hs_vs_bel = times_hs_gpu ./ times_bel_gpu_opt
    local max_speedup_gpu = ceil(maximum(speedup_gpu_hs_vs_bel))
    local y_ticks_speedup_gpu = 0:0.5:max_speedup_gpu
    local p3 = plot(sizes, speedup_gpu_hs_vs_bel, label="Hillis & Steele / Blelloch (GPU)", 
              marker=:cross, linewidth=2, xscale=:log2, 
              xticks=(x_ticks, x_tick_labels), yticks=y_ticks_speedup_gpu,
              legend=:topleft, size=(800, 600), fontsize=10)
    xlabel!("Array Size")
    ylabel!("Speedup")
    title!("Speedup: GPU Hillis & Steele vs Blelloch ($dist)")
    savefig(p3, "plots/$(dist)_speedup_gpu_vs_gpu.png")
    println("Saved as '$(dist)_speedup_gpu_vs_gpu.png'")
end