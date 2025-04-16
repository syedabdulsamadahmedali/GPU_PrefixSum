using CUDA
using Printf
using Plots

include("../../algorithms/hillis_steele.jl")
include("../../algorithms/blelloch.jl")
include("genomic_utils.jl")

using CUDA

function verify_implementations()
    println("Verifying implementations:\n")
    # Test cases
     test_cases = [
        ("Small Array", Float32[1, 2, 3, 4]),
        ("Empty Array", Float32[]),
        ("Single Element", Float32[5]),
        ("All Zeros", Float32[0, 0, 0, 0]),
        ("Negative Numbers", Float32[-1, -2, 3, -4]),
        ("Large Power of Two", Float32[i for i in 1:8]),
        ("Non-Power of Two", Float32[i for i in 1:7]),
        ("Alternating Positive and Negative", Float32[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
        ("Randomized Input", Float32[rand(-10:10) for _ in 1:10]),
        ("Non-Uniform Spacing", Float32[1.0, 10.0, 100.0, 1000.0])
    ]
    for (name, x) in test_cases
        println("Test Case: $name")
        println("Input: ", x)
        y_cpu = similar(x)
        y_gpu = CuArray(x)
        sequential_hillis_steele(y_cpu, x)
        println("Sequential Hillis & Steele: ", y_cpu)
        sequential_blelloch(y_cpu, x)
        println("Sequential Blelloch: ", y_cpu)
        hillis_steele(y_gpu, x)
        println("Hillis & Steele GPU: ", Array(y_gpu))
        blelloch(y_gpu, x)
        println("Blelloch GPU: ", Array(y_gpu))
        println()
    end
end

verify_implementations()
results = run_genomic_benchmarks()
sizes, times_seq_hs_cpu, times_seq_bel_cpu, times_hs_gpu, times_bel_gpu_opt = results

x_ticks = [2^i for i in 10:4:26]
x_tick_labels = ["2^{$i}" for i in 10:4:26]

y_ticks = [10.0^i for i in -5:0] 
y_tick_labels = ["10 μs", "100 μs", "1 ms", "10 ms", "100 ms", "1 s"]

# All Implementations
p1 = plot(sizes, times_seq_hs_cpu, label="Hillis & Steele (CPU)", 
          marker=:square, linewidth=2, xscale=:log2, yscale=:log10, 
          xticks=(x_ticks, x_tick_labels), yticks=(y_ticks, y_tick_labels),
          legend=:topleft, size=(800, 600), fontsize=10)
plot!(p1, sizes, times_seq_bel_cpu, label="Blelloch (CPU)", marker=:diamond, linewidth=2)
plot!(p1, sizes, times_hs_gpu, label="Hillis & Steele (GPU)", marker=:cross, linewidth=2)
plot!(p1, sizes, times_bel_gpu_opt, label="Blelloch (GPU)", marker=:circle, linewidth=2)
xlabel!("Array Size")
ylabel!("Time")
title!("Performance: Genomic Data")
savefig(p1, "plots/genomic_all_implementations.png")
println("Saved as 'genomic_all_implementations.png'")

# Speedup CPU vs GPU
speedup_hs = times_seq_hs_cpu ./ times_hs_gpu
speedup_bel = times_seq_bel_cpu ./ times_bel_gpu_opt
max_speedup = ceil(maximum([speedup_hs; speedup_bel]))
y_ticks_speedup = 0:1:max_speedup
p2 = plot(sizes, speedup_hs, label="Hillis & Steele (CPU/GPU)", 
          marker=:circle, linewidth=2, xscale=:log2, 
          xticks=(x_ticks, x_tick_labels), yticks=y_ticks_speedup,
          legend=:topleft, size=(800, 600), fontsize=10)
plot!(p2, sizes, speedup_bel, label="Blelloch (CPU/GPU)", marker=:diamond, linewidth=2)
xlabel!("Array Size")
ylabel!("Speedup")
title!("Speedup: CPU vs GPU (Genomic)")
savefig(p2, "plots/genomic_speedup_cpu_vs_gpu.png")
println("Saved as 'genomic_speedup_cpu_vs_gpu.png'")

# GPU vs GPU
speedup_gpu_hs_vs_bel = times_hs_gpu ./ times_bel_gpu_opt
max_speedup_gpu = ceil(maximum(speedup_gpu_hs_vs_bel))
y_ticks_speedup_gpu = 0:0.5:max_speedup_gpu
p3 = plot(sizes, speedup_gpu_hs_vs_bel, label="Hillis & Steele / Blelloch (GPU)", 
          marker=:cross, linewidth=2, xscale=:log2, 
          xticks=(x_ticks, x_tick_labels), yticks=y_ticks_speedup_gpu,
          legend=:topleft, size=(800, 600), fontsize=10)
xlabel!("Array Size")
ylabel!("Speedup")
title!("Speedup: GPU Hillis & Steele vs Blelloch (Genomic)")
savefig(p3, "plots/genomic_gpu_vs_gpu_speedup.png")
println("Saved as 'genomic_gpu_vs_gpu_speedup.png'")