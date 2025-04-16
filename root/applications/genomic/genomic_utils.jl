using BioSequences
using CUDA
using Random

include("../../common.jl")
using .Common
import .Common: measure_time_cpu, measure_time_gpu
import BioSequences: LongSequence, randdnaseq, DNAAlphabet, DNA_A, DNA_C, DNA_G, DNA_T

# Generating genomic data
function generate_reference_genome(size)
    base_ref = randdnaseq(16569)
    ref = LongSequence{DNAAlphabet{4}}("")
    while length(ref) < size
        ref = ref * base_ref
    end
    return ref[1:size]
end

function dna_to_arrays(seq)
    n = length(seq)
    counts_A = zeros(Float32, n)
    counts_C = zeros(Float32, n)
    counts_G = zeros(Float32, n)
    counts_T = zeros(Float32, n)
    for i in 1:n
        if seq[i] == DNA_A
            counts_A[i] = 1.0
        elseif seq[i] == DNA_C
            counts_C[i] = 1.0
        elseif seq[i] == DNA_G
            counts_G[i] = 1.0
        elseif seq[i] == DNA_T
            counts_T[i] = 1.0
        end
    end
    return counts_A, counts_C, counts_G, counts_T
end

function build_rank_table(seq, prefix_sum_func!, use_gpu=false)
    counts_A, counts_C, counts_G, counts_T = dna_to_arrays(seq)
    n = length(seq)
    if use_gpu
        counts_A_gpu = CuArray(counts_A)
        counts_C_gpu = CuArray(counts_C)
        counts_G_gpu = CuArray(counts_G)
        counts_T_gpu = CuArray(counts_T)
        result_A = similar(counts_A_gpu)
        result_C = similar(counts_C_gpu)
        result_G = similar(counts_G_gpu)
        result_T = similar(counts_T_gpu)
        prefix_sum_func!(result_A, counts_A_gpu)
        prefix_sum_func!(result_C, counts_C_gpu)
        prefix_sum_func!(result_G, counts_G_gpu)
        prefix_sum_func!(result_T, counts_T_gpu)
        return Array(result_A), Array(result_C), Array(result_G), Array(result_T)
    else
        result_A = similar(counts_A)
        result_C = similar(counts_C)
        result_G = similar(counts_G)
        result_T = similar(counts_T)
        prefix_sum_func!(result_A, counts_A)
        prefix_sum_func!(result_C, counts_C)
        prefix_sum_func!(result_G, counts_G)
        prefix_sum_func!(result_T, counts_T)
        return result_A, result_C, result_G, result_T
    end
end

function run_genomic_benchmarks()
    sizes = [2^i for i in 10:2:26]
    times_seq_hs_cpu = Float64[]
    times_seq_bel_cpu = Float64[]
    times_hs_gpu = Float64[]
    times_bel_gpu_opt = Float64[]
    for n in sizes
        println("Testing genome size: $n base pairs")
        ref = generate_reference_genome(n)
        counts_A, counts_C, counts_G, counts_T = dna_to_arrays(ref)
        y_cpu_A = similar(counts_A)
        y_cpu_C = similar(counts_C)
        y_cpu_G = similar(counts_G)
        y_cpu_T = similar(counts_T)
        counts_A_gpu = CuArray(counts_A)
        counts_C_gpu = CuArray(counts_C)
        counts_G_gpu = CuArray(counts_G)
        counts_T_gpu = CuArray(counts_T)
        y_gpu_A = similar(counts_A_gpu)
        y_gpu_C = similar(counts_C_gpu)
        y_gpu_G = similar(counts_G_gpu)
        y_gpu_T = similar(counts_T_gpu)
        t_seq_hs = Common.measure_time_cpu(sequential_hillis_steele, y_cpu_A, counts_A) +
                   Common.measure_time_cpu(sequential_hillis_steele, y_cpu_C, counts_C) +
                   Common.measure_time_cpu(sequential_hillis_steele, y_cpu_G, counts_G) +
                   Common.measure_time_cpu(sequential_hillis_steele, y_cpu_T, counts_T)
        t_seq_bel = Common.measure_time_cpu(sequential_blelloch, y_cpu_A, counts_A) +
                    Common.measure_time_cpu(sequential_blelloch, y_cpu_C, counts_C) +
                    Common.measure_time_cpu(sequential_blelloch, y_cpu_G, counts_G) +
                    Common.measure_time_cpu(sequential_blelloch, y_cpu_T, counts_T)
        t_hs_gpu = Common.measure_time_gpu(hillis_steele, y_gpu_A, counts_A_gpu) +
                   Common.measure_time_gpu(hillis_steele, y_gpu_C, counts_C_gpu) +
                   Common.measure_time_gpu(hillis_steele, y_gpu_G, counts_G_gpu) +
                   Common.measure_time_gpu(hillis_steele, y_gpu_T, counts_T_gpu)
        t_bel_gpu_opt = Common.measure_time_gpu(blelloch, y_gpu_A, counts_A_gpu) +
                        Common.measure_time_gpu(blelloch, y_gpu_C, counts_C_gpu) +
                        Common.measure_time_gpu(blelloch, y_gpu_G, counts_G_gpu) +
                        Common.measure_time_gpu(blelloch, y_gpu_T, counts_T_gpu)
        push!(times_seq_hs_cpu, t_seq_hs)
        push!(times_seq_bel_cpu, t_seq_bel)
        push!(times_hs_gpu, t_hs_gpu)
        push!(times_bel_gpu_opt, t_bel_gpu_opt)
        println("Speedup Hillis & Steele, CPU/GPU, genomic: ", t_seq_hs / t_hs_gpu)
        println("Speedup (Blelloch, CPU/GPU, genomic): ", t_seq_bel / t_bel_gpu_opt)
    end
    return sizes, times_seq_hs_cpu, times_seq_bel_cpu, times_hs_gpu, times_bel_gpu_opt
end