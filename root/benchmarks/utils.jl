using Random
using Distributions
using MLDatasets
using CUDA

include("../common.jl")
using .Common
import .Common: measure_time_cpu, measure_time_gpu

# Data generation functions
function generate_sparse_data(n)
    x = zeros(Float32, n)
    num_non_zeros = floor(Int, 0.1 * n)
    if num_non_zeros == 0
        num_non_zeros = 1
    end
    non_zero_indices = randperm(n)[1:num_non_zeros]
    x[non_zero_indices] .= rand(Float32, num_non_zeros)
    return x
end

function generate_skewed_data(n)
    return Float32.(rand(Exponential(1.0), n))
end

function generate_sorted_data(n)
    return Float32.(1:n)
end

function generate_alternating_data(n)
    return Float32.([i % 2 for i in 1:n])
end

function generate_real_world_data(n)
    x = zeros(Float32, n)
    segment_size = n ÷ 10
    for i in 1:10
        start_idx = (i-1) * segment_size + 1
        end_idx = i == 10 ? n : i * segment_size
        segment_length = end_idx - start_idx + 1
        pattern_type = rand(1:4)
        if pattern_type == 1
            num_non_zeros = floor(Int, 0.5 * segment_length)
            if num_non_zeros == 0
                num_non_zeros = 1
            end
            indices = start_idx .+ randperm(segment_length)[1:num_non_zeros] .- 1
            x[indices] .= rand(Float32, num_non_zeros)
        elseif pattern_type == 2
            x[start_idx:end_idx] .= Float32.(rand(Exponential(1.0), segment_length))
        elseif pattern_type == 3
            x[start_idx:end_idx] .= Float32.(1:segment_length)
        else
            x[start_idx:end_idx] .= Float32.([j % 2 for j in 1:segment_length])
        end
    end
    return x
end

function generate_mnist_data(n)
    mnist = MNIST(:train)
    images = mnist.features
    num_images = size(images, 3)
    image_size = 28 * 28
    flattened_images = reshape(images, (image_size, num_images))
    flattened_images = Float32.(flattened_images)
    num_images_needed = ceil(Int, n / image_size)
    x = zeros(Float32, n)
    for i in 1:num_images_needed
        img_idx = (i - 1) % num_images + 1
        start_idx = (i - 1) * image_size + 1
        end_idx = min(i * image_size, n)
        segment_length = end_idx - start_idx + 1
        x[start_idx:end_idx] = flattened_images[1:segment_length, img_idx]
    end
    return x
end

# Benchmark
function run_benchmarks()
    sizes = [2^i for i in 10:2:26]
    times_seq_hs_cpu_rand = Float64[]
    times_seq_bel_cpu_rand = Float64[]
    times_hs_gpu_rand = Float64[]
    times_bel_gpu_opt_rand = Float64[]
    times_seq_hs_cpu_sparse = Float64[]
    times_seq_bel_cpu_sparse = Float64[]
    times_hs_gpu_sparse = Float64[]
    times_bel_gpu_opt_sparse = Float64[]
    times_seq_hs_cpu_skewed = Float64[]
    times_seq_bel_cpu_skewed = Float64[]
    times_hs_gpu_skewed = Float64[]
    times_bel_gpu_opt_skewed = Float64[]
    times_seq_hs_cpu_sorted = Float64[]
    times_seq_bel_cpu_sorted = Float64[]
    times_hs_gpu_sorted = Float64[]
    times_bel_gpu_opt_sorted = Float64[]
    times_seq_hs_cpu_alt = Float64[]
    times_seq_bel_cpu_alt = Float64[]
    times_hs_gpu_alt = Float64[]
    times_bel_gpu_opt_alt = Float64[]
    times_seq_hs_cpu_real = Float64[]
    times_seq_bel_cpu_real = Float64[]
    times_hs_gpu_real = Float64[]
    times_bel_gpu_opt_real = Float64[]
    times_seq_hs_cpu_mnist = Float64[]
    times_seq_bel_cpu_mnist = Float64[]
    times_hs_gpu_mnist = Float64[]
    times_bel_gpu_opt_mnist = Float64[]
    for n in sizes
        println("Testing array size: $n")

        # Random data
        println("Random data:")
        x_cpu_rand = rand(Float32, n)
        y_cpu_rand = similar(x_cpu_rand)
        x_gpu_rand = CuArray(x_cpu_rand)
        y_gpu_rand = similar(x_gpu_rand)
        t_seq_hs_rand = measure_time_cpu(sequential_hillis_steele, y_cpu_rand, x_cpu_rand)
        t_seq_bel_rand = measure_time_cpu(sequential_blelloch, y_cpu_rand, x_cpu_rand)
        t_hs_gpu_rand = measure_time_gpu(hillis_steele, y_gpu_rand, x_gpu_rand)
        t_bel_gpu_opt_rand = measure_time_gpu(blelloch, y_gpu_rand, x_gpu_rand)
        push!(times_seq_hs_cpu_rand, t_seq_hs_rand)
        push!(times_seq_bel_cpu_rand, t_seq_bel_rand)
        push!(times_hs_gpu_rand, t_hs_gpu_rand)
        push!(times_bel_gpu_opt_rand, t_bel_gpu_opt_rand)
        println("Speedup Hillis & Steele, CPU/GPU, random: ", t_seq_hs_rand / t_hs_gpu_rand)
        println("Speedup Blelloch, CPU/GPU, random: ", t_seq_bel_rand / t_bel_gpu_opt_rand)

        # Sparse data
        println("Sparse data")
        x_cpu_sparse = generate_sparse_data(n)
        y_cpu_sparse = similar(x_cpu_sparse)
        x_gpu_sparse = CuArray(x_cpu_sparse)
        y_gpu_sparse = similar(x_gpu_sparse)
        t_seq_hs_sparse = measure_time_cpu(sequential_hillis_steele, y_cpu_sparse, x_cpu_sparse)
        t_seq_bel_sparse = measure_time_cpu(sequential_blelloch, y_cpu_sparse, x_cpu_sparse)
        t_hs_gpu_sparse = measure_time_gpu(hillis_steele, y_gpu_sparse, x_gpu_sparse)
        t_bel_gpu_opt_sparse = measure_time_gpu(blelloch, y_gpu_sparse, x_gpu_sparse)
        push!(times_seq_hs_cpu_sparse, t_seq_hs_sparse)
        push!(times_seq_bel_cpu_sparse, t_seq_bel_sparse)
        push!(times_hs_gpu_sparse, t_hs_gpu_sparse)
        push!(times_bel_gpu_opt_sparse, t_bel_gpu_opt_sparse)
        println("Speedup Hillis & Steele, CPU/GPU, sparse: ", t_seq_hs_sparse / t_hs_gpu_sparse)
        println("Speedup Blelloch, CPU/GPU, sparse: ", t_seq_bel_sparse / t_bel_gpu_opt_sparse)

        # Skewed data
        println("Skewed data")
        x_cpu_skewed = generate_skewed_data(n)
        y_cpu_skewed = similar(x_cpu_skewed)
        x_gpu_skewed = CuArray(x_cpu_skewed)
        y_gpu_skewed = similar(x_gpu_skewed)
        t_seq_hs_skewed = measure_time_cpu(sequential_hillis_steele, y_cpu_skewed, x_cpu_skewed)
        t_seq_bel_skewed = measure_time_cpu(sequential_blelloch, y_cpu_skewed, x_cpu_skewed)
        t_hs_gpu_skewed = measure_time_gpu(hillis_steele, y_gpu_skewed, x_gpu_skewed)
        t_bel_gpu_opt_skewed = measure_time_gpu(blelloch, y_gpu_skewed, x_gpu_skewed)
        push!(times_seq_hs_cpu_skewed, t_seq_hs_skewed)
        push!(times_seq_bel_cpu_skewed, t_seq_bel_skewed)
        push!(times_hs_gpu_skewed, t_hs_gpu_skewed)
        push!(times_bel_gpu_opt_skewed, t_bel_gpu_opt_skewed)
        println("Speedup Hillis & Steele, CPU/GPU, skewed: ", t_seq_hs_skewed / t_hs_gpu_skewed)
        println("Speedup Blelloch, CPU/GPU, skewed: ", t_seq_bel_skewed / t_bel_gpu_opt_skewed)

        # Sorted data
        println("Sorted data")
        x_cpu_sorted = generate_sorted_data(n)
        y_cpu_sorted = similar(x_cpu_sorted)
        x_gpu_sorted = CuArray(x_cpu_sorted)
        y_gpu_sorted = similar(x_gpu_sorted)
        t_seq_hs_sorted = measure_time_cpu(sequential_hillis_steele, y_cpu_sorted, x_cpu_sorted)
        t_seq_bel_sorted = measure_time_cpu(sequential_blelloch, y_cpu_sorted, x_cpu_sorted)
        t_hs_gpu_sorted = measure_time_gpu(hillis_steele, y_gpu_sorted, x_gpu_sorted)
        t_bel_gpu_opt_sorted = measure_time_gpu(blelloch, y_gpu_sorted, x_gpu_sorted)
        push!(times_seq_hs_cpu_sorted, t_seq_hs_sorted)
        push!(times_seq_bel_cpu_sorted, t_seq_bel_sorted)
        push!(times_hs_gpu_sorted, t_hs_gpu_sorted)
        push!(times_bel_gpu_opt_sorted, t_bel_gpu_opt_sorted)
        println("Speedup Hillis & Steele, CPU/GPU, sorted: ", t_seq_hs_sorted / t_hs_gpu_sorted)
        println("Speedup Blelloch, CPU/GPU, sorted: ", t_seq_bel_sorted / t_bel_gpu_opt_sorted)

        # Alternating data
        println("Alternating data")
        x_cpu_alt = generate_alternating_data(n)
        y_cpu_alt = similar(x_cpu_alt)
        x_gpu_alt = CuArray(x_cpu_alt)
        y_gpu_alt = similar(x_gpu_alt)
        t_seq_hs_alt = measure_time_cpu(sequential_hillis_steele, y_cpu_alt, x_cpu_alt)
        t_seq_bel_alt = measure_time_cpu(sequential_blelloch, y_cpu_alt, x_cpu_alt)
        t_hs_gpu_alt = measure_time_gpu(hillis_steele, y_gpu_alt, x_gpu_alt)
        t_bel_gpu_opt_alt = measure_time_gpu(blelloch, y_gpu_alt, x_gpu_alt)
        push!(times_seq_hs_cpu_alt, t_seq_hs_alt)
        push!(times_seq_bel_cpu_alt, t_seq_bel_alt)
        push!(times_hs_gpu_alt, t_hs_gpu_alt)
        push!(times_bel_gpu_opt_alt, t_bel_gpu_opt_alt)
        println("Speedup Hillis & Steele, CPU/GPU, alternating: ", t_seq_hs_alt / t_hs_gpu_alt)
        println("Speedup Blelloch, CPU/GPU, alternating: ", t_seq_bel_alt / t_bel_gpu_opt_alt)

        # Real-world data
        println("Real-world data")
        x_cpu_real = generate_real_world_data(n)
        y_cpu_real = similar(x_cpu_real)
        x_gpu_real = CuArray(x_cpu_real)
        y_gpu_real = similar(x_gpu_real)
        t_seq_hs_real = measure_time_cpu(sequential_hillis_steele, y_cpu_real, x_cpu_real)
        t_seq_bel_real = measure_time_cpu(sequential_blelloch, y_cpu_real, x_cpu_real)
        t_hs_gpu_real = measure_time_gpu(hillis_steele, y_gpu_real, x_gpu_real)
        t_bel_gpu_opt_real = measure_time_gpu(blelloch, y_gpu_real, x_gpu_real)
        push!(times_seq_hs_cpu_real, t_seq_hs_real)
        push!(times_seq_bel_cpu_real, t_seq_bel_real)
        push!(times_hs_gpu_real, t_hs_gpu_real)
        push!(times_bel_gpu_opt_real, t_bel_gpu_opt_real)
        println("Speedup Hillis & Steele, CPU/GPU, real-world: ", t_seq_hs_real / t_hs_gpu_real)
        println("Speedup Blelloch, CPU/GPU, real-world: ", t_seq_bel_real / t_bel_gpu_opt_real)

        # MNIST data
        println("MNIST data")
        x_cpu_mnist = generate_mnist_data(n)
        y_cpu_mnist = similar(x_cpu_mnist)
        x_gpu_mnist = CuArray(x_cpu_mnist)
        y_gpu_mnist = similar(x_gpu_mnist)
        t_seq_hs_mnist = measure_time_cpu(sequential_hillis_steele, y_cpu_mnist, x_cpu_mnist)
        t_seq_bel_mnist = measure_time_cpu(sequential_blelloch, y_cpu_mnist, x_cpu_mnist)
        t_hs_gpu_mnist = measure_time_gpu(hillis_steele, y_gpu_mnist, x_gpu_mnist)
        t_bel_gpu_opt_mnist = measure_time_gpu(blelloch, y_gpu_mnist, x_gpu_mnist)
        push!(times_seq_hs_cpu_mnist, t_seq_hs_mnist)
        push!(times_seq_bel_cpu_mnist, t_seq_bel_mnist)
        push!(times_hs_gpu_mnist, t_hs_gpu_mnist)
        push!(times_bel_gpu_opt_mnist, t_bel_gpu_opt_mnist)
        println("Speedup Hillis & Steele, CPU/GPU, MNIST: ", t_seq_hs_mnist / t_hs_gpu_mnist)
        println("Speedup Blelloch, CPU/GPU, MNIST): ", t_seq_bel_mnist / t_bel_gpu_opt_mnist)
    end

    return (sizes, 
            times_seq_hs_cpu_rand, times_seq_bel_cpu_rand, times_hs_gpu_rand, times_bel_gpu_opt_rand,
            times_seq_hs_cpu_sparse, times_seq_bel_cpu_sparse, times_hs_gpu_sparse, times_bel_gpu_opt_sparse,
            times_seq_hs_cpu_skewed, times_seq_bel_cpu_skewed, times_hs_gpu_skewed, times_bel_gpu_opt_skewed,
            times_seq_hs_cpu_sorted, times_seq_bel_cpu_sorted, times_hs_gpu_sorted, times_bel_gpu_opt_sorted,
            times_seq_hs_cpu_alt, times_seq_bel_cpu_alt, times_hs_gpu_alt, times_bel_gpu_opt_alt,
            times_seq_hs_cpu_real, times_seq_bel_cpu_real, times_hs_gpu_real, times_bel_gpu_opt_real,
            times_seq_hs_cpu_mnist, times_seq_bel_cpu_mnist, times_hs_gpu_mnist, times_bel_gpu_opt_mnist)
end