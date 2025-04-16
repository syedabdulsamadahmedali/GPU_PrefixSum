using CUDA

include("kernels.jl")

# Sequential version
function sequential_blelloch(y, x)
    n = length(x)
    if n == 0
        return
    end
    if n == 1
        y[1] = 0
        return
    end

    k = ceil(Int, log2(n))
    padded_length = 2^k
    padded = zeros(eltype(x), padded_length)
    padded[1:n] .= x

    # Upsweep
    for d in 0:(k-1)
        stride = 2^d
        for i in (2*stride):2*stride:padded_length
            padded[i] += padded[i - stride]
        end
    end

    padded[padded_length] = 0

    # Downsweep
    for d in (k-1):-1:0
        stride = 2^d
        for i in (2*stride):2*stride:padded_length
            left = i - stride
            temp = padded[left]
            padded[left] = padded[i] 
            padded[i] = temp + padded[i]
        end
    end

    y[1:n] .= padded[1:n]
end


# Parallel version
function blelloch(y, x)
    n = length(x)
    if n == 0
        return
    end
    threads_per_block = 256
    blocks = ceil(Int, n / threads_per_block)
    shmem_size = threads_per_block
    CUDA.@sync begin
        copyto!(y, x)
        
        # Block level scans
        @cuda blocks=blocks threads=threads_per_block shmem=(shmem_size * sizeof(Float32)) blelloch_block_scan_kernel(y, n, shmem_size)
        
        # Collecting block sums
        block_sums = CUDA.zeros(Float32, blocks)
        @cuda blocks=blocks threads=1 collect_block_sums_kernel(y, block_sums, n, threads_per_block)
        
        # Scanning block sums
        block_sums_scan = CUDA.zeros(Float32, blocks)
        max_threads = 1024
        if blocks <= max_threads
            @cuda blocks=1 threads=blocks shmem=(max_threads * sizeof(Float32)) scan_block_sums_kernel(block_sums, block_sums_scan, blocks)
        else
            group_size = max_threads
            num_groups = ceil(Int, blocks / group_size)
            group_sums = CUDA.zeros(Float32, num_groups)
            
            # Scanning block sums in groups
            @cuda blocks=num_groups threads=group_size shmem=(group_size * sizeof(Float32)) scan_block_sums_kernel(block_sums, block_sums_scan, group_size)
            
            # Collecting group sums
            @cuda blocks=num_groups threads=1 collect_group_sums_kernel(block_sums_scan, group_sums, blocks, group_size)
            
            # Scanning group sums
            group_sums_scan = CUDA.zeros(Float32, num_groups)
            @cuda blocks=1 threads=num_groups shmem=(max_threads * sizeof(Float32)) scan_block_sums_kernel(group_sums, group_sums_scan, num_groups)
            
            # Adding group sums
            @cuda blocks=num_groups threads=group_size combine_group_sums_kernel(block_sums_scan, group_sums_scan, blocks, group_size)
        end
        
        # Adding block sums
        @cuda blocks=blocks threads=threads_per_block blelloch_combine_blocks_kernel(y, block_sums_scan, n, threads_per_block)
    end
end