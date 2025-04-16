using CUDA

# Hillis and Steele kernel
function hillis_steele_step_kernel(y, offset, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n && i > offset
        y[i] = y[i] + y[i - offset]
    end
    return
end

# Blelloch kernels
function blelloch_block_scan_kernel(y, n, shmem_size)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    tid = threadIdx().x
    shared = @cuDynamicSharedMem(Float32, shmem_size)
    
    # Loading data in shared memory
    if idx <= n
        shared[tid] = y[idx]
    else
        shared[tid] = 0.0f0
    end
    CUDA.sync_threads()
    
    # upsweep within block
    for d in 0:(Int(log2(shmem_size))-1)
        stride = 2^d
        if tid <= shmem_size && tid % (2 * stride) == 0 && (tid - stride) >= 1
            shared[tid] = shared[tid] + shared[tid - stride]
        end
        CUDA.sync_threads()
    end
    
    # downsweep
    if tid == shmem_size
        shared[tid] = 0.0f0
    end
    CUDA.sync_threads()
    for d in (Int(log2(shmem_size))-1):-1:0
        stride = 2^d
        if tid <= shmem_size && tid % (2 * stride) == 0 && (tid - stride) >= 1
            left = tid - stride
            temp = shared[left]
            shared[left] = shared[tid]
            shared[tid] = temp + shared[tid]
        end
        CUDA.sync_threads()
    end
    if idx <= n
        y[idx] = shared[tid]
    end
    return
end

function collect_block_sums_kernel(y, block_sums, n, block_size)
    b = blockIdx().x
    idx = b * block_size
    if idx <= n
        block_sums[b] = y[idx]
    end
    return
end

function scan_block_sums_kernel(input, output, num_elements)
    tid = threadIdx().x
    shmem_size = 1 << ceil(Int, log2(num_elements))
    shared = @cuDynamicSharedMem(Float32, shmem_size)
    if tid <= num_elements
        shared[tid] = input[tid]
    else
        shared[tid] = 0.0f0
    end
    CUDA.sync_threads()
    
    # Upsweep
    for d in 0:(ceil(Int, log2(num_elements))-1)
        stride = 2^d
        if tid <= shmem_size && tid % (2 * stride) == 0 && (tid - stride) >= 1
            shared[tid] = shared[tid] + shared[tid - stride]
        end
        CUDA.sync_threads()
    end
    
    # Downsweep
    if tid == num_elements
        shared[tid] = 0.0f0
    end
    CUDA.sync_threads()
    for d in (ceil(Int, log2(num_elements))-1):-1:0
        stride = 2^d
        if tid <= shmem_size && tid % (2 * stride) == 0 && (tid - stride) >= 1
            left = tid - stride
            temp = shared[left]
            shared[left] = shared[tid]
            shared[tid] = temp + shared[tid]
        end
        CUDA.sync_threads()
    end
    if tid <= num_elements
        output[tid] = shared[tid]
    end
    return
end

function collect_group_sums_kernel(block_sums_scan, group_sums, num_blocks, group_size)
    g = blockIdx().x
    idx = g * group_size
    if idx <= num_blocks
        group_sums[g] = block_sums_scan[idx]
    end
    return
end

function combine_group_sums_kernel(block_sums_scan, group_sums_scan, num_blocks, group_size)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    group_idx = blockIdx().x
    if group_idx > 1 && idx <= num_blocks
        block_sums_scan[idx] = block_sums_scan[idx] + group_sums_scan[group_idx - 1]
    end
    return
end

function blelloch_combine_blocks_kernel(y, block_sums_scan, n, block_size)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    block_idx = blockIdx().x
    if block_idx > 1 && idx <= n
        y[idx] = y[idx] + block_sums_scan[block_idx - 1]
    end
    return
end