using CUDA

include("kernels.jl")

# Sequential version
function sequential_hillis_steele(y, x)
    n = length(x)
    if n == 0
        return
    end
    copyto!(y, x)
    temp = similar(y)
    for d in 1:ceil(Int, log2(n))
        copyto!(temp, y)
        offset = 2^(d-1)
        for i in (offset+1):n
            y[i] = temp[i] + temp[i - offset]
        end
    end
end

# Parallel version
function hillis_steele(y, x)
    n = length(x)
    if n == 0
        return
    end
    CUDA.@sync begin
        copyto!(y, x)
        for d in 1:ceil(Int, log2(n))
            offset = 2^(d-1)
            threads_per_block = 256
            blocks = ceil(Int, n / threads_per_block)
            @cuda blocks=blocks threads=threads_per_block hillis_steele_step_kernel(y, offset, n)
            CUDA.synchronize()
        end
    end
end