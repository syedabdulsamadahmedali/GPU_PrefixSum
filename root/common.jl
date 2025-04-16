module Common

using CUDA 

export measure_time_cpu, measure_time_gpu

function measure_time_cpu(func, y, x, runs=10)
    total_time = 0.0
    for _ in 1:runs
        copyto!(y, x)
        t = @elapsed func(y, x)
        total_time += t
    end
    return total_time / runs
end

function measure_time_gpu(func, y, x, runs=10)
    CUDA.synchronize()
    total_time = 0.0
    for _ in 1:runs
        copyto!(y, x)
        t = CUDA.@elapsed func(y, x)
        total_time += t
    end
    CUDA.synchronize()
    return total_time / runs
end
end