// kernels/reduction.cu
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "globals.cuh"

// Sum kernels
template<typename T>
__device__ void reduce_sum_all_kernel(const T* input, T* output, int size) {
    __shared__ T shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    // Each thread processes multiple elements
    T thread_result = T(0.0);
    while (idx < size) {
        thread_result += input[idx];
        idx += grid_size;
    }

    shared_data[tid] = thread_result;
    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

template<typename T>
__device__ void reduce_sum_axes_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int outer_size,
    int axis_size,
    int inner_size
) {
    int output_idx = blockIdx.x;

    if (output_idx >= outer_size * inner_size) return;

    int outer_idx = output_idx / inner_size;
    int inner_idx = output_idx % inner_size;

    T sum = T(0.0);

    // Each thread processes multiple elements along axis to improve memory efficiency
    for (int axis_idx = threadIdx.x; axis_idx < axis_size; axis_idx += blockDim.x) {
        int input_idx = outer_idx * axis_size * inner_size + axis_idx * inner_size + inner_idx;
        sum += input[input_idx];
    }

    // Warp-level reduction
    sum = warp_reduce_sum(sum);

    // Block-level reduction using shared memory
    __shared__ T block_sums[32];
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    if (lane == 0) {
        block_sums[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction
    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x >> 5)) ? block_sums[lane] : T(0.0);
        sum = warp_reduce_sum(sum);

        if (threadIdx.x == 0) {
            output[output_idx] = sum;
        }
    }
}

// Max kernels
template<typename T>
__device__ void reduce_max_all_kernel(const T* input, T* output, int size) {
    __shared__ T shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    T thread_result = -INFINITY;
    while (idx < size) {
        thread_result = max(thread_result, input[idx]);
        idx += grid_size;
    }

    shared_data[tid] = thread_result;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

template<typename T>
__device__ void reduce_max_axes_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int outer_size,
    int axis_size,
    int inner_size
) {
    int output_idx = blockIdx.x;

    if (output_idx >= outer_size * inner_size) return;

    int outer_idx = output_idx / inner_size;
    int inner_idx = output_idx % inner_size;

    T max_val = -FLT_MAX;

    for (int axis_idx = threadIdx.x; axis_idx < axis_size; axis_idx += blockDim.x) {
        int input_idx = outer_idx * axis_size * inner_size + axis_idx * inner_size + inner_idx;
        max_val = max(max_val, input[input_idx]);
    }

    max_val = warp_reduce_max(max_val);

    __shared__ T block_maxs[32];
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    if (lane == 0) {
        block_maxs[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (threadIdx.x < (blockDim.x >> 5)) ? block_maxs[lane] : -FLT_MAX;
        max_val = warp_reduce_max(max_val);

        if (threadIdx.x == 0) {
            output[output_idx] = max_val;
        }
    }
}

// Min kernels
template<typename T>
__device__ void reduce_min_all_kernel(const T* input, T* output, int size) {
    __shared__ T shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    T thread_result = INFINITY;
    while (idx < size) {
        thread_result = min(thread_result, input[idx]);
        idx += grid_size;
    }

    shared_data[tid] = thread_result;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = min(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

template<typename T>
__device__ void reduce_min_axes_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int outer_size,
    int axis_size,
    int inner_size
) {
    int output_idx = blockIdx.x;

    if (output_idx >= outer_size * inner_size) return;

    int outer_idx = output_idx / inner_size;
    int inner_idx = output_idx % inner_size;

    T min_val = FLT_MAX;

    for (int axis_idx = threadIdx.x; axis_idx < axis_size; axis_idx += blockDim.x) {
        int input_idx = outer_idx * axis_size * inner_size + axis_idx * inner_size + inner_idx;
        min_val = min(min_val, input[input_idx]);
    }

    min_val = warp_reduce_min(min_val);

    __shared__ T block_mins[32];
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    if (lane == 0) {
        block_mins[warp_id] = min_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        min_val = (threadIdx.x < (blockDim.x >> 5)) ? block_mins[lane] : FLT_MAX;
        min_val = warp_reduce_min(min_val);

        if (threadIdx.x == 0) {
            output[output_idx] = min_val;
        }
    }
}

// Product kernels
template<typename T>
__device__ void reduce_prod_all_kernel(const T* input, T* output, int size) {
    __shared__ T shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    T thread_result = T(1.0);
    while (idx < size) {
        thread_result *= input[idx];
        idx += grid_size;
    }

    shared_data[tid] = thread_result;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] *= shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

template<typename T>
__device__ void reduce_prod_axes_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int outer_size,
    int axis_size,
    int inner_size
) {
    int output_idx = blockIdx.x;

    if (output_idx >= outer_size * inner_size) return;

    int outer_idx = output_idx / inner_size;
    int inner_idx = output_idx % inner_size;

    T prod = T(1.0);

    for (int axis_idx = threadIdx.x; axis_idx < axis_size; axis_idx += blockDim.x) {
        int input_idx = outer_idx * axis_size * inner_size + axis_idx * inner_size + inner_idx;
        prod *= input[input_idx];
    }

    prod = warp_reduce_prod(prod);

    __shared__ T block_prods[32];
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    if (lane == 0) {
        block_prods[warp_id] = prod;
    }
    __syncthreads();

    if (warp_id == 0) {
        prod = (threadIdx.x < (blockDim.x >> 5)) ? block_prods[lane] : T(1.0);
        prod = warp_reduce_prod(prod);

        if (threadIdx.x == 0) {
            output[output_idx] = prod;
        }
    }
}

// Float32 wrapper functions
extern "C" __global__ void reduce_sum_all(const float* input, float* output, int size) {
    reduce_sum_all_kernel<float>(input, output, size);
}

extern "C" __global__ void reduce_sum_axes(
    const float* __restrict__ input,
    float* __restrict__ output,
    int outer_size,
    int axis_size,
    int inner_size
) {
    reduce_sum_axes_kernel<float>(input, output, outer_size, axis_size, inner_size);
}

extern "C" __global__ void reduce_max_all(const float* input, float* output, int size) {
    reduce_max_all_kernel<float>(input, output, size);
}

extern "C" __global__ void reduce_max_axes(
    const float* __restrict__ input,
    float* __restrict__ output,
    int outer_size,
    int axis_size,
    int inner_size
) {
    reduce_max_axes_kernel<float>(input, output, outer_size, axis_size, inner_size);
}

extern "C" __global__ void reduce_min_all(const float* input, float* output, int size) {
    reduce_min_all_kernel<float>(input, output, size);
}

extern "C" __global__ void reduce_min_axes(
    const float* __restrict__ input,
    float* __restrict__ output,
    int outer_size,
    int axis_size,
    int inner_size
) {
    reduce_min_axes_kernel<float>(input, output, outer_size, axis_size, inner_size);
}

extern "C" __global__ void reduce_prod_all(const float* input, float* output, int size) {
    reduce_prod_all_kernel<float>(input, output, size);
}

extern "C" __global__ void reduce_prod_axes(
    const float* __restrict__ input,
    float* __restrict__ output,
    int outer_size,
    int axis_size,
    int inner_size
) {
    reduce_prod_axes_kernel<float>(input, output, outer_size, axis_size, inner_size);
}

// Float64 wrapper functions
extern "C" __global__ void reduce_sum_all_f64(const double* input, double* output, int size) {
    reduce_sum_all_kernel<double>(input, output, size);
}

extern "C" __global__ void reduce_sum_axes_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    int outer_size,
    int axis_size,
    int inner_size
) {
    reduce_sum_axes_kernel<double>(input, output, outer_size, axis_size, inner_size);
}

extern "C" __global__ void reduce_max_all_f64(const double* input, double* output, int size) {
    reduce_max_all_kernel<double>(input, output, size);
}

extern "C" __global__ void reduce_max_axes_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    int outer_size,
    int axis_size,
    int inner_size
) {
    reduce_max_axes_kernel<double>(input, output, outer_size, axis_size, inner_size);
}

extern "C" __global__ void reduce_min_all_f64(const double* input, double* output, int size) {
    reduce_min_all_kernel<double>(input, output, size);
}

extern "C" __global__ void reduce_min_axes_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    int outer_size,
    int axis_size,
    int inner_size
) {
    reduce_min_axes_kernel<double>(input, output, outer_size, axis_size, inner_size);
}

extern "C" __global__ void reduce_prod_all_f64(const double* input, double* output, int size) {
    reduce_prod_all_kernel<double>(input, output, size);
}

extern "C" __global__ void reduce_prod_axes_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    int outer_size,
    int axis_size,
    int inner_size
) {
    reduce_prod_axes_kernel<double>(input, output, outer_size, axis_size, inner_size);
}
