#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "globals.cuh"

// ========== DIRECT KERNEL IMPLEMENTATIONS ==========

// Sum kernels
extern "C" __global__ void reduce_sum_all(const float* input, float* output, int size) {
    __shared__ float shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    // Each thread processes multiple elements
    float thread_result = 0.0f;
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

extern "C" __global__ void reduce_sum_all_f64(const double* input, double* output, int size) {
    __shared__ double shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    double thread_result = 0.0;
    while (idx < size) {
        thread_result += input[idx];
        idx += grid_size;
    }

    shared_data[tid] = thread_result;
    __syncthreads();

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

extern "C" __global__ void reduce_sum_axes(
    const float* input, float* output, int outer_size, int axis_size, int inner_size) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    float result = 0.0f;
    for (int axis_idx = 0; axis_idx < axis_size; axis_idx++) {
        int input_idx = outer_idx * axis_size * inner_size +
            axis_idx * inner_size + inner_idx;
        result += input[input_idx];
    }

    int output_idx = outer_idx * inner_size + inner_idx;
    output[output_idx] = result;
}

extern "C" __global__ void reduce_sum_axes_f64(
    const double* input, double* output, int outer_size, int axis_size, int inner_size) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    double result = 0.0;
    for (int axis_idx = 0; axis_idx < axis_size; axis_idx++) {
        int input_idx = outer_idx * axis_size * inner_size +
            axis_idx * inner_size + inner_idx;
        result += input[input_idx];
    }

    int output_idx = outer_idx * inner_size + inner_idx;
    output[output_idx] = result;
}

// Max kernels
extern "C" __global__ void reduce_max_all(const float* input, float* output, int size) {
    __shared__ float shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    float thread_result = -INFINITY;
    while (idx < size) {
        thread_result = fmaxf(thread_result, input[idx]);
        idx += grid_size;
    }

    shared_data[tid] = thread_result;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

extern "C" __global__ void reduce_max_all_f64(const double* input, double* output, int size) {
    __shared__ double shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    double thread_result = -INFINITY;
    while (idx < size) {
        thread_result = fmax(thread_result, input[idx]);
        idx += grid_size;
    }

    shared_data[tid] = thread_result;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fmax(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

extern "C" __global__ void reduce_max_axes(
    const float* input, float* output, int outer_size, int axis_size, int inner_size) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    float result = -INFINITY;
    for (int axis_idx = 0; axis_idx < axis_size; axis_idx++) {
        int input_idx = outer_idx * axis_size * inner_size +
            axis_idx * inner_size + inner_idx;
        result = fmaxf(result, input[input_idx]);
    }

    int output_idx = outer_idx * inner_size + inner_idx;
    output[output_idx] = result;
}

extern "C" __global__ void reduce_max_axes_f64(
    const double* input, double* output, int outer_size, int axis_size, int inner_size) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    double result = -INFINITY;
    for (int axis_idx = 0; axis_idx < axis_size; axis_idx++) {
        int input_idx = outer_idx * axis_size * inner_size +
            axis_idx * inner_size + inner_idx;
        result = fmax(result, input[input_idx]);
    }

    int output_idx = outer_idx * inner_size + inner_idx;
    output[output_idx] = result;
}

// Min kernels
extern "C" __global__ void reduce_min_all(const float* input, float* output, int size) {
    __shared__ float shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    float thread_result = INFINITY;
    while (idx < size) {
        thread_result = fminf(thread_result, input[idx]);
        idx += grid_size;
    }

    shared_data[tid] = thread_result;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fminf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

extern "C" __global__ void reduce_min_all_f64(const double* input, double* output, int size) {
    __shared__ double shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    double thread_result = INFINITY;
    while (idx < size) {
        thread_result = fmin(thread_result, input[idx]);
        idx += grid_size;
    }

    shared_data[tid] = thread_result;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fmin(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

extern "C" __global__ void reduce_min_axes(
    const float* input, float* output, int outer_size, int axis_size, int inner_size) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    float result = INFINITY;
    for (int axis_idx = 0; axis_idx < axis_size; axis_idx++) {
        int input_idx = outer_idx * axis_size * inner_size +
            axis_idx * inner_size + inner_idx;
        result = fminf(result, input[input_idx]);
    }

    int output_idx = outer_idx * inner_size + inner_idx;
    output[output_idx] = result;
}

extern "C" __global__ void reduce_min_axes_f64(
    const double* input, double* output, int outer_size, int axis_size, int inner_size) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    double result = INFINITY;
    for (int axis_idx = 0; axis_idx < axis_size; axis_idx++) {
        int input_idx = outer_idx * axis_size * inner_size +
            axis_idx * inner_size + inner_idx;
        result = fmin(result, input[input_idx]);
    }

    int output_idx = outer_idx * inner_size + inner_idx;
    output[output_idx] = result;
}

// Product kernels
extern "C" __global__ void reduce_prod_all(const float* input, float* output, int size) {
    __shared__ float shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    float thread_result = 1.0f;
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

extern "C" __global__ void reduce_prod_all_f64(const double* input, double* output, int size) {
    __shared__ double shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    double thread_result = 1.0;
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

extern "C" __global__ void reduce_prod_axes(
    const float* input, float* output, int outer_size, int axis_size, int inner_size) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    float result = 1.0f;
    for (int axis_idx = 0; axis_idx < axis_size; axis_idx++) {
        int input_idx = outer_idx * axis_size * inner_size +
            axis_idx * inner_size + inner_idx;
        result *= input[input_idx];
    }

    int output_idx = outer_idx * inner_size + inner_idx;
    output[output_idx] = result;
}

extern "C" __global__ void reduce_prod_axes_f64(
    const double* input, double* output, int outer_size, int axis_size, int inner_size) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    double result = 1.0;
    for (int axis_idx = 0; axis_idx < axis_size; axis_idx++) {
        int input_idx = outer_idx * axis_size * inner_size +
            axis_idx * inner_size + inner_idx;
        result *= input[input_idx];
    }

    int output_idx = outer_idx * inner_size + inner_idx;
    output[output_idx] = result;
}
