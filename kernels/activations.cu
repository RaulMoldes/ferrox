// kernels/activations.cu
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "globals.cuh"



template<typename T>
__device__ void relu_kernel(
    const T* input,
    T* output,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        output[idx] = max(T(0.0), input[idx]);
    }
}

template<typename T>
__device__ void sigmoid_kernel(const T* input, T* output, int size) {
    int idx = get_global_idx();
    if (idx < size) {
        // Element-wise sigmoid function: output = 1 / (1 + exp(-input))
        T x = input[idx];
        output[idx] = T(1.0) / (T(1.0) + exp(-x));
    }
}

template<typename T>
__device__ void hyperbolic_tangent_kernel(
    const T* input,
    T* output,
    int size
) {
    // Calculate thread index using standard CUDA pattern
    int idx = get_global_idx();

    // Boundary check for thread safety
    if (idx < size) {
        T x = input[idx];

        // Use the built-in tanh function for numerical stability
        // It handles overflow/underflow cases internally
        output[idx] = tanh(x);
    }
}

template <typename T>
__device__ void softmax_kernel(const T* input, T* output, int N) {
    int tid = threadIdx.x;
    int global_idx = get_global_idx();

    // S1: Find global maxima using block reduction
    T local_max = -T(FLT_MAX);

    // Initialize local max values.
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        local_max = max(local_max, input[i]);
    }

    // Reduce the maximum
    T block_max = block_reduce_max(local_max);

    // Broadcast the maximum to all threads
    __shared__ T s_max;
    if (tid == 0) s_max = block_max;
    __syncthreads();

    // S2: Compute local sums
    T local_sum = T(0.0);

    // Each thread calculates its local sum value.
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        local_sum += exp(input[i] - s_max);
    }

    // Reduce sum between all blocks.
    T block_sum = block_reduce_sum(local_sum);

    // Broadcast the sum to all threads.
    __shared__ T s_sum;
    if (tid == 0) s_sum = block_sum;
    __syncthreads();

    // S3: Compute final softmax and return the result.
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        output[i] = exp(input[i] - s_max) / s_sum;
    }
}

// Batch-aware softmax kernels
// This kernel processes multiple sequences in parallel, computing softmax
// along the specified axis while maintaining batch efficiency
// Each block processes one sequence from the batch
template <typename T>
__device__ void softmax_batch_axis_kernel(
    const T* input,
    T* output,
    int batch_size,
    int seq_length,
    int inner_size,
    int total_elements
) {
    int block_id = blockIdx.x;
    int tid = threadIdx.x;

    int inner_idx = block_id % inner_size;
    int batch_idx = block_id / inner_size;

    if (batch_idx >= batch_size) return;

    int sequence_start = batch_idx * seq_length * inner_size + inner_idx;

    // Find maximum
    T local_max = -T(FLT_MAX);

    for (int i = tid; i < seq_length; i += blockDim.x) {
        int global_idx = sequence_start + i * inner_size;
        local_max = max(local_max, input[global_idx]);
    }

    T block_max = block_reduce_max(local_max);

    __shared__ T s_max;
    if (tid == 0) s_max = block_max;
    __syncthreads();

    // Compute sum
    T local_sum = T(0.0);

    for (int i = tid; i < seq_length; i += blockDim.x) {
        int global_idx = sequence_start + i * inner_size;
        local_sum += exp(input[global_idx] - s_max);
    }

    T block_sum = block_reduce_sum(local_sum);

    __shared__ T s_sum;
    if (tid == 0) s_sum = block_sum;
    __syncthreads();

    // Final softmax computation
    for (int i = tid; i < seq_length; i += blockDim.x) {
        int global_idx = sequence_start + i * inner_size;
        output[global_idx] = exp(input[global_idx] - s_max) / s_sum;
    }
}

// Float32 wrapper functions
extern "C" __global__ void relu(
    const float* input,
    float* output,
    int size
) {
    relu_kernel<float>(input, output, size);
}

extern "C" __global__ void sigmoid(const float* input, float* output, int size) {
    sigmoid_kernel<float>(input, output, size);
}

extern "C" __global__ void hyperbolic_tangent(
    const float* input,
    float* output,
    int size
) {
    hyperbolic_tangent_kernel<float>(input, output, size);
}

extern "C" __global__ void softmax(const float* input, float* output, int N) {
    softmax_kernel<float>(input, output, N);
}

extern "C" __global__ void softmax_batch_axis(
    const float* input,
    float* output,
    int batch_size,
    int seq_length,     // Size of the axis we're computing softmax over
    int inner_size,     // Size of dimensions after the softmax axis
    int total_elements
) {
    softmax_batch_axis_kernel<float>(input, output, batch_size, seq_length, inner_size, total_elements);
}

// Float64 wrapper functions
extern "C" __global__ void relu_f64(
    const double* input,
    double* output,
    int size
) {
    relu_kernel<double>(input, output, size);
}

extern "C" __global__ void sigmoid_f64(const double* input, double* output, int size) {
    sigmoid_kernel<double>(input, output, size);
}

extern "C" __global__ void hyperbolic_tangent_f64(
    const double* input,
    double* output,
    int size
) {
    hyperbolic_tangent_kernel<double>(input, output, size);
}

extern "C" __global__ void softmax_f64(const double* input, double* output, int N) {
    softmax_kernel<double>(input, output, N);
}

extern "C" __global__ void softmax_batch_axis_f64(
    const double* input,
    double* output,
    int batch_size,
    int seq_length,     // Size of the axis we're computing softmax over
    int inner_size,     // Size of dimensions after the softmax axis
    int total_elements
) {
    softmax_batch_axis_kernel<double>(input, output, batch_size, seq_length, inner_size, total_elements);
}
