#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "globals.cuh"

extern "C" __global__ void relu(
    const float* input,
    float* output,
    int size
) {
    int idx = get_global_idx();

    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__device__ __forceinline__ float sigmoid_fused(float x) {
    // Clamp to prevent overflow - reduces branch divergence
    x = __saturatef(x * 0.0113636f) * 88.0f; // Fast clamp to [-88, 88]

    // Use intrinsic for fused reciprocal: 1.0 / (1.0 + exp(-x))
    return __frcp_rn(__fmaf_rn(expf(-x), 1.0f, 1.0f));
}

__device__ __forceinline__ double sigmoid_fused_f64(double x) {
    x = fmax(-709.0, fmin(709.0, x));
    return __drcp_rn(__fma_rn(exp(-x), 1.0, 1.0));
}


extern "C" __global__ void sigmoid(const float* input, float* output, int size) {
    int idx = get_global_idx();
    if (idx < size) {
        // Element-wise sigmoid function: output = 1 / (1 + exp(-input))
        float x = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}



extern "C" __global__ void sigmoid_f64(const double* input, double* output, int size) {
    int idx = get_global_idx();
    if (idx < size) {
        // Element-wise sigmoid function: output = 1 / (1 + exp(-input))
        double x = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}


extern "C" __global__ void hyperbolic_tangent(
    const float* input,
    float* output,
    int size
) {
    // Calculate thread index using standard CUDA pattern
    int idx = get_global_idx();

    // Boundary check for thread safety
    if (idx < size) {
        float x = input[idx];

        // Use the built-in tanhf function for numerical stability
        // It handles overflow/underflow cases internally
        output[idx] = tanhf(x);

        // Alternative manual implementation (commented for reference):
        // float exp_pos = expf(x);
        // float exp_neg = expf(-x);
        // output[idx] = (exp_pos - exp_neg) / (exp_pos + exp_neg);
    }
}

/// ADAPTED TO SUPPORT DOUBLE PRECISION
extern "C" __global__ void relu_f64(
    const double* input,
    double* output,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        output[idx] = fmax(0.0, input[idx]);  // 0.0 not 0.0d
    }
}

extern "C" __global__ void hyperbolic_tangent_f64(
    const double* input,
    double* output,
    int size
) {
    // Calculate thread index using standard CUDA pattern
    int idx = get_global_idx();

    // Boundary check for thread safety
    if (idx < size) {
        double x = input[idx];

        // Use the built-in tanhf function for numerical stability
        // It handles overflow/underflow cases internally
        output[idx] = tanh(x);

        // Alternative manual implementation (commented for reference):
        // float exp_pos = expf(x);
        // float exp_neg = expf(-x);
        // output[idx] = (exp_pos - exp_neg) / (exp_pos + exp_neg);
    }
}




// =============================================================================
// SOFTMAX KERNEL: WARP REDUCTION PRIMITIVES
// =============================================================================

template <typename T>
__inline__ __device__ T warpReduceMax(T val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    }
    return val;
}

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask, 32);
    }
    return val;
}

// =============================================================================
// BLOCK REDUCTION PRIMITIVES
// =============================================================================

template <typename T>
__inline__ __device__ T blockReduceMax(T val) {
    __shared__ T shared[33];

    int lane = threadIdx.x & 0x1f;  // threadIdx.x % 32
    int wid = threadIdx.x >> 5;     // threadIdx.x / 32

    // Reduce inside the warp
    val = warpReduceMax(val);

    // Thread 0 of each warp stores on shmem
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // First warp makes the final reduction
    if (wid == 0) {
        val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : -FLT_MAX;
        val = warpReduceMax(val);
    }

    return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
    __shared__ T shared[33];

    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : 0.0f;
        val = warpReduceSum(val);
    }

    return val;
}

// =============================================================================
// SOFTMAX KERNELS
// =============================================================================

extern "C" __global__ void softmax(const float* input, float* output, int N) {


    int tid = threadIdx.x;
    int global_idx = get_global_idx();

    // S1: Find global maxima using block reduction
    float local_max = -FLT_MAX;

    // Initialize local max values.
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        local_max = max(local_max, input[i]);
    }

    // Reduce the maximum
    float block_max = blockReduceMax(local_max);

    // Broadcast the maximum to all threads
    __shared__ float s_max;
    if (tid == 0) {
        s_max = block_max;
    }
    __syncthreads();

    // S2: Compute local sums
    float local_sum = 0.0f;

    // Each thread calculates its local sum value.
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        local_sum += expf(input[i] - s_max);
    }

    // Reduce sum between all blocks.
    float block_sum = blockReduceSum(local_sum);

    // Broadcast the sum to all threads.
    __shared__ float s_sum;
    if (tid == 0) {
        s_sum = block_sum;
    }
    __syncthreads();

    // S3: Compute final softmax and return the result.
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        output[i] = expf(input[i] - s_max) / s_sum;
    }
}




extern "C" __global__ void softmax_f64(const double* input, double* output, int N) {

    int tid = threadIdx.x;
    int global_idx = get_global_idx();

    // S1: Find global maxima using block reduction
    double local_max = -(double)FLT_MAX;

    // Initialize local max values.
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        local_max = max(local_max, input[i]);
    }

    // Reduce the maximum
    double block_max = blockReduceMax(local_max);

    // Broadcast the maximum to all threads
    __shared__ double s_max;
    if (tid == 0) {
        s_max = block_max;
    }
    __syncthreads();

    // S2: Compute local sums
    double local_sum = 0.0f;

    // Each thread calculates its local sum value.
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        local_sum += exp(input[i] - s_max);
    }

    // Reduce sum between all blocks.
    double block_sum = blockReduceSum(local_sum);

    // Broadcast the sum to all threads.
    __shared__ double s_sum;
    if (tid == 0) {
        s_sum = block_sum;
    }
    __syncthreads();

    // S3: Compute final softmax and return the result.
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        output[i] = exp(input[i] - s_max) / s_sum;
    }
}



// BATCH-AWARE SOFTMAX KERNELS
// =============================================================================
// This kernel processes multiple sequences in parallel, computing softmax
// along the specified axis while maintaining batch efficiency
// Each block processes one sequence from the batch

extern "C" __global__ void softmax_batch_axis(
    const float* input,
    float* output,
    int batch_size,
    int seq_length,     // Size of the axis we're computing softmax over
    int inner_size,     // Size of dimensions after the softmax axis
    int total_elements
) {
    // Each block handles one sequence (batch_idx, inner_idx combination)
    int block_id = blockIdx.x;
    int tid = threadIdx.x;

    // Calculate which batch and inner index this block handles
    int inner_idx = block_id % inner_size;
    int batch_idx = block_id / inner_size;

    // Skip if this block is beyond our data
    if (batch_idx >= batch_size) return;

    // Calculate starting position for this sequence
    // Layout: [batch_size, seq_length, inner_size]
    int sequence_start = batch_idx * seq_length * inner_size + inner_idx;

    // S1: Find maximum value in this sequence using block reduction
    float local_max = -FLT_MAX;

    // Each thread processes elements with stride inner_size
    for (int i = tid; i < seq_length; i += blockDim.x) {
        int global_idx = sequence_start + i * inner_size;
        local_max = max(local_max, input[global_idx]);
    }

    // Reduce maximum across the block
    float block_max = blockReduceMax(local_max);

    // Broadcast maximum to all threads in this block
    __shared__ float s_max;
    if (tid == 0) {
        s_max = block_max;
    }
    __syncthreads();

    // S2: Compute sum of exponentials
    float local_sum = 0.0f;

    for (int i = tid; i < seq_length; i += blockDim.x) {
        int global_idx = sequence_start + i * inner_size;
        local_sum += expf(input[global_idx] - s_max);
    }

    // Reduce sum across the block
    float block_sum = blockReduceSum(local_sum);

    // Broadcast sum to all threads in this block
    __shared__ float s_sum;
    if (tid == 0) {
        s_sum = block_sum;
    }
    __syncthreads();

    // S3: Compute final softmax values
    for (int i = tid; i < seq_length; i += blockDim.x) {
        int global_idx = sequence_start + i * inner_size;
        output[global_idx] = expf(input[global_idx] - s_max) / s_sum;
    }
}



extern "C" __global__ void softmax_batch_axis_f64(
    const double* input,
    double* output,
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
    double local_max = -(double)FLT_MAX;

    for (int i = tid; i < seq_length; i += blockDim.x) {
        int global_idx = sequence_start + i * inner_size;
        local_max = max(local_max, input[global_idx]);
    }

    double block_max = blockReduceMax(local_max);

    __shared__ double s_max;
    if (tid == 0) {
        s_max = block_max;
    }
    __syncthreads();

    // Compute sum
    double local_sum = 0.0;

    for (int i = tid; i < seq_length; i += blockDim.x) {
        int global_idx = sequence_start + i * inner_size;
        local_sum += exp(input[global_idx] - s_max);
    }

    double block_sum = blockReduceSum(local_sum);

    __shared__ double s_sum;
    if (tid == 0) {
        s_sum = block_sum;
    }
    __syncthreads();

    // Final softmax computation
    for (int i = tid; i < seq_length; i += blockDim.x) {
        int global_idx = sequence_start + i * inner_size;
        output[global_idx] = exp(input[global_idx] - s_max) / s_sum;
    }
}
