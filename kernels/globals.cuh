#pragma once
#include <cuda_runtime.h>

#ifdef CUDART_INF
#else
#define CUDART_INF INFINITY
#endif

#ifdef CUDART_NAN
#else
#define CUDART_NAN NAN
#endif

// Block size is always kept to 256 for easier configuration handling
#define BLOCK_SIZE 256
#define TILE_SIZE 16
__device__ inline int get_global_idx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ inline int get_global_idy() {
    return blockIdx.y * blockDim.y + threadIdx.y;
}


// Warp reduction primitives
template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
    // Use __shfl_down_sync for better performance than __shfl_xor_sync
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset, 32);
    }
    return val;
}

template <typename T>
__inline__ __device__ T warp_reduce_max(T val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset, 32));
    }
    return val;
}

template <typename T>
__inline__ __device__ T warp_reduce_min(T val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = min(val, __shfl_down_sync(0xffffffff, val, offset, 32));
    }
    return val;
}

template <typename T>
__inline__ __device__ T warp_reduce_prod(T val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val *= __shfl_down_sync(0xffffffff, val, offset, 32);
    }
    return val;
}



// Block reduction primitives
template <typename T>
__inline__ __device__ T block_reduce_max(T val) {
    __shared__ T shared[33];

    int lane = threadIdx.x & 0x1f;  // threadIdx.x % 32
    int wid = threadIdx.x >> 5;     // threadIdx.x / 32

    // Reduce inside the warp
    val = warp_reduce_max(val);

    // Thread 0 of each warp stores on shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // First warp makes the final reduction
    if (wid == 0) {
        val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : -FLT_MAX;
        val = warp_reduce_max(val);
    }

    return val;
}

template <typename T>
__inline__ __device__ T block_reduce_sum(T val) {
    __shared__ T shared[33];

    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : T(0.0);
        val = warp_reduce_sum(val);
    }

    return val;
}
