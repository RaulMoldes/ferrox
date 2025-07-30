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
