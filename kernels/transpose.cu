// transpose.cu
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define TILE_SIZE 16;

extern "C" __global__ void transpose_2d(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int input_idx = row * cols + col;      // input[row][col]
        int output_idx = col * rows + row;     // output[col][row]
        output[output_idx] = input[input_idx];
    }
}

extern "C" __global__ void transpose_2d_f64(
    const double* input,
    double* output,
    int rows,
    int cols
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int input_idx = row * cols + col;      // input[row][col]
        int output_idx = col * rows + row;     // output[col][row]
        output[output_idx] = input[input_idx];
    }
}


// Optimized kernels with shared memory
__global__ void transpose_2d_shared(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;
    int thread_row = block_row + threadIdx.y;
    int thread_col = block_col + threadIdx.x;

    if (thread_row < rows && thread_col < cols) {
        int input_idx = thread_row * cols + thread_col;
        tile[threadIdx.y][threadIdx.x] = input[input_idx];
    }
    else {
        tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    int output_row = block_col + threadIdx.y;
    int output_col = block_row + threadIdx.x;

    if (output_row < cols && output_col < rows) {
        int output_idx = output_row * rows + output_col;
        output[output_idx] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose_2d_shared_f64(
    const double* input,
    double* output,
    int rows,
    int cols
) {
    __shared__ double tile[TILE_SIZE][TILE_SIZE + 1];

    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;
    int thread_row = block_row + threadIdx.y;
    int thread_col = block_col + threadIdx.x;

    if (thread_row < rows && thread_col < cols) {
        int input_idx = thread_row * cols + thread_col;
        tile[threadIdx.y][threadIdx.x] = input[input_idx];
    }
    else {
        tile[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    int output_row = block_col + threadIdx.y;
    int output_col = block_row + threadIdx.x;

    if (output_row < cols && output_col < rows) {
        int output_idx = output_row * rows + output_col;
        output[output_idx] = tile[threadIdx.x][threadIdx.y];
    }
}
