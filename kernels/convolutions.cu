// kernels/conv2d_optimized.cu
#include <cuda_runtime.h>
#include "globals.cuh"


extern "C" __global__ void conv2d_forward(
    const float* input,
    const float* filter,
    float* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {

    // Use shared memory for input and filter
    // Shared memory size: input tile + filter
    // Total shared memory size: (TILE_SIZE + kernel_height - 1) *
    extern __shared__ float shared_mem[];

    // Compute the size of the input tile
    // Input tile size is the size of the output tile plus the kernel size minus 1
    // This accounts for the padding needed for the convolution
    int input_tile_h = TILE_SIZE + kernel_height - 1;
    int input_tile_w = TILE_SIZE + kernel_width - 1;
    int filter_size = kernel_height * kernel_width;

    // Partition shared memory
    // First part for input tile, second part for filter
    // shared_mem size: input_tile_h * input_tile_w + filter_size
    float* shared_input = shared_mem;
    float* shared_filter = shared_mem + input_tile_h * input_tile_w;

    int out_x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int out_y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int out_c = blockIdx.z % out_channels;
    int batch_idx = blockIdx.z / out_channels;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int thread_id = ty * blockDim.x + tx;
    int threads_per_block = blockDim.x * blockDim.y;

    // Compute the output coordinates
    bool valid_output = (out_x < out_width && out_y < out_height &&
                        out_c < out_channels && batch_idx < batch_size);

    float result = 0.0f;

    // Process each input channel
    for (int in_c = 0; in_c < in_channels; in_c++) {

        // 1. Load the filter cooperatively into shared memory
        // Each thread loads one element of the filter
        for (int i = thread_id; i < filter_size; i += threads_per_block) {
            int ky = i / kernel_width;
            int kx = i % kernel_width;

            int filter_idx = out_c * (in_channels * kernel_height * kernel_width) +
                           in_c * (kernel_height * kernel_width) +
                           ky * kernel_width + kx;
            shared_filter[i] = filter[filter_idx];
        }

        // 2. Load the input tile into shared memory
        // Each thread loads one element of the input tile
        int input_elements = input_tile_h * input_tile_w;
        for (int i = thread_id; i < input_elements; i += threads_per_block) {
            int tile_y = i / input_tile_w;
            int tile_x = i % input_tile_w;

            // Calculate the input coordinates based on the tile position
            int in_x = blockIdx.x * TILE_SIZE * stride_w - pad_w + tile_x;
            int in_y = blockIdx.y * TILE_SIZE * stride_h - pad_h + tile_y;

            if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
                int input_idx = batch_idx * (in_channels * in_height * in_width) +
                              in_c * (in_height * in_width) +
                              in_y * in_width + in_x;
                shared_input[i] = input[input_idx];
            } else {
                shared_input[i] = 0.0f; // Zero padding
            }
        }

        __syncthreads();

        // 3. Perform the convolution operation
        // Each thread computes a part of the output
        if (valid_output) {
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // Position in the shared memory tile
                    int shared_y = ty * stride_h + ky;
                    int shared_x = tx * stride_w + kx;

                    // Verify if the shared memory indices are within bounds
                    if (shared_y < input_tile_h && shared_x < input_tile_w) {
                        int input_idx = shared_y * input_tile_w + shared_x;
                        int filter_idx = ky * kernel_width + kx;

                        result += shared_input[input_idx] * shared_filter[filter_idx];
                    }
                }
            }
        }

        __syncthreads();
    }

    // Add bias if provided
    // Bias is added only if the output is valid
    if (valid_output) {

        int output_idx = batch_idx * (out_channels * out_height * out_width) +
                        out_c * (out_height * out_width) +
                        out_y * out_width + out_x;
        output[output_idx] = result;
    }
}



// F64 OPS

// Optimized 2D Convolution kernel with batch support
// Input: [batch, in_channels, height, width]
// Filter: [out_channels, in_channels, kernel_height, kernel_width]
// Output: [batch, out_channels, out_height, out_width]
extern "C" __global__ void conv2d_forward_f64(
    const double* input,
    const double* filter,
    double* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {

    // Use shared memory for input and filter
    // Shared memory size: input tile + filter
    // Total shared memory size: (TILE_SIZE + kernel_height - 1) *
    extern __shared__ double shared_mem_f64[]; // Be careful with variable name declarations

    // Compute the size of the input tile
    // Input tile size is the size of the output tile plus the kernel size minus 1
    // This accounts for the padding needed for the convolution
    int input_tile_h = TILE_SIZE + kernel_height - 1;
    int input_tile_w = TILE_SIZE + kernel_width - 1;
    int filter_size = kernel_height * kernel_width;

    // Partition shared memory
    // First part for input tile, second part for filter
    // shared_mem size: input_tile_h * input_tile_w + filter_size
    double* shared_input = shared_mem_f64;
    double* shared_filter = shared_mem_f64 + input_tile_h * input_tile_w;

    int out_x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int out_y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int out_c = blockIdx.z % out_channels;
    int batch_idx = blockIdx.z / out_channels;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int thread_id = ty * blockDim.x + tx;
    int threads_per_block = blockDim.x * blockDim.y;

    // Compute the output coordinates
    bool valid_output = (out_x < out_width && out_y < out_height &&
        out_c < out_channels && batch_idx < batch_size);

    double result = 0.0;

    // Process each input channel
    for (int in_c = 0; in_c < in_channels; in_c++) {

        // 1. Load the filter cooperatively into shared memory
        // Each thread loads one element of the filter
        for (int i = thread_id; i < filter_size; i += threads_per_block) {
            int ky = i / kernel_width;
            int kx = i % kernel_width;

            int filter_idx = out_c * (in_channels * kernel_height * kernel_width) +
                in_c * (kernel_height * kernel_width) +
                ky * kernel_width + kx;
            shared_filter[i] = filter[filter_idx];
        }

        // 2. Load the input tile into shared memory
        // Each thread loads one element of the input tile
        int input_elements = input_tile_h * input_tile_w;
        for (int i = thread_id; i < input_elements; i += threads_per_block) {
            int tile_y = i / input_tile_w;
            int tile_x = i % input_tile_w;

            // Calculate the input coordinates based on the tile position
            int in_x = blockIdx.x * TILE_SIZE * stride_w - pad_w + tile_x;
            int in_y = blockIdx.y * TILE_SIZE * stride_h - pad_h + tile_y;

            if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
                int input_idx = batch_idx * (in_channels * in_height * in_width) +
                    in_c * (in_height * in_width) +
                    in_y * in_width + in_x;
                shared_input[i] = input[input_idx];
            }
            else {
                shared_input[i] = 0.0f; // Zero padding
            }
        }

        __syncthreads();

        // 3. Perform the convolution operation
        // Each thread computes a part of the output
        if (valid_output) {
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // Position in the shared memory tile
                    int shared_y = ty * stride_h + ky;
                    int shared_x = tx * stride_w + kx;

                    // Verify if the shared memory indices are within bounds
                    if (shared_y < input_tile_h && shared_x < input_tile_w) {
                        int input_idx = shared_y * input_tile_w + shared_x;
                        int filter_idx = ky * kernel_width + kx;

                        result += shared_input[input_idx] * shared_filter[filter_idx];
                    }
                }
            }
        }

        __syncthreads();
    }

    // Add bias if provided
    // Bias is added only if the output is valid
    if (valid_output) {


        int output_idx = batch_idx * (out_channels * out_height * out_width) +
            out_c * (out_height * out_width) +
            out_y * out_width + out_x;
        output[output_idx] = result;
    }
}


extern "C" __global__ void conv1d_forward(const float* input, const float* kernel, float* output,
    int input_size, int kernel_size) {

    __shared__ float s_data[input_size - kernel_size + 1];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int output_size = input_size - kernel_size + 1;

    // Number of elements per block
    int elements_per_block = blockDim.x;
    int block_start = bid * elements_per_block;

    // Load the kernel to shared memory
    if (tid < kernel_size) {
        s_data[tid] = kernel[tid];
    }

    // Calculate where do we start to compute for this block.
    int input_needed = elements_per_block + kernel_size - 1;
    int input_start = block_start;

    // Load the input after the kernel
    float* s_input = &s_data[kernel_size];

    for (int i = tid; i < input_needed; i += blockDim.x) {
        int global_idx = input_start + i;
        if (global_idx < input_size) {
            s_input[i] = input[global_idx];
        }
        else {
            s_input[i] = 0.0f;
        }
    }

    __syncthreads();

    // Cada thread procesa un elemento de salida
    int output_idx = block_start + tid;

    if (output_idx < output_size) {
        float sum = 0.0f;

#pragma unroll 8
        for (int k = 0; k < kernel_size; k++) {
            sum += s_input[tid + k] * s_data[k];
        }

        output[output_idx] = sum;
    }

}


extern "C" __global__ void conv1d_forward(const double* input, const double* kernel, double* output,
    int input_size, int kernel_size) {

    __shared__ double s_data[input_size - kernel_size + 1];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int output_size = input_size - kernel_size + 1;

    // Número de elementos que procesará cada bloque
    int elements_per_block = blockDim.x;
    int block_start = bid * elements_per_block;

    // Cargar kernel a memoria compartida
    if (tid < kernel_size) {
        s_data[tid] = kernel[tid];
    }

    // Calcular cuántos elementos de input necesita este bloque
    int input_needed = elements_per_block + kernel_size - 1;
    int input_start = block_start;

    // Cargar input a memoria compartida (después del kernel)
    double* s_input = &s_data[kernel_size];

    for (int i = tid; i < input_needed; i += blockDim.x) {
        int global_idx = input_start + i;
        if (global_idx < input_size) {
            s_input[i] = input[global_idx];
        }
        else {
            s_input[i] = 0.0;
        }
    }

    __syncthreads();

    // Cada thread procesa un elemento de salida
    int output_idx = block_start + tid;

    if (output_idx < output_size) {
        double sum = 0.0;

#pragma unroll 8
        for (int k = 0; k < kernel_size; k++) {
            sum += s_input[tid + k] * s_data[k];
        }

        output[output_idx] = sum;
    }

}
