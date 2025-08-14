// kernels/conv2d_optimized.cu
#include <cuda_runtime.h>
#include "globals.cuh"

// 2D Convolution kernel with batch support
// Input: [batch, in_channels, height, width]
// Filter: [out_channels, in_channels, kernel_height, kernel_width]
// Output: [batch, out_channels, out_height, out_width]
template<typename T>
__device__ void conv2d_forward_kernel(
    const T* input,
    const T* filter,
    T* output,
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
    extern __shared__ unsigned char smem[];
    T *shared_mem = reinterpret_cast<T*>(smem);

    // Compute the size of the input tile
    // Input tile size is the size of the output tile plus the kernel size minus 1
    // This accounts for the padding needed for the convolution
    int input_tile_h = TILE_SIZE + kernel_height - 1;
    int input_tile_w = TILE_SIZE + kernel_width - 1;
    int filter_size = kernel_height * kernel_width;

    // Partition shared memory
    // First part for input tile, second part for filter
    // shared_mem size: input_tile_h * input_tile_w + filter_size
    T* shared_input = shared_mem;
    T* shared_filter = shared_mem + input_tile_h * input_tile_w;

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

    T result = 0.0;

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



template <typename T>
__device__ void conv1d_forward_kernel(const T* __restrict__ input,
    const T* __restrict__ kernel,
    T* __restrict__ output,
    int input_size, int kernel_size)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int outN = input_size - kernel_size + 1;           // total outputs
    if (outN <= 0) return;

    int outputs_per_block = blockDim.x;                 // one thread -> one output
    int block_start = bid * outputs_per_block;

    // Shared memory layout: [ kernel (kernel_size) | tile_input (outputs_per_block + kernel_size - 1) ]
    extern __shared__ unsigned char smem[];
    T* s_kernel = reinterpret_cast<T*>(smem);
    T* s_input = s_kernel + kernel_size;

    // Load kernel into shared (strided)
    for (int i = tid; i < kernel_size; i += blockDim.x) {
        s_kernel[i] = kernel[i]; // reverse here if you want true convolution
        // s_kernel[i] = kernel[kernel_size - 1 - i]; // <- uncomment for convolution
    }

    // Figure out how much input this block needs: the tile + halos
    int input_needed = outputs_per_block + kernel_size - 1;
    int input_start = block_start;

    // Load input tile (strided, with zero-padding beyond input_size)
    for (int i = tid; i < input_needed; i += blockDim.x) {
        int g = input_start + i;
        s_input[i] = (g < input_size) ? input[g] : T(0);
    }

    __syncthreads(); // ensure shared is ready

    // Each thread computes one output
    int out_idx = block_start + tid;
    if (out_idx < outN) {
        T sum = 0;
        // Unroll a bit; you can also use #pragma unroll if kernel_size is small/known
#pragma unroll 8
        for (int k = 0; k < kernel_size; ++k) {
            sum += s_input[tid + k] * s_kernel[k];
        }
        output[out_idx] = sum;
    }
}

// Convenience wrappers
extern "C" __global__
void conv1d_forward(const float* input, const float* kernel, float* output,
    int input_size, int kernel_size) {
    conv1d_forward_kernel<float>(input, kernel, output, input_size, kernel_size);
}

extern "C" __global__
void conv1d_forward_f64(const double* input, const double* kernel, double* output,
    int input_size, int kernel_size) {
    conv1d_forward_kernel<double>(input, kernel, output, input_size, kernel_size);
}


extern "C" __global__
void conv2d_forward(
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
    conv2d_forward_kernel<float>(input,
        filter,
        output,
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w);
}

extern "C" __global__
void conv2d_forward_f64(
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
    conv2d_forward_kernel<double>(input,
        filter,
        output,
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w);
}

/// Gradient w.r.t. input
/// TESTED AGAINST PYTORCH
template<typename T>
__device__ void deconv2d_kernel(
    const T* input,
    const T* filter,
    T* output,
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
    extern __shared__ unsigned char smem[];
    T* shared_filter = reinterpret_cast<T*>(smem);

    int in_x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int in_y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int in_c = blockIdx.z % in_channels;
    int batch_idx = blockIdx.z / in_channels;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int thread_id = ty * blockDim.x + tx;
    int threads_per_block = blockDim.x * blockDim.y;

    bool valid_input = (in_x < in_width && in_y < in_height &&
        in_c < in_channels && batch_idx < batch_size);

    T result = 0.0;

    for (int out_c = 0; out_c < out_channels; out_c++) {
        int filter_size = kernel_height * kernel_width;
        for (int i = thread_id; i < filter_size; i += threads_per_block) {
            int ky = i / kernel_width;
            int kx = i % kernel_width;
            int filter_idx = out_c * (in_channels * kernel_height * kernel_width) +
                in_c * (kernel_height * kernel_width) +
                ky * kernel_width + kx;
            shared_filter[i] = filter[filter_idx];
        }
        __syncthreads();

        if (valid_input) {
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    int out_y = (in_y + pad_h - ky);
                    int out_x = (in_x + pad_w - kx);

                    if (out_y >= 0 && out_x >= 0 &&
                        out_y % stride_h == 0 && out_x % stride_w == 0) {
                        out_y /= stride_h;
                        out_x /= stride_w;
                        if (out_y < out_height && out_x < out_width) {
                            int out_idx = batch_idx * (out_channels * out_height * out_width) +
                                out_c * (out_height * out_width) +
                                out_y * out_width + out_x;
                            int filter_idx = ky * kernel_width + kx;
                            result += input[out_idx] * shared_filter[filter_idx];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (valid_input) {
        int output_idx = batch_idx * (in_channels * in_height * in_width) +
            in_c * (in_height * in_width) +
            in_y * in_width + in_x;
        output[output_idx] = result;
    }
}




extern "C" __global__ void  deconv2d(
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
    int pad_w) {

    deconv2d_kernel<float>(input,
        filter,
        output,
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w);
}



extern "C" __global__ void deconv2d_f64(
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
    int pad_w) {

    deconv2d_kernel<double>(
        input,
        filter,
        output,
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w);
}


template<typename T>
__device__ void cross_correlation_kernel(
    const T* input1,         // [batch, in_channels, in_height, in_width]
    const T* input2,   // [batch, out_channels, out_height, out_width]
    T* output,         // [out_channels, in_channels, kernel_height, kernel_width]
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
    // Each thread computes one element of grad_filter
    int kx = blockIdx.x * blockDim.x + threadIdx.x;
    int ky = blockIdx.y * blockDim.y + threadIdx.y;
    int in_c = blockIdx.z % in_channels;
    int out_c = blockIdx.z / in_channels;

    if (kx >= kernel_width || ky >= kernel_height ||
        in_c >= in_channels || out_c >= out_channels) {
        return;
    }

    T sum = 0.0;

    // Accumulate gradients across all batch elements and spatial positions
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (int out_y = 0; out_y < out_height; out_y++) {
            for (int out_x = 0; out_x < out_width; out_x++) {
                // Calculate corresponding input position
                int in_y = out_y * stride_h - pad_h + ky;
                int in_x = out_x * stride_w - pad_w + kx;

                // Check if input position is valid (within bounds)
                if (in_y >= 0 && in_y < in_height &&
                    in_x >= 0 && in_x < in_width) {

                    // Get input value
                    int input_idx = batch_idx * (in_channels * in_height * in_width) +
                        in_c * (in_height * in_width) +
                        in_y * in_width + in_x;

                    // Get grad_output value
                    int grad_out_idx = batch_idx * (out_channels * out_height * out_width) +
                        out_c * (out_height * out_width) +
                        out_y * out_width + out_x;

                    // Accumulate: grad_filter[out_c][in_c][ky][kx] += input * grad_output
                    sum += input1[input_idx] * input2[grad_out_idx];
                }
            }
        }
    }

    // Write result to grad_filter
    int grad_filter_idx = out_c * (in_channels * kernel_height * kernel_width) +
        in_c * (kernel_height * kernel_width) +
        ky * kernel_width + kx;
    output[grad_filter_idx] = sum;
}



extern "C" __global__ void  cross_correlation(
    const float* input1,
    const float* input2,
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
    int pad_w) {

    cross_correlation_kernel<float>(input1,
        input2,
        output,
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w);
}



extern "C" __global__ void cross_correlation_f64(
    const double* input1,
    const double* input2,
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
    int pad_w) {

    cross_correlation_kernel<double>(input1,
        input2,
        output,
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w);
}
