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
    T* shared_mem = reinterpret_cast<T*>(smem);

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

template<typename T>
__device__ void cross_correlation2d_kernel(
    const T* input1,
    const T* input2,
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
    // Each block processes multiple kernel elements
    extern __shared__ unsigned char smem[];
    T* shared_data = reinterpret_cast<T*>(smem);

    // Block processes a 2D region of kernel elements
    int kernel_block_w = min(blockDim.x, kernel_width - blockIdx.x * blockDim.x);
    int kernel_block_h = min(blockDim.y, kernel_height - blockIdx.y * blockDim.y);

    int kx_base = blockIdx.x * blockDim.x;
    int ky_base = blockIdx.y * blockDim.y;
    int in_c = blockIdx.z % in_channels;
    int out_c = blockIdx.z / in_channels;

    // Each thread within block handles one kernel position
    int local_kx = threadIdx.x;
    int local_ky = threadIdx.y;
    int kx = kx_base + local_kx;
    int ky = ky_base + local_ky;

    if (kx >= kernel_width || ky >= kernel_height) return;

    T sum = T(0.0);

    // Process in chunks to improve memory access patterns
    int chunk_size = 32; // Process 32 output positions at a time

    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (int chunk_start = 0; chunk_start < out_height * out_width; chunk_start += chunk_size) {

            // Load data for current chunk cooperatively
            for (int i = 0; i < chunk_size && (chunk_start + i) < out_height * out_width; i++) {
                int linear_pos = chunk_start + i;
                int out_y = linear_pos / out_width;
                int out_x = linear_pos % out_width;

                // Calculate input position
                int in_y = out_y * stride_h - pad_h + ky;
                int in_x = out_x * stride_w - pad_w + kx;

                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int input_idx = batch_idx * (in_channels * in_height * in_width) +
                        in_c * (in_height * in_width) +
                        in_y * in_width + in_x;

                    int grad_idx = batch_idx * (out_channels * out_height * out_width) +
                        out_c * (out_height * out_width) +
                        out_y * out_width + out_x;

                    sum += input1[input_idx] * input2[grad_idx];
                }
            }
        }
    }

    // Write result
    int output_idx = out_c * (in_channels * kernel_height * kernel_width) +
        in_c * (kernel_height * kernel_width) +
        ky * kernel_width + kx;
    output[output_idx] = sum;
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



extern "C" __global__ void  cross_correlation2d(
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

    cross_correlation2d_kernel<float>(input1,
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



extern "C" __global__ void cross_correlation2d_f64(
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

    cross_correlation2d_kernel<double>(input1,
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



template <typename T>
__device__ void cross_correlation1d_kernel(const T* __restrict__ input1,
    const T* __restrict__ input2,
    T* __restrict__ output,
    int input1_size, int input2_size, int output_size)
{
    int k = get_global_idx();
    if (k >= output_size) return;

    T sum = 0;
    for (int i = 0; i < input2_size; ++i) {
        sum += input1[i + k] * input2[i];
    }
    output[k] = sum;
}

// Convenience wrappers
extern "C" __global__
void cross_correlation1d(const float* input1, const float* input2, float* output,
    int input1_size, int input2_size, int output_size) {
    cross_correlation1d_kernel<float>(input1, input2, output, input1_size, input2_size, output_size);
}



// Convenience wrappers
extern "C" __global__
void cross_correlation1d_f64(const double* input1, const double* input2, double* output,
    int input1_size, int input2_size, int output_size) {
    cross_correlation1d_kernel<double>(input1, input2, output, input1_size, input2_size, output_size);
}



template <typename T>
__device__ void deconv1d_kernel(const T* __restrict__ input,
    const T* __restrict__ filter,
    T* __restrict__ output,
    int input_size, int kernel_size, int output_size)
{
    int i = get_global_idx();
    if (i >= output_size) return;

    T sum = 0;

    // For each position in output, accumulate from relevant input positions
    // output[i] = sum over j where j+k=i and j is valid input index
    for (int j = 0; j < input_size; ++j) {
        int k = i - j;  // Filter index (flipped)
        if (k >= 0 && k < kernel_size) {
            // Use flipped filter: filter[kernel_size - 1 - k]
            sum += input[j] * filter[kernel_size - 1 - k];
        }
    }

    output[i] = sum;
}

// Convenience wrappers
extern "C" __global__
void deconv1d(const float* input, const float* filter, float* output,
    int input_size, int kernel_size, int output_size) {
    deconv1d_kernel<float>(input, filter, output, input_size, kernel_size, output_size);
}


extern "C" __global__
void deconv1d_f64(const double* input, const double* filter, double* output,
    int input_size, int kernel_size, int output_size) {
    deconv1d_kernel<double>(input, filter, output, input_size, kernel_size, output_size);
}



// Pooling
template <typename T>
__device__ void maxpool2d_kernel(const T* input, T* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding) {

    // Compute global indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H_out * W_out;

    if (idx >= total_elements) return;

    // Decode indices (n, c, h_out, w_out)
    int w_out_pos = idx % W_out;
    int h_out_pos = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (W_out * H_out * C);

    // Compute its position
    int h_start = h_out_pos * stride - padding;
    int w_start = w_out_pos * stride - padding;

    // Each thread loads the kernel to shared memory
    extern __shared__ unsigned char shmem[];
    T* shared_input = reinterpret_cast<T*>(shmem);

    // Compute thread offset
    int tid = threadIdx.x;
    int shared_offset = tid * kernel_size * kernel_size;

    // Load data from kernel to shmem
    int input_base = n * C * H * W + c * H * W;

    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int h_pos = h_start + kh;
            int w_pos = w_start + kw;

            T val = -(T)FLT_MAX;  // Default value for padding

            // Check if we are within the limits
            if (h_pos >= 0 && h_pos < H && w_pos >= 0 && w_pos < W) {
                int input_idx = input_base + h_pos * W + w_pos;
                val = input[input_idx];
            }

            // Load to shmem
            int shared_idx = shared_offset + kh * kernel_size + kw;
            shared_input[shared_idx] = val;
        }
    }

    // Sincronizae this block
    __syncthreads();

    // Find the maximum in shmem
    T max_val = -(T)FLT_MAX;

    for (int i = 0; i < kernel_size * kernel_size; i++) {
        int shared_idx = shared_offset + i;
        max_val = max(max_val, shared_input[shared_idx]);
    }

    // Write result back to HBM
    output[idx] = max_val;
}


extern "C" __global__ void  maxpool2d(const float* input, float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding) {

    maxpool2d_kernel<float>(input,output,
        N,  C, H, W,
        H_out, W_out,
        kernel_size, stride,  padding);
}


extern "C" __global__ void  maxpool2d_f64(const double* input, double* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding) {

    maxpool2d_kernel<double>(input, output,
        N, C, H, W,
        H_out, W_out,
        kernel_size, stride, padding);
}
