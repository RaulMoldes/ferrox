// kernels/conv2d_optimized.cu
#include <cuda_runtime.h>

__device__ inline int get_global_idx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ inline int get_global_idy() {
    return blockIdx.y * blockDim.y + threadIdx.y;
}



extern "C" __global__ void conv2d_forward(
    const float* input,
    const float* filter,
    const float* bias,
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
    const int TILE_SIZE = 16;

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
        if (bias != nullptr) {
            result += bias[out_c];
        }

        int output_idx = batch_idx * (out_channels * out_height * out_width) +
                        out_c * (out_height * out_width) +
                        out_y * out_width + out_x;
        output[output_idx] = result;
    }
}





// ============================================================================
// FUSED DEPTHWISE SEPARABLE CONVOLUTION KERNEL
// ============================================================================
// Performs both depthwise and pointwise convolution in a single kernel
extern "C" __global__ void depthwise_separable_conv2d_fused(
    const float* input,
    const float* depthwise_filter,
    const float* pointwise_filter,
    const float* depthwise_bias,
    const float* pointwise_bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_height,
    int kernel_width,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    // Calculate output dimensions
    int out_height = (in_height + 2 * pad_h - kernel_height) / stride_h + 1;
    int out_width = (in_width + 2 * pad_w - kernel_width) / stride_w + 1;

    // Thread mapping: each thread computes one output pixel for one output channel
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z % out_channels;
    int batch_idx = blockIdx.z / out_channels;

    if (out_x >= out_width || out_y >= out_height ||
        out_c >= out_channels || batch_idx >= batch_size) {
        return;
    }

    float final_result = 0.0f;

    // Calculate input coordinates for depthwise convolution
    int in_x_start = out_x * stride_w - pad_w;
    int in_y_start = out_y * stride_h - pad_h;

    // STEP 1: Depthwise convolution for each input channel
    // We need to compute the intermediate values and immediately use them for pointwise
    for (int in_c = 0; in_c < in_channels; in_c++) {
        float depthwise_result = 0.0f;

        // Perform depthwise convolution for this input channel
        for (int ky = 0; ky < kernel_height; ky++) {
            for (int kx = 0; kx < kernel_width; kx++) {
                int in_y = in_y_start + ky;
                int in_x = in_x_start + kx;

                // Check bounds (zero padding)
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    // Input index: [batch, in_channel, height, width]
                    int input_idx = batch_idx * (in_channels * in_height * in_width) +
                                  in_c * (in_height * in_width) +
                                  in_y * in_width + in_x;

                    // Depthwise filter index: [in_channel, kernel_y, kernel_x]
                    int filter_idx = in_c * (kernel_height * kernel_width) +
                                   ky * kernel_width + kx;

                    depthwise_result += input[input_idx] * depthwise_filter[filter_idx];
                }
            }
        }

        // Add depthwise bias
        if (depthwise_bias != nullptr) {
            depthwise_result += depthwise_bias[in_c];
        }

        // STEP 2: Immediately apply pointwise convolution
        // Pointwise filter index: [out_channel, in_channel]
        int pointwise_filter_idx = out_c * in_channels + in_c;
        final_result += depthwise_result * pointwise_filter[pointwise_filter_idx];
    }

    // Add pointwise bias
    if (pointwise_bias != nullptr) {
        final_result += pointwise_bias[out_c];
    }

    // Write final result
    int output_idx = batch_idx * (out_channels * out_height * out_width) +
                    out_c * (out_height * out_width) +
                    out_y * out_width + out_x;
    output[output_idx] = final_result;
}



// F64 OPS

// Optimized 2D Convolution kernel with batch support
// Input: [batch, in_channels, height, width]
// Filter: [out_channels, in_channels, kernel_height, kernel_width]
// Output: [batch, out_channels, out_height, out_width]
extern "C" __global__ void conv2d_forward_f64(
    const double* input,
    const double* filter,
    const double* bias,
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
    const int TILE_SIZE = 16;

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
        if (bias != nullptr) {
            result += bias[out_c];
        }

        int output_idx = batch_idx * (out_channels * out_height * out_width) +
            out_c * (out_height * out_width) +
            out_y * out_width + out_x;
        output[output_idx] = result;
    }
}




// ============================================================================
// FUSED DEPTHWISE SEPARABLE CONVOLUTION KERNEL
// ============================================================================
// Performs both depthwise and pointwise convolution in a single kernel
extern "C" __global__ void depthwise_separable_conv2d_fused_f64(
    const double* input,
    const double* depthwise_filter,
    const double* pointwise_filter,
    const double* depthwise_bias,
    const double* pointwise_bias,
    double* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_height,
    int kernel_width,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    // Calculate output dimensions
    int out_height = (in_height + 2 * pad_h - kernel_height) / stride_h + 1;
    int out_width = (in_width + 2 * pad_w - kernel_width) / stride_w + 1;

    // Thread mapping: each thread computes one output pixel for one output channel
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z % out_channels;
    int batch_idx = blockIdx.z / out_channels;

    if (out_x >= out_width || out_y >= out_height ||
        out_c >= out_channels || batch_idx >= batch_size) {
        return;
    }

    double final_result = 0.0;

    // Calculate input coordinates for depthwise convolution
    int in_x_start = out_x * stride_w - pad_w;
    int in_y_start = out_y * stride_h - pad_h;

    // STEP 1: Depthwise convolution for each input channel
    // We need to compute the intermediate values and immediately use them for pointwise
    for (int in_c = 0; in_c < in_channels; in_c++) {
        float depthwise_result = 0.0f;

        // Perform depthwise convolution for this input channel
        for (int ky = 0; ky < kernel_height; ky++) {
            for (int kx = 0; kx < kernel_width; kx++) {
                int in_y = in_y_start + ky;
                int in_x = in_x_start + kx;

                // Check bounds (zero padding)
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    // Input index: [batch, in_channel, height, width]
                    int input_idx = batch_idx * (in_channels * in_height * in_width) +
                        in_c * (in_height * in_width) +
                        in_y * in_width + in_x;

                    // Depthwise filter index: [in_channel, kernel_y, kernel_x]
                    int filter_idx = in_c * (kernel_height * kernel_width) +
                        ky * kernel_width + kx;

                    depthwise_result += input[input_idx] * depthwise_filter[filter_idx];
                }
            }
        }

        // Add depthwise bias
        if (depthwise_bias != nullptr) {
            depthwise_result += depthwise_bias[in_c];
        }

        // STEP 2: Immediately apply pointwise convolution
        // Pointwise filter index: [out_channel, in_channel]
        int pointwise_filter_idx = out_c * in_channels + in_c;
        final_result += depthwise_result * pointwise_filter[pointwise_filter_idx];
    }

    // Add pointwise bias
    if (pointwise_bias != nullptr) {
        final_result += pointwise_bias[out_c];
    }

    // Write final result
    int output_idx = batch_idx * (out_channels * out_height * out_width) +
        out_c * (out_height * out_width) +
        out_y * out_width + out_x;
    output[output_idx] = final_result;
}
