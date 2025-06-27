// kernels/conv2d.cu
#include <cuda_runtime.h>

__device__ inline int get_global_idx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ inline int get_global_idy() {
    return blockIdx.y * blockDim.y + threadIdx.y;
}

// 2D Convolution kernel with shared memory optimization
// Input: [batch, in_channels, height, width]
// Filter: [out_channels, in_channels, kernel_height, kernel_width] 
// Output: [batch, out_channels, out_height, out_width]
extern "C" __global__ void conv2d_forward(
    const float* input,      // Input tensor
    const float* filter,     // Convolution filters
    const float* bias,       // Bias (can be null)
    float* output,           // Output tensor
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
    // Thread coordinates
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z;
    
    // Check bounds
    if (out_x >= out_width || out_y >= out_height || out_c >= out_channels) {
        return;
    }
    
    // Shared memory for input tile
    extern __shared__ float shared_input[];
    
    // Calculate input coordinates
    int in_x_start = out_x * stride_w - pad_w;
    int in_y_start = out_y * stride_h - pad_h;
    
    float result = 0.0f;
    
    // Convolve over all input channels
    for (int in_c = 0; in_c < in_channels; in_c++) {
        // Convolve with kernel
        for (int ky = 0; ky < kernel_height; ky++) {
            for (int kx = 0; kx < kernel_width; kx++) {
                int in_y = in_y_start + ky;
                int in_x = in_x_start + kx;
                
                // Check bounds (padding)
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    // Input index: [batch=0, in_c, in_y, in_x]
                    int input_idx = in_c * in_height * in_width + in_y * in_width + in_x;
                    
                    // Filter index: [out_c, in_c, ky, kx]  
                    int filter_idx = out_c * in_channels * kernel_height * kernel_width +
                                   in_c * kernel_height * kernel_width +
                                   ky * kernel_width + kx;
                    
                    result += input[input_idx] * filter[filter_idx];
                }
            }
        }
    }
    
    // Add bias if provided
    if (bias != nullptr) {
        result += bias[out_c];
    }
    
    // Output index: [batch=0, out_c, out_y, out_x]
    int output_idx = out_c * out_height * out_width + out_y * out_width + out_x;
    output[output_idx] = result;
}

// Optimized version with shared memory for larger kernels
extern "C" __global__ void conv2d_forward_shared(
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
    const int FILTER_SIZE = 5; // Max kernel size for shared memory
    
    __shared__ float shared_input[TILE_SIZE + FILTER_SIZE - 1][TILE_SIZE + FILTER_SIZE - 1];
    __shared__ float shared_filter[FILTER_SIZE][FILTER_SIZE];
    
    int out_x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int out_y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int out_c = blockIdx.z;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    if (out_x >= out_width || out_y >= out_height || out_c >= out_channels) {
        return;
    }
    
    float result = 0.0f;
    
    // Process each input channel
    for (int in_c = 0; in_c < in_channels; in_c++) {
        // Load filter weights to shared memory
        if (tx < kernel_width && ty < kernel_height) {
            int filter_idx = out_c * in_channels * kernel_height * kernel_width +
                           in_c * kernel_height * kernel_width +
                           ty * kernel_width + tx;
            shared_filter[ty][tx] = filter[filter_idx];
        }
        
        // Load input tile to shared memory with padding
        int in_x = out_x * stride_w - pad_w + tx;
        int in_y = out_y * stride_h - pad_h + ty;
        
        if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
            int input_idx = in_c * in_height * in_width + in_y * in_width + in_x;
            shared_input[ty][tx] = input[input_idx];
        } else {
            shared_input[ty][tx] = 0.0f; // Zero padding
        }
        
        __syncthreads();
        
        // Perform convolution using shared memory
        for (int ky = 0; ky < kernel_height; ky++) {
            for (int kx = 0; kx < kernel_width; kx++) {
                if (ty + ky < TILE_SIZE + FILTER_SIZE - 1 && tx + kx < TILE_SIZE + FILTER_SIZE - 1) {
                    result += shared_input[ty + ky][tx + kx] * shared_filter[ky][kx];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Add bias
    if (bias != nullptr) {
        result += bias[out_c];
    }
    
    // Write result
    int output_idx = out_c * out_height * out_width + out_y * out_width + out_x;
    output[output_idx] = result;
}

// Separable convolution (for efficiency with large kernels)
extern "C" __global__ void conv2d_separable_h(
    const float* input,
    const float* filter_h,
    float* temp_output,
    int channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int pad
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    
    if (x >= width || y >= height || c >= channels) return;
    
    float result = 0.0f;
    
    for (int k = 0; k < kernel_size; k++) {
        int in_x = x * stride - pad + k;
        if (in_x >= 0 && in_x < width) {
            int idx = c * height * width + y * width + in_x;
            result += input[idx] * filter_h[k];
        }
    }
    
    int out_idx = c * height * width + y * width + x;
    temp_output[out_idx] = result;
}

extern "C" __global__ void conv2d_separable_v(
    const float* temp_input,
    const float* filter_v,
    float* output,
    int channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int pad
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    
    if (x >= width || y >= height || c >= channels) return;
    
    float result = 0.0f;
    
    for (int k = 0; k < kernel_size; k++) {
        int in_y = y * stride - pad + k;
        if (in_y >= 0 && in_y < height) {
            int idx = c * height * width + in_y * width + x;
            result += temp_input[idx] * filter_v[k];
        }
    }
    
    int out_idx = c * height * width + y * width + x;
    output[out_idx] = result;
}

// Depthwise convolution (each input channel convolved separately)
extern "C" __global__ void conv2d_depthwise(
    const float* input,
    const float* filter,
    const float* bias,
    float* output,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    
    if (out_x >= out_width || out_y >= out_height || c >= channels) return;
    
    float result = 0.0f;
    
    int in_x_start = out_x * stride_w - pad_w;
    int in_y_start = out_y * stride_h - pad_h;
    
    // Single channel convolution
    for (int ky = 0; ky < kernel_height; ky++) {
        for (int kx = 0; kx < kernel_width; kx++) {
            int in_y = in_y_start + ky;
            int in_x = in_x_start + kx;
            
            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                int input_idx = c * in_height * in_width + in_y * in_width + in_x;
                int filter_idx = c * kernel_height * kernel_width + ky * kernel_width + kx;
                
                result += input[input_idx] * filter[filter_idx];
            }
        }
    }
    
    if (bias != nullptr) {
        result += bias[c];
    }
    
    int output_idx = c * out_height * out_width + out_y * out_width + out_x;
    output[output_idx] = result;
}