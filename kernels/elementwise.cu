// kernels/elementwise.cu
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "globals.cuh"

template<typename T>
__device__ void elementwise_add_kernel(
    const T* a,
    const T* b,
    T* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

template<typename T>
__device__ void elementwise_abs_kernel(
    const T* input,
    T* output,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        output[idx] = abs(input[idx]);
    }
}

template<typename T>
__device__ void elementwise_mul_kernel(const T* a, const T* b, T* c, int size) {
    int idx = get_global_idx();
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

template<typename T>
__device__ void elementwise_div_kernel(const T* a, const T* b, T* c, int size) {
    int idx = get_global_idx();
    if (idx < size) {
        // Check for division by zero
        if (b[idx] != T(0.0)) {
            c[idx] = a[idx] / b[idx];
        }
        else {
            // Handle division by zero - set to infinity
            c[idx] = INFINITY;
        }
    }
}

template<typename T>
__device__ void elementwise_reciprocal_kernel(const T* a, T* c, int size) {
    int idx = get_global_idx();
    if (idx < size) {
        // Check for division by zero
        if (a[idx] != T(0.0)) {
            c[idx] = T(1.0) / a[idx];
        }
        else {
            // Handle division by zero - set to infinity
            c[idx] = INFINITY;
        }
    }
}

template<typename T>
__device__ void elementwise_exp_kernel(const T* input, T* output, int size) {
    int idx = get_global_idx();
    // This is a basic element-wise exponential kernel
    if (idx < size) {
        output[idx] = exp(input[idx]);
    }
}

template<typename T>
__device__ void elementwise_log_kernel(
    const T* input,
    T* output,
    int size
) {
    // Calculate thread index - standard CUDA indexing pattern
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check to avoid out-of-bounds memory access
    if (idx < size) {
        T x = input[idx];

        // Handle edge cases appropriately
        if (x <= T(0.0)) {
            // For x <= 0, natural log is undefined or -infinity
            // Following standard conventions: ln(0) = -inf, ln(negative) = NaN
            output[idx] = (x == T(0.0)) ? -INFINITY : NAN;
        }
        else {
            // Standard natural logarithm for positive values
            output[idx] = log(x);
        }
    }
}

template<typename T>
__device__ void elementwise_pow_kernel(
    const T* a,        // Base array
    const T* b,        // Exponent array
    T* output,         // Result array
    int size           // Number of elements
) {
    // Calculate thread index using standard CUDA pattern
    int idx = get_global_idx();

    // Boundary check for thread safety
    if (idx < size) {
        T base = a[idx];
        T exponent = b[idx];

        // Handle special cases for numerical stability
        if (base == T(0.0) && exponent == T(0.0)) {
            // Mathematical convention: 0^0 = 1
            output[idx] = T(1.0);
        }
        else if (base == T(0.0) && exponent > T(0.0)) {
            // 0^positive = 0
            output[idx] = T(0.0);
        }
        else if (base == T(0.0) && exponent < T(0.0)) {
            // 0^negative = infinity
            output[idx] = INFINITY;
        }
        else if (base < T(0.0) && floor(exponent) != exponent) {
            // Negative base with non-integer exponent results in complex number
            // Return NaN following IEEE 754 standard
            output[idx] = NAN;
        }
        else {
            // Standard power computation using built-in pow
            output[idx] = pow(base, exponent);
        }
    }
}

template<typename T>
__device__ void elementwise_max_kernel(
    const T* a,
    const T* b,
    T* c,
    int size
) {
    __shared__ T shared_a[BLOCK_SIZE];
    __shared__ T shared_b[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = get_global_idx();
    int block_size = blockDim.x;

    for (int offset = 0; offset < size; offset += gridDim.x * block_size) {
        int global_idx = offset + idx;

        if (global_idx < size) {
            shared_a[tid] = a[global_idx];
            shared_b[tid] = b[global_idx];
        }

        __syncthreads();

        if (global_idx < size) {
            c[global_idx] = max(shared_a[tid], shared_b[tid]);
        }

        __syncthreads();
    }
}

template<typename T>
__device__ void elementwise_min_kernel(
    const T* a,
    const T* b,
    T* c,
    int size
) {
    __shared__ T shared_a[BLOCK_SIZE];
    __shared__ T shared_b[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = get_global_idx();
    int block_size = blockDim.x;

    // Load data into shared memory in chunks
    for (int offset = 0; offset < size; offset += gridDim.x * block_size) {
        int global_idx = offset + idx;

        if (global_idx < size) {
            shared_a[tid] = a[global_idx];
            shared_b[tid] = b[global_idx];

            __syncthreads();

            // Compute and store result
            c[global_idx] = min(shared_a[tid], shared_b[tid]);
        }

        __syncthreads();
    }
}

template<typename T>
__device__ void elementwise_negate_kernel(
    const T* input,
    T* output,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        output[idx] = -input[idx];
    }
}

template<typename T>
__device__ void elementwise_sqrt_kernel(
    const T* input,
    T* output,
    int size
) {
    int idx = get_global_idx();

    if (idx < size) {
        T val = input[idx];
        // Check for negative values and handle them gracefully
        if (val >= T(0.0)) {
            output[idx] = sqrt(val);
        }
        else {
            // Set to NaN to indicate error - you could also set to 0.0
            output[idx] = NAN;
        }
    }
}

template<typename T>
__device__ void elementwise_sub_kernel(
    const T* a,
    const T* b,
    T* c,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        c[idx] = a[idx] - b[idx];
    }
}

// Float32 wrapper functions
extern "C" __global__ void elementwise_add(
    const float* a,
    const float* b,
    float* result,
    int size
) {
    elementwise_add_kernel<float>(a, b, result, size);
}

extern "C" __global__ void elementwise_abs(
    const float* input,
    float* output,
    int size
) {
    elementwise_abs_kernel<float>(input, output, size);
}

extern "C" __global__ void elementwise_mul(const float* a, const float* b, float* c, int size) {
    elementwise_mul_kernel<float>(a, b, c, size);
}

extern "C" __global__ void elementwise_div(const float* a, const float* b, float* c, int size) {
    elementwise_div_kernel<float>(a, b, c, size);
}

extern "C" __global__ void elementwise_reciprocal(const float* a, float* c, int size) {
    elementwise_reciprocal_kernel<float>(a, c, size);
}

extern "C" __global__ void elementwise_exp(const float* input, float* output, int size) {
    elementwise_exp_kernel<float>(input, output, size);
}

extern "C" __global__ void elementwise_log(
    const float* input,
    float* output,
    int size
) {
    elementwise_log_kernel<float>(input, output, size);
}

extern "C" __global__ void elementwise_pow(
    const float* a,
    const float* b,
    float* output,
    int size
) {
    elementwise_pow_kernel<float>(a, b, output, size);
}

extern "C" __global__ void elementwise_max(
    const float* a,
    const float* b,
    float* c,
    int size
) {
    elementwise_max_kernel<float>(a, b, c, size);
}

extern "C" __global__ void elementwise_min(
    const float* a,
    const float* b,
    float* c,
    int size
) {
    elementwise_min_kernel<float>(a, b, c, size);
}

extern "C" __global__ void elementwise_negate(
    const float* input,
    float* output,
    int size
) {
    elementwise_negate_kernel<float>(input, output, size);
}

extern "C" __global__ void elementwise_sqrt(
    const float* input,
    float* output,
    int size
) {
    elementwise_sqrt_kernel<float>(input, output, size);
}

extern "C" __global__ void elementwise_sub(
    const float* a,
    const float* b,
    float* c,
    int size
) {
    elementwise_sub_kernel<float>(a, b, c, size);
}

// Float64 wrapper functions
extern "C" __global__ void elementwise_add_f64(
    const double* a,
    const double* b,
    double* result,
    int size
) {
    elementwise_add_kernel<double>(a, b, result, size);
}

extern "C" __global__ void elementwise_abs_f64(
    const double* input,
    double* output,
    int size
) {
    elementwise_abs_kernel<double>(input, output, size);
}

extern "C" __global__ void elementwise_mul_f64(
    const double* a,
    const double* b,
    double* c,
    int size
) {
    elementwise_mul_kernel<double>(a, b, c, size);
}

extern "C" __global__ void elementwise_div_f64(
    const double* a,
    const double* b,
    double* c,
    int size
) {
    elementwise_div_kernel<double>(a, b, c, size);
}

extern "C" __global__ void elementwise_reciprocal_f64(const double* a, double* c, int size) {
    elementwise_reciprocal_kernel<double>(a, c, size);
}

extern "C" __global__ void elementwise_exp_f64(
    const double* input,
    double* output,
    int size
) {
    elementwise_exp_kernel<double>(input, output, size);
}

extern "C" __global__ void elementwise_log_f64(
    const double* input,
    double* output,
    int size
) {
    elementwise_log_kernel<double>(input, output, size);
}

// Special handling for f64 pow to maintain the original vectorized optimization
extern "C" __global__ void elementwise_pow_f64(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ output,
    int size
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2; // Process 2 elements per thread

    if (idx + 1 < size) {
        // Load 2 doubles at once from both arrays (reduces L1TEX pressure)
        double2 base_data = reinterpret_cast<const double2*>(a)[idx >> 1];
        double2 exp_data = reinterpret_cast<const double2*>(b)[idx >> 1];

        // Compute both using helper function
        double2 result_data;

        // Handle special cases for first element
        if (base_data.x == 0.0 && exp_data.x == 0.0) {
            result_data.x = 1.0;
        }
        else if (base_data.x == 0.0 && exp_data.x > 0.0) {
            result_data.x = 0.0;
        }
        else if (base_data.x == 0.0 && exp_data.x < 0.0) {
            result_data.x = INFINITY;
        }
        else if (base_data.x < 0.0 && floor(exp_data.x) != exp_data.x) {
            result_data.x = NAN;
        }
        else {
            result_data.x = pow(base_data.x, exp_data.x);
        }

        // Handle special cases for second element
        if (base_data.y == 0.0 && exp_data.y == 0.0) {
            result_data.y = 1.0;
        }
        else if (base_data.y == 0.0 && exp_data.y > 0.0) {
            result_data.y = 0.0;
        }
        else if (base_data.y == 0.0 && exp_data.y < 0.0) {
            result_data.y = INFINITY;
        }
        else if (base_data.y < 0.0 && floor(exp_data.y) != exp_data.y) {
            result_data.y = NAN;
        }
        else {
            result_data.y = pow(base_data.y, exp_data.y);
        }

        // Store 2 doubles at once
        reinterpret_cast<double2*>(output)[idx >> 1] = result_data;
    }
    else if (idx < size) {
        // Handle remaining element
        elementwise_pow_kernel<double>(a, b, output, 1);
    }
}

extern "C" __global__ void elementwise_max_f64(
    const double* a,
    const double* b,
    double* c,
    int size
) {
    elementwise_max_kernel<double>(a, b, c, size);
}

extern "C" __global__ void elementwise_min_f64(
    const double* a,
    const double* b,
    double* c,
    int size
) {
    elementwise_min_kernel<double>(a, b, c, size);
}

extern "C" __global__ void elementwise_negate_f64(
    const double* input,
    double* output,
    int size
) {
    elementwise_negate_kernel<double>(input, output, size);
}

extern "C" __global__ void elementwise_sqrt_f64(
    const double* input,
    double* output,
    int size
) {
    elementwise_sqrt_kernel<double>(input, output, size);
}

extern "C" __global__ void elementwise_sub_f64(
    const double* a,
    const double* b,
    double* c,
    int size
) {
    elementwise_sub_kernel<double>(a, b, c, size);
}
