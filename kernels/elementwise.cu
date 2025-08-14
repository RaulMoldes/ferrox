// Helper function to get global thread index
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "globals.cuh"

// TESTED
extern "C" __global__ void elementwise_add(
    const float* a,
    const float* b,
    float* result,
    int size
) {
    int idx = get_global_idx();

    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

// TESTED
// Element-wise absolute value: output[i] = |input[i]|
extern "C" __global__ void elementwise_abs(
    const float* input,
    float* output,
    int size
) {
    int idx = get_global_idx();

    if (idx < size) {
        output[idx] = fabsf(input[idx]);
    }
}

// TESTED
// Element-wise muliplication: output[i] = a[i] x b[i]
extern "C" __global__ void elementwise_mul(const float* a, const float* b, float* c, int size) {
    int idx = get_global_idx();
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}



// TESTED
// Element-wise division: output[i] = a[i] / b[i]
extern "C" __global__ void elementwise_div(const float* a, const float* b, float* c, int size) {
    int idx = get_global_idx();

    if (idx < size) {
        // Check for division by zero
        if (b[idx] != 0.0f) {
            c[idx] = a[idx] / b[idx];
        } else {
            // Handle division by zero - set to infinity
            c[idx] = INFINITY; // or use CUDART_INF_F with proper include
        }
    }
}


extern "C" __global__ void elementwise_reciprocal(const float* a, float* c, int size) {
    int idx = get_global_idx();

    if (idx < size) {
        // Check for division by zero
        if (a[idx] != 0.0f) {
            c[idx] = 1.0f / a[idx];
        }
        else {
            // Handle division by zero - set to infinity
            c[idx] = INFINITY;
        }
    }
}



extern "C" __global__ void elementwise_reciprocal_f64(const double* a, double* c, int size) {
    int idx = get_global_idx();

    if (idx < size) {
        // Check for division by zero
        if (a[idx] != 0.0) {
            c[idx] = 1.0 / a[idx];
        }
        else {
            // Handle division by zero - set to infinity
            c[idx] = INFINITY;
        }
    }
}

// TESTED
extern "C" __global__ void elementwise_exp(const float* input, float* output, int size) {
    int idx = get_global_idx();
    // THis is a basic element-wise exponential kernel
    if (idx < size) {
        output[idx] = expf(input[idx]);
    }
}

// TESTED
extern "C" __global__ void elementwise_log(
    const float* input,
    float* output,
    int size
) {
    // Calculate thread index - standard CUDA indexing pattern
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check to avoid out-of-bounds memory access
    if (idx < size) {
        float x = input[idx];

        // Handle edge cases appropriately
        if (x <= 0.0f) {
            // For x <= 0, natural log is undefined or -infinity
            // Following standard conventions: ln(0) = -inf, ln(negative) = NaN
            output[idx] = (x == 0.0f) ? -INFINITY : NAN;
        } else {
            // Standard natural logarithm for positive values
            output[idx] = logf(x);
        }
    }
}


// TESTED
extern "C" __global__ void elementwise_pow(
    const float* a,        // Base array
    const float* b,        // Exponent array
    float* output,         // Result array
    int size               // Number of elements
) {
    // Calculate thread index using standard CUDA pattern
    int idx =get_global_idx();

    // Boundary check for thread safety
    if (idx < size) {
        float base = a[idx];
        float exponent = b[idx];

        // Handle special cases for numerical stability
        if (base == 0.0f && exponent == 0.0f) {
            // Mathematical convention: 0^0 = 1
            output[idx] = 1.0f;
        } else if (base == 0.0f && exponent > 0.0f) {
            // 0^positive = 0
            output[idx] = 0.0f;
        } else if (base == 0.0f && exponent < 0.0f) {
            // 0^negative = infinity
            output[idx] = INFINITY;
        } else if (base < 0.0f && floorf(exponent) != exponent) {
            // Negative base with non-integer exponent results in complex number
            // Return NaN following IEEE 754 standard
            output[idx] = NAN;
        } else {
            // Standard power computation using built-in powf
            output[idx] = powf(base, exponent);
        }
    }
}

// TESTED
extern "C" __global__ void elementwise_max(
    const float* a,
    const float* b,
    float* c,
    int size
) {
    __shared__ float shared_a[BLOCK_SIZE];
    __shared__ float shared_b[BLOCK_SIZE];

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
            c[global_idx] = fmaxf(shared_a[tid], shared_b[tid]);
        }

        __syncthreads();
    }
}


//TESTED
extern "C" __global__ void elementwise_min(
    const float* a,
    const float* b,
    float* c,
    int size
) {
    __shared__ float shared_a[BLOCK_SIZE];
    __shared__ float shared_b[BLOCK_SIZE];

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
            c[global_idx] = fminf(shared_a[tid], shared_b[tid]);
        }

        __syncthreads();
    }
}

// TESTED
extern "C" __global__ void elementwise_negate(
    const float* input,
    float* output,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        output[idx] = -input[idx];
    }
}


// Element-wise square root: output[i] = sqrt(input[i])
extern "C" __global__ void elementwise_sqrt(
    const float* input,
    float* output,
    int size
) {
    int idx = get_global_idx();

    if (idx < size) {
        float val = input[idx];
        // Check for negative values and handle them gracefully
        if (val >= 0.0f) {
            output[idx] = sqrtf(val);
        } else {
            // Set to NaN to indicate error - you could also set to 0.0f
            output[idx] = nanf("");
        }
    }
}

extern "C" __global__ void elementwise_sub(
    const float* a,
    const float* b,
    float* c,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        c[idx] = a[idx] - b[idx];
    }
}


// ------------------------------------------------------
// Double precision version of the elementwise operations
// ------------------------------------------------------

extern "C" __global__ void elementwise_add_f64(
    const double* a,
    const double* b,
    double* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

extern "C" __global__ void elementwise_abs_f64(
    const double* input,
    double* output,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        output[idx] = fabs(input[idx]);
    }
}

extern "C" __global__ void elementwise_mul_f64(
    const double* a,
    const double* b,
    double* c,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

extern "C" __global__ void elementwise_div_f64(
    const double* a,
    const double* b,
    double* c,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        if (b[idx] != 0.0) {
            c[idx] = a[idx] / b[idx];
        } else {
            c[idx] = CUDART_INF;
        }
    }
}

extern "C" __global__ void elementwise_exp_f64(
    const double* input,
    double* output,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        output[idx] = exp(input[idx]);
    }
}

extern "C" __global__ void elementwise_log_f64(
    const double* input,
    double* output,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        double x = input[idx];
        if (x <= 0.0) {
            output[idx] = (x == 0.0) ? -CUDART_INF : CUDART_NAN;
        } else {
            output[idx] = log(x);
        }
    }
}

// Helper function to compute power with edge case handling
__device__ __forceinline__ double safe_pow(double base, double exponent) {
    if (base == 0.0 && exponent == 0.0) {
        return 1.0;
    } else if (base == 0.0 && exponent > 0.0) {
        return 0.0;
    } else if (base == 0.0 && exponent < 0.0) {
        return CUDART_INF;
    } else if (base < 0.0 && floor(exponent) != exponent) {
        return CUDART_NAN;
    } else {
        return pow(base, exponent);
    }
}

// Decided to change this kernel as L1TEX pipeline had become a limiter
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
        result_data.x = safe_pow(base_data.x, exp_data.x);
        result_data.y = safe_pow(base_data.y, exp_data.y);

        // Store 2 doubles at once
        reinterpret_cast<double2*>(output)[idx >> 1] = result_data;
    } else if (idx < size) {
        // Handle remaining element
        output[idx] = safe_pow(a[idx], b[idx]);
    }
}

extern "C" __global__ void elementwise_max_f64(
    const double* a,
    const double* b,
    double* c,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        c[idx] = fmax(a[idx], b[idx]);
    }
}

extern "C" __global__ void elementwise_min_f64(
    const double* a,
    const double* b,
    double* c,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        c[idx] = fmin(a[idx], b[idx]);
    }
}

extern "C" __global__ void elementwise_negate_f64(
    const double* input,
    double* output,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        output[idx] = -input[idx];
    }
}

extern "C" __global__ void elementwise_sqrt_f64(
    const double* input,
    double* output,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        double val = input[idx];
        if (val >= 0.0) {
            output[idx] = sqrt(val);
        } else {
            output[idx] = CUDART_NAN;
        }
    }
}

extern "C" __global__ void elementwise_sub_f64(
    const double* a,
    const double* b,
    double* c,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        c[idx] = a[idx] - b[idx];
    }
}
