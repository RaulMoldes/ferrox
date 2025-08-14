// kernels/comparison.cu
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "globals.cuh"

// CUDA kernels for comparison operations

template<typename T>
__device__ void clamp_kernel(
    const T* input,
    T* output,
    T min_val,
    T max_val,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        T val = input[idx];
        val = val < min_val ? min_val : val;
        val = val > max_val ? max_val : val;
        output[idx] = val;
    }
}

template<typename T>
__device__ void greater_equal_kernel(
    const T* a,
    const T* b,
    T* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (a[idx] >= b[idx]) ? T(1.0) : T(0.0);
    }
}

template<typename T>
__device__ void greater_equal_scalar_kernel(
    const T* a,
    const T scalar,
    T* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (a[idx] >= scalar) ? T(1.0) : T(0.0);
    }
}

template<typename T>
__device__ void greater_kernel(
    const T* a,
    const T* b,
    T* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (a[idx] > b[idx]) ? T(1.0) : T(0.0);
    }
}

template<typename T>
__device__ void greater_scalar_kernel(
    const T* a,
    const T scalar,
    T* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (a[idx] > scalar) ? T(1.0) : T(0.0);
    }
}

template<typename T>
__device__ void less_equal_kernel(
    const T* a,
    const T* b,
    T* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (a[idx] <= b[idx]) ? T(1.0) : T(0.0);
    }
}

template<typename T>
__device__ void less_equal_scalar_kernel(
    const T* a,
    const T scalar,
    T* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (a[idx] <= scalar) ? T(1.0) : T(0.0);
    }
}

template<typename T>
__device__ void less_kernel(
    const T* a,
    const T* b,
    T* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (a[idx] < b[idx]) ? T(1.0) : T(0.0);
    }
}

template<typename T>
__device__ void less_scalar_kernel(
    const T* a,
    const T scalar,
    T* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (a[idx] < scalar) ? T(1.0) : T(0.0);
    }
}

template<typename T>
__device__ void equal_kernel(
    const T* a,
    const T* b,
    T* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (a[idx] == b[idx]) ? T(1.0) : T(0.0);
    }
}

template<typename T>
__device__ void logical_not_kernel(
    const T* input,
    T* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (input[idx] == T(0.0)) ? T(1.0) : T(0.0);
    }
}

template<typename T>
__device__ void in_range_kernel(
    const T* input,
    T min_val,
    T max_val,
    T* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (input[idx] >= min_val && input[idx] <= max_val) ? T(1.0) : T(0.0);
    }
}

template<typename T>
__device__ void sign_kernel(
    const T* input,
    T* result,
    int size
) {
    int idx = get_global_idx();

    if (idx < size) {
        T x = input[idx];
        if (x > T(0.0)) {
            result[idx] = T(1.0);
        }
        else if (x < T(0.0)) {
            result[idx] = T(-1.0);
        }
        else {
            result[idx] = T(0.0);  // x == 0
        }
    }
}

template<typename T>
__device__ void where_condition_kernel(
    const T* condition,
    const T* true_val,
    const T* false_val,
    T* output,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        output[idx] = (condition[idx] > T(0)) ? true_val[idx] : false_val[idx];
    }
}

// Float32 wrapper functions
extern "C" __global__ void clamp(
    const float* input,
    float* output,
    float min_val,
    float max_val,
    int size
) {
    clamp_kernel<float>(input, output, min_val, max_val, size);
}

extern "C" __global__ void greater_equal(
    const float* a,
    const float* b,
    float* result,
    int size
) {
    greater_equal_kernel<float>(a, b, result, size);
}

extern "C" __global__ void greater_equal_scalar(
    const float* a,
    const float scalar,
    float* result,
    int size
) {
    greater_equal_scalar_kernel<float>(a, scalar, result, size);
}

extern "C" __global__ void greater(
    const float* a,
    const float* b,
    float* result,
    int size
) {
    greater_kernel<float>(a, b, result, size);
}

extern "C" __global__ void greater_scalar(
    const float* a,
    const float scalar,
    float* result,
    int size
) {
    greater_scalar_kernel<float>(a, scalar, result, size);
}

extern "C" __global__ void less_equal(
    const float* a,
    const float* b,
    float* result,
    int size
) {
    less_equal_kernel<float>(a, b, result, size);
}

extern "C" __global__ void less_equal_scalar(
    const float* a,
    const float scalar,
    float* result,
    int size
) {
    less_equal_scalar_kernel<float>(a, scalar, result, size);
}

extern "C" __global__ void less(
    const float* a,
    const float* b,
    float* result,
    int size
) {
    less_kernel<float>(a, b, result, size);
}

extern "C" __global__ void less_scalar(
    const float* a,
    const float scalar,
    float* result,
    int size
) {
    less_scalar_kernel<float>(a, scalar, result, size);
}

extern "C" __global__ void equal(
    const float* a,
    const float* b,
    float* result,
    int size
) {
    equal_kernel<float>(a, b, result, size);
}

extern "C" __global__ void logical_not(
    const float* input,
    float* result,
    int size
) {
    logical_not_kernel<float>(input, result, size);
}

extern "C" __global__ void in_range(
    const float* input,
    float min_val,
    float max_val,
    float* result,
    int size
) {
    in_range_kernel<float>(input, min_val, max_val, result, size);
}

extern "C" __global__ void sign(
    const float* input,
    float* result,
    int size
) {
    sign_kernel<float>(input, result, size);
}

extern "C" __global__ void where_condition(
    const float* condition,
    const float* true_val,
    const float* false_val,
    float* output,
    int size
) {
    where_condition_kernel<float>(condition, true_val, false_val, output, size);
}

// Float64 wrapper functions
extern "C" __global__ void clamp_f64(
    const double* input,
    double* output,
    double min_val,
    double max_val,
    int size
) {
    clamp_kernel<double>(input, output, min_val, max_val, size);
}

extern "C" __global__ void greater_equal_f64(
    const double* a,
    const double* b,
    double* result,
    int size
) {
    greater_equal_kernel<double>(a, b, result, size);
}

extern "C" __global__ void greater_f64(
    const double* a,
    const double* b,
    double* result,
    int size
) {
    greater_kernel<double>(a, b, result, size);
}

extern "C" __global__ void greater_equal_scalar_f64(
    const double* a,
    const double scalar,
    double* result,
    int size
) {
    greater_equal_scalar_kernel<double>(a, scalar, result, size);
}

extern "C" __global__ void greater_scalar_f64(
    const double* a,
    const double scalar,
    double* result,
    int size
) {
    greater_scalar_kernel<double>(a, scalar, result, size);
}

extern "C" __global__ void less_equal_f64(
    const double* a,
    const double* b,
    double* result,
    int size
) {
    less_equal_kernel<double>(a, b, result, size);
}

extern "C" __global__ void less_equal_scalar_f64(
    const double* a,
    const double scalar,
    double* result,
    int size
) {
    less_equal_scalar_kernel<double>(a, scalar, result, size);
}

extern "C" __global__ void less_f64(
    const double* a,
    const double* b,
    double* result,
    int size
) {
    less_kernel<double>(a, b, result, size);
}

extern "C" __global__ void less_scalar_f64(
    const double* a,
    const double scalar,
    double* result,
    int size
) {
    less_scalar_kernel<double>(a, scalar, result, size);
}

extern "C" __global__ void equal_f64(
    const double* a,
    const double* b,
    double* result,
    int size
) {
    equal_kernel<double>(a, b, result, size);
}

extern "C" __global__ void logical_not_f64(
    const double* input,
    double* result,
    int size
) {
    logical_not_kernel<double>(input, result, size);
}

extern "C" __global__ void in_range_f64(
    const double* input,
    double min_val,
    double max_val,
    double* result,
    int size
) {
    in_range_kernel<double>(input, min_val, max_val, result, size);
}

extern "C" __global__ void sign_f64(
    const double* input,
    double* result,
    int size
) {
    sign_kernel<double>(input, result, size);
}

extern "C" __global__ void where_condition_f64(
    const double* condition,
    const double* true_val,
    const double* false_val,
    double* output,
    int size
) {
    where_condition_kernel<double>(condition, true_val, false_val, output, size);
}
