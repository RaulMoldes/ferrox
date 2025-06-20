// kernels/comparison.cu
// CUDA kernels for comparison operations

extern "C" __global__ void greater_equal_kernel(
    const float* a,
    const float* b,
    float* result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = (a[idx] >= b[idx]) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void greater_equal_kernel_f64(
    const double* a,
    const double* b,
    double* result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = (a[idx] >= b[idx]) ? 1.0 : 0.0;
    }
}

extern "C" __global__ void less_equal_kernel(
    const float* a,
    const float* b,
    float* result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = (a[idx] <= b[idx]) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void less_equal_kernel_f64(
    const double* a,
    const double* b,
    double* result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = (a[idx] <= b[idx]) ? 1.0 : 0.0;
    }
}

extern "C" __global__ void equal_kernel(
    const float* a,
    const float* b,
    float* result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = (a[idx] == b[idx]) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void equal_kernel_f64(
    const double* a,
    const double* b,
    double* result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = (a[idx] == b[idx]) ? 1.0 : 0.0;
    }
}

extern "C" __global__ void logical_not_kernel(
    const float* input,
    float* result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = (input[idx] == 0.0f) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void logical_not_kernel_f64(
    const double* input,
    double* result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = (input[idx] == 0.0) ? 1.0 : 0.0;
    }
}

extern "C" __global__ void in_range_kernel(
    const float* input,
    float min_val,
    float max_val,
    float* result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = (input[idx] >= min_val && input[idx] <= max_val) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void in_range_kernel_f64(
    const double* input,
    double min_val,
    double max_val,
    double* result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = (input[idx] >= min_val && input[idx] <= max_val) ? 1.0 : 0.0;
    }
}