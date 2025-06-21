// kernels/comparison.cu
// CUDA kernels for comparison operations

// Helper function to get global thread index
__device__ inline int get_global_idx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

extern "C" __global__ void clamp(
    const float* input,
    float* output, 
    float min_val,
    float max_val,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        float val = input[idx];
        val = val < min_val ? min_val : val;
        val = val > max_val ? max_val : val;
        output[idx] = val;
    }
}

extern "C" __global__ void greater_equal(
    const float* a,
    const float* b,
    float* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (a[idx] >= b[idx]) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void greater_equal_f64(
    const double* a,
    const double* b,
    double* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (a[idx] >= b[idx]) ? 1.0 : 0.0;
    }
}

extern "C" __global__ void less_equal(
    const float* a,
    const float* b,
    float* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (a[idx] <= b[idx]) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void less_equal_f64(
    const double* a,
    const double* b,
    double* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (a[idx] <= b[idx]) ? 1.0 : 0.0;
    }
}

extern "C" __global__ void equal(
    const float* a,
    const float* b,
    float* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (a[idx] == b[idx]) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void equal_f64(
    const double* a,
    const double* b,
    double* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (a[idx] == b[idx]) ? 1.0 : 0.0;
    }
}

extern "C" __global__ void logical_not(
    const float* input,
    float* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (input[idx] == 0.0f) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void logical_not_f64(
    const double* input,
    double* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (input[idx] == 0.0) ? 1.0 : 0.0;
    }
}

extern "C" __global__ void in_range(
    const float* input,
    float min_val,
    float max_val,
    float* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (input[idx] >= min_val && input[idx] <= max_val) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void in_range_f64(
    const double* input,
    double min_val,
    double max_val,
    double* result,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        result[idx] = (input[idx] >= min_val && input[idx] <= max_val) ? 1.0 : 0.0;
    }
}

// kernels/sign.cu
// CUDA kernel for sign operation

extern "C" __global__ void sign(
    const float* input,
    float* result,
    int size
) {
    int idx = get_global_idx();
    
    if (idx < size) {
        float x = input[idx];
        if (x > 0.0f) {
            result[idx] = 1.0f;
        } else if (x < 0.0f) {
            result[idx] = -1.0f;
        } else {
            result[idx] = 0.0f;  // x == 0
        }
    }
}

extern "C" __global__ void sign_kernel_f64(
    const double* input,
    double* result,
    int size
) {
    int idx = get_global_idx();
    
    if (idx < size) {
        double x = input[idx];
        if (x > 0.0) {
            result[idx] = 1.0;
        } else if (x < 0.0) {
            result[idx] = -1.0;
        } else {
            result[idx] = 0.0;  // x == 0
        }
    }
}