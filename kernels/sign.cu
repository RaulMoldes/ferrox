// kernels/sign.cu
// CUDA kernel for sign operation

extern "C" __global__ void sign_kernel(
    const float* input,
    float* result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
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