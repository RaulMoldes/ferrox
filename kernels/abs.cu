// Helper function to get global thread index
__device__ inline int get_global_idx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}


// Element-wise absolute value: output[i] = |input[i]|
extern "C" __global__ void element_abs(
    const float* input, 
    float* output, 
    int size
) {
    int idx = get_global_idx();
    
    if (idx < size) {
        output[idx] = fabsf(input[idx]);
    }
}

// Double precision version
extern "C" __global__ void element_abs_f64(
    const double* input, 
    double* output, 
    int size
) {
    int idx = get_global_idx();
    
    if (idx < size) {
        output[idx] = fabs(input[idx]);
    }
}
