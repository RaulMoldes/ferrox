// add.cu
extern "C" __global__ void elementwise_add(
    const float* a,
    const float* b,
    float* result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

extern "C" __global__ void add_scalar(
    const float* input,
    float scalar,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = input[idx] + scalar;
    }
}

extern "C" __global__ void add_backward(
    const float* grad_output,
    float* grad_a,
    float* grad_b,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        grad_a[idx] = grad_output[idx];
        grad_b[idx] = grad_output[idx];
    }
}