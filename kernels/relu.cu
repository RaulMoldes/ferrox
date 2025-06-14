// relu.cu
extern "C" __global__ void relu_forward(
    const float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

extern "C" __global__ void relu_backward(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
    }
}