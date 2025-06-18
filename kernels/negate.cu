// kernels/negate.cu - Unary negation kernel
extern "C" __global__ void negate_kernel(
    const float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = -input[idx];
    }
}