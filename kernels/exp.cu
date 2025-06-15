extern "C" __global__ void exp_forward(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // THis is a basic element-wise exponential kernel
    if (idx < size) {
        output[idx] = expf(input[idx]);
    }
}
