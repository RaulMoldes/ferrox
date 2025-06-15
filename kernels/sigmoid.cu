extern "C" __global__ void sigmoid_forward(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Element-wise sigmoid function: output = 1 / (1 + exp(-input))
        float x = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}
