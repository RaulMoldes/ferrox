extern "C" __global__ void clamp_kernel(
    const float* input,
    float* output, 
    float min_val,
    float max_val,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        val = val < min_val ? min_val : val;
        val = val > max_val ? max_val : val;
        output[idx] = val;
    }
}
