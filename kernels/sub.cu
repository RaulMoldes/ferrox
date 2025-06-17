// src/backend/cuda/sub_kernel.cu
extern "C" __global__ void elementwise_sub(
    const float* a, 
    const float* b, 
    float* c, 
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] - b[idx];
    }
}
