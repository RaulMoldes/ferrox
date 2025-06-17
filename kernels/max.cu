// src/backend/cuda/max_kernel.cu
extern "C" __global__ void max_reduce_kernel(
    const float* input,
    float* output,
    int* indices,  // for argmax
    int batch_size,
    int dim_size
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    float max_val = -INFINITY;
    int max_idx = 0;
    
    for (int i = 0; i < dim_size; i++) {
        float val = input[batch_idx * dim_size + i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    
    output[batch_idx] = max_val;
    if (indices) indices[batch_idx] = max_idx;
}
