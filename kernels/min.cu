// kernels/min.cu - Min reduction kernel (complement to max.cu)
extern "C" __global__ void min_reduce_kernel(
    const float* input,
    float* output,
    int* indices,  // for argmin
    int batch_size,
    int dim_size
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    float min_val = INFINITY;  // Start with positive infinity for min
    int min_idx = 0;
    
    for (int i = 0; i < dim_size; i++) {
        float val = input[batch_idx * dim_size + i];
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
    }
    
    output[batch_idx] = min_val;
    if (indices) indices[batch_idx] = min_idx;
}