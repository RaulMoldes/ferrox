// kernels/max_along_dim.cu
__device__ inline int get_global_idx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

// Simplified max reduction along a dimension
extern "C" __global__ void max_reduce_along_dim(
    const float* input,
    float* output,
    int outer_size,
    int axis_size,
    int inner_size
) {
    int idx = get_global_idx();
    int total_output = outer_size * inner_size;
    
    if (idx >= total_output) return;
    
    // Calculate position in output
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    
    // Find maximum along the axis
    float max_val = -INFINITY;
    for (int axis_idx = 0; axis_idx < axis_size; axis_idx++) {
        int input_idx = outer_idx * axis_size * inner_size + axis_idx * inner_size + inner_idx;
        float val = input[input_idx];
        if (val > max_val) {
            max_val = val;
        }
    }
    
    output[idx] = max_val;
}