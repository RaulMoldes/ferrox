extern "C" __global__ void sum_axis_kernel(
    const float* input,
    float* output,
    int outer_size,
    int axis_size, 
    int inner_size
) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    float sum = 0.0f;
    for (int axis_idx = 0; axis_idx < axis_size; axis_idx++) {
        int input_idx = outer_idx * axis_size * inner_size + 
                       axis_idx * inner_size + inner_idx;
        sum += input[input_idx];
    }
    
    int output_idx = outer_idx * inner_size + inner_idx;
    output[output_idx] = sum;
}
