// reduce.cu
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__device__ inline int get_global_idx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

extern "C" __global__ void max_along_dim(
    const float* input,
    float* output,
    int outer_size,
    int axis_size,
    int inner_size
) {
    int idx = get_global_idx();
    int total_output = outer_size * inner_size;
    
    if (idx >= total_output) return;
    
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    
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

extern "C" __global__ void max_along_dim_f64(
    const double* input,
    double* output,
    int outer_size,
    int axis_size,
    int inner_size
) {
    int idx = get_global_idx();
    int total_output = outer_size * inner_size;
    
    if (idx >= total_output) return;
    
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    
    double max_val = -CUDART_INF;
    for (int axis_idx = 0; axis_idx < axis_size; axis_idx++) {
        int input_idx = outer_idx * axis_size * inner_size + axis_idx * inner_size + inner_idx;
        double val = input[input_idx];
        if (val > max_val) {
            max_val = val;
        }
    }
    
    output[idx] = max_val;
}

extern "C" __global__ void sum_axis(
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

extern "C" __global__ void sum_axis_f64(
    const double* input,
    double* output,
    int outer_size,
    int axis_size,
    int inner_size
) {
    int outer_idx = blockIdx.x;
    int inner_idx = threadIdx.x;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    double sum = 0.0;
    for (int axis_idx = 0; axis_idx < axis_size; axis_idx++) {
        int input_idx = outer_idx * axis_size * inner_size + 
                       axis_idx * inner_size + inner_idx;
        sum += input[input_idx];
    }
    
    int output_idx = outer_idx * inner_size + inner_idx;
    output[output_idx] = sum;
}