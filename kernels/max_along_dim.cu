__device__ inline int get_global_idx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

// Helper function to calculate flat index from multi-dimensional coordinates
__device__ inline int coords_to_flat_idx(
    const int* coords,
    const int* strides,
    int ndim
) {
    int flat_idx = 0;
    for (int i = 0; i < ndim; i++) {
        flat_idx += coords[i] * strides[i];
    }
    return flat_idx;
}

// Helper function to convert flat index to multi-dimensional coordinates
__device__ inline void flat_idx_to_coords(
    int flat_idx,
    const int* shape,
    int* coords,
    int ndim
) {
    for (int i = ndim - 1; i >= 0; i--) {
        coords[i] = flat_idx % shape[i];
        flat_idx /= shape[i];
    }
}

// Calculate strides for a given shape
__device__ inline void calculate_strides(
    const int* shape,
    int* strides,
    int ndim
) {
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

// Maximum reduction along a specific dimension
// This is a complex kernel that handles arbitrary dimensional tensors
extern "C" __global__ void max_reduce_along_dim(
    const float* input,
    float* output,
    const int* input_shape,
    const int* output_shape,
    int reduce_dim,
    int input_ndim,
    int output_ndim,
    int output_size
) {
    int output_idx = get_global_idx();
    
    if (output_idx >= output_size) return;
    
    // Convert output flat index to output coordinates
    int output_coords[8]; // Support up to 8 dimensions
    flat_idx_to_coords(output_idx, output_shape, output_coords, output_ndim);
    
    // Create input coordinates by inserting the reduce dimension
    int input_coords[8];
    for (int i = 0, j = 0; i < input_ndim; i++) {
        if (i == reduce_dim) {
            input_coords[i] = 0; // Start from 0 for the dimension we're reducing
        } else {
            input_coords[i] = output_coords[j++];
        }
    }
    
    // Calculate input strides
    int input_strides[8];
    calculate_strides(input_shape, input_strides, input_ndim);
    
    // Initialize with first element along the reduce dimension
    int base_input_idx = coords_to_flat_idx(input_coords, input_strides, input_ndim);
    float max_val = input[base_input_idx];
    
    // Iterate through all elements along the reduce dimension
    for (int dim_idx = 1; dim_idx < input_shape[reduce_dim]; dim_idx++) {
        input_coords[reduce_dim] = dim_idx;
        int input_idx = coords_to_flat_idx(input_coords, input_strides, input_ndim);
        float val = input[input_idx];
        
        if (val > max_val) {
            max_val = val;
        }
    }
    
    output[output_idx] = max_val;
}
