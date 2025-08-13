#include "globals.cuh"

/// Convert linear index to multi-dimensional coordinates
/// This handles the output tensor indexing
__device__ void linear_to_coords(int linear_idx, const int* shape, int ndim, int* coords) {
    for (int i = ndim - 1; i >= 0; i--) {
        coords[i] = linear_idx % shape[i];
        linear_idx /= shape[i];
    }
}

/// Convert coordinates to strided input index
/// This handles broadcast strides (0 for repeated dimensions)
__device__ int coords_to_strided_idx(const int* coords, const int* strides, int ndim) {
    int strided_idx = 0;
    for (int i = 0; i < ndim; i++) {
        strided_idx += coords[i] * strides[i];
    }
    return strided_idx;
}

/// Materialize strided/broadcast tensor into contiguous memory
/// Each thread handles one output element



extern "C" __global__ void materialize(
    const float* __restrict__ input,     // Original small data
    float* __restrict__ output,          // Expanded contiguous output
    const int* __restrict__ shape,   // Target shape dimensions
    const int* __restrict__ strides, // Input strides (0 for broadcast dims)
    int ndim,                        // Number of dimensions
    int total_elements               // Total output elements
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= total_elements) return;

    // Convert linear output index to coordinates in target shape
    int coords[8]; // Support up to 8D tensors
    linear_to_coords(tid, shape, ndim, coords);

    // Convert coordinates to input index using strides
    int input_idx = coords_to_strided_idx(coords, strides, ndim);

    // Copy data from input to output
    output[tid] = input[input_idx];
}

extern "C" __global__ void materialize_f64(
    const double* __restrict__ input,     // Original small data
    double* __restrict__ output,          // Expanded contiguous output
    const int* __restrict__ shape,   // Target shape dimensions
    const int* __restrict__ strides, // Input strides (0 for broadcast dims)
    int ndim,                        // Number of dimensions
    int total_elements               // Total output elements
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= total_elements) return;

    // Convert linear output index to coordinates in target shape
    int coords[8]; // Support up to 8D tensors
    linear_to_coords(tid, shape, ndim, coords);

    // Convert coordinates to input index using strides
    int input_idx = coords_to_strided_idx(coords, strides, ndim);

    // Copy data from input to output
    output[tid] = input[input_idx];
}

