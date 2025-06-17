// kernels/log.cu
// Natural logarithm activation function
// Computes ln(x) element-wise on GPU tensors
// Handles edge cases: ln(0) = -inf, ln(negative) = NaN

extern "C" __global__ void log_forward(
    const float* input,
    float* output,
    int size
) {
    // Calculate thread index - standard CUDA indexing pattern
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check to avoid out-of-bounds memory access
    if (idx < size) {
        float x = input[idx];
        
        // Handle edge cases appropriately
        if (x <= 0.0f) {
            // For x <= 0, natural log is undefined or -infinity
            // Following standard conventions: ln(0) = -inf, ln(negative) = NaN
            output[idx] = (x == 0.0f) ? -INFINITY : NAN;
        } else {
            // Standard natural logarithm for positive values
            output[idx] = logf(x);
        }
    }
}
