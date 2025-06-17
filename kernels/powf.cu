// kernels/power.cu
// Element-wise power function: a^b
// Computes powf(a[i], b[i]) for each element in the arrays
// Handles edge cases: 0^0 = 1, negative base with non-integer exponent = NaN

extern "C" __global__ void power_forward(
    const float* a,        // Base array
    const float* b,        // Exponent array  
    float* output,         // Result array
    int size               // Number of elements
) {
    // Calculate thread index using standard CUDA pattern
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check for thread safety
    if (idx < size) {
        float base = a[idx];
        float exponent = b[idx];
        
        // Handle special cases for numerical stability
        if (base == 0.0f && exponent == 0.0f) {
            // Mathematical convention: 0^0 = 1
            output[idx] = 1.0f;
        } else if (base == 0.0f && exponent > 0.0f) {
            // 0^positive = 0
            output[idx] = 0.0f;
        } else if (base == 0.0f && exponent < 0.0f) {
            // 0^negative = infinity
            output[idx] = INFINITY;
        } else if (base < 0.0f && floorf(exponent) != exponent) {
            // Negative base with non-integer exponent results in complex number
            // Return NaN following IEEE 754 standard
            output[idx] = NAN;
        } else {
            // Standard power computation using built-in powf
            output[idx] = powf(base, exponent);
        }
    }
}
