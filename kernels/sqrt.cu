// Element-wise square root: output[i] = sqrt(input[i])
extern "C" __global__ void sqrt(
    const float* input, 
    float* output, 
    int size
) {
    int idx = get_global_idx();
    
    if (idx < size) {
        float val = input[idx];
        // Check for negative values and handle them gracefully
        if (val >= 0.0f) {
            output[idx] = sqrtf(val);
        } else {
            // Set to NaN to indicate error - you could also set to 0.0f
            output[idx] = nanf("");
        }
    }
}

// Double precision version
extern "C" __global__ void sqrt_f64(
    const double* input, 
    double* output, 
    int size
) {
    int idx = get_global_idx();
    
    if (idx < size) {
        double val = input[idx];
        if (val >= 0.0) {
            output[idx] = sqrt(val);
        } else {
            output[idx] = nan("");
        }
    }
}