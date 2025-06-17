// kernels/tanh.cu
// Hyperbolic tangent activation function
// Computes tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)) element-wise
// Maps input to range (-1, 1), making it zero-centered

extern "C" __global__ void tanh_forward(
    const float* input,
    float* output,
    int size
) {
    // Calculate thread index using standard CUDA pattern
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check for thread safety
    if (idx < size) {
        float x = input[idx];
        
        // Use the built-in tanhf function for numerical stability
        // It handles overflow/underflow cases internally
        output[idx] = tanhf(x);
        
        // Alternative manual implementation (commented for reference):
        // float exp_pos = expf(x);
        // float exp_neg = expf(-x);
        // output[idx] = (exp_pos - exp_neg) / (exp_pos + exp_neg);
    }
}
