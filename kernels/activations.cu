#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
// Helper function to get global thread index
__device__ inline int get_global_idx() {
    return  blockIdx.x * blockDim.x + threadIdx.x;
}

extern "C" __global__ void relu(
    const float* input,
    float* output,
    int size
) {
    int idx = get_global_idx();
    
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

extern "C" __global__ void sigmoid(const float* input, float* output, int size) {
    int idx = get_global_idx();
    if (idx < size) {
        // Element-wise sigmoid function: output = 1 / (1 + exp(-input))
        float x = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}


extern "C" __global__ void hyperbolic_tangent(
    const float* input,
    float* output,
    int size
) {
    // Calculate thread index using standard CUDA pattern
    int idx =  get_global_idx();
    
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

/// ADAPTED TO SUPPORT DOUBLE PRECISION
extern "C" __global__ void relu_f64(
    const double* input,
    double* output,
    int size
) {
    int idx = get_global_idx();
    if (idx < size) {
        output[idx] = fmax(0.0, input[idx]);  // 0.0 not 0.0d
    }
}
extern "C" __global__ void sigmoid_f64(const double* input, double* output, int size) {
    int idx = get_global_idx();
    if (idx < size) {
        double x = input[idx];  // Should be double, not float
        output[idx] = 1.0 / (1.0 + exp(-x));  // Use exp() not expf()
    }
}

extern "C" __global__ void hyperbolic_tangent_f64(
    const double* input,
    double* output,
    int size
) {
    // Calculate thread index using standard CUDA pattern
    int idx =  get_global_idx();
    
    // Boundary check for thread safety
    if (idx < size) {
        double x = input[idx];
        
        // Use the built-in tanhf function for numerical stability
        // It handles overflow/underflow cases internally
        output[idx] = tanh(x);
        
        // Alternative manual implementation (commented for reference):
        // float exp_pos = expf(x);
        // float exp_neg = expf(-x);
        // output[idx] = (exp_pos - exp_neg) / (exp_pos + exp_neg);
    }
}