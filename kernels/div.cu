
__global__ void elementwise_div(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Check for division by zero
        if (b[idx] != 0.0f) {
            c[idx] = a[idx] / b[idx];
        } else {
            // Handle division by zero - set to infinity
            c[idx] = INFINITY; // or use CUDART_INF_F with proper include
        }
    }
}
