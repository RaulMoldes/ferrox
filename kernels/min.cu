// Helper function to get global thread index
__device__ inline int get_global_idx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

// Element-wise minimum between two tensors: c[i] = min(a[i], b[i])
// Uses shared memory for performance optimization
extern "C" __global__ void element_min(
    const float* a, 
    const float* b, 
    float* c, 
    int size
) {
    __shared__ float shared_a[256];
    __shared__ float shared_b[256];
    
    int tid = threadIdx.x;
    int idx = get_global_idx();
    int block_size = blockDim.x;
    
    // Load data into shared memory in chunks
    for (int offset = 0; offset < size; offset += gridDim.x * block_size) {
        int global_idx = offset + idx;
        
        if (global_idx < size) {
            shared_a[tid] = a[global_idx];
            shared_b[tid] = b[global_idx];
            
            __syncthreads();
            
            // Compute and store result
            c[global_idx] = fminf(shared_a[tid], shared_b[tid]);
        }
        
        __syncthreads();
    }
}