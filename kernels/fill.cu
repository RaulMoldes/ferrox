#include <cuda.h>



__global__ void fill_random(float* data, int size, unsigned long seed) {
    int idx = get_global_idx();
    if (idx < size) {
        unsigned int state = seed + idx + 1; // Avoid 0

        // XORShift algorithm
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;

        // Convert to float [0,1]
        float uniform = (float)state / (float)UINT_MAX;
        data[idx] = uniform * 2.0f - 1.0f; // Simple uniform to normal
    }
}

__global__ void fill_random_f64(double* data, int size, unsigned long seed) {
    int idx = get_global_idx();
    if (idx < size) {
        unsigned int state = seed + idx + 1; // Avoid 0

        // XORShift algorithm
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;

        // Convert to float [0,1]
        float uniform = (double)state / (double)UINT_MAX;
        data[idx] = uniform * 2.0f - 1.0f; // Simple uniform to normal
    }
}


__global__ void fill(float* data, float value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

__global__ void fill_f64(double* data, double value, int size) {  // Fixed: double parameter, not float
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}
