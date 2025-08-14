
// kernels/fill.cu
#include <cuda.h>
#include "globals.cuh"

template<typename T>
__device__ void fill_random_kernel(T* data, int size, unsigned long seed) {
    int idx = get_global_idx();
    if (idx < size) {
        unsigned int state = seed + idx + 1; // Avoid 0

        // XORShift algorithm
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;

        // Convert to float [0,1]
        T uniform = (T)state / (T)UINT_MAX;
        data[idx] = uniform * T(2.0) - T(1.0); // Simple uniform to normal
    }
}

template<typename T>
__device__ void fill_kernel(T* data, T value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

extern "C" __global__ void fill_random(float* data, int size, unsigned long seed) {
    fill_random_kernel<float>(data, size, seed);
}

extern "C" __global__ void fill_random_f64(double* data, int size, unsigned long seed) {
    fill_random_kernel<double>(data, size, seed);
}

extern "C" __global__ void fill(float* data, float value, int size) {
    fill_kernel<float>(data, value, size);
}

extern "C" __global__ void fill_f64(double* data, double value, int size) {
    fill_kernel<double>(data, value, size);
}
