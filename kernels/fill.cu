#include <curand.h>
#include <cuda.h>
#include <curand_kernel.h>

__global__ void fill_random(float* data, int size, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        // Generate normal distribution (mean=0, std=1)
        data[idx] = curand_normal(&state);
    }
}

__global__ void fill_random_f64(double* data, int size, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        // Generate normal distribution for f64
        data[idx] = curand_normal_double(&state);
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
