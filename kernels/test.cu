#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include "elementwise.cu"
#include "activations.cu"
#include "comparison.cu"
#include "fill.cu"
#include "reduction.cu"
#include "matmul.cu"
#include "convolutions.cu"
#include "globals.cuh"

#define BLOCK_SIZE 256
#define TILE_SIZE 16
#define SIZE (1 << 26) // ~64 million elements
#define REDUCTION_SIZE (1 << 20) // ~1 million elements for reduction tests
#define MATRIX_SIZE 1024 // Matrix dimensions for matmul tests
#define CONV_SIZE 128 // Input size for convolution tests
#define CHECK_CUDA(call) \
  if ((call) != cudaSuccess) { \
      fprintf(stderr, "CUDA Error: %s (at %s:%d)\n", cudaGetErrorString(call), __FILE__, __LINE__); \
      exit(EXIT_FAILURE); \
  }

template <typename T>
void run_and_time_kernel_binary(const char* name, void (*kernel)(const T*, const T*, T*, int),
    const T* d_a, const T* d_b, T* d_out, int size) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel << <(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (d_a, d_b, d_out, size);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("%-25s took %.3f ms\n", name, ms);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

template <typename T>
void run_and_time_kernel_unary(const char* name, void (*kernel)(const T*, T*, int),
    const T* d_in, T* d_out, int size) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel << <(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (d_in, d_out, size);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("%-25s took %.3f ms\n", name, ms);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// Template for fill operations with scalar value
template <typename T>
void run_and_time_fill(const char* name, void (*kernel)(T*, T, int),
    T* d_out, T value, int size) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel << <(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (d_out, value, size);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("%-25s took %.3f ms\n", name, ms);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// Template for random fill operations
template <typename T>
void run_and_time_random_fill(const char* name, void (*kernel)(T*, int, unsigned long),
    T* d_out, int size, unsigned long seed) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel << <(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (d_out, size, seed);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("%-25s took %.3f ms\n", name, ms);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// Template for clamp operations with scalar parameters
template <typename T>
void run_and_time_clamp(const char* name, void (*kernel)(const T*, T*, T, T, int),
    const T* d_in, T* d_out, T min_val, T max_val, int size) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel << <(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (d_in, d_out, min_val, max_val, size);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("%-25s took %.3f ms\n", name, ms);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// Template for in_range operations with scalar parameters
template <typename T>
void run_and_time_in_range(const char* name, void (*kernel)(const T*, T, T, T*, int),
    const T* d_in, T min_val, T max_val, T* d_out, int size) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel << <(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (d_in, min_val, max_val, d_out, size);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("%-25s took %.3f ms\n", name, ms);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// Template for where_condition operations with three arrays
template <typename T>
void run_and_time_where(const char* name, void (*kernel)(const T*, const T*, const T*, T*, int),
    const T* d_cond, const T* d_true, const T* d_false, T* d_out, int size) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel << <(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (d_cond, d_true, d_false, d_out, size);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("%-25s took %.3f ms\n", name, ms);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// Template for reduction operations - full reduction to scalar
template <typename T>
void run_and_time_reduce_all(const char* name, void (*kernel)(const T*, T*, int),
    const T* d_in, T* d_out, int size) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CHECK_CUDA(cudaEventRecord(start));
    kernel << <num_blocks, BLOCK_SIZE >> > (d_in, d_out, size);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("%-25s took %.3f ms (blocks: %d)\n", name, ms, num_blocks);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// Template for reduction operations - axis reduction
template <typename T>
void run_and_time_reduce_axes(const char* name, void (*kernel)(const T*, T*, int, int, int),
    const T* d_in, T* d_out, int outer_size, int axis_size, int inner_size) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel << <outer_size, inner_size >> > (d_in, d_out, outer_size, axis_size, inner_size);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("%-25s took %.3f ms (shape: %dx%dx%d)\n", name, ms, outer_size, axis_size, inner_size);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// Template for matrix multiplication
template <typename T>
void run_and_time_matmul(const char* name, void (*kernel)(const T*, const T*, T*, int, int, int),
    const T* d_a, const T* d_b, T* d_c, int M, int N, int K) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    CHECK_CUDA(cudaEventRecord(start));
    kernel << <grid, block >> > (d_a, d_b, d_c, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // Calculate GFLOPS
    double gflops = (2.0 * M * N * K) / (ms * 1e6);
    printf("%-25s took %.3f ms (%.1f GFLOPS, %dx%dx%d)\n", name, ms, gflops, M, N, K);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// Template for convolution
template <typename T>
void run_and_time_conv2d(const char* name,
    void (*kernel)(const T*, const T*, const T*, T*, int, int, int, int, int, int, int, int, int, int, int, int, int),
    const T* d_input, const T* d_filter, const T* d_bias, T* d_output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_height, int kernel_width, int stride_h, int stride_w, int pad_h, int pad_w) {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels);

    // Calculate shared memory size
    int input_tile_h = TILE_SIZE + kernel_height - 1;
    int input_tile_w = TILE_SIZE + kernel_width - 1;
    int filter_size = kernel_height * kernel_width;
    size_t shared_mem_size = (input_tile_h * input_tile_w + filter_size) * sizeof(T);

    CHECK_CUDA(cudaEventRecord(start));
    kernel << <grid, block, shared_mem_size >> > (
        d_input, d_filter, d_bias, d_output,
        batch_size, in_channels, in_height, in_width,
        out_channels, out_height, out_width,
        kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // Calculate GFLOPS
    long long ops = (long long)batch_size * out_channels * out_height * out_width *
        in_channels * kernel_height * kernel_width * 2; // 2 for MAC
    double gflops = ops / (ms * 1e6);

    printf("%-25s took %.3f ms (%.1f GFLOPS, %dx%dx%dx%d)\n",
        name, ms, gflops, batch_size, out_channels, out_height, out_width);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    size_t bytes_f32 = SIZE * sizeof(float);
    size_t bytes_f64 = SIZE * sizeof(double);
    size_t reduce_bytes_f32 = REDUCTION_SIZE * sizeof(float);
    size_t reduce_bytes_f64 = REDUCTION_SIZE * sizeof(double);
    size_t matrix_bytes_f32 = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    size_t matrix_bytes_f64 = MATRIX_SIZE * MATRIX_SIZE * sizeof(double);

    // Allocate device memory for float32
    float* d_a_f32, * d_b_f32, * d_c_f32, * d_out_f32;
    CHECK_CUDA(cudaMalloc(&d_a_f32, bytes_f32));
    CHECK_CUDA(cudaMalloc(&d_b_f32, bytes_f32));
    CHECK_CUDA(cudaMalloc(&d_c_f32, bytes_f32));
    CHECK_CUDA(cudaMalloc(&d_out_f32, bytes_f32));

    // Allocate device memory for float64
    double* d_a_f64, * d_b_f64, * d_c_f64, * d_out_f64;
    CHECK_CUDA(cudaMalloc(&d_a_f64, bytes_f64));
    CHECK_CUDA(cudaMalloc(&d_b_f64, bytes_f64));
    CHECK_CUDA(cudaMalloc(&d_c_f64, bytes_f64));
    CHECK_CUDA(cudaMalloc(&d_out_f64, bytes_f64));

    // Allocate smaller arrays for reduction tests
    float* d_reduce_in_f32, * d_reduce_out_f32;
    double* d_reduce_in_f64, * d_reduce_out_f64;
    CHECK_CUDA(cudaMalloc(&d_reduce_in_f32, reduce_bytes_f32));
    CHECK_CUDA(cudaMalloc(&d_reduce_out_f32, reduce_bytes_f32));
    CHECK_CUDA(cudaMalloc(&d_reduce_in_f64, reduce_bytes_f64));
    CHECK_CUDA(cudaMalloc(&d_reduce_out_f64, reduce_bytes_f64));

    // Allocate matrices for matmul tests
    float* d_mat_a_f32, * d_mat_b_f32, * d_mat_c_f32;
    double* d_mat_a_f64, * d_mat_b_f64, * d_mat_c_f64;
    CHECK_CUDA(cudaMalloc(&d_mat_a_f32, matrix_bytes_f32));
    CHECK_CUDA(cudaMalloc(&d_mat_b_f32, matrix_bytes_f32));
    CHECK_CUDA(cudaMalloc(&d_mat_c_f32, matrix_bytes_f32));
    CHECK_CUDA(cudaMalloc(&d_mat_a_f64, matrix_bytes_f64));
    CHECK_CUDA(cudaMalloc(&d_mat_b_f64, matrix_bytes_f64));
    CHECK_CUDA(cudaMalloc(&d_mat_c_f64, matrix_bytes_f64));

    // Allocate memory for convolution tests
    int conv_input_size = 1 * 3 * CONV_SIZE * CONV_SIZE; // batch=1, channels=3
    int conv_filter_size = 32 * 3 * 3 * 3; // 32 filters, 3 channels, 3x3 kernel
    int conv_output_size = 1 * 32 * CONV_SIZE * CONV_SIZE; // approximate

    float* d_conv_input_f32, * d_conv_filter_f32, * d_conv_bias_f32, * d_conv_output_f32;
    double* d_conv_input_f64, * d_conv_filter_f64, * d_conv_bias_f64, * d_conv_output_f64;

    CHECK_CUDA(cudaMalloc(&d_conv_input_f32, conv_input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_conv_filter_f32, conv_filter_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_conv_bias_f32, 32 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_conv_output_f32, conv_output_size * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_conv_input_f64, conv_input_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_conv_filter_f64, conv_filter_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_conv_bias_f64, 32 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_conv_output_f64, conv_output_size * sizeof(double)));

    // Initialize arrays using GPU kernels
    printf("=== FLOAT32 UTILITY OPERATIONS ===\n");
    run_and_time_fill("fill", fill, d_a_f32, 3.14159f, SIZE);
    run_and_time_fill("fill (zeros)", fill, d_b_f32, 0.0f, SIZE);
    run_and_time_random_fill("fill_random", fill_random, d_c_f32, SIZE, 12345UL);

    printf("\n=== FLOAT64 UTILITY OPERATIONS ===\n");
    run_and_time_fill("fill_f64", fill_f64, d_a_f64, 3.14159, SIZE);
    run_and_time_fill("fill_f64 (zeros)", fill_f64, d_b_f64, 0.0, SIZE);
    run_and_time_random_fill("fill_random_f64", fill_random_f64, d_c_f64, SIZE, 12345UL);

    // Initialize all test arrays
    run_and_time_random_fill("init reduce f32", fill_random, d_reduce_in_f32, REDUCTION_SIZE, 98765UL);
    run_and_time_random_fill("init reduce f64", fill_random_f64, d_reduce_in_f64, REDUCTION_SIZE, 98765UL);
    run_and_time_random_fill("init matrix A f32", fill_random, d_mat_a_f32, MATRIX_SIZE * MATRIX_SIZE, 11111UL);
    run_and_time_random_fill("init matrix B f32", fill_random, d_mat_b_f32, MATRIX_SIZE * MATRIX_SIZE, 22222UL);
    run_and_time_random_fill("init matrix A f64", fill_random_f64, d_mat_a_f64, MATRIX_SIZE * MATRIX_SIZE, 11111UL);
    run_and_time_random_fill("init matrix B f64", fill_random_f64, d_mat_b_f64, MATRIX_SIZE * MATRIX_SIZE, 22222UL);
    run_and_time_random_fill("init conv input f32", fill_random, d_conv_input_f32, conv_input_size, 33333UL);
    run_and_time_random_fill("init conv filter f32", fill_random, d_conv_filter_f32, conv_filter_size, 44444UL);
    run_and_time_fill("init conv bias f32", fill, d_conv_bias_f32, 0.1f, 32);
    run_and_time_random_fill("init conv input f64", fill_random_f64, d_conv_input_f64, conv_input_size, 33333UL);
    run_and_time_random_fill("init conv filter f64", fill_random_f64, d_conv_filter_f64, conv_filter_size, 44444UL);
    run_and_time_fill("init conv bias f64", fill_f64, d_conv_bias_f64, 0.1, 32);

    // Re-initialize with better values for operations
    run_and_time_random_fill("reinit random f32", fill_random, d_a_f32, SIZE, 54321UL);
    run_and_time_fill("reinit values f32", fill, d_b_f32, 2.0f, SIZE);
    run_and_time_random_fill("reinit random f64", fill_random_f64, d_a_f64, SIZE, 54321UL);
    run_and_time_fill("reinit values f64", fill_f64, d_b_f64, 2.0, SIZE);

    printf("\n=== FLOAT32 BINARY OPERATIONS ===\n");
    run_and_time_kernel_binary("elementwise_add", elementwise_add, d_a_f32, d_b_f32, d_out_f32, SIZE);
    run_and_time_kernel_binary("elementwise_sub", elementwise_sub, d_a_f32, d_b_f32, d_out_f32, SIZE);
    run_and_time_kernel_binary("elementwise_mul", elementwise_mul, d_a_f32, d_b_f32, d_out_f32, SIZE);
    run_and_time_kernel_binary("elementwise_div", elementwise_div, d_a_f32, d_b_f32, d_out_f32, SIZE);
    run_and_time_kernel_binary("elementwise_pow", elementwise_pow, d_a_f32, d_b_f32, d_out_f32, SIZE);
    run_and_time_kernel_binary("elementwise_max", elementwise_max, d_a_f32, d_b_f32, d_out_f32, SIZE);
    run_and_time_kernel_binary("elementwise_min", elementwise_min, d_a_f32, d_b_f32, d_out_f32, SIZE);

    printf("\n=== FLOAT32 COMPARISON OPERATIONS ===\n");
    run_and_time_kernel_binary("greater_equal", greater_equal, d_a_f32, d_b_f32, d_out_f32, SIZE);
    run_and_time_kernel_binary("greater", greater, d_a_f32, d_b_f32, d_out_f32, SIZE);
    run_and_time_kernel_binary("less_equal", less_equal, d_a_f32, d_b_f32, d_out_f32, SIZE);
    run_and_time_kernel_binary("less", less, d_a_f32, d_b_f32, d_out_f32, SIZE);
    run_and_time_kernel_binary("equal", equal, d_a_f32, d_b_f32, d_out_f32, SIZE);

    printf("\n=== FLOAT32 UNARY OPERATIONS ===\n");
    run_and_time_kernel_unary("logical_not", logical_not, d_a_f32, d_out_f32, SIZE);
    run_and_time_kernel_unary("sign", sign, d_a_f32, d_out_f32, SIZE);
    run_and_time_clamp("clamp", clamp, d_a_f32, d_out_f32, -2.0f, 2.0f, SIZE);
    run_and_time_in_range("in_range", in_range, d_a_f32, -1.0f, 1.0f, d_out_f32, SIZE);
    run_and_time_where("where_condition", where_condition, d_a_f32, d_b_f32, d_c_f32, d_out_f32, SIZE);

    printf("\n=== FLOAT32 ACTIVATION FUNCTIONS ===\n");
    run_and_time_kernel_unary("relu", relu, d_a_f32, d_out_f32, SIZE);
    run_and_time_kernel_unary("sigmoid", sigmoid, d_a_f32, d_out_f32, SIZE);
    run_and_time_kernel_unary("hyperbolic_tangent", hyperbolic_tangent, d_a_f32, d_out_f32, SIZE);

    printf("\n=== FLOAT32 REDUCTION OPERATIONS (FULL) ===\n");
    run_and_time_reduce_all("reduce_sum_all", reduce_sum_all, d_reduce_in_f32, d_reduce_out_f32, REDUCTION_SIZE);
    run_and_time_reduce_all("reduce_max_all", reduce_max_all, d_reduce_in_f32, d_reduce_out_f32, REDUCTION_SIZE);
    run_and_time_reduce_all("reduce_min_all", reduce_min_all, d_reduce_in_f32, d_reduce_out_f32, REDUCTION_SIZE);
    run_and_time_reduce_all("reduce_prod_all", reduce_prod_all, d_reduce_in_f32, d_reduce_out_f32, REDUCTION_SIZE);

    printf("\n=== FLOAT32 REDUCTION OPERATIONS (AXES) ===\n");
    run_and_time_reduce_axes("reduce_sum_axes", reduce_sum_axes, d_reduce_in_f32, d_reduce_out_f32, 1000, 1024, 1);
    run_and_time_reduce_axes("reduce_max_axes", reduce_max_axes, d_reduce_in_f32, d_reduce_out_f32, 1000, 1024, 1);
    run_and_time_reduce_axes("reduce_min_axes", reduce_min_axes, d_reduce_in_f32, d_reduce_out_f32, 1000, 1024, 1);
    run_and_time_reduce_axes("reduce_prod_axes", reduce_prod_axes, d_reduce_in_f32, d_reduce_out_f32, 1000, 1024, 1);

    printf("\n=== FLOAT32 MATRIX MULTIPLICATION ===\n");
    run_and_time_matmul("matmul", matmul, d_mat_a_f32, d_mat_b_f32, d_mat_c_f32, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE);

    printf("\n=== FLOAT32 CONVOLUTION ===\n");
    run_and_time_conv2d("conv2d_forward", conv2d_forward,
        d_conv_input_f32, d_conv_filter_f32, d_conv_bias_f32, d_conv_output_f32,
        1, 3, CONV_SIZE, CONV_SIZE, 32, CONV_SIZE, CONV_SIZE, 3, 3, 1, 1, 1, 1);

    printf("\n=== FLOAT64 BINARY OPERATIONS ===\n");
    run_and_time_kernel_binary("elementwise_add_f64", elementwise_add_f64, d_a_f64, d_b_f64, d_out_f64, SIZE);
    run_and_time_kernel_binary("elementwise_sub_f64", elementwise_sub_f64, d_a_f64, d_b_f64, d_out_f64, SIZE);
    run_and_time_kernel_binary("elementwise_mul_f64", elementwise_mul_f64, d_a_f64, d_b_f64, d_out_f64, SIZE);
    run_and_time_kernel_binary("elementwise_div_f64", elementwise_div_f64, d_a_f64, d_b_f64, d_out_f64, SIZE);
    run_and_time_kernel_binary("elementwise_pow_f64", elementwise_pow_f64, d_a_f64, d_b_f64, d_out_f64, SIZE);
    run_and_time_kernel_binary("elementwise_max_f64", elementwise_max_f64, d_a_f64, d_b_f64, d_out_f64, SIZE);
    run_and_time_kernel_binary("elementwise_min_f64", elementwise_min_f64, d_a_f64, d_b_f64, d_out_f64, SIZE);

    printf("\n=== FLOAT64 COMPARISON OPERATIONS ===\n");
    run_and_time_kernel_binary("greater_equal_f64", greater_equal_f64, d_a_f64, d_b_f64, d_out_f64, SIZE);
    run_and_time_kernel_binary("greater_f64", greater_f64, d_a_f64, d_b_f64, d_out_f64, SIZE);
    run_and_time_kernel_binary("less_equal_f64", less_equal_f64, d_a_f64, d_b_f64, d_out_f64, SIZE);
    run_and_time_kernel_binary("less_f64", less_f64, d_a_f64, d_b_f64, d_out_f64, SIZE);
    run_and_time_kernel_binary("equal_f64", equal_f64, d_a_f64, d_b_f64, d_out_f64, SIZE);

    printf("\n=== FLOAT64 UNARY OPERATIONS ===\n");
    run_and_time_kernel_unary("logical_not_f64", logical_not_f64, d_a_f64, d_out_f64, SIZE);
    run_and_time_kernel_unary("sign_f64", sign_f64, d_a_f64, d_out_f64, SIZE);
    run_and_time_clamp("clamp_f64", clamp_f64, d_a_f64, d_out_f64, -2.0, 2.0, SIZE);
    run_and_time_in_range("in_range_f64", in_range_f64, d_a_f64, -1.0, 1.0, d_out_f64, SIZE);
    run_and_time_where("where_condition_f64", where_condition_f64, d_a_f64, d_b_f64, d_c_f64, d_out_f64, SIZE);

    printf("\n=== FLOAT64 ACTIVATION FUNCTIONS ===\n");
    run_and_time_kernel_unary("relu_f64", relu_f64, d_a_f64, d_out_f64, SIZE);
    run_and_time_kernel_unary("sigmoid_f64", sigmoid_f64, d_a_f64, d_out_f64, SIZE);
    run_and_time_kernel_unary("hyperbolic_tangent_f64", hyperbolic_tangent_f64, d_a_f64, d_out_f64, SIZE);

    printf("\n=== FLOAT64 REDUCTION OPERATIONS (FULL) ===\n");
    run_and_time_reduce_all("reduce_sum_all_f64", reduce_sum_all_f64, d_reduce_in_f64, d_reduce_out_f64, REDUCTION_SIZE);
    run_and_time_reduce_all("reduce_max_all_f64", reduce_max_all_f64, d_reduce_in_f64, d_reduce_out_f64, REDUCTION_SIZE);
    run_and_time_reduce_all("reduce_min_all_f64", reduce_min_all_f64, d_reduce_in_f64, d_reduce_out_f64, REDUCTION_SIZE);
    run_and_time_reduce_all("reduce_prod_all_f64", reduce_prod_all_f64, d_reduce_in_f64, d_reduce_out_f64, REDUCTION_SIZE);

    printf("\n=== FLOAT64 REDUCTION OPERATIONS (AXES) ===\n");
    run_and_time_reduce_axes("reduce_sum_axes_f64", reduce_sum_axes_f64, d_reduce_in_f64, d_reduce_out_f64, 1000, 1024, 1);
    run_and_time_reduce_axes("reduce_max_axes_f64", reduce_max_axes_f64, d_reduce_in_f64, d_reduce_out_f64, 1000, 1024, 1);
    run_and_time_reduce_axes("reduce_min_axes_f64", reduce_min_axes_f64, d_reduce_in_f64, d_reduce_out_f64, 1000, 1024, 1);
    run_and_time_reduce_axes("reduce_prod_axes_f64", reduce_prod_axes_f64, d_reduce_in_f64, d_reduce_out_f64, 1000, 1024, 1);

    printf("\n=== FLOAT64 MATRIX MULTIPLICATION ===\n");
    run_and_time_matmul("matmul_f64", matmul_f64, d_mat_a_f64, d_mat_b_f64, d_mat_c_f64, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE);

    printf("\n=== FLOAT64 CONVOLUTION ===\n");
    run_and_time_conv2d("conv2d_forward_f64", conv2d_forward_f64,
        d_conv_input_f64, d_conv_filter_f64, d_conv_bias_f64, d_conv_output_f64,
        1, 3, CONV_SIZE, CONV_SIZE, 32, CONV_SIZE, CONV_SIZE, 3, 3, 1, 1, 1, 1);

    // Cleanup all allocated memory
    printf("\n=== CLEANUP ===\n");

    // Basic arrays
    cudaFree(d_a_f32);
    cudaFree(d_b_f32);
    cudaFree(d_c_f32);
    cudaFree(d_out_f32);
    cudaFree(d_a_f64);
    cudaFree(d_b_f64);
    cudaFree(d_c_f64);
    cudaFree(d_out_f64);

    // Reduction arrays
    cudaFree(d_reduce_in_f32);
    cudaFree(d_reduce_out_f32);
    cudaFree(d_reduce_in_f64);
    cudaFree(d_reduce_out_f64);

    // Matrix arrays
    cudaFree(d_mat_a_f32);
    cudaFree(d_mat_b_f32);
    cudaFree(d_mat_c_f32);
    cudaFree(d_mat_a_f64);
    cudaFree(d_mat_b_f64);
    cudaFree(d_mat_c_f64);

    // Convolution arrays
    cudaFree(d_conv_input_f32);
    cudaFree(d_conv_filter_f32);
    cudaFree(d_conv_bias_f32);
    cudaFree(d_conv_output_f32);
    cudaFree(d_conv_input_f64);
    cudaFree(d_conv_filter_f64);
    cudaFree(d_conv_bias_f64);
    cudaFree(d_conv_output_f64);

    printf("All tests completed successfully!\n");
    return 0;
}
