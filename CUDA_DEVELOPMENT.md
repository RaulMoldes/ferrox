# Ferrox CUDA Development Guide

This document outlines my process for developing, benchmarking, and validating CUDA kernels in Ferrox against PyTorch implementations.

## Overview

I maintain CUDA kernels for performance-critical operations in the `kernels/` directory. Each kernel is thoroughly benchmarked using my custom CUDA benchmark suite and validated against PyTorch's implementations to ensure correctness.

## Directory Structure

```
kernels/
├── benchmark.cu         # Main benchmarking executable
├── elementwise.cu       # Elementwise operations kernels
├── activations.cu       # Activation function kernels
├── comparison.cu        # Comparison operation kernels
├── fill.cu              # Tensor fill kernels
├── reduction.cu         # Reduction operation kernels
├── matmul.cu           # Matrix multiplication kernels
├── convolutions.cu     # Convolution kernels
├── materialize.cu      # Tensor materialization kernels
├── globals.cuh         # Global constants and definitions
└── *.ptx               # Compiled PTX files
```

## Development Workflow

### 1. Kernel Implementation

CUDA kernels are written following these conventions:

```cuda
// File: kernels/elementwise.cu

__global__ void elementwise_add_f32(
    const float* a,
    const float* b,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    output[idx] = a[idx] + b[idx];
}

__global__ void elementwise_add_f64(
    const double* a,
    const double* b,
    double* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    output[idx] = a[idx] + b[idx];
}
```

**Critical conventions I follow:**
- **Launch configurations**: Must match exactly with `backend/cuda/ops.rs`
- **Block size**: `BLOCK_SIZE = 256` for standard kernels
- **F64 optimization**: `F64_BLOCK_SIZE = 512` with vectorization for double precision
- **2D operations**: `TILE_SIZE = 16` for matrix operations
- **Naming**: `operation_name_dtype` format (e.g., `elementwise_add_f32`)

### 2. Benchmark Integration

Integrate kernels into the benchmark suite in `benchmark.cu`:

```cuda
#include "elementwise.cu"
#include "activations.cu"
#include "comparison.cu"
// ... other includes

// Constants must match backend/cuda/ops.rs exactly
#define BLOCK_SIZE 256      // From get_launch_config()
#define TILE_SIZE 16        // From get_2d_launch_config()
#define F64_BLOCK_SIZE 512  // Special vectorized config for f64

#define SIZE (1 << 26)      // ~64 million elements
#define REDUCTION_SIZE (1 << 20)  // ~1 million for reductions
#define MATRIX_SIZE 1024    // Matrix dimensions
#define CONV_SIZE 128       // Convolution input size
```

Each kernel type has a specialized timing function:

```cuda
template <typename T>
void run_and_time_kernel_binary(
    const char* name,
    void (*kernel)(const T*, const T*, T*, int),
    const T* d_a, const T* d_b, T* d_out, int size
) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // Use exact same config as ops.rs
    if (sizeof(T) == 8) {  // f64 operations
        // Special vectorized config to reduce L1TEX stalls
        int grid_size = ((size / 2) + F64_BLOCK_SIZE - 1) / F64_BLOCK_SIZE;
        kernel<<<grid_size, F64_BLOCK_SIZE>>>(d_a, d_b, d_out, size);
    } else {  // f32 operations
        int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_out, size);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("%-25s took %.3f ms\n", name, ms);
}
```

### 3. Compilation and Profiling

You can compile and profile kernels using this workflow:

```bash
# Compile benchmark suite
nvcc -o benchmark benchmark.cu -lcurand

# Run basic timing benchmarks
./benchmark

# Profile with NVIDIA Nsight Compute for detailed analysis
ncu --set full --target-processes all ./benchmark

# Profile specific kernels
ncu --set full --kernel-regex "elementwise_add" ./benchmark

# Memory bandwidth analysis
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ./benchmark

# Occupancy analysis
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./benchmark
```

### 4. Performance Metrics I Track

The benchmark suite provides comprehensive metrics:

```
=== FLOAT32 ELEMENTWISE OPERATIONS ===
elementwise_add_f32      took 12.345 ms
elementwise_sub_f32      took 12.456 ms
elementwise_mul_f32      took 12.567 ms

=== FLOAT32 MATRIX MULTIPLICATION ===
matmul_f32              took 45.123 ms (234.5 GFLOPS, 1024x1024x1024)

=== FLOAT32 CONVOLUTION ===
conv2d_forward_f32      took 78.901 ms (batch=1, in=3, out=32, size=128x128)
```

For matrix operations, I calculate GFLOPS:
```cuda
double gflops = (2.0 * M * N * K) / (ms * 1e6);
printf("%-25s took %.3f ms (%.1f GFLOPS, %dx%dx%d)\n",
       name, ms, gflops, M, N, K);
```

## Validation Against PyTorch

### Rust-Based Validation Tests

I validate kernel correctness directly in my Rust test suite using PyTorch reference values:

```rust
#[test]
fn test_conv2d_pytorch_reference() {
    println!("Testing Conv2D against PyTorch reference...");

    // Exact same input data as PyTorch test
    let input_data = vec![
        0.100, 0.150, 0.200, 0.250, 0.300,
        0.350, 0.400, 0.450, 0.500, 0.550,
        // ... rest of data
    ];

    let inputs = vec![Tensor::from_vec_with_device(
        input_data,
        &[1, 3, 5, 5], // batch=1, channels=3, height=5, width=5
        device,
    ).expect("Failed to create input tensor")];


    // PyTorch reference output (generated separately)
    let expected_values = vec![
        0.7780, 1.1820, 1.3320, 1.4820, 0.9700,
        1.1345, 1.7940, 2.0185, 2.2430, 1.6215,
        // ... rest of expected values
    ];

    // Compare with tight tolerance
    let tolerance = 1e-3;
    let result_data = result.to_vec().expect("Failed to get result data");
        test_op_with_values!(
            Conv2d::new((1, 1), (1, 1)),
            inputs,
            &[1, 4, 5, 5],
            &expected_values,
            tolerance,
            "Conv2d_Test"
        );
}
```

NOTE: I also have macros to test forward and backward passes of autograd operations.

### PyTorch Reference Generation

**Important**: I generate PyTorch reference values manually by running separate Python scripts, then hard-coding the results into my Rust tests. This is intentional - I don't want automated PyTorch integration that could mask subtle bugs.

The process you should follow:

**Workflow for new operations:**

1. Write the Rust implementation
2. Create Python script with identical test data
3. Run Python script to generate reference values
4. Copy-paste the output into Rust test as hard-coded arrays
5. Run Rust test to validate against references
6. If test fails, debug the Rust implementation (not the reference generation)

## Optimization Guidelines

### Launch Configuration Strategy

I optimize launch configurations based on operation type:

**Elementwise operations:**
```cuda
// Standard config for most kernels
int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
kernel<<<grid_size, BLOCK_SIZE>>>(args);
```

**F64 operations (special optimization):**
```cuda
// Vectorized approach to reduce L1TEX cache stalls
int grid_size = ((size / 2) + F64_BLOCK_SIZE - 1) / F64_BLOCK_SIZE;
kernel<<<grid_size, F64_BLOCK_SIZE>>>(args); // Processes 2 elements per thread
```

NOTE: I have seen many of my kernels being limited by l1tex stalls. I think it may be due to my gpu's (GeForce RTX 3050 compute capability 8.0) limited double precision capability. If you run them on a more specialized hardware, let me know.

### Performance Debugging with NCU

I use NVIDIA Nsight Compute for performance analysis. This is not a requisite but I find it super-useful.

```bash
# Memory bandwidth analysis
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum \
    --csv ./benchmark > memory_analysis.csv

# Occupancy analysis
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    --kernel-regex "elementwise" ./benchmark

# L1 cache analysis (important for f64 optimization)
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
    --kernel-regex "f64" ./benchmark

# Instruction throughput
ncu --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum \
    ./benchmark
```

## Testing Checklist

Before submitting a new kernel, make sure:

### Correctness
- [ ] Kernel produces identical results to PyTorch (within 1e-3 tolerance for f32, 1e-6 for f64)
- [ ] Forward pass validation passes
- [ ] Gradient validation passes (if applicable)
- [ ] Edge cases tested (NaN, infinity, zero, large tensors)
- [ ] All data types supported (f32, f64)

### Performance
- [ ] Benchmarked in `benchmark.cu` suite
- [ ] NCU profiling completed
- [ ] Memory bandwidth utilization > 80% of peak
- [ ] Launch configuration matches `backend/cuda/ops.rs` exactly
- [ ] F64 vectorization applied where beneficial

### Integration
- [ ] Rust backend integration tested
- [ ] Error handling covers CUDA launch failures
- [ ] Unit tests added to appropriate `src/ops/` module
- [ ] Documentation updated
