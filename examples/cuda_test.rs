// examples/cuda_test.rs
#[cfg(feature = "cuda")]
use cudarc::driver::{LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use ferrox::backend::cuda::CudaBackend;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA backend
    let mut cuda_backend = CudaBackend::new(0)?;
    let device = cuda_backend.device().clone();

    // Test vector addition
    test_vector_addition(&cuda_backend)?;

    // Test ReLU activation
    test_relu_activation(&cuda_backend)?;

    // Test benchmark
    benchmark_kernels(&cuda_backend)?;

    println!("All CUDA tests passed!");
    Ok(())
}

#[cfg(feature = "cuda")]
fn test_vector_addition(backend: &CudaBackend) -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing vector addition...");

    let size = 1024;
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

    // Allocate GPU memory
    let a_gpu = backend.device().htod_copy(a.clone())?;
    let b_gpu = backend.device().htod_copy(b.clone())?;
    let mut c_gpu = backend.device().alloc_zeros::<f32>(size)?;

    // Launch configuration
    let cfg = LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (((size + 255) / 256).try_into().unwrap(), 1, 1),
        shared_mem_bytes: 0,
    };

    // Method 1: Using specific kernel method (recommended)
    backend
        .kernels()
        .launch_add(cfg, &a_gpu, &b_gpu, &mut c_gpu, size as i32)?;

    // Synchronize and get results
    backend.synchronize()?;
    let result = backend.device().dtoh_sync_copy(&c_gpu)?;

    // Verify results
    for i in 0..size.min(10) {
        let expected = a[i] + b[i];
        let actual = result[i];
        assert!(
            (expected - actual).abs() < 1e-6,
            "Mismatch at index {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }

    println!("✓ Vector addition test passed");
    Ok(())
}

#[cfg(feature = "cuda")]
/// Tests the ReLU activation function on the GPU
fn test_relu_activation(backend: &CudaBackend) -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing ReLU activation...");

    let size = 1024;
    let input: Vec<f32> = (0..size).map(|i| i as f32 - 512.0).collect(); // Mix of positive and negative

    // Allocate GPU memory
    let input_gpu = backend.device().htod_copy(input.clone())?;
    let mut output_gpu = backend.device().alloc_zeros::<f32>(size)?;

    // Launch configuration
    let cfg = LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (((size + 255) / 256).try_into().unwrap(), 1, 1),
        shared_mem_bytes: 0,
    };

    // Method 2: Using specific kernel method (alternative approach)
    backend
        .kernels()
        .launch_relu(cfg, &input_gpu, &mut output_gpu, size as i32)?;

    // Synchronize and get results
    backend.synchronize()?;
    let result = backend.device().dtoh_sync_copy(&output_gpu)?;

    // Verify results (ReLU should clamp negative values to 0)
    for i in 0..size.min(10) {
        let expected = input[i].max(0.0);
        let actual = result[i];
        assert!(
            (expected - actual).abs() < 1e-6,
            "ReLU mismatch at index {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }

    println!("✓ ReLU activation test passed");
    Ok(())
}

#[cfg(feature = "cuda")]
/// Benchmarks the CUDA kernels by running a simple vector addition multiple times
fn benchmark_kernels(backend: &CudaBackend) -> Result<(), Box<dyn std::error::Error>> {
    println!("Benchmarking kernels...");

    let size = 1_000_000;
    let iterations = 100;

    // Prepare data
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

    let a_gpu = backend.device().htod_copy(a)?;
    let b_gpu = backend.device().htod_copy(b)?;
    let mut c_gpu = backend.device().alloc_zeros::<f32>(size)?;

    let cfg = LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (((size + 255) / 256).try_into().unwrap(), 1, 1),
        shared_mem_bytes: 0,
    };

    // Warm up
    backend
        .kernels()
        .launch_add(cfg, &a_gpu, &b_gpu, &mut c_gpu, size as i32)?;
    backend.synchronize()?;

    // Benchmark
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        // Method 3: Direct cloning in a loop (shows the pattern)
        let add_kernel = backend
            .kernels()
            .get_function_cloned("add")
            .ok_or("Add kernel not found")?;

        unsafe {
            add_kernel.launch(cfg, (&a_gpu, &b_gpu, &mut c_gpu, size as i32))?;
        }
    }
    backend.synchronize()?;
    let elapsed = start.elapsed();

    let avg_time = elapsed.as_secs_f64() / iterations as f64;
    let throughput = (size as f64 * 3.0 * 4.0) / (avg_time * 1e9); // 3 arrays * 4 bytes/float / time in GB/s

    println!("✓ Benchmark results:");
    println!("  Average kernel time: {:.3} ms", avg_time * 1000.0);
    println!("  Memory throughput: {:.2} GB/s", throughput);

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature is not enabled. Please enable it to run CUDA tests.");
}
