#[cfg(feature = "cuda")]
/// Comprehensive CUDA benchmark with multiple kernel tests and performance metrics
fn benchmark_kernels(backend: &CudaBackend) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CUDA Kernels Benchmark ===\n");

    // Test configurations
    let sizes = [1_000, 100_000, 1_000_000];
    let iterations = 100;

    for &size in &sizes {
        println!("Testing with {} elements:", size);
        
        // Test 1: Vector Addition
        benchmark_vector_addition(backend, size, iterations)?;
        
        // Test 2: ReLU Activation
        benchmark_relu(backend, size, iterations)?;
        
        // Test 3: Matrix Multiplication (if size allows)
        if size <= 1024 * 1024 {
            let dim = (size as f64).sqrt() as usize;
            if dim * dim == size {
                benchmark_matmul(backend, dim, iterations / 10)?; // Fewer iterations for matmul
            }
        }
        
        println!();
    }

    // Test 4: Memory Transfer Benchmark
    benchmark_memory_transfers(backend)?;
    
    println!("✓ All benchmarks completed successfully!");
    Ok(())
}

#[cfg(feature = "cuda")]
/// Benchmark vector addition with detailed metrics
fn benchmark_vector_addition(
    backend: &CudaBackend, 
    size: usize, 
    iterations: usize
) -> Result<(), Box<dyn std::error::Error>> {
    
    // Prepare data
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

    let a_gpu = backend.memory_manager().host_to_device(a.clone())?;
    let b_gpu = backend.memory_manager().host_to_device(b.clone())?;
    let mut c_gpu = backend.memory_manager().alloc_zeros::<f32>(size)?;

    let cfg = LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (((size + 255) / 256).try_into().unwrap(), 1, 1),
        shared_mem_bytes: 0,
    };

    // Warm up
    backend.kernels().launch_add(cfg, &a_gpu, &b_gpu, &mut c_gpu, size as i32)?;
    backend.synchronize()?;

    // Benchmark using high-level API
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        backend.kernels().launch_add(cfg, &a_gpu, &b_gpu, &mut c_gpu, size as i32)?;
    }
    backend.synchronize()?;
    let high_level_time = start.elapsed();

    // Benchmark using low-level API for comparison
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let add_kernel = backend
            .kernels()
            .get_function_cloned("elementwise_add")
            .ok_or("Add kernel not found")?;

        unsafe {
            backend.context().default_stream()
                .launch_builder(&add_kernel)
                .arg(&a_gpu)
                .arg(&b_gpu) 
                .arg(&mut c_gpu)
                .arg(&(size as i32))
                .launch(cfg)?;
        }
    }
    backend.synchronize()?;
    let low_level_time = start.elapsed();

    // Verify correctness
    let result = backend.memory_manager().device_to_host(&c_gpu)?;
    for i in 0..10.min(size) {
        let expected = a[i] + b[i];
        assert!((result[i] - expected).abs() < 1e-6, "Incorrect result at index {}", i);
    }

    // Calculate metrics
    let high_level_avg = high_level_time.as_secs_f64() / iterations as f64;
    let low_level_avg = low_level_time.as_secs_f64() / iterations as f64;
    let memory_bandwidth = (size as f64 * 3.0 * 4.0) / (high_level_avg * 1e9); // GB/s
    
    println!("  Vector Addition:");
    println!("    High-level API: {:.3} ms/op", high_level_avg * 1000.0);
    println!("    Low-level API:  {:.3} ms/op", low_level_avg * 1000.0);
    println!("    Memory BW:      {:.2} GB/s", memory_bandwidth);
    
    Ok(())
}

#[cfg(feature = "cuda")]
/// Benchmark ReLU activation
fn benchmark_relu(
    backend: &CudaBackend, 
    size: usize, 
    iterations: usize
) -> Result<(), Box<dyn std::error::Error>> {
    
    let input: Vec<f32> = (0..size).map(|i| i as f32 - (size as f32 / 2.0)).collect();
    let input_gpu = backend.memory_manager().host_to_device(input.clone())?;
    let mut output_gpu = backend.memory_manager().alloc_zeros::<f32>(size)?;

    let cfg = LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (((size + 255) / 256).try_into().unwrap(), 1, 1),
        shared_mem_bytes: 0,
    };

    // Warm up
    backend.kernels().launch_relu(cfg, &input_gpu, &mut output_gpu, size as i32)?;
    backend.synchronize()?;

    // Benchmark
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        backend.kernels().launch_relu(cfg, &input_gpu, &mut output_gpu, size as i32)?;
    }
    backend.synchronize()?;
    let elapsed = start.elapsed();

    // Verify correctness
    let result = backend.memory_manager().device_to_host(&output_gpu)?;
    for i in 0..10.min(size) {
        let expected = input[i].max(0.0);
        assert!((result[i] - expected).abs() < 1e-6, "Incorrect ReLU at index {}", i);
    }

    let avg_time = elapsed.as_secs_f64() / iterations as f64;
    let memory_bandwidth = (size as f64 * 2.0 * 4.0) / (avg_time * 1e9); // GB/s
    
    println!("  ReLU Activation:");
    println!("    Average time:   {:.3} ms/op", avg_time * 1000.0);
    println!("    Memory BW:      {:.2} GB/s", memory_bandwidth);
    
    Ok(())
}

#[cfg(feature = "cuda")]
/// Benchmark matrix multiplication
fn benchmark_matmul(
    backend: &CudaBackend, 
    dim: usize, 
    iterations: usize
) -> Result<(), Box<dyn std::error::Error>> {
    
    let size = dim * dim;
    let a: Vec<f32> = (0..size).map(|i| (i % 100) as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| ((i + 50) % 100) as f32).collect();
    
    let a_gpu = backend.memory_manager().host_to_device(a)?;
    let b_gpu = backend.memory_manager().host_to_device(b)?;
    let mut c_gpu = backend.memory_manager().alloc_zeros::<f32>(size)?;

    let cfg = LaunchConfig {
        block_dim: (16, 16, 1),
        grid_dim: (
            ((dim + 15) / 16).try_into().unwrap(), 
            ((dim + 15) / 16).try_into().unwrap(), 
            1
        ),
        shared_mem_bytes: 0,
    };

    // Warm up
    backend.kernels().launch_matmul(
        cfg, &a_gpu, &b_gpu, &mut c_gpu, 
        dim as i32, dim as i32, dim as i32
    )?;
    backend.synchronize()?;

    // Benchmark
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        backend.kernels().launch_matmul(
            cfg, &a_gpu, &b_gpu, &mut c_gpu, 
            dim as i32, dim as i32, dim as i32
        )?;
    }
    backend.synchronize()?;
    let elapsed = start.elapsed();

    let avg_time = elapsed.as_secs_f64() / iterations as f64;
    let flops = 2.0 * (dim as f64).powi(3); // 2*N^3 for N×N matmul
    let gflops = flops / (avg_time * 1e9);
    
    println!("  Matrix Mul {}×{}:", dim, dim);
    println!("    Average time:   {:.3} ms/op", avg_time * 1000.0);
    println!("    Performance:    {:.2} GFLOPS", gflops);
    
    Ok(())
}

#[cfg(feature = "cuda")]
/// Benchmark memory transfer operations
fn benchmark_memory_transfers(backend: &CudaBackend) -> Result<(), Box<dyn std::error::Error>> {
    println!("Memory Transfer Benchmark:");
    
    let sizes = [1_000, 100_000, 10_000_000];
    let iterations = 20;
    
    for &size in &sizes {
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        
        // Host to Device
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _gpu_data = backend.memory_manager().host_to_device(data.clone())?;
        }
        let h2d_time = start.elapsed().as_secs_f64() / iterations as f64;
        
        // Device to Host (reuse last allocation)
        let gpu_data = backend.memory_manager().host_to_device(data.clone())?;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _result = backend.memory_manager().device_to_host(&gpu_data)?;
        }
        let d2h_time = start.elapsed().as_secs_f64() / iterations as f64;
        
        let memory_size_mb = (size * 4) as f64 / 1e6; // MB
        let h2d_bandwidth = memory_size_mb / (h2d_time * 1000.0); // GB/s
        let d2h_bandwidth = memory_size_mb / (d2h_time * 1000.0); // GB/s
        
        println!("  {} elements ({:.1} MB):", size, memory_size_mb);
        println!("    H2D: {:.3} ms ({:.2} GB/s)", h2d_time * 1000.0, h2d_bandwidth);
        println!("    D2H: {:.3} ms ({:.2} GB/s)", d2h_time * 1000.0, d2h_bandwidth);
    }
    
    Ok(())
}



#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Ferrox CUDA Test Suite");
    println!("========================\n");

    // Initialize CUDA backend
    println!("Initializing CUDA backend...");
    let cuda_backend = match CudaBackend::new(0) {
        Ok(backend) => {
            println!("CUDA backend initialized successfully");
            println!("   Device: {}", backend.name());
            backend
        },
        Err(e) => {
            eprintln!("Failed to initialize CUDA backend: {}", e);
            eprintln!("   Make sure you have CUDA drivers installed and a compatible GPU");
            return Err(e.into());
        }
    };

    // Print loaded kernels info
    let loaded_kernels = cuda_backend.kernels().loaded_kernels();
    println!("   Loaded {} CUDA kernels\n", loaded_kernels.len());

    // Run all tests
    let mut tests_passed = 0;
    let mut tests_failed = 0;

    // Test 1: Basic vector addition
    println!("Test 1: Vector Addition");
    match test_vector_addition(&cuda_backend) {
        Ok(_) => {
            println!("Vector addition test passed\n");
            tests_passed += 1;
        },
        Err(e) => {
            eprintln!("Vector addition test failed: {}\n", e);
            tests_failed += 1;
        }
    }

    // Test 2: ReLU activation
    println!("Test 2: ReLU Activation");
    match test_relu_activation(&cuda_backend) {
        Ok(_) => {
            println!("ReLU activation test passed\n");
            tests_passed += 1;
        },
        Err(e) => {
            eprintln!("ReLU activation test failed: {}\n", e);
            tests_failed += 1;
        }
    }

    // Test 3: Matrix multiplication
    println!("Test 3: Matrix Multiplication");
    match test_matrix_multiplication(&cuda_backend) {
        Ok(_) => {
            println!("Matrix multiplication test passed\n");
            tests_passed += 1;
        },
        Err(e) => {
            eprintln!("Matrix multiplication test failed: {}\n", e);
            tests_failed += 1;
        }
    }

    // Test 4: Multiple operations chaining
    println!("Test 4: Operation Chaining");
    match test_operation_chaining(&cuda_backend) {
        Ok(_) => {
            println!("Operation chaining test passed\n");
            tests_passed += 1;
        },
        Err(e) => {
            eprintln!("Operation chaining test failed: {}\n", e);
            tests_failed += 1;
        }
    }

    // Performance benchmarks
    println!("⚡ Performance Benchmarks");
    match benchmark_kernels(&cuda_backend) {
        Ok(_) => {
            println!("Benchmarks completed successfully\n");
        },
        Err(e) => {
            eprintln!("⚠️  Benchmarks failed: {}\n", e);
        }
    }

    // Summary
    println!("Test Summary");
    println!("===============");
    println!("Tests passed: {}", tests_passed);
    println!("Tests failed: {}", tests_failed);
    
    if tests_failed == 0 {
        println!(" All tests passed! CUDA backend is working correctly.");
        Ok(())
    } else {
        eprintln!(" {} test(s) failed. Please check your CUDA setup.", tests_failed);
        std::process::exit(1);
    }
}

#[cfg(feature = "cuda")]
fn test_vector_addition(backend: &CudaBackend) -> Result<(), Box<dyn std::error::Error>> {
    let size = 1024;
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

    // Allocate GPU memory
    let a_gpu = backend.memory_manager().host_to_device(a.clone())?;
    let b_gpu = backend.memory_manager().host_to_device(b.clone())?;
    let mut c_gpu = backend.memory_manager().alloc_zeros::<f32>(size)?;

    // Launch configuration
    let cfg = LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (((size + 255) / 256).try_into().unwrap(), 1, 1),
        shared_mem_bytes: 0,
    };

    // Execute kernel
    backend.kernels().launch_add(cfg, &a_gpu, &b_gpu, &mut c_gpu, size as i32)?;
    backend.synchronize()?;
    
    // Get results and verify
    let result = backend.memory_manager().device_to_host(&c_gpu)?;
    for i in 0..size.min(10) {
        let expected = a[i] + b[i];
        let actual = result[i];
        assert!(
            (expected - actual).abs() < 1e-6,
            "Mismatch at index {}: expected {}, got {}",
            i, expected, actual
        );
    }

    println!("   ✓ Computed {} element additions", size);
    Ok(())
}

#[cfg(feature = "cuda")]
fn test_relu_activation(backend: &CudaBackend) -> Result<(), Box<dyn std::error::Error>> {
    let size = 1024;
    let input: Vec<f32> = (0..size).map(|i| i as f32 - 512.0).collect(); // Mix of positive/negative

    let input_gpu = backend.memory_manager().host_to_device(input.clone())?;
    let mut output_gpu = backend.memory_manager().alloc_zeros::<f32>(size)?;

    let cfg = LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (((size + 255) / 256).try_into().unwrap(), 1, 1),
        shared_mem_bytes: 0,
    };

    backend.kernels().launch_relu(cfg, &input_gpu, &mut output_gpu, size as i32)?;
    backend.synchronize()?;
    
    let result = backend.memory_manager().device_to_host(&output_gpu)?;
    for i in 0..size.min(10) {
        let expected = input[i].max(0.0);
        let actual = result[i];
        assert!(
            (expected - actual).abs() < 1e-6,
            "ReLU mismatch at index {}: expected {}, got {}",
            i, expected, actual
        );
    }

    println!("   ✓ Applied ReLU to {} elements", size);
    Ok(())
}

#[cfg(feature = "cuda")]
fn test_matrix_multiplication(backend: &CudaBackend) -> Result<(), Box<dyn std::error::Error>> {
    let dim = 64; // 64x64 matrices
    let size = dim * dim;
    
    // Create test matrices
    let a: Vec<f32> = (0..size).map(|i| (i % 10) as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| ((i + 5) % 10) as f32).collect();
    
    let a_gpu = backend.memory_manager().host_to_device(a.clone())?;
    let b_gpu = backend.memory_manager().host_to_device(b.clone())?;
    let mut c_gpu = backend.memory_manager().alloc_zeros::<f32>(size)?;

    let cfg = LaunchConfig {
        block_dim: (16, 16, 1),
        grid_dim: (
            ((dim + 15) / 16).try_into().unwrap(), 
            ((dim + 15) / 16).try_into().unwrap(), 
            1
        ),
        shared_mem_bytes: 0,
    };

    backend.kernels().launch_matmul(
        cfg, &a_gpu, &b_gpu, &mut c_gpu,
        dim as i32, dim as i32, dim as i32
    )?;
    backend.synchronize()?;
    
    let result = backend.memory_manager().device_to_host(&c_gpu)?;
    
    // Verify a few elements (simple sanity check)
    assert!(result.len() == size, "Result size mismatch");
    assert!(result[0] != 0.0, "Matrix multiplication produced zero result");
    
    println!("   ✓ Multiplied {}×{} matrices", dim, dim);
    Ok(())
}

#[cfg(feature = "cuda")]
fn test_operation_chaining(backend: &CudaBackend) -> Result<(), Box<dyn std::error::Error>> {
    let size = 512;
    let input: Vec<f32> = (0..size).map(|i| i as f32 - 256.0).collect();

    let input_gpu = backend.memory_manager().host_to_device(input.clone())?;
    let mut temp_gpu = backend.memory_manager().alloc_zeros::<f32>(size)?;
    let mut output_gpu = backend.memory_manager().alloc_zeros::<f32>(size)?;

    let cfg = LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (((size + 255) / 256).try_into().unwrap(), 1, 1),
        shared_mem_bytes: 0,
    };

    // Chain: input -> ReLU -> add with itself (double positive values)
    backend.kernels().launch_relu(cfg, &input_gpu, &mut temp_gpu, size as i32)?;
    backend.synchronize()?;
    
    backend.kernels().launch_add(cfg, &temp_gpu, &temp_gpu, &mut output_gpu, size as i32)?;
    backend.synchronize()?;
    
    let result = backend.memory_manager().device_to_host(&output_gpu)?;
    
    // Verify chained operations
    for i in 0..size.min(10) {
        let expected = input[i].max(0.0) * 2.0;
        let actual = result[i];
        assert!(
            (expected - actual).abs() < 1e-6,
            "Chained ops mismatch at index {}: expected {}, got {}",
            i, expected, actual
        );
    }

    println!("   ✓ Chained ReLU + Addition on {} elements", size);
    Ok(())
}

// [Include the benchmark_kernels function and its helper functions from the previous artifact here]

#[cfg(not(feature = "cuda"))]
fn main() {
    println!(" CUDA feature is not enabled!");
    println!("   To run CUDA tests, compile with: cargo run --features cuda --example cuda_test");
    println!("   Make sure you have:");
    println!("   - NVIDIA GPU with CUDA support");
    println!("   - CUDA toolkit installed");
    println!("   - Proper CUDA drivers");
}