// examples/cuda_test.rs
#[cfg(feature = "cuda")]
use ferrox::backend::{Device, cuda, CudaBackend};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced CUDA Test - Kernel Execution");
    println!("==========================================");

    // Test 1: Basic CUDA device creation
    println!("\n1. Testing CUDA Device Creation...");
    let cuda_device = cuda(0);
    println!("✓ Created CUDA device: {:?}", cuda_device);
    
    // Test 2: Create CUDA backend with kernels
    println!("\n2. Testing CUDA Backend Initialization...");
    let mut backend = CudaBackend::new(0)?;
    println!("✓ CUDA Backend created: {}", backend.name());
    println!("✓ Device ID: {}", backend.device_id());
    
    // Test 3: Verify kernels are loaded
    println!("\n3. Testing Kernel Loading...");
    let kernels = backend.kernels();
    
    // Check if kernels are available
    let available_kernels = ["add", "matmul", "relu"];
    for kernel_name in &available_kernels {
        if let Some(_) = kernels.get_function(kernel_name) {
            println!("✓ Kernel '{}' loaded successfully", kernel_name);
        } else {
            println!("✗ Kernel '{}' not found", kernel_name);
        }
    }
    
    // Test 4: Launch ADD kernel
    println!("\n4. Testing ADD Kernel Execution...");
    test_add_kernel(&backend)?;
    
    // Test 5: Launch RELU kernel  
    println!("\n5. Testing RELU Kernel Execution...");
    test_relu_kernel(&backend)?;
    
    // Test 6: Performance benchmark
    println!("\n6. Performance Benchmark...");
    benchmark_kernels(&backend)?;
    
    // Test 7: Memory operations
    println!("\n7. Testing GPU Memory Operations...");
    test_memory_operations(&backend)?;
    
    println!("\n🎉 All CUDA tests completed successfully!");
    Ok(())
}

#[cfg(feature = "cuda")]
fn test_add_kernel(backend: &CudaBackend) -> Result<(), Box<dyn std::error::Error>> {
    let device = backend.device();
    let kernels = backend.kernels();
    
    // Prepare test data
    let size = 1024;
    let a_host = vec![1.0f32; size];
    let b_host = vec![2.0f32; size];
    
    // Allocate GPU memory
    let a_gpu = device.htod_copy(a_host.clone())?;
    let b_gpu = device.htod_copy(b_host.clone())?;
    let mut c_gpu = device.alloc_zeros::<f32>(size)?;
    
    // Get the add kernel
    if let Some(add_kernel) = kernels.get_function("add") {
        // Configure launch parameters
        let threads_per_block = 256;
        let blocks = (size + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig::for_num_elems(size as u32);
        
        // Launch kernel: c = a + b
        unsafe {
            add_kernel.launch(
                cfg,
                (
                    &a_gpu,
                    &b_gpu, 
                    &mut c_gpu,
                    size as i32,
                )
            )?;
        }
        
        // Synchronize and copy result back
        device.synchronize()?;
        let result = device.dtoh_sync_copy(&c_gpu)?;
        
        // Verify results
        let expected = 3.0f32; // 1.0 + 2.0
        let all_correct = result.iter().all(|&x| (x - expected).abs() < 1e-6);
        
        if all_correct {
            println!("✓ ADD kernel: {} elements processed correctly", size);
            println!("  Sample: {} + {} = {} ✓", a_host[0], b_host[0], result[0]);
        } else {
            println!("✗ ADD kernel failed verification");
            println!("  Expected: {}, Got: {}", expected, result[0]);
        }
    } else {
        println!("✗ ADD kernel not available");
    }
    
    Ok(())
}

#[cfg(feature = "cuda")]
fn test_relu_kernel(backend: &CudaBackend) -> Result<(), Box<dyn std::error::Error>> {
    let device = backend.device();
    let kernels = backend.kernels();
    
    // Prepare test data with negative and positive values
    let size = 1024;
    let input_host: Vec<f32> = (0..size)
        .map(|i| if i % 2 == 0 { -(i as f32) } else { i as f32 })
        .collect();
    
    // Allocate GPU memory
    let input_gpu = device.htod_copy(input_host.clone())?;
    let mut output_gpu = device.alloc_zeros::<f32>(size)?;
    
    // Get the ReLU kernel
    if let Some(relu_kernel) = kernels.get_function("relu") {
        let cfg = LaunchConfig::for_num_elems(size as u32);
        
        // Launch kernel: output = max(0, input)
        unsafe {
            relu_kernel.launch(
                cfg,
                (
                    &input_gpu,
                    &mut output_gpu,
                    size as i32,
                )
            )?;
        }
        
        // Synchronize and copy result back
        device.synchronize()?;
        let result = device.dtoh_sync_copy(&output_gpu)?;
        
        // Verify ReLU: negative values should be 0, positive unchanged
        let mut correct_count = 0;
        for (i, (&input, &output)) in input_host.iter().zip(result.iter()).enumerate() {
            let expected = if input < 0.0 { 0.0 } else { input };
            if (output - expected).abs() < 1e-6 {
                correct_count += 1;
            } else if i < 10 { // Show first few errors
                println!("  Error at {}: input={}, expected={}, got={}", i, input, expected, output);
            }
        }
        
        println!("✓ ReLU kernel: {}/{} elements correct", correct_count, size);
        if correct_count == size {
            println!("  Sample: ReLU({}) = {} ✓", input_host[0], result[0]);
            println!("  Sample: ReLU({}) = {} ✓", input_host[1], result[1]);
        }
    } else {
        println!("✗ ReLU kernel not available");
    }
    
    Ok(())
}

#[cfg(feature = "cuda")]
fn benchmark_kernels(backend: &CudaBackend) -> Result<(), Box<dyn std::error::Error>> {
    let device = backend.device();
    let sizes = vec![1024, 10240, 102400, 1024000];
    
    println!("Benchmarking kernel performance:");
    println!("Size\t\tTime (ms)\tGBPS");
    println!("--------------------------------");
    
    for &size in &sizes {
        let start = Instant::now();
        
        // Create test data
        let a_host = vec![1.0f32; size];
        let b_host = vec![2.0f32; size];
        
        // GPU operations
        let a_gpu = device.htod_copy(a_host)?;
        let b_gpu = device.htod_copy(b_host)?;
        let mut c_gpu = device.alloc_zeros::<f32>(size)?;
        
        // Launch ADD kernel if available
        if let Some(add_kernel) = backend.kernels().get_function("add") {
            let cfg = LaunchConfig::for_num_elems(size as u32);
            unsafe {
                add_kernel.launch(cfg, (&a_gpu, &b_gpu, &mut c_gpu, size as i32))?;
            }
        }
        
        device.synchronize()?;
        let _result = device.dtoh_sync_copy(&c_gpu)?;
        
        let elapsed = start.elapsed();
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        
        // Calculate bandwidth (3 arrays * 4 bytes/float * size)
        let bytes_transferred = 3 * 4 * size;
        let gbps = (bytes_transferred as f64) / (elapsed.as_secs_f64() * 1e9);
        
        println!("{}\t\t{:.2}\t\t{:.2}", size, elapsed_ms, gbps);
    }
    
    Ok(())
}

#[cfg(feature = "cuda")]
fn test_memory_operations(backend: &CudaBackend) -> Result<(), Box<dyn std::error::Error>> {
    let device = backend.device();
    
    // Test different memory operations
    let size = 1000;
    
    // Test 1: Host to Device copy
    println!("Testing memory operations with {} elements:", size);
    let host_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    
    let start = Instant::now();
    let gpu_data = device.htod_copy(host_data.clone())?;
    let h2d_time = start.elapsed();
    
    // Test 2: Device to Host copy
    let start = Instant::now();
    let retrieved_data = device.dtoh_sync_copy(&gpu_data)?;
    let d2h_time = start.elapsed();
    
    // Test 3: GPU allocation
    let start = Instant::now();
    let _zeros_gpu = device.alloc_zeros::<f32>(size)?;
    let alloc_time = start.elapsed();
    
    // Verify data integrity
    let data_matches = host_data.iter()
        .zip(retrieved_data.iter())
        .all(|(a, b)| (a - b).abs() < 1e-6);
    
    println!("✓ Host→Device copy: {:.2}ms", h2d_time.as_secs_f64() * 1000.0);
    println!("✓ Device→Host copy: {:.2}ms", d2h_time.as_secs_f64() * 1000.0);
    println!("✓ GPU allocation: {:.2}ms", alloc_time.as_secs_f64() * 1000.0);
    println!("✓ Data integrity: {}", if data_matches { "PASS" } else { "FAIL" });
    
    // Calculate memory bandwidth
    let bytes = size * std::mem::size_of::<f32>();
    let h2d_bandwidth = bytes as f64 / (h2d_time.as_secs_f64() * 1e9);
    let d2h_bandwidth = bytes as f64 / (d2h_time.as_secs_f64() * 1e9);
    
    println!("  H2D Bandwidth: {:.2} GB/s", h2d_bandwidth);
    println!("  D2H Bandwidth: {:.2} GB/s", d2h_bandwidth);
    
    Ok(())
}

// If CUDA feature is not enabled, provide a message
#[cfg(not(feature = "cuda"))]
fn main() {
    println!("❌ CUDA feature is not enabled.");
    println!("💡 Compile with --features cuda to enable CUDA support.");
    println!("📝 Example: cargo run --features cuda --example cuda_test");
}