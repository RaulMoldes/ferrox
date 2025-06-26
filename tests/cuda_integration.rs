#![cfg(test)]
#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
#[cfg(feature = "cuda")]
use ferrox::backend::Device;
#[cfg(feature = "cuda")]
use ferrox::backend::cuda::{CudaBackend, CudaTensor};
#[cfg(feature = "cuda")]
use ferrox::backend::cuda::{CudaKernels, load_all_kernels};
#[cfg(feature = "cuda")]
use ferrox::backend::manager::get_backend;
#[cfg(feature = "cuda")]
use ferrox::tensor::Tensor;
#[cfg(feature = "cuda")]
use ndarray::{ArrayD, IxDyn};

#[cfg(feature = "cuda")]
#[test]
fn test_basic_cuda_tensor_operations() {
    // Test tensor creation and device transfer
    let cpu_tensor = Tensor::new(ndarray::arr2(&[[1.0f32, 2.0], [3.0, 4.0]]).into_dyn());
    println!("Created CPU tensor: {:?}", cpu_tensor.shape());

    // Transfer to CUDA
    match cpu_tensor.to_cuda() {
        Ok(cuda_tensor) => {
            println!("Successfully moved tensor to CUDA");
            assert!(cuda_tensor.is_cuda());

            // Test basic arithmetic on CUDA
            test_cuda_arithmetic(&cuda_tensor);
        }
        Err(e) => {
            println!("CUDA not available: {}", e);
            // Skip test if CUDA is not available
        }
    }
}

#[cfg(feature = "cuda")]
fn test_cuda_arithmetic(tensor: &Tensor<f32>) {
    // Test addition
    match tensor.add_cuda(tensor) {
        Ok(result) => {
            println!("CUDA addition successful");
            // Convert back to CPU to verify results
            let cpu_result = result.to_cpu().unwrap();
            println!("Addition result shape: {:?}", cpu_result.shape());
        }
        Err(e) => panic!("CUDA addition failed: {}", e),
    }

    // Test scalar multiplication
    match tensor.mul_scalar_cuda(2.0) {
        Ok(result) => {
            println!("CUDA scalar multiplication successful");
        }
        Err(e) => panic!("CUDA scalar multiplication failed: {}", e),
    }

    // Test ReLU activation
    match tensor.relu_cuda() {
        Ok(result) => {
            println!("CUDA ReLU successful");
        }
        Err(e) => panic!("CUDA ReLU failed: {}", e),
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_vs_cpu_consistency() {
    let data = ndarray::arr1(&[1.0f32, -2.0, 3.0, -4.0, 5.0]).into_dyn();
    let cpu_tensor = Tensor::new(data);

    if let Ok(cuda_tensor) = cpu_tensor.to_cuda() {
        // Test that CPU and CUDA operations give same results
        let cpu_relu = cpu_tensor.relu();
        let cuda_relu = cuda_tensor.relu_cuda().unwrap().to_cpu().unwrap();

        // Compare results (implement approximate equality)
        assert_tensors_approx_equal(&cpu_relu, &cuda_relu, 1e-6);
        println!("CPU and CUDA ReLU results match!");
    }
}

#[cfg(feature = "cuda")]
fn assert_tensors_approx_equal(a: &Tensor<f32>, b: &Tensor<f32>, tolerance: f32) {
    assert_eq!(a.shape(), b.shape());

    for (val_a, val_b) in a.data.iter().zip(b.data.iter()) {
        assert!(
            (val_a - val_b).abs() < tolerance,
            "Values differ: {} vs {}",
            val_a,
            val_b
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_large_tensor_operations() {
    // Test with larger tensors to ensure memory handling works
    let size = 1_000_000;
    let data = ndarray::Array::zeros((size,)).into_dyn();
    let cpu_tensor = Tensor::new(data);

    match cpu_tensor.to_cuda() {
        Ok(cuda_tensor) => {
            println!("Successfully allocated {} element tensor on GPU", size);

            // Test operations don't crash with large tensors
            let result = cuda_tensor.add_scalar_cuda(1.0);
            assert!(result.is_ok(), "Large tensor scalar addition failed");
        }
        Err(e) => {
            println!(
                "Large tensor allocation failed (expected on low-memory GPUs): {}",
                e
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_shape_mismatch_errors() {
    let tensor_a = Tensor::new(ndarray::arr2(&[[1.0f32, 2.0], [3.0, 4.0]]).into_dyn());
    let tensor_b = Tensor::new(ndarray::arr1(&[1.0f32, 2.0, 3.0]).into_dyn());

    if let (Ok(cuda_a), Ok(cuda_b)) = (tensor_a.to_cuda(), tensor_b.to_cuda()) {
        // This should fail due to shape mismatch
        let result = cuda_a.add_cuda(&cuda_b);
        assert!(result.is_err(), "Shape mismatch should cause error");
        println!("Correctly detected shape mismatch: {:?}", result.err());
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_multiple_device_transfers() {
    let cpu_tensor = Tensor::new(ndarray::arr1(&[1.0f32, 2.0, 3.0]).into_dyn());

    // Multiple transfers should work correctly
    for i in 0..5 {
        match cpu_tensor.to_cuda() {
            Ok(cuda_tensor) => {
                let back_to_cpu = cuda_tensor.to_cpu().unwrap();

                // Verify data integrity after transfer
                for (original, transferred) in cpu_tensor.data.iter().zip(back_to_cpu.data.iter()) {
                    assert!(
                        (original - transferred).abs() < 1e-6,
                        "Data corruption after transfer {}: {} vs {}",
                        i,
                        original,
                        transferred
                    );
                }
            }
            Err(e) => {
                println!("Transfer {} failed: {}", i, e);
                break;
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_mixed_device_operations() {
    let cpu_tensor = Tensor::new(ndarray::arr1(&[1.0f32, 2.0, 3.0]).into_dyn());

    if let Ok(cuda_tensor) = cpu_tensor.to_cuda() {
        // Test automatic fallback to CPU when CUDA operation fails
        let result = cuda_tensor.add(&cpu_tensor); // Should handle device mismatch gracefully

        // The add method should either work or fail gracefully
        match result {
            Ok(_) => println!("Mixed device operation succeeded"),
            Err(e) => println!("Mixed device operation failed as expected: {}", e),
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_concurrent_cuda_operations() {
    use std::sync::Arc;
    use std::thread;

    let cpu_tensor = Arc::new(Tensor::new(ndarray::arr1(&[1.0f32, 2.0, 3.0]).into_dyn()));
    let mut handles = vec![];

    // Test concurrent CUDA operations
    for i in 0..4 {
        let tensor_clone = Arc::clone(&cpu_tensor);

        let handle = thread::spawn(move || {
            if let Ok(cuda_tensor) = tensor_clone.to_cuda() {
                // Each thread performs its own CUDA operations
                let result = cuda_tensor.mul_scalar_cuda(i as f32 + 1.0);
                match result {
                    Ok(_) => println!("Thread {} CUDA operation succeeded", i),
                    Err(e) => println!("Thread {} CUDA operation failed: {}", i, e),
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    println!("All concurrent operations completed");
}

#[test]
fn test_cuda_environment() {
    println!("=== CUDA Environment Debug ===");

    // Check if CUDA feature is enabled
    #[cfg(feature = "cuda")]
    println!("✓ CUDA feature is enabled");
    #[cfg(not(feature = "cuda"))]
    println!("✗ CUDA feature is NOT enabled");

    // Try direct cudarc device creation
    #[cfg(feature = "cuda")]
    {
        println!("Attempting direct CUDA device creation...");
        match cudarc::driver::CudaContext::new(0) {
            Ok(device) => {
                println!("✓ Direct CUDA device creation successful");
                println!("  Device name: {:?}", device.name());
            }
            Err(e) => {
                println!("✗ Direct CUDA device creation failed: {}", e);
            }
        }

        // Try backend manager
        println!("Checking backend manager...");
        let backend = get_backend();
        println!("Backend has CUDA: {}", backend.has_cuda());

        if let Some(cuda_backend) = backend.cuda_backend() {
            println!("✓ CUDA backend available in manager");
            println!("  Device ID: {}", cuda_backend.id());
            println!("  Device name: {}", cuda_backend.name());
        } else {
            println!("✗ CUDA backend NOT available in manager");
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_gpu_chaining_efficiency() {
    let a = Tensor::new(ndarray::arr1(&[1.0f32, 2.0, 3.0]).into_dyn());
    let b = Tensor::new(ndarray::arr1(&[2.0f32, 3.0, 4.0]).into_dyn());

    if let (Ok(cuda_a), Ok(cuda_b)) = (a.to_cuda(), b.to_cuda()) {
        // Chain operations on GPU - these should create GPU-only tensors
        if let Ok(result1) = cuda_a.add_cuda(&cuda_b) {
            if let Ok(result2) = result1.mul_scalar_cuda(2.0) {
                // Verify it's still on GPU and CPU data is empty
                assert!(result2.is_cuda());
                assert!(result2.cuda_storage.is_some());

                // Only transfer when needed
                let cpu_result = result2.to_cpu().unwrap();
                // (1+2)*2 = 6, (2+3)*2 = 10, (3+4)*2 = 14
                assert!((cpu_result.data()[[0]] - 6.0).abs() < 1e-6);
                assert!((cpu_result.data()[[1]] - 10.0).abs() < 1e-6);
                assert!((cpu_result.data()[[2]] - 14.0).abs() < 1e-6);

                println!("GPU chaining test passed!");
            } else {
                panic!("GPU chaining failed: could not multiply by scalar");
            }
        } else {
            panic!("GPU chaining failed: could not add tensors");
        }
    } else {
        println!("CUDA not available, skipping GPU chaining test");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_gpu_operations_create_gpu_only_tensors() {
    let a = Tensor::new(ndarray::arr1(&[1.0f32, 2.0]).into_dyn());
    let b = Tensor::new(ndarray::arr1(&[3.0f32, 4.0]).into_dyn());

    if let (Ok(cuda_a), Ok(cuda_b)) = (a.to_cuda(), b.to_cuda()) {
        // Test that CUDA operations create GPU-only results
        if let Ok(result) = cuda_a.add_cuda(&cuda_b) {
            assert!(result.is_cuda());
            assert!(result.cuda_storage.is_some());

            // Verify that accessing CPU data would require explicit conversion
            // (We can't test the panic here since it would fail the test)

            println!("GPU-only tensor creation test passed!");
        }
    } else {
        println!("CUDA not available, skipping GPU-only tensor test");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_backward_compatibility() {
    // Test that existing patterns still work
    let cpu_tensor = Tensor::new(ndarray::arr1(&[1.0f32, 2.0, 3.0]).into_dyn());

    // Traditional pattern should still work
    if let Ok(cuda_tensor) = cpu_tensor.to_cuda() {
        // This tensor has both CPU and GPU data
        assert!(cuda_tensor.is_cuda());
        assert!(cuda_tensor.cuda_storage.is_some());
        assert!(!cuda_tensor.data.is_empty()); // CPU data still exists

        // So indexing still works
        assert!((cuda_tensor[0] - 1.0).abs() < 1e-6);

        // And operations that return to CPU still work
        if let Ok(result) = cuda_tensor.add_cuda(&cuda_tensor) {
            if let Ok(cpu_result) = result.to_cpu() {
                assert!((cpu_result[0] - 2.0).abs() < 1e-6);
            }
        }

        println!("Backward compatibility test passed!");
    } else {
        println!("CUDA not available, skipping backward compatibility test");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_to_vec_works_on_gpu_tensors() {
    let cpu_tensor = Tensor::new(ndarray::arr1(&[1.0f32, 2.0, 3.0]).into_dyn());

    if let Ok(cuda_tensor) = cpu_tensor.to_cuda() {
        // to_vec should work even on GPU tensors
        let vec_data = cuda_tensor.to_vec().unwrap();
        assert_eq!(vec_data, vec![1.0, 2.0, 3.0]);

        // Test chained GPU operations
        if let Ok(result) = cuda_tensor.add_cuda(&cuda_tensor) {
            let result_vec = result.to_vec().unwrap();
            assert_eq!(result_vec, vec![2.0, 4.0, 6.0]);
        }

        println!("to_vec GPU test passed!");
    } else {
        println!("CUDA not available, skipping to_vec GPU test");
    }
}

#[cfg(feature = "cuda")]
#[test]
#[should_panic(expected = "Cannot index GPU tensor. Call .to_cpu() first")]
fn test_gpu_only_tensor_indexing_panics() {
    let backend = get_backend();
    if let Some(cuda_backend) = backend.cuda_backend() {
        if let Ok(cuda_tensor) = CudaTensor::from_vec(
            cuda_backend.memory_manager(),
            vec![1.0f32, 2.0, 3.0],
            vec![3],
        ) {
            // Create GPU-only tensor
            let gpu_tensor = Tensor {
                data: ArrayD::zeros(IxDyn(&[0])), // Empty CPU data
                device: Device::CUDA(0),
                cuda_storage: Some(cuda_tensor),
            };

            let _ = gpu_tensor[0]; // Should panic
        }
    } else {
        println!("CUDA backend not available, skipping GPU-only tensor indexing test");
        // Force panic without CUDA if backend unavailable
        let gpu_tensor = Tensor::<f64> {
            data: ArrayD::zeros(IxDyn(&[0])),
            device: Device::CUDA(0),
            cuda_storage: None,
        };
        let _ = gpu_tensor[0]; // Should panic
    }
}

// Helper function to load a single kernel with detailed error reporting
#[cfg(feature = "cuda")]
fn load_single_kernel(kernels: &mut CudaKernels, name: &str) -> Result<(), String> {
    use ferrox::backend::cuda::kernels::*;

    let (ptx_bytes, expected_functions) = match name {
        "elementwise" => (
            ELEMENTWISE_PTX,
            vec![
                // f32 versions
                "elementwise_add",
                "elementwise_sqrt",
                "elementwise_abs",
                "elementwise_mul",
                "elementwise_div",
                "elementwise_sub",
                "elementwise_pow",
                "elementwise_min",
                "elementwise_max",
                "elementwise_exp",
                "elementwise_log",
                "elementwise_negate",
                // f64 versions
                "elementwise_add_f64",
                "elementwise_sqrt_f64",
                "elementwise_abs_f64",
                "elementwise_mul_f64",
                "elementwise_div_f64",
                "elementwise_sub_f64",
                "elementwise_pow_f64",
                "elementwise_min_f64",
                "elementwise_max_f64",
                "elementwise_exp_f64",
                "elementwise_log_f64",
                "elementwise_negate_f64",
            ],
        ),
        "matmul" => (MATMUL_PTX, vec!["matmul", "matmul_f64"]),
        "activations" => (
            ACTIVATIONS_PTX,
            vec![
                "relu",
                "sigmoid",
                "hyperbolic_tangent",
                "relu_f64",
                "sigmoid_f64",
                "hyperbolic_tangent_f64",
            ],
        ),
        "reduces" => (
            REDUCES_PTX,
            vec![
                "sum_axis",
                "max_along_dim",
                "sum_axis_f64",
                "max_along_dim_f64",
            ],
        ),
        "transpose" => (TRANSPOSE_PTX, vec!["transpose_2d", "transpose_2d_f64"]),
        "comparison" => (
            COMPARISON_PTX,
            vec![
                "greater_equal",
                "equal",
                "sign",
                "greater_equal_f64",
                "equal_f64",
                "sign_f64",
                "less_equal",
                "less_equal_f64",
                "clamp",
                "clamp_f64",
            ],
        ),
        _ => return Err(format!("Unknown kernel: {}", name)),
    };

    println!("  PTX size: {} bytes", ptx_bytes.len());

    // Check if PTX is valid UTF-8
    let ptx_str = match std::str::from_utf8(ptx_bytes) {
        Ok(s) => {
            println!("  ✓ PTX is valid UTF-8");
            println!("  PTX preview: {}", &s[..s.len().min(200)]);
            s
        }
        Err(e) => {
            return Err(format!("Invalid PTX UTF-8: {}", e));
        }
    };

    // Try to load the PTX
    let module_name = format!("{}_module", name);
    let functions: Vec<&str> = expected_functions.iter().take(25).cloned().collect(); // Test with fewer functions first

    println!("  Attempting to load PTX into module: {}", module_name);
    println!("  Expected functions: {:?}", functions);

    match kernels
        .context()
        .load_ptx(ptx_str.into(), &module_name, &functions)
    {
        Ok(_) => {
            println!("  ✓ PTX loaded into device successfully");

            // Check if functions are accessible
            for func_name in &functions {
                if let Some(_func) = kernels.context().get_func(&module_name, func_name) {
                    println!("    ✓ Function {} found", func_name);
                } else {
                    println!("    ✗ Function {} NOT found", func_name);
                }
            }
            Ok(())
        }
        Err(e) => {
            println!("  ✗ PTX loading failed: {}", e);
            Err(format!("Failed to load {} kernel: {}", name, e))
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_detailed_kernel_loading() {
    println!("=== Detailed Kernel Loading Debug ===");

    // Step 1: Create device (we know this works)
    let device = CudaContext::new(0).expect("CUDA device creation failed");
    println!("✓ CUDA device created: {:?}", device.name());

    // Step 2: Create empty kernel manager
    let mut kernels = CudaKernels::new(device.clone());
    println!("✓ Empty kernel manager created");

    // Step 3: Try loading kernels one by one to find which one fails
    let kernel_names = [
        "elementwise",
        "matmul",
        "activations",
        "reduces",
        "transpose",
        "comparison",
    ];

    for kernel_name in &kernel_names {
        println!("Loading kernel: {}", kernel_name);
        match load_single_kernel(&mut kernels, kernel_name) {
            Ok(_) => println!("  ✓ {} loaded successfully", kernel_name),
            Err(e) => {
                println!("  ✗ {} failed: {}", kernel_name, e);
                // This tells us exactly which kernel and why it failed
                break;
            }
        }
    }

    println!("Loaded kernels: {:?}", kernels.loaded_kernels());
}
