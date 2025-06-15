// src/backend/cuda/tests.rs
#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::super::{CudaBackend, CudaKernels, load_all_kernels};
    use cudarc::driver::{CudaDevice, LaunchConfig};
    use std::sync::Arc;

    /// Helper function to create a test CUDA backend
    /// Skips test if CUDA is not available on the system
    fn setup_cuda_backend() -> Option<CudaBackend> {
        match CudaBackend::new(0) {
            Ok(backend) => Some(backend),
            Err(_) => {
                println!("CUDA not available, skipping CUDA tests");
                None
            }
        }
    }

    /// Helper function to create test vectors with known patterns
    fn create_test_vectors(size: usize) -> (Vec<f32>, Vec<f32>) {
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
        (a, b)
    }

    /// Helper function to verify results with tolerance
    fn assert_float_eq(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "Length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tolerance,
                "Mismatch at index {}: expected {}, got {}, diff: {}",
                i, e, a, (a - e).abs()
            );
        }
    }

    #[test]
    fn test_cuda_backend_initialization() {
        if let Some(backend) = setup_cuda_backend() {
            assert_eq!(backend.device_id(), 0);
            assert!(backend.name().contains("CUDA Device"));
            
            // Test device synchronization
            assert!(backend.synchronize().is_ok());
        }
    }

    #[test]
    fn test_kernel_loading() {
        if let Some(backend) = setup_cuda_backend() {
            let kernels = backend.kernels();
            
            // Check that all expected kernels are loaded
            let loaded_kernels = kernels.loaded_kernels();
            assert!(loaded_kernels.contains(&&"add".to_string()));
            assert!(loaded_kernels.contains(&&"relu".to_string()));
            assert!(loaded_kernels.contains(&&"matmul".to_string()));
            
            // Check that functions can be retrieved
            assert!(kernels.get_function("add").is_some());
            assert!(kernels.get_function("relu").is_some());
            assert!(kernels.get_function("matmul").is_some());
            assert!(kernels.get_function("nonexistent").is_none());
        }
    }

    #[test]
    fn test_kernel_cloning() {
        if let Some(backend) = setup_cuda_backend() {
            let kernels = backend.kernels();
            
            // Test that cloned functions work
            let add_kernel1 = kernels.get_function_cloned("add");
            let add_kernel2 = kernels.get_function_cloned("add");
            
            assert!(add_kernel1.is_some());
            assert!(add_kernel2.is_some());
            
            // Test cloning non-existent kernel
            let fake_kernel = kernels.get_function_cloned("fake_kernel");
            assert!(fake_kernel.is_none());
        }
    }

    #[test]
    fn test_add_kernel_small() {
        if let Some(backend) = setup_cuda_backend() {
            let size = 16;
            let (a_host, b_host) = create_test_vectors(size);
            
            // Allocate GPU memory
            let a_gpu = backend.device().htod_copy(a_host.clone()).unwrap();
            let b_gpu = backend.device().htod_copy(b_host.clone()).unwrap();
            let mut c_gpu = backend.device().alloc_zeros::<f32>(size).unwrap();
            
            // Configure launch parameters for small test
            let cfg = LaunchConfig {
                block_dim: (16, 1, 1),
                grid_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            
            // Launch add kernel
            backend.kernels().launch_add(cfg, &a_gpu, &b_gpu, &mut c_gpu, size as i32).unwrap();
            backend.synchronize().unwrap();
            
            // Get results and verify
            let result = backend.device().dtoh_sync_copy(&c_gpu).unwrap();
            let expected: Vec<f32> = a_host.iter().zip(b_host.iter()).map(|(a, b)| a + b).collect();
            
            assert_float_eq(&result, &expected, 1e-6);
        }
    }

    #[test]
    fn test_add_kernel_large() {
        if let Some(backend) = setup_cuda_backend() {
            let size = 1024;
            let (a_host, b_host) = create_test_vectors(size);
            
            let a_gpu = backend.device().htod_copy(a_host.clone()).unwrap();
            let b_gpu = backend.device().htod_copy(b_host.clone()).unwrap();
            let mut c_gpu = backend.device().alloc_zeros::<f32>(size).unwrap();
            
            // Multi-block configuration
            let cfg = LaunchConfig {
                block_dim: (256, 1, 1),
                grid_dim: ((size + 255) / 256, 1, 1),
                shared_mem_bytes: 0,
            };
            
            backend.kernels().launch_add(cfg, &a_gpu, &b_gpu, &mut c_gpu, size as i32).unwrap();
            backend.synchronize().unwrap();
            
            let result = backend.device().dtoh_sync_copy(&c_gpu).unwrap();
            let expected: Vec<f32> = a_host.iter().zip(b_host.iter()).map(|(a, b)| a + b).collect();
            
            assert_float_eq(&result, &expected, 1e-6);
        }
    }

    #[test]
    fn test_relu_kernel_positive_values() {
        if let Some(backend) = setup_cuda_backend() {
            let size = 256;
            let input: Vec<f32> = (0..size).map(|i| i as f32).collect(); // All positive
            
            let input_gpu = backend.device().htod_copy(input.clone()).unwrap();
            let mut output_gpu = backend.device().alloc_zeros::<f32>(size).unwrap();
            
            let cfg = LaunchConfig {
                block_dim: (256, 1, 1),
                grid_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            
            backend.kernels().launch_relu(cfg, &input_gpu, &mut output_gpu, size as i32).unwrap();
            backend.synchronize().unwrap();
            
            let result = backend.device().dtoh_sync_copy(&output_gpu).unwrap();
            let expected: Vec<f32> = input.iter().map(|&x| x.max(0.0)).collect();
            
            assert_float_eq(&result, &expected, 1e-6);
        }
    }

    #[test]
    fn test_relu_kernel_mixed_values() {
        if let Some(backend) = setup_cuda_backend() {
            let size = 512;
            let input: Vec<f32> = (0..size).map(|i| i as f32 - 256.0).collect(); // Mix of pos/neg
            
            let input_gpu = backend.device().htod_copy(input.clone()).unwrap();
            let mut output_gpu = backend.device().alloc_zeros::<f32>(size).unwrap();
            
            let cfg = LaunchConfig {
                block_dim: (256, 1, 1),
                grid_dim: (2, 1, 1),
                shared_mem_bytes: 0,
            };
            
            backend.kernels().launch_relu(cfg, &input_gpu, &mut output_gpu, size as i32).unwrap();
            backend.synchronize().unwrap();
            
            let result = backend.device().dtoh_sync_copy(&output_gpu).unwrap();
            let expected: Vec<f32> = input.iter().map(|&x| x.max(0.0)).collect();
            
            assert_float_eq(&result, &expected, 1e-6);
            
            // Verify that negative values are zeroed
            for (i, (&input_val, &result_val)) in input.iter().zip(result.iter()).enumerate() {
                if input_val < 0.0 {
                    assert_eq!(result_val, 0.0, "Negative value not zeroed at index {}", i);
                } else {
                    assert_eq!(result_val, input_val, "Positive value changed at index {}", i);
                }
            }
        }
    }

    #[test]
    fn test_matmul_kernel_small() {
        if let Some(backend) = setup_cuda_backend() {
            // Test 2x2 * 2x2 = 2x2 matrix multiplication
            let m = 2;
            let n = 2; 
            let k = 2;
            
            // Matrix A (2x2): [[1, 2], [3, 4]]
            let a_host = vec![1.0, 2.0, 3.0, 4.0];
            // Matrix B (2x2): [[5, 6], [7, 8]]
            let b_host = vec![5.0, 6.0, 7.0, 8.0];
            // Expected C: [[19, 22], [43, 50]]
            let expected = vec![19.0, 22.0, 43.0, 50.0];
            
            let a_gpu = backend.device().htod_copy(a_host).unwrap();
            let b_gpu = backend.device().htod_copy(b_host).unwrap();
            let mut c_gpu = backend.device().alloc_zeros::<f32>((m * n) as usize).unwrap();
            
            let cfg = LaunchConfig {
                block_dim: (16, 16, 1),
                grid_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            
            backend.kernels().launch_matmul(cfg, &a_gpu, &b_gpu, &mut c_gpu, m, n, k).unwrap();
            backend.synchronize().unwrap();
            
            let result = backend.device().dtoh_sync_copy(&c_gpu).unwrap();
            assert_float_eq(&result, &expected, 1e-5);
        }
    }

    #[test]
    fn test_matmul_kernel_identity() {
        if let Some(backend) = setup_cuda_backend() {
            // Test multiplication with identity matrix
            let size = 3;
            let m = size;
            let n = size;
            let k = size;
            
            // Matrix A (3x3): [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            let a_host = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
            // Identity matrix B (3x3): [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            let b_host = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            // Result should be A itself
            let expected = a_host.clone();
            
            let a_gpu = backend.device().htod_copy(a_host).unwrap();
            let b_gpu = backend.device().htod_copy(b_host).unwrap();
            let mut c_gpu = backend.device().alloc_zeros::<f32>((m * n) as usize).unwrap();
            
            let cfg = LaunchConfig {
                block_dim: (16, 16, 1),
                grid_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            
            backend.kernels().launch_matmul(cfg, &a_gpu, &b_gpu, &mut c_gpu, m, n, k).unwrap();
            backend.synchronize().unwrap();
            
            let result = backend.device().dtoh_sync_copy(&c_gpu).unwrap();
            assert_float_eq(&result, &expected, 1e-5);
        }
    }

    #[test]
    fn test_kernel_error_handling() {
        if let Some(backend) = setup_cuda_backend() {
            let size = 16;
            let (a_host, b_host) = create_test_vectors(size);
            
            let a_gpu = backend.device().htod_copy(a_host).unwrap();
            let b_gpu = backend.device().htod_copy(b_host).unwrap();
            let mut c_gpu = backend.device().alloc_zeros::<f32>(size).unwrap();
            
            // Test with invalid grid configuration (should still work with CUDA's error handling)
            let cfg = LaunchConfig {
                block_dim: (1024, 1, 1), // Max block size
                grid_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            
            // This should succeed even with large block size (CUDA will handle it)
            let result = backend.kernels().launch_add(cfg, &a_gpu, &b_gpu, &mut c_gpu, size as i32);
            
            // The result might be ok or error depending on GPU capabilities
            // We just test that it doesn't panic
            match result {
                Ok(_) => println!("Large block size accepted"),
                Err(e) => println!("Large block size rejected: {}", e),
            }
        }
    }

    #[test]
    fn test_multiple_kernel_launches() {
        if let Some(backend) = setup_cuda_backend() {
            let size = 128;
            let input: Vec<f32> = (0..size).map(|i| i as f32 - 64.0).collect();
            
            let input_gpu = backend.device().htod_copy(input.clone()).unwrap();
            let mut temp_gpu = backend.device().alloc_zeros::<f32>(size).unwrap();
            let mut output_gpu = backend.device().alloc_zeros::<f32>(size).unwrap();
            
            let cfg = LaunchConfig {
                block_dim: (128, 1, 1),
                grid_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            
            // Chain operations: input -> ReLU -> add with itself
            backend.kernels().launch_relu(cfg, &input_gpu, &mut temp_gpu, size as i32).unwrap();
            backend.synchronize().unwrap();
            
            backend.kernels().launch_add(cfg, &temp_gpu, &temp_gpu, &mut output_gpu, size as i32).unwrap();
            backend.synchronize().unwrap();
            
            let result = backend.device().dtoh_sync_copy(&output_gpu).unwrap();
            
            // Expected: ReLU(input) * 2
            let expected: Vec<f32> = input.iter().map(|&x| x.max(0.0) * 2.0).collect();
            assert_float_eq(&result, &expected, 1e-6);
        }
    }

    #[test]
    fn test_kernel_manager_creation() {
        if let Some(backend) = setup_cuda_backend() {
            let device = CudaDevice::new(0).unwrap();
            let device_arc = Arc::new(device);
            let mut kernels = CudaKernels::new(device_arc.clone());
            
            // Initially no kernels loaded
            assert_eq!(kernels.loaded_kernels().len(), 0);
            
            // Load kernels manually
            load_all_kernels(&mut kernels).unwrap();
            
            // Now kernels should be loaded
            assert_eq!(kernels.loaded_kernels().len(), 3);
        }
    }

    #[test]
    fn test_concurrent_kernel_access() {
        use std::thread;
        use std::sync::Arc;
        
        if let Some(backend) = setup_cuda_backend() {
            let backend = Arc::new(backend);
            let mut handles = vec![];
            
            // Test concurrent access to kernel functions
            for _ in 0..4 {
                let backend_clone = Arc::clone(&backend);
                let handle = thread::spawn(move || {
                    let kernels = backend_clone.kernels();
                    
                    // Multiple threads trying to clone kernels simultaneously
                    for _ in 0..10 {
                        let _add_kernel = kernels.get_function_cloned("add");
                        let _relu_kernel = kernels.get_function_cloned("relu");
                        let _matmul_kernel = kernels.get_function_cloned("matmul");
                    }
                });
                handles.push(handle);
            }
            
            // Wait for all threads to complete
            for handle in handles {
                handle.join().unwrap();
            }
        }
    }
}