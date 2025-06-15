// src/backend/cuda/tests.rs
#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::super::{CudaBackend, CudaKernels, load_all_kernels};
    use crate::backend::cuda::memory::CudaMemoryManager;
    use crate::backend::cuda::memory::CudaTensor;
    use crate::backend::cuda::memory::compute_strides;
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
                i,
                e,
                a,
                (a - e).abs()
            );
        }
    }

    fn setup_memory_manager() -> Option<CudaMemoryManager> {
        match CudaDevice::new(0) {
            Ok(device) => Some(CudaMemoryManager::new(device)),
            Err(_) => None,
        }
    }

    #[test]
    fn test_memory_allocation() {
        if let Some(manager) = setup_memory_manager() {
            let size = 1024;

            // Test zero allocation
            let zeros = manager.alloc_zeros::<f32>(size);
            assert!(zeros.is_ok());

            let buffer = unsafe { manager.alloc::<f32>(size) };
            // Test regular allocation
            assert!(buffer.is_ok());
        }
    }

    #[test]
    fn test_host_device_transfers() {
        if let Some(manager) = setup_memory_manager() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
            let original_data = data.clone();

            // Host to device
            let gpu_data = manager.host_to_device(data);
            assert!(gpu_data.is_ok());

            if let Ok(gpu_buffer) = gpu_data {
                // Device to host
                let retrieved_data = manager.device_to_host(&gpu_buffer);
                assert!(retrieved_data.is_ok());

                if let Ok(host_data) = retrieved_data {
                    assert_eq!(host_data, original_data);
                }
            }
        }
    }

    #[test]
    fn test_cuda_tensor_creation() {
        if let Some(manager) = setup_memory_manager() {
            let shape = vec![2, 3, 4];
            let tensor = CudaTensor::<f32>::zeros(&manager, shape.clone());

            assert!(tensor.is_ok());

            if let Ok(t) = tensor {
                assert_eq!(t.shape(), &shape);
                assert_eq!(t.size(), 24);
                assert_eq!(t.ndim(), 3);
            }
        }
    }

    #[test]
    fn test_tensor_reshape() {
        if let Some(manager) = setup_memory_manager() {
            let shape = vec![2, 6];
            let mut tensor = CudaTensor::<f32>::zeros(&manager, shape).unwrap();

            // Valid reshape
            assert!(tensor.reshape(vec![3, 4]).is_ok());
            assert_eq!(tensor.shape(), &[3, 4]);

            // Invalid reshape (different size)
            assert!(tensor.reshape(vec![2, 7]).is_err());
        }
    }

    #[test]
    fn test_stride_computation() {
        let shape = vec![2, 3, 4];
        let strides = compute_strides(&shape);
        assert_eq!(strides, vec![12, 4, 1]);

        let shape2 = vec![5, 7];
        let strides2 = compute_strides(&shape2);
        assert_eq!(strides2, vec![7, 1]);
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
    fn test_element_wise_operations() {
        if let Some(backend) = setup_cuda_backend() {
            let memory = backend.memory_manager();
            let ops = CudaOps::new(backend.kernels(), memory);

            // Test data
            let a_data = vec![1.0, 2.0, 3.0, 4.0];
            let b_data = vec![2.0, 3.0, 4.0, 5.0];

            // Create CUDA tensors
            let a = CudaTensor::from_vec(memory, a_data, vec![2, 2]).unwrap();
            let b = CudaTensor::from_vec(memory, b_data, vec![2, 2]).unwrap();

            // Test addition
            let result = ops.add(&a, &b).unwrap();
            let host_result = memory.device_to_host(&result.data).unwrap();
            assert_eq!(host_result, vec![3.0, 5.0, 7.0, 9.0]);

            // Test multiplication
            let result = ops.mul(&a, &b).unwrap();
            let host_result = memory.device_to_host(&result.data).unwrap();
            assert_eq!(host_result, vec![2.0, 6.0, 12.0, 20.0]);
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
            backend
                .kernels()
                .launch_add(cfg, &a_gpu, &b_gpu, &mut c_gpu, size as i32)
                .unwrap();
            backend.synchronize().unwrap();

            // Get results and verify
            let result = backend.device().dtoh_sync_copy(&c_gpu).unwrap();
            let expected: Vec<f32> = a_host
                .iter()
                .zip(b_host.iter())
                .map(|(a, b)| a + b)
                .collect();

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
                grid_dim: (((size + 255) / 256).try_into().unwrap(), 1, 1),
                shared_mem_bytes: 0,
            };

            backend
                .kernels()
                .launch_add(cfg, &a_gpu, &b_gpu, &mut c_gpu, size as i32)
                .unwrap();
            backend.synchronize().unwrap();

            let result = backend.device().dtoh_sync_copy(&c_gpu).unwrap();
            let expected: Vec<f32> = a_host
                .iter()
                .zip(b_host.iter())
                .map(|(a, b)| a + b)
                .collect();

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

            backend
                .kernels()
                .launch_relu(cfg, &input_gpu, &mut output_gpu, size as i32)
                .unwrap();
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

            backend
                .kernels()
                .launch_relu(cfg, &input_gpu, &mut output_gpu, size as i32)
                .unwrap();
            backend.synchronize().unwrap();

            let result = backend.device().dtoh_sync_copy(&output_gpu).unwrap();
            let expected: Vec<f32> = input.iter().map(|&x| x.max(0.0)).collect();

            assert_float_eq(&result, &expected, 1e-6);

            // Verify that negative values are zeroed
            for (i, (&input_val, &result_val)) in input.iter().zip(result.iter()).enumerate() {
                if input_val < 0.0 {
                    assert_eq!(result_val, 0.0, "Negative value not zeroed at index {}", i);
                } else {
                    assert_eq!(
                        result_val, input_val,
                        "Positive value changed at index {}",
                        i
                    );
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
            let mut c_gpu = backend
                .device()
                .alloc_zeros::<f32>((m * n) as usize)
                .unwrap();

            let cfg = LaunchConfig {
                block_dim: (16, 16, 1),
                grid_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };

            backend
                .kernels()
                .launch_matmul(cfg, &a_gpu, &b_gpu, &mut c_gpu, m, n, k)
                .unwrap();
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
            let mut c_gpu = backend
                .device()
                .alloc_zeros::<f32>((m * n) as usize)
                .unwrap();

            let cfg = LaunchConfig {
                block_dim: (16, 16, 1),
                grid_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };

            backend
                .kernels()
                .launch_matmul(cfg, &a_gpu, &b_gpu, &mut c_gpu, m, n, k)
                .unwrap();
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
            let result = backend
                .kernels()
                .launch_add(cfg, &a_gpu, &b_gpu, &mut c_gpu, size as i32);

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
            backend
                .kernels()
                .launch_relu(cfg, &input_gpu, &mut temp_gpu, size as i32)
                .unwrap();
            backend.synchronize().unwrap();

            backend
                .kernels()
                .launch_add(cfg, &temp_gpu, &temp_gpu, &mut output_gpu, size as i32)
                .unwrap();
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

            let mut kernels = CudaKernels::new(device.clone());

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
        use std::sync::Arc;
        use std::thread;

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

    #[cfg(feature = "cuda")]
    #[test]
    fn test_create_tensor_from_cpu() {
        if let Some(backend) = setup_cuda_backend() {
            // Test data
            let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let shape = vec![2, 3];

            // Create CUDA tensor from CPU data
            let cuda_tensor = backend.create_tensor_from_cpu(data.clone(), shape.clone());

            assert!(
                cuda_tensor.is_ok(),
                "Failed to create CUDA tensor from CPU data"
            );

            if let Ok(tensor) = cuda_tensor {
                // Verify shape and size
                assert_eq!(tensor.shape(), &shape);
                assert_eq!(tensor.size(), 6);
                assert_eq!(tensor.ndim(), 2);

                // Verify data by transferring back to CPU
                let cpu_data = tensor.to_cpu().unwrap();
                assert_eq!(
                    cpu_data, data,
                    "Data doesn't match after round-trip transfer"
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_create_tensor_from_cpu_errors() {
        if let Some(backend) = setup_cuda_backend() {
            // Test shape mismatch error
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let wrong_shape = vec![2, 3]; // Should be [4] or [2, 2]

            let result = backend.create_tensor_from_cpu(data, wrong_shape);
            assert!(result.is_err(), "Should fail with shape mismatch");

            // Verify error message contains useful information
            let error_msg = result.err().unwrap();
            assert!(
                error_msg.contains("doesn't match shape"),
                "Error message should mention shape mismatch"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_tensor_from_vec() {
        if let Some(manager) = setup_memory_manager() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let shape = vec![2, 2];

            // Create tensor using from_vec
            let tensor = CudaTensor::from_vec(&manager, data.clone(), shape.clone());

            assert!(tensor.is_ok(), "Failed to create tensor from vec");

            if let Ok(t) = tensor {
                assert_eq!(t.shape(), &shape);
                assert_eq!(t.size(), 4);

                // Test round-trip
                let cpu_data = t.to_cpu().unwrap();
                assert_eq!(cpu_data, data);
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_integration_with_backend_methods() {
        if let Some(backend) = setup_cuda_backend() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let shape = vec![2, 2];

            // Create tensor using backend method
            let cuda_tensor = backend
                .create_tensor_from_cpu(data.clone(), shape.clone())
                .unwrap();

            // Test that we can use this tensor with operations
            let memory = backend.memory_manager();
            let ops = backend.ops();

            // Test scalar addition
            let result = ops.add_scalar(&cuda_tensor, 5.0);
            assert!(result.is_ok(), "Scalar addition should work");

            if let Ok(result_tensor) = result {
                let result_data = result_tensor.to_cpu().unwrap();
                let expected: Vec<f32> = data.iter().map(|x| x + 5.0).collect();
                assert_eq!(result_data, expected, "Scalar addition result incorrect");
            }
        }
    }
}
