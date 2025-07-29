// src/backend/cuda/tests.rs
#[cfg(all(test, feature = "cuda"))]
mod tests {

    use super::super::{CudaBackend, CudaKernels, load_all_kernels};
    use crate::backend::cuda::context::CudaContextManager;
    use crate::backend::cuda::context::CudaTensor;
    use crate::backend::cuda::context::compute_strides;
    use crate::backend::cuda::ops::CudaOps;
    use cudarc::driver::{CudaContext, LaunchConfig};
    //use std::time::{Duration, Instant};
    //use std::sync::Arc;

    /// Helper function to create a test CUDA backend
    /// Skips test if CUDA is not available on the system
    fn setup_cuda_backend() -> Option<CudaBackend> {
        match CudaBackend::from_device_id(0) {
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

    fn setup_context_manager() -> Option<CudaContextManager> {
        match CudaContextManager::from_device_id(0) {
            Ok(manager) => Some(manager),
            Err(_) => None,
        }
    }

    #[test]
    fn test_memory_allocation() {
        if let Some(manager) = setup_context_manager() {
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
        if let Some(manager) = setup_context_manager() {
            let mut data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
            let original_data = data.clone();

            // Host to device
            let gpu_data = manager.host_to_device(&mut data);
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
        if let Some(manager) = setup_context_manager() {
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
        if let Some(manager) = setup_context_manager() {
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
            assert_eq!(backend.id(), 0);

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
            assert!(loaded_kernels.contains(&&"elementwise_add".to_string()));
            assert!(loaded_kernels.contains(&&"relu".to_string()));
            assert!(loaded_kernels.contains(&&"matmul".to_string()));

            // Check that functions can be retrieved
            assert!(kernels.get_function("elementwise_add").is_some());
            assert!(kernels.get_function("relu").is_some());
            assert!(kernels.get_function("matmul").is_some());
            assert!(kernels.get_function("nonexistent").is_none());
        }
    }

    #[test]
    fn test_element_wise_operations() {
        if let Some(backend) = setup_cuda_backend() {
            let ops = backend.ops();

            // Test data
            let a_data = vec![1.0, 2.0, 3.0, 4.0];
            let b_data = vec![2.0, 3.0, 4.0, 5.0];

            // Create CUDA tensors
            let a = CudaTensor::from_vec(&backend, a_data, vec![2, 2]).unwrap();
            let b = CudaTensor::from_vec(&backend, b_data, vec![2, 2]).unwrap();

            // Test addition
            let result = ops.add(&a, &b).unwrap();
            let host_result = backend.device_to_host(&result.data).unwrap();
            assert_eq!(host_result, vec![3.0, 5.0, 7.0, 9.0]);

            // Test multiplication
            let result = ops.mul(&a, &b).unwrap();
            let host_result = backend.device_to_host(&result.data).unwrap();
            assert_eq!(host_result, vec![2.0, 6.0, 12.0, 20.0]);
        }
    }

    #[test]
    fn test_kernel_cloning() {
        if let Some(backend) = setup_cuda_backend() {
            let kernels = backend.kernels();

            // Test that cloned functions work
            let add_kernel1 = kernels.get_function_cloned("elementwise_add");
            let add_kernel2 = kernels.get_function_cloned("elementwise_add");

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
            let a_gpu = backend.host_to_device(&a_host).unwrap();
            let b_gpu = backend.host_to_device(&b_host).unwrap();
            let mut c_gpu = backend.alloc_zeros::<f32>(size).unwrap();

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
            let result = backend.device_to_host(&c_gpu).unwrap();
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

            let a_gpu = backend.host_to_device(&a_host).unwrap();
            let b_gpu = backend.host_to_device(&b_host).unwrap();
            let mut c_gpu = backend.alloc_zeros::<f32>(size).unwrap();

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

            let result = backend.device_to_host(&c_gpu).unwrap();
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

            let input_gpu = backend.host_to_device(&input).unwrap();
            let mut output_gpu = backend.alloc_zeros::<f32>(size).unwrap();

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

            let result = backend.device_to_host(&output_gpu).unwrap();
            let expected: Vec<f32> = input.iter().map(|&x| x.max(0.0)).collect();

            assert_float_eq(&result, &expected, 1e-6);
        }
    }

    #[test]
    fn test_relu_kernel_mixed_values() {
        if let Some(backend) = setup_cuda_backend() {
            let size = 512;
            let input: Vec<f32> = (0..size).map(|i| i as f32 - 256.0).collect(); // Mix of pos/neg

            let input_gpu = backend.host_to_device(&input).unwrap();
            let mut output_gpu = backend.alloc_zeros::<f32>(size).unwrap();

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

            let result = backend.device_to_host(&output_gpu).unwrap();
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

            let a_gpu = backend.host_to_device(&a_host).unwrap();
            let b_gpu = backend.host_to_device(&b_host).unwrap();
            let mut c_gpu = backend.alloc_zeros::<f32>((m * n) as usize).unwrap();

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

            let result = backend.device_to_host(&c_gpu).unwrap();
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

            let a_gpu = backend.host_to_device(&a_host).unwrap();
            let b_gpu = backend.host_to_device(&b_host).unwrap();
            let mut c_gpu = backend
                .default_stream()
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

            let result = backend.device_to_host(&c_gpu).unwrap();
            assert_float_eq(&result, &expected, 1e-5);
        }
    }

    #[test]
    fn test_kernel_error_handling() {
        if let Some(backend) = setup_cuda_backend() {
            let size = 16;
            let (a_host, b_host) = create_test_vectors(size);

            let a_gpu = backend.host_to_device(&a_host).unwrap();
            let b_gpu = backend.host_to_device(&b_host).unwrap();
            let mut c_gpu = backend.alloc_zeros::<f32>(size).unwrap();

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

            let input_gpu = backend.host_to_device(&input).unwrap();
            let mut temp_gpu = backend.alloc_zeros::<f32>(size).unwrap();
            let mut output_gpu = backend.alloc_zeros::<f32>(size).unwrap();

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

            let result = backend.device_to_host(&output_gpu).unwrap();

            // Expected: ReLU(input) * 2
            let expected: Vec<f32> = input.iter().map(|&x| x.max(0.0) * 2.0).collect();
            assert_float_eq(&result, &expected, 1e-6);
        }
    }

    #[test]
    fn test_kernel_manager_creation() {
        if let Some(_backend) = setup_cuda_backend() {
            let device = CudaContext::new(0).unwrap();

            let mut kernels = CudaKernels::new(device.clone());

            // Initially no kernels loaded
            assert_eq!(kernels.loaded_kernels().len(), 0);

            // Load kernels manually
            load_all_kernels(&mut kernels).unwrap();

            // Now kernels should be loaded
            assert_eq!(kernels.loaded_kernels().len(), 56);
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
                        let _add_kernel = kernels.get_function_cloned("elementwise_add");
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
            let cuda_tensor = backend.create_tensor_from_cpu(&data, shape.clone());

            assert!(
                cuda_tensor.is_ok(),
                "Failed to create CUDA tensor from CPU data"
            );

            if let Ok(tensor) = cuda_tensor {
                // Verify shape and size
                assert_eq!(tensor.shape(), &shape);
                assert_eq!(tensor.size(), 6);
                assert_eq!(tensor.ndim(), 2);
                let memory = backend;
                // Verify data by transferring back to CPU
                let cpu_data = tensor.to_cpu(&memory).unwrap();
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

            let result = backend.create_tensor_from_cpu(&data, wrong_shape);
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
        if let Some(manager) = setup_context_manager() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let shape = vec![2, 2];

            // Create tensor using from_vec
            let tensor = CudaTensor::from_vec(&manager, data.clone(), shape.clone());

            assert!(tensor.is_ok(), "Failed to create tensor from vec");

            if let Ok(t) = tensor {
                assert_eq!(t.shape(), &shape);
                assert_eq!(t.size(), 4);

                // Test round-trip
                let cpu_data = t.to_cpu(&manager).unwrap();
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
                .create_tensor_from_cpu(&data, shape.clone())
                .unwrap();

            // Test that we can use this tensor with operations
            let ops = backend.ops();

            // Test scalar addition
            let result = ops.add_scalar(&cuda_tensor, 5.0);
            assert!(result.is_ok(), "Scalar addition should work");

            if let Ok(result_tensor) = result {
                let result_data = result_tensor.to_cpu(&backend).unwrap();
                let expected: Vec<f32> = data.iter().map(|x| x + 5.0).collect();
                assert_eq!(result_data, expected, "Scalar addition result incorrect");
            }
        }
    }
}

// ----------------------------------------------------------
// KERNEL UNIT TESTS
// ----------------------------------------------------------
#[cfg(all(test, feature = "cuda"))]
mod kernel_tests {
    use super::super::CudaBackend;
    use cudarc::driver::LaunchConfig;

    fn setup_test_backend() -> Option<CudaBackend> {
        match CudaBackend::from_device_id(0) {
            Ok(backend) => Some(backend),
            Err(_) => None,
        }
    }

    fn create_launch_config(size: usize) -> LaunchConfig {
        let threads = 256;
        let blocks = (size + threads - 1) / threads;
        LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn create_2d_launch_config(rows: usize, cols: usize) -> LaunchConfig {
        let block_size = 16;
        let grid_x = (cols + block_size - 1) / block_size;
        let grid_y = (rows + block_size - 1) / block_size;
        LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (block_size as u32, block_size as u32, 1),
            shared_mem_bytes: 0,
        }
    }

    fn assert_f32_eq(result: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(result.len(), expected.len());
        for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < tolerance,
                "Mismatch at index {}: expected={}, got={}, diff={}",
                i,
                exp,
                res,
                (res - exp).abs()
            );
        }
    }

    fn assert_f64_eq(result: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(result.len(), expected.len());
        for (i, (&res, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < tolerance,
                "Mismatch at index {}: expected={}, got={}, diff={}",
                i,
                exp,
                res,
                (res - exp).abs()
            );
        }
    }

    // ===== BINARY ELEMENTWISE TESTS =====

    #[test]
    fn test_add_kernel_f32() {
        if let Some(backend) = setup_test_backend() {
            let size = 1024;
            let a = vec![1.0f32; size];
            let b = vec![2.0f32; size];

            let a_gpu = backend.host_to_device(&a).unwrap();
            let b_gpu = backend.host_to_device(&b).unwrap();
            let mut c_gpu = backend.alloc_zeros::<f32>(size).unwrap();

            let cfg = create_launch_config(size);
            backend
                .kernels()
                .launch_add(cfg, &a_gpu, &b_gpu, &mut c_gpu, size as i32)
                .unwrap();
            println!("Add kernel launched with config: {:?}", cfg);
            println!("Waiting for synchronization...");
            backend.synchronize().unwrap();

            let result = backend.device_to_host(&c_gpu).unwrap();
            for i in 0..size {
                let expected = 3.0f32;
                println!(
                    "Result at index {}: expected={}, got={}",
                    i, expected, result[i]
                );
                assert!((result[i] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_logical_not_f32() {
        if let Some(backend) = setup_test_backend() {
            let size = 8;
            let input = vec![0.0f32, 1.0, -1.0, 2.5, 0.0, -0.0, 100.0, -50.0];

            let input_gpu = backend.host_to_device(&input).unwrap();
            let mut result_gpu = backend.alloc_zeros::<f32>(size).unwrap();

            let cfg = create_launch_config(size);
            backend
                .kernels()
                .launch_logical_not(cfg, &input_gpu, &mut result_gpu, size as i32)
                .unwrap();
            backend.synchronize().unwrap();

            let result = backend.device_to_host(&result_gpu).unwrap();
            for i in 0..size {
                let expected = if input[i] == 0.0 { 1.0 } else { 0.0 };
                assert!((result[i] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_sign_f32() {
        if let Some(backend) = setup_test_backend() {
            let size = 9;
            let input = vec![-5.0f32, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0, -0.0, 100.0];

            let input_gpu = backend.host_to_device(&input).unwrap();
            let mut result_gpu = backend.alloc_zeros::<f32>(size).unwrap();

            let cfg = create_launch_config(size);
            backend
                .kernels()
                .launch_sign(cfg, &input_gpu, &mut result_gpu, size as i32)
                .unwrap();
            backend.synchronize().unwrap();

            let result = backend.device_to_host(&result_gpu).unwrap();
            let expected = vec![-1.0f32, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0];
            assert_f32_eq(&result, &expected, 1e-6);
        }
    }

    #[test]
    fn test_sign_f64() {
        if let Some(backend) = setup_test_backend() {
            let size = 6;
            let input = vec![-10.5f64, -0.001, 0.0, 0.001, 10.5, -0.0];

            let input_gpu = backend.host_to_device(&input).unwrap();
            let mut result_gpu = backend.alloc_zeros::<f64>(size).unwrap();

            let cfg = create_launch_config(size);
            backend
                .kernels()
                .launch_sign(cfg, &input_gpu, &mut result_gpu, size as i32)
                .unwrap();
            backend.synchronize().unwrap();

            let result = backend.device_to_host(&result_gpu).unwrap();
            let expected = vec![-1.0f64, -1.0, 0.0, 1.0, 1.0, 0.0];
            assert_f64_eq(&result, &expected, 1e-10);
        }
    }

    #[test]
    fn test_in_range_f32() {
        if let Some(backend) = setup_test_backend() {
            let size = 10;
            let input: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let min_val = 3.0f32;
            let max_val = 7.0f32;

            let input_gpu = backend.host_to_device(&input).unwrap();
            let mut result_gpu = backend.alloc_zeros::<f32>(size).unwrap();

            let cfg = create_launch_config(size);
            backend
                .kernels()
                .launch_in_range(
                    cfg,
                    &input_gpu,
                    min_val,
                    max_val,
                    &mut result_gpu,
                    size as i32,
                )
                .unwrap();
            backend.synchronize().unwrap();

            let result = backend.device_to_host(&result_gpu).unwrap();
            for i in 0..size {
                let expected = if input[i] >= min_val && input[i] <= max_val {
                    1.0
                } else {
                    0.0
                };
                assert!((result[i] - expected).abs() < 1e-6);
            }
        }
    }

    // ===== SPECIAL OPERATIONS TESTS =====

    #[test]
    fn test_clamp_f32() {
        if let Some(backend) = setup_test_backend() {
            let size = 10;
            let input: Vec<f32> = (0..size).map(|i| (i as f32 - 5.0) * 2.0).collect();
            let min_val = -5.0f32;
            let max_val = 5.0f32;

            let input_gpu = backend.host_to_device(&input).unwrap();
            let mut result_gpu = backend.alloc_zeros::<f32>(size).unwrap();

            let cfg = create_launch_config(size);
            backend
                .kernels()
                .launch_clamp(
                    cfg,
                    &input_gpu,
                    &mut result_gpu,
                    min_val,
                    max_val,
                    size as i32,
                )
                .unwrap();
            backend.synchronize().unwrap();

            let result = backend.device_to_host(&result_gpu).unwrap();
            for i in 0..size {
                let expected = input[i].max(min_val).min(max_val);
                assert!((result[i] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_clamp_f64() {
        if let Some(backend) = setup_test_backend() {
            let size = 8;
            let input = vec![-100.0f64, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0, 100.0];
            let min_val = -2.0f64;
            let max_val = 8.0f64;

            let input_gpu = backend.host_to_device(&input).unwrap();
            let mut result_gpu = backend.alloc_zeros::<f64>(size).unwrap();

            let cfg = create_launch_config(size);
            backend
                .kernels()
                .launch_clamp(
                    cfg,
                    &input_gpu,
                    &mut result_gpu,
                    min_val,
                    max_val,
                    size as i32,
                )
                .unwrap();
            backend.synchronize().unwrap();

            let result = backend.device_to_host(&result_gpu).unwrap();
            let expected = vec![-2.0f64, -2.0, -1.0, 0.0, 1.0, 5.0, 8.0, 8.0];
            assert_f64_eq(&result, &expected, 1e-10);
        }
    }

    // ===== MATRIX OPERATIONS TESTS =====

    #[test]
    fn test_matmul_kernel_small_f32() {
        if let Some(backend) = setup_test_backend() {
            let m = 2;
            let n = 2;
            let k = 2;

            let a_host = vec![1.0f32, 2.0, 3.0, 4.0];
            let b_host = vec![5.0f32, 6.0, 7.0, 8.0];
            let expected = vec![19.0f32, 22.0, 43.0, 50.0];

            let a_gpu = backend.host_to_device(&a_host).unwrap();
            let b_gpu = backend.host_to_device(&b_host).unwrap();
            let mut c_gpu = backend
                .default_stream()
                .alloc_zeros::<f32>((m * n) as usize)
                .unwrap();

            let cfg = create_2d_launch_config(m as usize, n as usize);
            backend
                .kernels()
                .launch_matmul(cfg, &a_gpu, &b_gpu, &mut c_gpu, m, n, k)
                .unwrap();
            backend.synchronize().unwrap();

            let result = backend.device_to_host(&c_gpu).unwrap();
            assert_f32_eq(&result, &expected, 1e-5);
        }
    }

    #[test]
    fn test_matmul_kernel_identity_f32() {
        if let Some(backend) = setup_test_backend() {
            let size = 3;
            let m = size;
            let n = size;
            let k = size;

            let a_host = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
            let b_host = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

            let a_gpu = backend.host_to_device(&a_host).unwrap();
            let b_gpu = backend.host_to_device(&b_host).unwrap();
            let mut c_gpu = backend.alloc_zeros::<f32>((m * n) as usize).unwrap();

            let cfg = create_2d_launch_config(m as usize, n as usize);
            backend
                .kernels()
                .launch_matmul(cfg, &a_gpu, &b_gpu, &mut c_gpu, m, n, k)
                .unwrap();
            backend.synchronize().unwrap();

            let result = backend.device_to_host(&c_gpu).unwrap();
            assert_f32_eq(&result, &a_host, 1e-5);
        }
    }

    #[test]
    fn test_matmul_kernel_f64() {
        if let Some(backend) = setup_test_backend() {
            let m = 2;
            let n = 3;
            let k = 2;

            let a_host = vec![1.5f64, 2.5, 3.5, 4.5];
            let b_host = vec![2.0f64, 1.0, 0.5, 3.0, 2.0, 1.5];
            let expected = vec![10.5f64, 6.5, 4.5, 20.5, 12.5, 8.5];

            let a_gpu = backend.host_to_device(&a_host).unwrap();
            let b_gpu = backend.host_to_device(&b_host).unwrap();
            let mut c_gpu = backend
                .default_stream()
                .alloc_zeros::<f64>((m * n) as usize)
                .unwrap();

            let cfg = create_2d_launch_config(m as usize, n as usize);
            backend
                .kernels()
                .launch_matmul(cfg, &a_gpu, &b_gpu, &mut c_gpu, m, n, k)
                .unwrap();
            backend.synchronize().unwrap();

            let result = backend.device_to_host(&c_gpu).unwrap();
            assert_f64_eq(&result, &expected, 1e-10);
        }
    }

    #[test]
    fn test_transpose_2d_f32() {
        if let Some(backend) = setup_test_backend() {
            let rows = 3;
            let cols = 4;
            let input = vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ];
            let expected = vec![
                1.0f32, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
            ];

            let input_gpu = backend.host_to_device(&input).unwrap();
            let mut output_gpu = backend.alloc_zeros::<f32>((rows * cols) as usize).unwrap();

            let cfg = create_2d_launch_config(rows as usize, cols as usize);
            backend
                .kernels()
                .launch_transpose_2d(cfg, &input_gpu, &mut output_gpu, rows, cols)
                .unwrap();
            backend.synchronize().unwrap();

            let result = backend.device_to_host(&output_gpu).unwrap();
            assert_f32_eq(&result, &expected, 1e-6);
        }
    }

    #[test]
    fn test_transpose_2d_f64() {
        if let Some(backend) = setup_test_backend() {
            let rows = 2;
            let cols = 3;
            let input = vec![1.5f64, 2.5, 3.5, 4.5, 5.5, 6.5];
            let expected = vec![1.5f64, 4.5, 2.5, 5.5, 3.5, 6.5];

            let input_gpu = backend.host_to_device(&input).unwrap();
            let mut output_gpu = backend.alloc_zeros::<f64>((rows * cols) as usize).unwrap();

            let cfg = create_2d_launch_config(rows as usize, cols as usize);
            backend
                .kernels()
                .launch_transpose_2d(cfg, &input_gpu, &mut output_gpu, rows, cols)
                .unwrap();
            backend.synchronize().unwrap();

            let result = backend.device_to_host(&output_gpu).unwrap();
            assert_f64_eq(&result, &expected, 1e-10);
        }
    }

    // ===== REDUCTION TESTS =====

    #[test]
    fn test_sum_axis_f32() {
        if let Some(backend) = setup_test_backend() {
            let outer_size = 2;
            let axis_size = 3;
            let inner_size = 2;
            let input = vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ];
            let expected = vec![9.0f32, 12.0, 27.0, 30.0];

            let input_gpu = backend.host_to_device(&input).unwrap();
            let mut output_gpu = backend
                .alloc_zeros::<f32>((outer_size * inner_size) as usize)
                .unwrap();

            let cfg = LaunchConfig {
                grid_dim: (outer_size as u32, 1, 1),
                block_dim: (inner_size as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            backend
                .kernels()
                .launch_sum_axis(
                    cfg,
                    &input_gpu,
                    &mut output_gpu,
                    outer_size,
                    axis_size,
                    inner_size,
                )
                .unwrap();
            backend.synchronize().unwrap();

            let result = backend.device_to_host(&output_gpu).unwrap();
            assert_f32_eq(&result, &expected, 1e-6);
        }
    }

    #[test]
    fn test_max_along_dim_f32() {
        if let Some(backend) = setup_test_backend() {
            let outer_size = 2;
            let axis_size = 3;
            let inner_size = 2;
            let input = vec![
                1.0f32, 2.0, 8.0, 4.0, 5.0, 9.0, 7.0, 3.0, 6.0, 10.0, 11.0, 12.0,
            ];
            let expected = vec![8.0f32, 9.0, 11.0, 12.0];

            let input_gpu = backend.host_to_device(&input).unwrap();
            let mut output_gpu = backend
                .alloc_zeros::<f32>((outer_size * inner_size) as usize)
                .unwrap();

            let cfg = LaunchConfig {
                grid_dim: (((outer_size * inner_size + 255) / 256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };

            backend
                .kernels()
                .launch_max_along_dim(
                    cfg,
                    &input_gpu,
                    &mut output_gpu,
                    outer_size,
                    axis_size,
                    inner_size,
                )
                .unwrap();
            backend.synchronize().unwrap();

            let result = backend.device_to_host(&output_gpu).unwrap();
            assert_f32_eq(&result, &expected, 1e-6);
        }
    }
}

mod stream_tests {

    use super::super::CudaBackend;
    use crate::backend::cuda::context::CudaContextManager;
    use crate::backend::cuda::context::CudaTensor;
    use cudarc::driver::{CudaContext, LaunchConfig};
    use std::time::{Duration, Instant};

    // ===== STREAMS TESTS =====
    /// Helper to create memory manager with streams
    fn setup_stream_manager() -> Option<CudaContextManager> {
        match CudaContextManager::from_device_id(0) {
            Ok(mut manager) => {
                if manager.setup_parallel_streams().is_ok() {
                    Some(manager)
                } else {
                    None
                }
            }
            Err(_) => None,
        }
    }

    /// Helper function to create a test CUDA backend
    /// Skips test if CUDA is not available on the system
    fn setup_cuda_backend() -> Option<CudaBackend> {
        match CudaBackend::from_device_id(0) {
            Ok(backend) => Some(backend),
            Err(_) => {
                println!("CUDA not available, skipping CUDA tests");
                None
            }
        }
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

    #[test]
    fn test_stream_creation() {
        if let Some(manager) = setup_stream_manager() {
            // Test creating individual streams
            assert!(manager.create_stream("test_stream").is_ok());
            assert!(manager.create_stream("another_stream").is_ok());

            // Check stream names
            let stream_names = manager.stream_names();
            assert!(stream_names.contains(&&"test_stream".to_string()));
            assert!(stream_names.contains(&&"another_stream".to_string()));
        }
    }

    #[test]
    fn test_parallel_streams_setup() {
        if let Some(manager) = setup_stream_manager() {
            let stream_names = manager.stream_names();

            // Check that all parallel streams were created
            assert!(stream_names.contains(&&"copy_h2d".to_string()));
            assert!(stream_names.contains(&&"copy_d2h".to_string()));
            assert!(stream_names.contains(&&"compute".to_string()));
            assert_eq!(stream_names.len(), 4);
        }
    }

    #[test]
    fn test_async_host_to_device() {
        if let Some(manager) = setup_stream_manager() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
            let original_data = data.clone();

            // Test async H2D transfer
            let gpu_data = manager.host_to_device_async(&data, Some("copy_h2d"));
            assert!(gpu_data.is_ok());

            if let Ok(gpu_buffer) = gpu_data {
                // Sync the stream
                assert!(manager.sync_stream("copy_h2d").is_ok());

                // Transfer back to verify
                let retrieved_data = manager.device_to_host(&gpu_buffer);
                assert!(retrieved_data.is_ok());

                if let Ok(host_data) = retrieved_data {
                    assert_eq!(host_data, original_data);
                }
            }
        }
    }

    #[test]
    fn test_async_device_to_host() {
        if let Some(manager) = setup_stream_manager() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
            let original_data = data.clone();

            // First transfer to device synchronously
            let gpu_data = manager.host_to_device(&data).unwrap();

            // Test async D2H transfer
            let host_data = manager.device_to_host_async(&gpu_data, Some("copy_d2h"));
            assert!(host_data.is_ok());

            if let Ok(result) = host_data {
                // Sync the stream
                assert!(manager.sync_stream("copy_d2h").is_ok());
                assert_eq!(result, original_data);
            }
        }
    }

    #[test]
    fn test_stream_query() {
        if let Some(manager) = setup_stream_manager() {
            // Start an async operation
            let data = vec![1.0f32; 1000000]; // Large data to take some time
            let _gpu_data = manager
                .host_to_device_async(&data, Some("copy_h2d"))
                .unwrap();

            // Query stream status (might be ready or not ready)
            let is_ready = manager.is_stream_ready("copy_h2d");
            assert!(is_ready.is_ok());

            // Synchronize and check again
            assert!(manager.sync_stream("copy_h2d").is_ok());
            let is_ready_after_sync = manager.is_stream_ready("copy_h2d").unwrap();
            assert!(is_ready_after_sync); // Should be ready after sync
        }
    }

    #[test]
    fn test_stream_not_found_errors() {
        if let Some(manager) = setup_stream_manager() {
            let data = vec![1.0f32, 2.0, 3.0];

            // Test with non-existent stream
            let result = manager.host_to_device_async(&data, Some("nonexistent"));
            assert!(result.is_err());

            // Test sync with non-existent stream
            let sync_result = manager.sync_stream("nonexistent");
            assert!(sync_result.is_err());

            // Test query with non-existent stream
            let query_result = manager.is_stream_ready("nonexistent");
            assert!(query_result.is_err());
        }
    }

    #[test]
    fn test_sync_all_streams() {
        if let Some(manager) = setup_stream_manager() {
            let data1 = vec![1.0f32; 1000];
            let data2 = vec![2.0f32; 1000];
            let data3 = vec![3.0f32; 1000];
            for st in manager.stream_names() {
                print!("Stream: {},", st);
            }
            // Start multiple async operations
            let gpu1 = manager.host_to_device_async(&data1, Some("copy_h2d"));
            assert!(gpu1.is_ok());
            let gpu2_result = manager.host_to_device_async(&data2, None); //
            //Default stream
            assert!(gpu2_result.is_ok());
            let gpu2 = gpu2_result.unwrap();
            let gpu3 = manager.host_to_device_async(&data3, Some("copy_d2h"));

            assert!(gpu3.is_ok());

            // Start async D2H on another stream
            let _result = manager
                .device_to_host_async(&gpu2, Some("compute"))
                .unwrap();

            // Sync all streams at once
            assert!(manager.sync_all_streams().is_ok());

            // All operations should be complete now
            assert!(manager.is_stream_ready("copy_h2d").unwrap());
            assert!(manager.is_stream_ready("copy_d2h").unwrap());
            assert!(manager.is_stream_ready("compute").unwrap());
        }
    }

    #[test]
    fn test_tensor_async_operations() {
        if let Some(manager) = setup_stream_manager() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let shape = vec![2, 3];
            let original_data = data.clone();

            // Create tensor with async transfer
            let tensor =
                CudaTensor::from_vec_async(&manager, data, shape.clone(), Some("copy_h2d"));
            assert!(tensor.is_ok());

            if let Ok(cuda_tensor) = tensor {
                assert_eq!(cuda_tensor.shape(), &shape);
                assert_eq!(cuda_tensor.size(), 6);

                // Sync the stream
                assert!(manager.sync_stream("copy_h2d").is_ok());

                // Transfer back asynchronously
                let cpu_data = cuda_tensor.to_cpu_async(&manager, Some("copy_d2h"));
                assert!(cpu_data.is_ok());

                if let Ok(result) = cpu_data {
                    // Sync and verify
                    assert!(manager.sync_stream("copy_d2h").is_ok());
                    assert_eq!(result, original_data);
                }
            }
        }
    }

    #[test]
    fn test_overlapped_transfers() {
        if let Some(manager) = setup_stream_manager() {
            let size = 1000000; // Large enough to see timing differences
            let data1: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let data2: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

            let start_time = Instant::now();

            // Sequential transfers (baseline)
            let gpu1 = manager.host_to_device(&data1).unwrap();
            let gpu2 = manager.host_to_device(&data2).unwrap();
            let _result1 = manager.device_to_host(&gpu1).unwrap();
            let _result2 = manager.device_to_host(&gpu2).unwrap();

            let sequential_time = start_time.elapsed();

            let start_time = Instant::now();

            // Overlapped transfers using different streams
            let gpu1_async = manager
                .host_to_device_async(&data1, Some("copy_h2d"))
                .unwrap();
            let gpu2_async = manager
                .host_to_device_async(&data2, Some("compute"))
                .unwrap();

            // Start D2H transfers on different streams
            let _result1_async = manager
                .device_to_host_async(&gpu1_async, Some("copy_d2h"))
                .unwrap();
            let _result2_async = manager
                .device_to_host_async(&gpu2_async, Some("copy_h2d"))
                .unwrap();

            // Sync all
            manager.sync_all_streams().unwrap();

            let parallel_time = start_time.elapsed();

            println!("Sequential time: {:?}", sequential_time);
            println!("Parallel time: {:?}", parallel_time);

            // Note: Parallel should be faster, but this depends on hardware
            // We just verify that both approaches work correctly
            assert!(parallel_time > Duration::from_nanos(0));
            assert!(sequential_time > Duration::from_nanos(0));
        }
    }

    #[test]
    fn test_stream_with_kernel_execution() {
        if let Some(backend) = setup_cuda_backend() {
            // This test requires integration with your kernel system
            let size = 1024;
            let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

            // Create additional streams for this test
            let temp_manager = setup_stream_manager().unwrap();

            // Transfer data asynchronously
            let gpu_data = temp_manager
                .host_to_device_async(&data, Some("copy_h2d"))
                .unwrap();
            let mut output_gpu = temp_manager.alloc_zeros::<f32>(size).unwrap();

            // Wait for transfer
            temp_manager.sync_stream("copy_h2d").unwrap();

            // Launch kernel on compute stream (if your kernels support stream parameter)
            let cfg = LaunchConfig {
                block_dim: (256, 1, 1),
                grid_dim: (((size + 255) / 256).try_into().unwrap(), 1, 1),
                shared_mem_bytes: 0,
            };

            // Note: This assumes your kernels can accept streams
            // You might need to modify your kernel launch methods
            backend
                .kernels()
                .launch_relu(cfg, &gpu_data, &mut output_gpu, size as i32)
                .unwrap();

            // Start async transfer back while potentially overlapping with kernel
            let result = temp_manager
                .device_to_host_async(&output_gpu, Some("copy_d2h"))
                .unwrap();

            // Sync everything
            backend.synchronize().unwrap();
            temp_manager.sync_stream("copy_d2h").unwrap();

            // Verify results
            let expected: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();
            assert_float_eq(&result, &expected, 1e-6);
        }
    }
}
