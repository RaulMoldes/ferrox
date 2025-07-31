// src/backend/cuda/tests.rs
#[cfg(all(test, feature = "cuda"))]
mod tests {

    use super::super::{CudaBackend, CudaKernels, load_all_kernels};
    use crate::backend::cuda::KernelManager;
    use crate::backend::cuda::context::CudaContextManager;
    use crate::backend::cuda::context::CudaTensor;
    use crate::backend::cuda::context::compute_strides;
    use crate::backend::cuda::ops::CudaOps;
    use cudarc::driver::{CudaContext, LaunchConfig};
    use std::sync::Arc;
    use std::thread;
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
            let tensor: Result<CudaTensor<f32>, String> =
                CudaTensor::alloc_init(&manager, shape.clone());

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
            let mut tensor: CudaTensor<f32> = CudaTensor::alloc_init(&manager, shape).unwrap();

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
    #[test]
    fn test_kernel_manager_creation() {
        if let Some(_backend) = setup_cuda_backend() {
            let ctx = CudaContext::new(0).unwrap();

            let mut kernels = KernelManager::new(ctx.default_stream());

            // Initially no kernels loaded
            assert_eq!(kernels.loaded_kernels().len(), 0);

            // Load kernels manually
            load_all_kernels(&mut kernels, &ctx).unwrap();

            // Now kernels should be loaded
            assert_eq!(kernels.loaded_kernels().len(), 56);
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
