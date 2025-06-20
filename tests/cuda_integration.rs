#![cfg(test)]
#[cfg(feature = "cuda")]
use ferrox::backend::Device;
#[cfg(feature = "cuda")]
use ferrox::tensor::Tensor;

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
