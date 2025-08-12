// src/backend/cuda/mod.rs
#[cfg(feature = "cuda")]
pub mod context;

#[cfg(feature = "cuda")]
pub mod kernels;
#[cfg(feature = "cuda")]
pub mod ops;
#[cfg(feature = "cuda")]
pub mod stream_manager;

#[cfg(feature = "cuda")]
pub use context::{CudaContextManager, CudaTensor};

#[cfg(feature = "cuda")]
pub use kernels::{load_all_kernels, KernelManager};

// ALIAS para compatibilidad - CudaContextManager es ahora el "backend principal"
#[cfg(feature = "cuda")]
pub use context::CudaContextManager as CudaBackend;

// Dummy implementations when CUDA is not available
#[cfg(not(feature = "cuda"))]
pub struct CudaBackend;

#[cfg(not(feature = "cuda"))]
impl CudaBackend {
    pub fn new(_device_id: usize) -> Result<Self, String> {
        Err("CUDA support not compiled".to_string())
    }
}

#[cfg(test)]
mod cuda_tests {

    use crate::backend::cuda::ops::CudaOps;
    use crate::backend::manager::{best_device, with_cuda_ops};
    use crate::backend::Tensor;

#[test]
fn test_partition_operation() {
    // Test partition operation through storage backend - follows tensor API pattern
    // Unlike slice_range which returns borrowed data, partition creates new tensor
    let device = best_device::<f32>();

    // ======================
    // 1D TENSOR TESTS
    // ======================

    // Create test input: 1D tensor with values [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    // This provides sufficient data to test different partition ranges
    let input_tensor = Tensor::from_vec_with_device(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[6],
        device
    ).expect("Failed to create input tensor");

    // Test Case 1: Extract middle elements [2.0, 3.0, 4.0] (indices 1-4)
    // Uses storage backend pattern like other tensor operations
    let result1 = {
        let storage = input_tensor.storage.as_ref()
            .ok_or("Tensor has no storage backend")
            .expect("Input tensor must have storage");

        // Call partition through storage backend - creates new tensor storage
        // partition_dim=0 for 1D tensor (partition along first/only dimension)
        let result_storage = storage.partition(0, 1, 4)
            .expect("Partition operation failed");

        // Create new tensor from result storage, preserving device context
        Tensor::from_storage_backend(result_storage, input_tensor.device)
            .expect("Failed to create result tensor")
    };

    // Verify result shape and data - partition creates independent tensor
    assert_eq!(result1.shape(), &[3], "Partition result shape mismatch");
    let result_array = result1.into_data().unwrap();
    let result_data1 = result_array.as_slice().expect("Failed to get result slice");
    assert_eq!(result_data1[0], 2.0, "First partitioned element incorrect");
    assert_eq!(result_data1[1], 3.0, "Second partitioned element incorrect");
    assert_eq!(result_data1[2], 4.0, "Third partitioned element incorrect");

    // Test Case 2: Extract first two elements [1.0, 2.0] (indices 0-2)
    let result2 = {
        let storage = input_tensor.storage.as_ref()
            .ok_or("Tensor has no storage backend")
            .expect("Input tensor must have storage");

        let result_storage = storage.partition(0, 0, 2)
            .expect("Second partition operation failed");

        Tensor::from_storage_backend(result_storage, input_tensor.device)
            .expect("Failed to create second result tensor")
    };

    assert_eq!(result2.shape(), &[2], "Second partition result shape mismatch");

    let result_array = result2.into_data().unwrap();
    let result_data2 = result_array.as_slice().expect("Failed to get result slice");
    assert_eq!(result_data2[0], 1.0, "First element of second partition incorrect");
    assert_eq!(result_data2[1], 2.0, "Second element of second partition incorrect");

    // Test Case 3: Extract last elements [5.0, 6.0] (indices 4-6)
    let result3 = {
        let storage = input_tensor.storage.as_ref()
            .ok_or("Tensor has no storage backend")
            .expect("Input tensor must have storage");

        let result_storage = storage.partition(0, 4, 6)
            .expect("Third partition operation failed");

        Tensor::from_storage_backend(result_storage, input_tensor.device)
            .expect("Failed to create third result tensor")
    };

    assert_eq!(result3.shape(), &[2], "Third partition result shape mismatch");

    let result_array = result3.into_data().unwrap();
    let result_data3 = result_array.as_slice().expect("Failed to get result slice");

  
    assert_eq!(result_data3[0], 5.0, "First element of third partition incorrect");
    assert_eq!(result_data3[1], 6.0, "Second element of third partition incorrect");

    // ======================
    // 2D TENSOR TESTS - Testing reshape functionality
    // ======================

    // Create 2D tensor [2, 3] = [[1,2,3], [4,5,6]]
    let input_tensor_2d = Tensor::from_vec_with_device(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        device
    ).expect("Failed to create 2D tensor");

    // Test Case 4: Partition along dimension 0 (rows) - NO reshape needed (last dim)
    let result4 = {
        let storage = input_tensor_2d.storage.as_ref().expect("2D tensor must have storage");
        let result_storage = storage.partition(0, 0, 1)
            .expect("2D partition along dim 0 failed");
        Tensor::from_storage_backend(result_storage, input_tensor_2d.device)
            .expect("Failed to create 2D result tensor")
    };

    // Should extract first row: [1, 2, 3] -> shape [1, 3]
    assert_eq!(result4.shape(), &[1, 3], "2D partition dim 0 shape mismatch");
    let result4_data = result4.into_data().unwrap();
    let result4_slice = result4_data.as_slice().expect("Failed to get 2D result slice");
    assert_eq!(result4_slice, &[1.0, 2.0, 3.0], "2D partition dim 0 data incorrect");

    // Test Case 5: Partition along dimension 1 (columns) - REQUIRES reshape
    let result5 = {
        let storage = input_tensor_2d.storage.as_ref().expect("2D tensor must have storage");
        let result_storage = storage.partition(1, 1, 3)
            .expect("2D partition along dim 1 failed");
        Tensor::from_storage_backend(result_storage, input_tensor_2d.device)
            .expect("Failed to create 2D column result tensor")
    };

    // Should extract columns 1-2: [[2,3], [5,6]] -> shape [2, 2]
    assert_eq!(result5.shape(), &[2, 2], "2D partition dim 1 shape mismatch");
    let result5_data = result5.into_data().unwrap();
    let result5_slice = result5_data.as_slice().expect("Failed to get 2D column result slice");
    assert_eq!(result5_slice, &[2.0, 3.0, 5.0, 6.0], "2D partition dim 1 data incorrect");

    // ======================
    // 3D TENSOR TESTS - Complex reshape scenarios
    // ======================

    // Create 3D tensor [2, 2, 3] = [[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]
    let input_tensor_3d = Tensor::from_vec_with_device(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        &[2, 2, 3],
        device
    ).expect("Failed to create 3D tensor");

    // Test Case 6: Partition along dimension 2 (last dim) - NO reshape needed
    let result6 = {
        let storage = input_tensor_3d.storage.as_ref().expect("3D tensor must have storage");
        let result_storage = storage.partition(2, 0, 2)
            .expect("3D partition along last dim failed");
        Tensor::from_storage_backend(result_storage, input_tensor_3d.device)
            .expect("Failed to create 3D last dim result tensor")
    };

    // Should extract first 2 elements of last dimension -> shape [2, 2, 2]
    assert_eq!(result6.shape(), &[2, 2, 2], "3D partition last dim shape mismatch");
    let result6_data = result6.into_data().unwrap();
    let result6_slice = result6_data.as_slice().expect("Failed to get 3D last dim result slice");
    assert_eq!(result6_slice, &[1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0],
              "3D partition last dim data incorrect");

    // Test Case 7: Partition along dimension 0 (first dim) - REQUIRES reshape
    let result7 = {
        let storage = input_tensor_3d.storage.as_ref().expect("3D tensor must have storage");
        let result_storage = storage.partition(0, 1, 2)
            .expect("3D partition along first dim failed");
        Tensor::from_storage_backend(result_storage, input_tensor_3d.device)
            .expect("Failed to create 3D first dim result tensor")
    };

    // Should extract second batch: [[[7,8,9], [10,11,12]]] -> shape [1, 2, 3]
    assert_eq!(result7.shape(), &[1, 2, 3], "3D partition first dim shape mismatch");
    let result7_data = result7.into_data().unwrap();
    let result7_slice = result7_data.as_slice().expect("Failed to get 3D first dim result slice");
    assert_eq!(result7_slice, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
              "3D partition first dim data incorrect");

    // Test Case 8: Partition along dimension 1 (middle dim) - REQUIRES reshape
    let result8 = {
        let storage = input_tensor_3d.storage.as_ref().expect("3D tensor must have storage");
        let result_storage = storage.partition(1, 0, 1)
            .expect("3D partition along middle dim failed");
        Tensor::from_storage_backend(result_storage, input_tensor_3d.device)
            .expect("Failed to create 3D middle dim result tensor")
    };

    // Should extract first row of each batch: [[[1,2,3]], [[7,8,9]]] -> shape [2, 1, 3]
    assert_eq!(result8.shape(), &[2, 1, 3], "3D partition middle dim shape mismatch");
    let result8_data = result8.into_data().unwrap();
    let result8_slice = result8_data.as_slice().expect("Failed to get 3D middle dim result slice");
    assert_eq!(result8_slice, &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
              "3D partition middle dim data incorrect");

    // ======================
    // ERROR CASES
    // ======================

    // Test Case 9: Invalid dimension
    let error_result = {
        let storage = input_tensor.storage.as_ref().expect("Tensor must have storage");
        storage.partition(1, 0, 2) // dimension 1 doesn't exist in 1D tensor
    };
    assert!(error_result.is_err(), "Should fail for invalid dimension");
    assert!(error_result.unwrap_err().contains("out of bounds"),
           "Error message should mention out of bounds");

    // Test Case 10: Invalid indices
    let error_result2 = {
        let storage = input_tensor.storage.as_ref().expect("Tensor must have storage");
        storage.partition(0, 2, 1) // end_index <= start_index
    };
    assert!(error_result2.is_err(), "Should fail for invalid indices");

    // Test Case 11: Index out of range
    let error_result3 = {
        let storage = input_tensor.storage.as_ref().expect("Tensor must have storage");
        storage.partition(0, 0, 10) // end_index > dimension size
    };
    assert!(error_result3.is_err(), "Should fail for index out of range");

    println!("✓ Partition operation test passed - all slice extractions work correctly");
    println!("✓ 1D, 2D, and 3D tensor partitioning validated");
    println!("✓ Reshape functionality for non-last dimensions verified");
    println!("✓ Error handling for invalid inputs confirmed");
}
}
