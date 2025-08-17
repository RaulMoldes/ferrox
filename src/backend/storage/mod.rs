mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
mod extensions;

#[cfg(feature = "cuda")]
use crate::backend::cuda::context::CudaTensor;
#[cfg(feature = "cuda")]
use crate::backend::cuda::ops::CudaOps;
use crate::{FerroxCudaF, FerroxF};

pub use cpu::CPUStorage;
#[cfg(feature = "cuda")]
pub use cuda::CUDAStorage;

use crate::backend::FerroxCudaN;

use ndarray::ArrayD;
use std::any::Any;
use std::fmt::Debug;

/// Trait for different storage ownership patterns
/// This allows us to have different storage implementations without enum overhead
pub trait StorageBackend<T>: Debug + Any
where
    T: FerroxCudaN,
{
    /// Get tensor shape
    fn shape(&self) -> &[usize];

    // Add this method for downcasting
    fn as_any(&self) -> &dyn Any;

    fn into_any(self: Box<Self>) -> Box<dyn Any>;

    /// Get number of dimensions
    fn ndim(&self) -> usize;

    /// Get total number of elements
    fn size(&self) -> usize;

    /// Check if storage is on GPU
    fn is_gpu(&self) -> bool;

    /// Get CPU data if available (may fail for GPU-only storage)
    fn cpu_data(&self) -> Result<&ArrayD<T>, String>;

    /// Get mutable CPU data if available and owned
    fn cpu_data_mut(&mut self) -> Result<&mut ArrayD<T>, String>;

    /// Check if this storage owns its data
    fn owns_data(&self) -> bool;

    fn clone_storage(&self) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise addition: self + other
    /// Returns new storage with result - doesn't modify inputs
    fn add(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise subtraction: self - other
    /// Returns new storage with result - doesn't modify inputs
    fn sub(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise multiplication: self * other
    /// Returns new storage with result - doesn't modify inputs
    fn mul(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise division: self / other
    /// Returns new storage with result - doesn't modify inputs
    fn div(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise minimum: min(self, other)
    /// Returns new storage with element-wise minimum values
    fn min(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise maximum: max(self, other)
    /// Returns new storage with element-wise maximum values
    fn max(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Scalar addition: self + scalar
    /// More efficient than broadcasting scalar to tensor shape
    fn add_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Scalar multiplication: self * scalar
    /// More efficient than broadcasting scalar to tensor shape
    fn mul_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Scalar division: self / scalar
    /// More efficient than broadcasting scalar to tensor shape
    fn div_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String>;

    // Scalar substraction: self - scalar
    /// More efficient than broadcasting scalar to tensor shape
    fn sub_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise negation: -self
    /// Unary operation that negates all elements
    fn neg(&self) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise absolute value: |self|
    /// Unary operation that returns absolute values
    fn abs(&self) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise clamp: clamp(self, min_val, max_val)
    /// Constrains all values to the range [min_val, max_val]
    fn clamp(&self, min_val: T, max_val: T) -> Result<Box<dyn StorageBackend<T>>, String>;

    fn sqrt(&self) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxCudaF;

    fn reciprocal(&self) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise greater than or equal comparison: self >= other
    /// Returns new storage with 1.0 for true, 0.0 for false
    fn greater_equal(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String>;

    fn greater_equal_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise less than or equal comparison: self <= other
    /// Returns new storage with 1.0 for true, 0.0 for false
    fn less_equal(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String>;

    fn less_equal_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise greater than comparison: self > other
    /// Returns new storage with 1.0 for true, 0.0 for false
    fn greater(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>;

    fn greater_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise less than  comparison: self < other
    /// Returns new storage with 1.0 for true, 0.0 for false
    fn less(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>;

    fn less_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise equality comparison: self == other
    /// Returns new storage with 1.0 for true, 0.0 for false
    fn equal(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Logical NOT operation: !self
    /// Flips 0s to 1s and non-zeros to 0s
    fn logical_not(&self) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Range check operation: min_val <= self <= max_val
    /// Returns new storage with 1.0 for values in range, 0.0 otherwise
    fn in_range(&self, min_val: T, max_val: T) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Sign function: sign(self)
    /// Returns 1.0 for positive, -1.0 for negative, 0.0 for zero
    fn sign(&self) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Matrix multiplication: self @ other
    /// Requires 2D tensors with compatible shapes
    fn matmul(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Sigmoid activation function: 1 / (1 + exp(-self))
    /// Returns new storage with sigmoid applied element-wise
    fn sigmoid(&self) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxF;

    /// Softmax activation function.
    /// Returns new storage with softmax applied element-wise
    fn softmax(&self) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxF;

    /// Batch-aware softmax along specified axis
    /// More efficient than element-wise softmax for batched data
    fn softmax_batched(&self, axis: usize) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxCudaF;

    /// ReLU activation function: max(0, self)
    /// Returns new storage with ReLU applied element-wise
    fn relu(&self) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Exponential function: exp(self)
    /// Returns new storage with exponential applied element-wise
    fn exp(&self) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxCudaF;

    /// Natural logarithm: ln(self)
    /// Returns new storage with natural log applied element-wise
    fn log(&self) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxCudaF;

    /// Hyperbolic tangent: tanh(self)
    /// Returns new storage with tanh applied element-wise
    fn tanh(&self) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxCudaF;

    /// Element-wise power: self ^ other
    /// Returns new storage with element-wise power operation
    fn powf(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxCudaF;

    /// Scalar power: self ^ scalar
    /// Returns new storage with scalar power applied element-wise
    fn power_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxCudaF;

    /// Sum reduction along multiple axes
    fn sum(
        &self,
        axes: Option<&[usize]>,
        keep_dims: bool,
    ) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Mean reduction along multiple axes
    fn mean(
        &self,
        axes: Option<&[usize]>,
        keep_dims: bool,
    ) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Maximum values along multiple axes
    fn max_reduce(
        &self,
        axes: Option<&[usize]>,
        keep_dims: bool,
    ) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Minimum values along multiple axes
    fn min_reduce(
        &self,
        axes: Option<&[usize]>,
        keep_dims: bool,
    ) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Broadcasting operation - expands tensor to target shape following broadcasting rules
    /// In-place operation that only changes the view, not the underlying data
    fn broadcast_to(&mut self, target_shape: &[usize]) -> Result<(), String>;

    /// Reshape operation - changes tensor shape while preserving total elements
    /// In-place operation that only changes the view, not the underlying data
    fn reshape(&mut self, new_shape: &[usize]) -> Result<(), String>;

    /// Transpose operation - permutes tensor axes
    /// If axes is None, reverses all axes (default transpose)
    /// In-place operation that only changes the view, not the underlying data
    fn transpose(&mut self, axes: Option<&[usize]>) -> Result<(), String>;

    /// Unsqueeze operation - adds dimension of size 1 at specified axis
    /// Similar to tf.expand_dims - in-place operation
    fn unsqueeze(&mut self, axis: usize) -> Result<(), String>;

    /// Squeeze operation - removes dimensions of size 1
    /// If axis is None, removes all dimensions of size 1
    /// If axis is Some(ax), removes only specified axis if it has size 1
    /// In-place operation that only changes the view, not the underlying data
    fn squeeze(&mut self, axis: Option<usize>) -> Result<(), String>;

    /// Expand_dims operation - alias for unsqueeze for TensorFlow compatibility
    /// Adds dimension of size 1 at specified axis
    fn expand_dims(&mut self, axis: usize) -> Result<(), String> {
        // Reuse unsqueeze implementation - no code duplication
        self.unsqueeze(axis)
    }
    /// 2D Convolution operation
    /// Performs standard convolution using im2col transformation for efficiency
    /// Returns new storage with convolution result - doesn't modify inputs
    fn conv2d(
        &self,
        filter: &dyn StorageBackend<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Box<dyn StorageBackend<T>>, String>;

    fn conv1d(&self, filter: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>;

    fn deconv1d(
        &self,
        filter: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String>;

    fn cross_correlation2d(
        &self,
        other: &dyn StorageBackend<T>,
        output_shape: &[usize],
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Box<dyn StorageBackend<T>>, String>;

    fn cross_correlation1d(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String>;

    fn deconv2d(
        &self, // I AM THE INPUT to the deconv (grad:output)
        filter: &dyn StorageBackend<T>,
        output_shape: &[usize],
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Iterator over storage elements - returns owned values
    /// This is the most flexible iteration method that works for both CPU and GPU
    fn iter_values(&self) -> Result<Vec<T>, String>;

    /// Get flat index access to elements (if supported by storage)
    /// Returns None if storage doesn't support flat indexing
    fn get_flat(&self, index: usize) -> Result<Option<T>, String>;

    /// Get multi-dimensional index access to elements (if supported)
    /// Returns None if storage doesn't support multi-dim indexing
    fn get_multi(&self, indices: &[usize]) -> Result<Option<T>, String>;


    /// 2D Max Pooling operation
    /// Performs max pooling over spatial dimensions
    /// Input shape: [batch, channels, height, width]
    /// Output shape: [batch, channels, height_out, width_out]
    fn maxpool2d(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// 2D Average Pooling operation
    /// Performs average pooling over spatial dimensions
    /// Input shape: [batch, channels, height, width]
    /// Output shape: [batch, channels, height_out, width_out]
    fn avgpool2d(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// 1D Max Pooling operation
    /// Performs max pooling over temporal/sequential dimension
    /// Input shape: [batch, channels, length]
    /// Output shape: [batch, channels, length_out]
    fn maxpool1d(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// 1D Average Pooling operation
    /// Performs average pooling over temporal/sequential dimension
    /// Input shape: [batch, channels, length]
    /// Output shape: [batch, channels, length_out]
    fn avgpool1d(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Box<dyn StorageBackend<T>>, String>;

    /*fn execute_custom_op<R>(&self, op: Box<dyn CustomOperation<T, R>>) -> Result<R, String>;*/
}

#[cfg(test)]
mod storage_backend_tests {

    use crate::backend::storage::{CPUStorage, StorageBackend};
    use ndarray::ArrayD;

    #[test]
    fn test_cpu_storage_creation() {
        // Create test data array
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let array = ArrayD::from_shape_vec(shape.clone(), data.clone()).unwrap();

        // Create CPU storage
        let storage = CPUStorage::<f32>::new(array);

        // Verify storage properties
        assert_eq!(storage.shape(), &shape);
        assert_eq!(storage.size(), 6);
        assert_eq!(storage.ndim(), 2);
        assert!(!storage.is_gpu()); // CPU storage should return false for is_gpu
    }

    #[test]
    fn test_storage_element_operations() {
        let data = vec![2.0f32, 4.0, 6.0, 8.0];
        let array = ArrayD::from_shape_vec(vec![2, 2], data).unwrap();
        let storage_a = CPUStorage::<f32>::new(array);

        let data_b = vec![1.0f32, 2.0, 3.0, 4.0];
        let array_b = ArrayD::from_shape_vec(vec![2, 2], data_b).unwrap();
        let storage_b = CPUStorage::<f32>::new(array_b);

        // Test addition
        let add_result = storage_a.add(&storage_b).unwrap();
        assert_eq!(add_result.shape(), &[2, 2]);

        // Test multiplication
        let mul_result = storage_a.mul(&storage_b).unwrap();
        assert_eq!(mul_result.shape(), &[2, 2]);

        // Test division
        let div_result = storage_a.div(&storage_b).unwrap();
        assert_eq!(div_result.shape(), &[2, 2]);
    }

    #[test]
    fn test_storage_scalar_operations() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let array = ArrayD::from_shape_vec(vec![2, 2], data).unwrap();
        let storage = CPUStorage::<f32>::new(array);

        // Test scalar addition
        let add_scalar_result = storage.add_scalar(5.0).unwrap();
        assert_eq!(add_scalar_result.shape(), storage.shape());

        // Test scalar multiplication
        let mul_scalar_result = storage.mul_scalar(2.0).unwrap();
        assert_eq!(mul_scalar_result.shape(), storage.shape());

        // Test scalar division
        let div_scalar_result = storage.div_scalar(2.0).unwrap();
        assert_eq!(div_scalar_result.shape(), storage.shape());
    }

    #[test]
    fn test_matrix_multiplication() {
        // Create compatible matrices for multiplication
        let data_a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let array_a = ArrayD::from_shape_vec(vec![2, 3], data_a).unwrap();
        let storage_a = CPUStorage::<f32>::new(array_a);

        let data_b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2 matrix
        let array_b = ArrayD::from_shape_vec(vec![3, 2], data_b).unwrap();
        let storage_b = CPUStorage::<f32>::new(array_b);

        // Perform matrix multiplication
        let matmul_result = storage_a.matmul(&storage_b).unwrap();

        // Result should be 2x2 matrix
        assert_eq!(matmul_result.shape(), &[2, 2]);
        assert_eq!(matmul_result.size(), 4);
    }
}
