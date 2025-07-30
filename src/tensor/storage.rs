// src/tensor/storage.rs
use crate::backend::manager::get_backend;
use ndarray::Axis;
use ndarray::{ArrayD, IxDyn};
use rand::Rng;
use rand_distr::StandardUniform;
use std::borrow::Cow;
use std::fmt::Debug;

#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaTensor;

#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;

use crate::backend::number::{CPUFloat, GPUFloat};

/// Trait for different storage ownership patterns
/// This allows us to have different storage implementations without enum overhead
pub trait StorageBackend<T>: Debug
where
    T: GPUFloat,
{
    /// Get tensor shape
    fn shape(&self) -> &[usize];

    // Add this method for downcasting
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None // Default implementation returns None for borrowed storage
    }

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

    fn sqrt(&self) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise greater than or equal comparison: self >= other
    /// Returns new storage with 1.0 for true, 0.0 for false
    fn greater_equal(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise less than or equal comparison: self <= other
    /// Returns new storage with 1.0 for true, 0.0 for false
    fn less_equal(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise greater than comparison: self > other
    /// Returns new storage with 1.0 for true, 0.0 for false
    fn greater(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise less than  comparison: self < other
    /// Returns new storage with 1.0 for true, 0.0 for false
    fn less(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>;

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
    fn sigmoid(&self) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// ReLU activation function: max(0, self)
    /// Returns new storage with ReLU applied element-wise
    fn relu(&self) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Exponential function: exp(self)
    /// Returns new storage with exponential applied element-wise
    fn exp(&self) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Natural logarithm: ln(self)
    /// Returns new storage with natural log applied element-wise
    fn log(&self) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Hyperbolic tangent: tanh(self)
    /// Returns new storage with tanh applied element-wise
    fn tanh(&self) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise power: self ^ other
    /// Returns new storage with element-wise power operation
    fn powf(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Scalar power: self ^ scalar
    /// Returns new storage with scalar power applied element-wise
    fn power_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Sum reduction along multiple axes
    fn sum(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Mean reduction along multiple axes
    fn mean(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Maximum values along multiple axes
    fn max_reduce(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Minimum values along multiple axes
    fn min_reduce(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Broadcasting operation - expands tensor to target shape following broadcasting rules
    /// Returns new storage with broadcasted shape - doesn't modify input
    fn broadcast_to(&self, target_shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Reshape operation - changes tensor shape while preserving total elements
    /// Returns new storage with new shape - doesn't modify input
    fn reshape(&self, new_shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Transpose operation - permutes tensor axes
    /// If axes is None, reverses all axes (default transpose)
    /// Returns new storage with transposed data - doesn't modify input
    fn transpose(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Unsqueeze operation - adds dimension of size 1 at specified axis
    /// Similar to tf.expand_dims - returns new storage with added dimension
    fn unsqueeze(&self, axis: usize) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Squeeze operation - removes dimensions of size 1
    /// If axis is None, removes all dimensions of size 1
    /// If axis is Some(ax), removes only specified axis if it has size 1
    fn squeeze(&self, axis: Option<usize>) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Expand_dims operation - alias for unsqueeze for TensorFlow compatibility
    /// Adds dimension of size 1 at specified axis
    fn expand_dims(&self, axis: usize) -> Result<Box<dyn StorageBackend<T>>, String> {
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

    /// Iterator over storage elements - returns owned values
    /// This is the most flexible iteration method that works for both CPU and GPU
    fn iter_values(&self) -> Result<Vec<T>, String>;

    /// Get flat index access to elements (if supported by storage)
    /// Returns None if storage doesn't support flat indexing
    fn get_flat(&self, index: usize) -> Result<Option<T>, String>;

    /// Get multi-dimensional index access to elements (if supported)
    /// Returns None if storage doesn't support multi-dim indexing
    fn get_multi(&self, indices: &[usize]) -> Result<Option<T>, String>;
}

/// Owned CPU storage
#[derive(Debug, Clone)]
pub struct CPUOwnedStorage<T> {
    pub data: ArrayD<T>,
}

impl<T> CPUOwnedStorage<T> {
    pub fn new(data: ArrayD<T>) -> Self {
        Self { data }
    }
}

impl<T> CPUOwnedStorage<T>
where
    T: crate::backend::number::GPUFloat + Clone,
{
    /// Convert image patches to column matrix (im2col) - reused from original impl
    /// This transforms 4D convolution into efficient 2D matrix multiplication
    fn im2col(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<T>, String> {
        let input_shape = self.shape();
        let (batch, channels, in_h, in_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (kernel_h, kernel_w) = kernel_size;

        let out_h = (in_h + 2 * padding.0 - kernel_h) / stride.0 + 1;
        let out_w = (in_w + 2 * padding.1 - kernel_w) / stride.1 + 1;

        let col_height = channels * kernel_h * kernel_w;
        let col_width = batch * out_h * out_w;

        let mut col_data = vec![<T as CPUFloat>::zero(); col_height * col_width];
        let input_data = if let Some(data) = self.data.as_slice() {
            data
        } else {
            return Err("Input data is empty!".to_string());
        };

        // Extract patches and arrange them as columns for matrix multiplication
        for b in 0..batch {
            for c in 0..channels {
                for ky in 0..kernel_h {
                    for kx in 0..kernel_w {
                        let col_row = c * kernel_h * kernel_w + ky * kernel_w + kx;

                        for out_y in 0..out_h {
                            for out_x in 0..out_w {
                                let in_y = out_y * stride.0 + ky;
                                let in_x = out_x * stride.1 + kx;
                                let col_col = b * out_h * out_w + out_y * out_w + out_x;

                                // Handle padding by checking bounds
                                if in_y >= padding.0
                                    && in_y < in_h + padding.0
                                    && in_x >= padding.1
                                    && in_x < in_w + padding.1
                                {
                                    let actual_y = in_y - padding.0;
                                    let actual_x = in_x - padding.1;

                                    if actual_y < in_h && actual_x < in_w {
                                        let input_idx = b * (channels * in_h * in_w)
                                            + c * (in_h * in_w)
                                            + actual_y * in_w
                                            + actual_x;
                                        col_data[col_row * col_width + col_col] =
                                            input_data[input_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        ArrayD::from_shape_vec(IxDyn(&[col_height, col_width]), col_data)
            .map_err(|e| format!("Failed to create im2col matrix: {}", e))
    }

    /// Standard 2D convolution implementation using im2col + GEMM
    fn conv2d_impl(
        &self,
        filter: &ArrayD<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<T>, String> {
        let input_shape = self.shape();
        let filter_shape = filter.shape();

        let (batch, in_channels, in_h, in_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (out_channels, _, kernel_h, kernel_w) = (
            filter_shape[0],
            filter_shape[1],
            filter_shape[2],
            filter_shape[3],
        );

        let out_h = (in_h + 2 * padding.0 - kernel_h) / stride.0 + 1;
        let out_w = (in_w + 2 * padding.1 - kernel_w) / stride.1 + 1;

        // Transform input to column matrix for efficient GEMM
        let col_matrix = self.im2col((kernel_h, kernel_w), stride, padding)?;

        // Reshape filter for matrix multiplication
        let filter_reshaped = filter
            .clone()
            .into_shape_with_order(IxDyn(&[out_channels, in_channels * kernel_h * kernel_w]))
            .map_err(|e| format!("Filter reshape failed: {}", e))?;

        // Perform convolution as matrix multiplication: filter @ col_matrix
        let im2col_view: ndarray::ArrayView2<T> = col_matrix
            .view()
            .into_dimensionality()
            .map_err(|e| format!("Shape error: {}", e))?;

        let filter_view: ndarray::ArrayView2<T> = filter_reshaped
            .view()
            .into_dimensionality()
            .map_err(|e| format!("Shape error: {}", e))?;

        let output_2d = filter_view.dot(&im2col_view);

        // Transpose result from [out_channels, batch * out_h * out_w] to [batch, out_channels, out_h, out_w]
        let output_data: Vec<T> = if let Some(out_slice) = output_2d.as_slice() {
            out_slice.to_vec()
        } else {
            panic!("Attempted to slice an empty output!")
        };
        let mut final_output = vec![<T as CPUFloat>::zero(); batch * out_channels * out_h * out_w];

        for out_c in 0..out_channels {
            for b in 0..batch {
                for y in 0..out_h {
                    for x in 0..out_w {
                        let src_idx =
                            out_c * (batch * out_h * out_w) + b * (out_h * out_w) + y * out_w + x;
                        let dst_idx = b * (out_channels * out_h * out_w)
                            + out_c * (out_h * out_w)
                            + y * out_w
                            + x;
                        final_output[dst_idx] = output_data[src_idx];
                    }
                }
            }
        }

        ArrayD::from_shape_vec(IxDyn(&[batch, out_channels, out_h, out_w]), final_output)
            .map_err(|e| format!("Failed to create output tensor: {}", e))
    }
}

impl<T> StorageBackend<T> for CPUOwnedStorage<T>
where
    T: GPUFloat,
{
    fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn ndim(&self) -> usize {
        self.data.ndim()
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn is_gpu(&self) -> bool {
        false
    }

    fn cpu_data(&self) -> Result<&ArrayD<T>, String> {
        Ok(&self.data)
    }

    fn cpu_data_mut(&mut self) -> Result<&mut ArrayD<T>, String> {
        Ok(&mut self.data)
    }

    fn owns_data(&self) -> bool {
        true
    }

    fn clone_storage(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        Ok(Box::new(self.clone()))
    }

    fn add(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Get other's CPU data - this handles CPU-CPU operations
        let other_data = other.cpu_data()?;

        // Shape broadcasting check - ndarray handles this automatically but we validate first
        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for addition: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        // Element-wise addition using ndarray's efficient implementation
        let result = &self.data + other_data;
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn sub(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for subtraction: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        let result = &self.data - other_data;
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn mul(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for multiplication: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        let result = &self.data * other_data;
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn div(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for division: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        let result = &self.data / other_data;
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn min(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for min operation: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        // Use flat iteration for efficiency - works with any dimensional tensor
        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| if a <= b { a } else { b })
            .collect();

        let result_array = ndarray::Array::from_shape_vec(self.data.raw_dim(), result_data)
            .map_err(|e| format!("Failed to create result tensor: {e}"))?;

        Ok(Box::new(CPUOwnedStorage::new(result_array)))
    }

    fn max(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for max operation: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| if a >= b { a } else { b })
            .collect();

        let result_array = ndarray::Array::from_shape_vec(self.data.raw_dim(), result_data)
            .map_err(|e| format!("Failed to create result tensor: {e}"))?;

        Ok(Box::new(CPUOwnedStorage::new(result_array)))
    }

    fn add_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        // ndarray's scalar operations are very efficient - no broadcasting overhead
        let result = &self.data + scalar;
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn mul_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = &self.data * scalar;
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn sub_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        // ndarray's scalar operations are very efficient - no broadcasting overhead
        let result = &self.data - scalar;
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn div_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = &self.data / scalar;
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn neg(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Unary negation - ndarray handles this efficiently
        let result = self.data.mapv(|x| -x);
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn abs(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Element-wise absolute value using mapv for efficiency
        let result_data: Vec<T> = self.data.iter().map(|&x| x.abs()).collect();

        let result_array = ndarray::Array::from_shape_vec(self.data.raw_dim(), result_data)
            .map_err(|e| format!("Failed to create result tensor: {e}"))?;

        Ok(Box::new(CPUOwnedStorage::new(result_array)))
    }

    fn clamp(&self, min_val: T, max_val: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Element-wise clamping using mapv - efficient vectorized operation
        let result_data = self.data.mapv(|x| {
            if x < min_val {
                min_val
            } else if x > max_val {
                max_val
            } else {
                x
            }
        });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn sqrt(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_data = self.data.mapv(|x| x.sqrt());
        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn greater_equal(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        let storage = self.compare(other, |&a, &b| {
            if a >= b {
                <T as CPUFloat>::one()
            } else {
                <T as CPUFloat>::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn less_equal(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        let storage = self.compare(other, |&a, &b| {
            if a <= b {
                <T as CPUFloat>::one()
            } else {
                <T as CPUFloat>::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn greater(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let storage = self.compare(other, |&a, &b| {
            if a > b {
                <T as CPUFloat>::one()
            } else {
                <T as CPUFloat>::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn less(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let storage = self.compare(other, |&a, &b| {
            if a < b {
                <T as CPUFloat>::one()
            } else {
                <T as CPUFloat>::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn equal(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let storage = self.compare(other, |&a, &b| {
            if a == b {
                <T as CPUFloat>::one()
            } else {
                <T as CPUFloat>::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn logical_not(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Flip 0s to 1s and non-zeros to 0s
        let result_data = self.data.mapv(|x| {
            if x == <T as crate::backend::number::CPUNumber>::zero() {
                <T as crate::backend::number::CPUNumber>::one()
            } else {
                <T as crate::backend::number::CPUNumber>::zero()
            }
        });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn in_range(&self, min_val: T, max_val: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Check if values are in range [min_val, max_val]
        let result_data = self.data.mapv(|x| {
            if x >= min_val && x <= max_val {
                <T as crate::backend::number::CPUNumber>::one()
            } else {
                <T as crate::backend::number::CPUNumber>::zero()
            }
        });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn sign(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Return 1 for positive, -1 for negative, 0 for zero
        let result_data = self.data.mapv(|x| {
            if x > <T as crate::backend::number::CPUNumber>::zero() {
                <T as crate::backend::number::CPUNumber>::one()
            } else if x < <T as crate::backend::number::CPUNumber>::zero() {
                -<T as crate::backend::number::CPUNumber>::one()
            } else {
                <T as crate::backend::number::CPUNumber>::zero()
            }
        });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn matmul(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: Clone + ndarray::LinalgScalar,
    {
        let other_data = other.cpu_data()?;

        if self.data.ndim() != 2 || other_data.ndim() != 2 {
            return Err("Matrix multiplication requires 2D tensors".to_string());
        }

        let a_shape = self.data.shape();
        let b_shape = other_data.shape();

        if a_shape[1] != b_shape[0] {
            return Err(format!(
                "Matrix multiplication shape mismatch: ({}, {}) @ ({}, {})",
                a_shape[0], a_shape[1], b_shape[0], b_shape[1]
            ));
        }

        // Convert to 2D views for matrix multiplication
        let a: ndarray::ArrayView2<T> = self
            .data
            .view()
            .into_dimensionality()
            .map_err(|e| format!("Failed to convert to 2D view: {}", e))?;
        let b: ndarray::ArrayView2<T> = other_data
            .view()
            .into_dimensionality()
            .map_err(|e| format!("Failed to convert to 2D view: {}", e))?;

        // Perform matrix multiplication using ndarray's dot product
        let result = a.dot(&b);

        Ok(Box::new(CPUOwnedStorage::new(result.into_dyn())))
    }

    fn sigmoid(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Sigmoid function: 1 / (1 + exp(-x))
        let result_data = self.data.mapv(|x| {
            let one = <T as crate::backend::number::CPUNumber>::one();
            let neg_x = -x;
            one / (one + neg_x.exp())
        });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn relu(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // ReLU activation: max(0, x)
        let result_data = self.data.mapv(|x| {
            let zero = <T as crate::backend::number::CPUNumber>::zero();
            if x > zero { x } else { zero }
        });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn exp(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Element-wise exponential
        let result_data = self.data.mapv(|x| x.exp());
        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn log(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Element-wise natural logarithm
        let result_data = self.data.mapv(|x| x.ln());
        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn tanh(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Hyperbolic tangent using the same formula as your original
        let result_data = self.data.mapv(|x| {
            let e_x = x.exp();
            let e_neg_x = (-x).exp();
            (e_x - e_neg_x) / (e_x + e_neg_x)
        });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn powf(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for powf: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        // Element-wise power using ndarray's Zip
        let result_data = ndarray::Zip::from(&self.data)
            .and(&*other_data)
            .map_collect(|&a, &b| a.powf(b));

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn power_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Scalar power operation
        let result_data = self.data.mapv(|x| x.powf(scalar));
        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn sum(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Use reduce_axes with ndarray's sum_axis function
        let result = self.reduce(axes, |array, ax| array.sum_axis(ax))?;
        Ok(Box::new(result) as Box<dyn StorageBackend<T>>)
    }

    fn mean(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        // First compute sum using reduce_axes
        let sum_result = self.sum(axes)?;

        // Calculate the number of elements being averaged over
        let divisor = match axes {
            Some(axes_list) => {
                // Product of dimensions being reduced
                axes_list
                    .iter()
                    .map(|&ax| self.data.shape()[ax])
                    .product::<usize>() as f64
            }
            None => {
                // All elements if no axes specified
                self.data.len() as f64
            }
        };

        // Convert divisor to tensor type and divide
        let divisor_scalar = <T as CPUFloat>::from_f64(1.0 / divisor)
            .ok_or("Failed to convert divisor to tensor type")?;

        sum_result.mul_scalar(divisor_scalar)
    }

    fn max_reduce(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Use reduce_axes with custom max reduction function
        // ndarray doesn't have a direct max_axis function, so we implement our own
        let result = self.reduce(axes, |array, ax| {
            let first = if let Some(f) = array.first() {
                f.clone()
            } else {
                panic!("Array is empty! Cannot reduce over empty array");
            };

            // Fold along the specified axis to find maximum values
            array.fold_axis(ax, first, |&acc, &x| if x > acc { x } else { acc })
        })?;

        Ok(Box::new(result) as Box<dyn StorageBackend<T>>)
    }

    fn min_reduce(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Use reduce_axes with custom min reduction function
        // Similar to max_axes but finding minimum values
        let result = self.reduce(axes, |array, ax| {
            let first = if let Some(f) = array.first() {
                f.clone()
            } else {
                panic!("Array is empty! Cannot reduce over empty array");
            };
            // Fold along the specified axis to find minimum values
            array.fold_axis(ax, first, |&acc, &x| if x < acc { x } else { acc })
        })?;
        Ok(Box::new(result) as Box<dyn StorageBackend<T>>)
    }

    fn broadcast_to(&self, target_shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Use ndarray's efficient broadcasting - handles all broadcasting rules automatically
        match self.data.broadcast(target_shape) {
            Some(broadcasted) => Ok(Box::new(CPUOwnedStorage::new(broadcasted.to_owned()))),
            None => Err(format!(
                "Cannot broadcast {:?} to {:?}",
                self.data.shape(),
                target_shape
            )),
        }
    }

    fn reshape(&self, new_shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Validate total elements remain the same
        let total_elements: usize = self.data.shape().iter().product();
        let new_total_elements: usize = new_shape.iter().product();

        if total_elements != new_total_elements {
            return Err(format!(
                "Cannot reshape tensor with {} elements to shape with {} elements",
                total_elements, new_total_elements
            ));
        }

        // Use ndarray's efficient reshape with order preservation
        match self.data.clone().into_shape_with_order(IxDyn(new_shape)) {
            Ok(reshaped) => Ok(Box::new(CPUOwnedStorage::new(reshaped))),
            Err(e) => Err(format!("Failed to reshape tensor: {e}")),
        }
    }

    fn transpose(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        match axes {
            Some(axes_order) => {
                // Validate axes specification
                if axes_order.len() != self.data.ndim() {
                    return Err(format!(
                        "Axes length {} doesn't match tensor dimensions {}",
                        axes_order.len(),
                        self.data.ndim()
                    ));
                }

                // Verify axes is valid permutation (contains each index 0..ndim exactly once)
                let mut sorted_axes = axes_order.to_vec();
                sorted_axes.sort_unstable();
                let expected: Vec<usize> = (0..self.data.ndim()).collect();
                if sorted_axes != expected {
                    return Err(format!(
                        "Invalid axes permutation: {:?}. Must be a permutation of 0..{}",
                        axes_order,
                        self.data.ndim()
                    ));
                }

                // Perform transpose with specified axes order
                let transposed = self.data.clone().permuted_axes(axes_order);
                Ok(Box::new(CPUOwnedStorage::new(transposed)))
            }
            None => {
                // Default transpose - reverse all axes order
                match self.data.ndim() {
                    0 | 1 => {
                        // 0D and 1D tensors unchanged by transpose
                        Ok(Box::new(self.clone()))
                    }
                    2 => {
                        // 2D matrices use efficient reversed_axes
                        let transposed = self.data.clone().reversed_axes();
                        Ok(Box::new(CPUOwnedStorage::new(transposed)))
                    }
                    _ => {
                        // Higher dimensions require explicit axis permutation
                        let axes_order: Vec<usize> = (0..self.data.ndim()).rev().collect();
                        let transposed = self.data.clone().permuted_axes(axes_order.as_slice());
                        Ok(Box::new(CPUOwnedStorage::new(transposed)))
                    }
                }
            }
        }
    }

    fn unsqueeze(&self, axis: usize) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Validate axis bounds - can insert at positions 0..ndim (inclusive)
        if axis > self.data.ndim() {
            return Err(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                self.data.ndim()
            ));
        }

        // Insert new dimension of size 1 at specified axis
        let expanded = self.data.clone().insert_axis(Axis(axis));
        Ok(Box::new(CPUOwnedStorage::new(expanded)))
    }

    fn squeeze(&self, axis: Option<usize>) -> Result<Box<dyn StorageBackend<T>>, String> {
        match axis {
            Some(ax) => {
                // Squeeze specific axis - validate it exists and has size 1
                if ax >= self.data.ndim() {
                    return Err(format!(
                        "Axis {} out of bounds for tensor with {} dimensions",
                        ax,
                        self.data.ndim()
                    ));
                }

                if self.data.shape()[ax] != 1 {
                    return Err(format!(
                        "Cannot squeeze axis {} with size {}",
                        ax,
                        self.data.shape()[ax]
                    ));
                }

                let squeezed = self.data.clone().remove_axis(Axis(ax));
                Ok(Box::new(CPUOwnedStorage::new(squeezed)))
            }
            None => {
                // Remove all dimensions of size 1
                let mut result = self.data.clone();
                let mut axes_to_remove = Vec::new();

                // Collect axes with size 1
                for (i, &size) in self.data.shape().iter().enumerate() {
                    if size == 1 {
                        axes_to_remove.push(i);
                    }
                }

                // Remove axes in reverse order to maintain valid indices
                for &ax in axes_to_remove.iter().rev() {
                    result = result.remove_axis(Axis(ax));
                }

                Ok(Box::new(CPUOwnedStorage::new(result)))
            }
        }
    }

    fn conv2d(
        &self,
        filter: &dyn StorageBackend<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Ensure filter is also CPU storage
        let filter_data = filter.cpu_data()?;

        let input_shape = self.shape();
        let filter_shape = filter.shape();

        // Validate input dimensions for conv2d
        if input_shape.len() != 4 || filter_shape.len() != 4 {
            return Err("Conv2D requires 4D tensors [batch, channels, height, width]".to_string());
        }

        let result = self.conv2d_impl(filter_data, stride, padding)?;
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn iter_values(&self) -> Result<Vec<T>, String> {
        // Efficient cloning of all values for iteration
        Ok(self.data.iter().cloned().collect())
    }

    fn get_flat(&self, index: usize) -> Result<Option<T>, String> {
        // Flat indexing using ndarray's slice functionality
        match self.data.as_slice() {
            Some(slice) => {
                if index < slice.len() {
                    Ok(Some(slice[index]))
                } else {
                    Ok(None)
                }
            }
            None => Err("Data is not contiguous for flat indexing".to_string()),
        }
    }

    fn get_multi(&self, indices: &[usize]) -> Result<Option<T>, String> {
        // Multi-dimensional indexing using ndarray
        if indices.len() != self.data.ndim() {
            return Ok(None);
        }

        // Check bounds before accessing
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.data.shape()[i] {
                return Ok(None);
            }
        }

        Ok(Some(self.data[ndarray::IxDyn(indices)]))
    }
}

impl<T> CPUOwnedStorage<T>
where
    T: GPUFloat,
{
    // Move the generic reduce method to the concrete implementation
    // This avoids making StorageBackend non-dyn-compatible
    fn reduce<F>(
        &self,
        axes: Option<&[usize]>,
        reduction_fn: F,
    ) -> Result<CPUOwnedStorage<T>, String>
    where
        F: Fn(&ndarray::ArrayD<T>, ndarray::Axis) -> ndarray::ArrayD<T>,
    {
        match axes {
            Some(axes_list) => {
                // Validate axes bounds before processing
                for &ax in axes_list {
                    if ax >= self.data.ndim() {
                        return Err(format!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            ax,
                            self.data.ndim()
                        ));
                    }
                }

                let mut result = self.data.clone();
                // Sort in descending order to prevent index shifting during reduction
                let mut sorted_axes = axes_list.to_vec();
                sorted_axes.sort_unstable();
                sorted_axes.reverse();
                sorted_axes.dedup();

                // Apply reduction sequentially along each axis
                for &ax in &sorted_axes {
                    result = reduction_fn(&result, ndarray::Axis(ax));
                }

                Ok(CPUOwnedStorage::new(result))
            }
            None => {
                // Reduce across all dimensions to get scalar result
                let mut result = self.data.clone();
                for dim in (0..result.ndim()).rev() {
                    result = reduction_fn(&result, ndarray::Axis(dim));
                }
                Ok(CPUOwnedStorage::new(result))
            }
        }
    }

    /// Generic comparison method using ndarray::Zip for efficiency
    /// This is a helper method to avoid code duplication in comparison operations
    fn compare<F>(
        &self,
        other: &dyn StorageBackend<T>,
        comparison_fn: F,
    ) -> Result<CPUOwnedStorage<T>, String>
    where
        F: Fn(&T, &T) -> T,
    {
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for comparison: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        // Use ndarray's Zip for efficient element-wise comparison
        let result_data = ndarray::Zip::from(&self.data)
            .and(other_data)
            .map_collect(|&a, &b| comparison_fn(&a, &b));

        // Return the owned storage result
        Ok(CPUOwnedStorage::new(result_data))
    }

    pub fn zeros(shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
        T: rand_distr::num_traits::Zero,
    {
        // Create ndarray with zeros directly - more efficient than device layer
        let data = ndarray::ArrayD::zeros(ndarray::IxDyn(shape));
        Ok(Box::new(CPUOwnedStorage::new(data)))
    }

    pub fn ones(shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
        T: rand_distr::num_traits::One,
    {
        // Create ndarray with ones directly
        let data = ndarray::ArrayD::ones(ndarray::IxDyn(shape));
        Ok(Box::new(CPUOwnedStorage::new(data)))
    }

    pub fn full(shape: &[usize], value: T) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
    {
        // Create ndarray filled with specific value
        let data = ndarray::ArrayD::from_elem(ndarray::IxDyn(shape), value);
        Ok(Box::new(CPUOwnedStorage::new(data)))
    }

    pub fn randn(shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
        StandardUniform: rand_distr::Distribution<T>,
    {
        let mut rng = rand::rng();
        let total_elements: usize = shape.iter().product();
        let two = <T as CPUFloat>::from_f64(2.0).expect("Cannot cast from f64");
        let one = <T as CPUFloat>::one();
        let data: Vec<T> = (0..total_elements)
            .map(|_| rng.random::<T>() * two - one) // Simple random between -1 and 1
            .collect();
        let data_array = ArrayD::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| format!("Failed to create array from data: {}", e))?;
        Ok(Box::new(CPUOwnedStorage::new(data_array)))
    }

    /// Conditional selection operation
    pub fn where_condition(
        condition: &dyn StorageBackend<T>,
        true_vals: &dyn StorageBackend<T>,
        false_vals: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
    {
        // Get data from all three storages
        let condition_data = condition.cpu_data()?;
        let true_data = true_vals.cpu_data()?;
        let false_data = false_vals.cpu_data()?;

        // Validate shapes match
        if condition_data.shape() != true_data.shape()
            || condition_data.shape() != false_data.shape()
        {
            return Err("Shape mismatch in where_condition".to_string());
        }

        // Element-wise selection using iterator zip
        let result_data: Vec<T> = condition_data
            .iter()
            .zip(true_data.iter())
            .zip(false_data.iter())
            .map(|((&cond, &true_val), &false_val)| {
                if cond > <T as CPUFloat>::zero() {
                    true_val
                } else {
                    false_val
                }
            })
            .collect();

        // Create result array with same shape
        let result_array = ndarray::Array::from_shape_vec(condition_data.raw_dim(), result_data)
            .map_err(|e| format!("Failed to create result: {}", e))?;

        Ok(Box::new(CPUOwnedStorage::new(result_array)))
    }
}

/// Borrowed CPU storage - new functionality for non-ownership
#[derive(Debug)]
pub struct CPUBorrowedStorage<'a, T> {
    pub data: &'a ArrayD<T>,
}

impl<'a, T> CPUBorrowedStorage<'a, T> {
    pub fn new(data: &'a ArrayD<T>) -> Self {
        Self { data }
    }
}

// Clone implementation for borrowed storage
impl<'a, T> Clone for CPUBorrowedStorage<'a, T> {
    fn clone(&self) -> Self {
        Self { data: self.data }
    }
}

impl<'a, T> CPUBorrowedStorage<'a, T>
where
    T: GPUFloat,
{
    // Borrowed storage reduce always returns owned storage
    // because reduction operations create new data that can't be borrowed
    fn reduce<F>(
        &self,
        axes: Option<&[usize]>,
        reduction_fn: F,
    ) -> Result<CPUOwnedStorage<T>, String>
    where
        F: Fn(&ndarray::ArrayD<T>, ndarray::Axis) -> ndarray::ArrayD<T>,
    {
        match axes {
            Some(axes_list) => {
                // Validate axes bounds before processing
                for &ax in axes_list {
                    if ax >= self.data.ndim() {
                        return Err(format!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            ax,
                            self.data.ndim()
                        ));
                    }
                }

                let mut result = self.data.clone();
                // Sort in descending order to prevent index shifting during reduction
                let mut sorted_axes = axes_list.to_vec();
                sorted_axes.sort_unstable();
                sorted_axes.reverse();
                sorted_axes.dedup();

                // Apply reduction sequentially along each axis
                for &ax in &sorted_axes {
                    result = reduction_fn(&result, ndarray::Axis(ax));
                }

                Ok(CPUOwnedStorage::new(result))
            }
            None => {
                // Reduce across all dimensions to get scalar result
                let mut result = self.data.clone();
                for dim in (0..result.ndim()).rev() {
                    result = reduction_fn(&result, ndarray::Axis(dim));
                }
                Ok(CPUOwnedStorage::new(result))
            }
        }
    }

    /// Generic comparison method using ndarray::Zip for efficiency
    /// This is a helper method to avoid code duplication in comparison operations
    fn compare<F>(
        &self,
        other: &dyn StorageBackend<T>,
        comparison_fn: F,
    ) -> Result<CPUOwnedStorage<T>, String>
    where
        F: Fn(&T, &T) -> T,
    {
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for comparison: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        // Use ndarray's Zip for efficient element-wise comparison
        let result_data = ndarray::Zip::from(&*self.data)
            .and(other_data)
            .map_collect(|&a, &b| comparison_fn(&a, &b));

        // Return the owned storage result
        Ok(CPUOwnedStorage::new(result_data))
    }

    pub fn zeros(shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
        T: rand_distr::num_traits::Zero,
    {
        // Create ndarray with zeros directly - more efficient than device layer
        let data = ndarray::ArrayD::zeros(ndarray::IxDyn(shape));
        Ok(Box::new(CPUOwnedStorage::new(data)))
    }

    pub fn ones(shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
        T: rand_distr::num_traits::One,
    {
        // Create ndarray with ones directly
        let data = ndarray::ArrayD::ones(ndarray::IxDyn(shape));
        Ok(Box::new(CPUOwnedStorage::new(data)))
    }

    pub fn full(shape: &[usize], value: T) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
    {
        // Create ndarray filled with specific value
        let data = ndarray::ArrayD::from_elem(ndarray::IxDyn(shape), value);
        Ok(Box::new(CPUOwnedStorage::new(data)))
    }

    pub fn randn(shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
        StandardUniform: rand_distr::Distribution<T>,
    {
        let mut rng = rand::rng();
        let total_elements: usize = shape.iter().product();
        let two = <T as CPUFloat>::from_f64(2.0).expect("Cannot cast from f64");
        let one = <T as CPUFloat>::one();
        let data: Vec<T> = (0..total_elements)
            .map(|_| rng.random::<T>() * two - one) // Simple random between -1 and 1
            .collect();
        let data_array = ArrayD::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| format!("Failed to create array from data: {}", e))?;
        Ok(Box::new(CPUOwnedStorage::new(data_array)))
    }

    /// Conditional selection operation
    pub fn where_condition(
        condition: &dyn StorageBackend<T>,
        true_vals: &dyn StorageBackend<T>,
        false_vals: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
    {
        // Get data from all three storages
        let condition_data = condition.cpu_data()?;
        let true_data = true_vals.cpu_data()?;
        let false_data = false_vals.cpu_data()?;

        // Validate shapes match
        if condition_data.shape() != true_data.shape()
            || condition_data.shape() != false_data.shape()
        {
            return Err("Shape mismatch in where_condition".to_string());
        }

        // Element-wise selection using iterator zip
        let result_data: Vec<T> = condition_data
            .iter()
            .zip(true_data.iter())
            .zip(false_data.iter())
            .map(|((&cond, &true_val), &false_val)| {
                if cond > <T as CPUFloat>::zero() {
                    true_val
                } else {
                    false_val
                }
            })
            .collect();

        // Create result array with same shape
        let result_array = ndarray::Array::from_shape_vec(condition_data.raw_dim(), result_data)
            .map_err(|e| format!("Failed to create result: {}", e))?;

        Ok(Box::new(CPUOwnedStorage::new(result_array)))
    }
}
// Add these private implementation methods to CPUBorrowedStorage

impl<'a, T> CPUBorrowedStorage<'a, T>
where
    T: crate::backend::number::GPUFloat,
{
    /// Convert image patches to column matrix (im2col) for borrowed data
    /// Same algorithm as owned version but works with borrowed references
    fn im2col(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<T>, String> {
        let input_shape = self.shape();
        let (batch, channels, in_h, in_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (kernel_h, kernel_w) = kernel_size;

        let out_h = (in_h + 2 * padding.0 - kernel_h) / stride.0 + 1;
        let out_w = (in_w + 2 * padding.1 - kernel_w) / stride.1 + 1;

        let col_height = channels * kernel_h * kernel_w;
        let col_width = batch * out_h * out_w;

        let mut col_data = vec![<T as CPUFloat>::zero(); col_height * col_width];
        // Use borrowed data reference instead of owned data
        let input_data = if let Some(data) = self.data.as_slice() {
            data
        } else {
            return Err("Input data is empty!".to_string());
        };

        // Same patch extraction logic as owned version
        for b in 0..batch {
            for c in 0..channels {
                for ky in 0..kernel_h {
                    for kx in 0..kernel_w {
                        let col_row = c * kernel_h * kernel_w + ky * kernel_w + kx;

                        for out_y in 0..out_h {
                            for out_x in 0..out_w {
                                let in_y = out_y * stride.0 + ky;
                                let in_x = out_x * stride.1 + kx;
                                let col_col = b * out_h * out_w + out_y * out_w + out_x;

                                // Handle padding by checking bounds
                                if in_y >= padding.0
                                    && in_y < in_h + padding.0
                                    && in_x >= padding.1
                                    && in_x < in_w + padding.1
                                {
                                    let actual_y = in_y - padding.0;
                                    let actual_x = in_x - padding.1;

                                    if actual_y < in_h && actual_x < in_w {
                                        let input_idx = b * (channels * in_h * in_w)
                                            + c * (in_h * in_w)
                                            + actual_y * in_w
                                            + actual_x;
                                        col_data[col_row * col_width + col_col] =
                                            input_data[input_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        ArrayD::from_shape_vec(IxDyn(&[col_height, col_width]), col_data)
            .map_err(|e| format!("Failed to create im2col matrix: {}", e))
    }

    /// Standard 2D convolution implementation for borrowed data
    fn conv2d_impl(
        &self,
        filter: &ArrayD<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<T>, String> {
        let input_shape = self.shape();
        let filter_shape = filter.shape();

        let (batch, in_channels, in_h, in_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (out_channels, _, kernel_h, kernel_w) = (
            filter_shape[0],
            filter_shape[1],
            filter_shape[2],
            filter_shape[3],
        );

        let out_h = (in_h + 2 * padding.0 - kernel_h) / stride.0 + 1;
        let out_w = (in_w + 2 * padding.1 - kernel_w) / stride.1 + 1;

        // Transform borrowed input to column matrix
        let col_matrix = self.im2col((kernel_h, kernel_w), stride, padding)?;

        // Same reshaping and computation logic as owned version
        let filter_reshaped = filter
            .clone()
            .into_shape_with_order(IxDyn(&[out_channels, in_channels * kernel_h * kernel_w]))
            .map_err(|e| format!("Filter reshape failed: {}", e))?;

        // Perform convolution as matrix multiplication
        let im2col_view: ndarray::ArrayView2<T> = col_matrix
            .view()
            .into_dimensionality()
            .map_err(|e| format!("IM2col reshape failed: {}", e))?;
        let filter_view: ndarray::ArrayView2<T> = filter_reshaped
            .view()
            .into_dimensionality()
            .map_err(|e| format!("Filter reshape failed: {}", e))?;
        let output_2d = filter_view.dot(&im2col_view);

        // Transpose result to correct output layout
        let output_data: Vec<T> = if let Some(out_slice) = output_2d.as_slice() {
            out_slice.to_vec()
        } else {
            panic!("Attempted to slice an empty output!")
        };
        let mut final_output = vec![<T as CPUFloat>::zero(); batch * out_channels * out_h * out_w];

        for out_c in 0..out_channels {
            for b in 0..batch {
                for y in 0..out_h {
                    for x in 0..out_w {
                        let src_idx =
                            out_c * (batch * out_h * out_w) + b * (out_h * out_w) + y * out_w + x;
                        let dst_idx = b * (out_channels * out_h * out_w)
                            + out_c * (out_h * out_w)
                            + y * out_w
                            + x;
                        final_output[dst_idx] = output_data[src_idx];
                    }
                }
            }
        }

        ArrayD::from_shape_vec(IxDyn(&[batch, out_channels, out_h, out_w]), final_output)
            .map_err(|e| format!("Failed to create output tensor: {}", e))
    }
}

impl<'a, T> StorageBackend<T> for CPUBorrowedStorage<'a, T>
where
    T: GPUFloat,
{
    fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }

    fn ndim(&self) -> usize {
        self.data.ndim()
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn is_gpu(&self) -> bool {
        false
    }

    fn cpu_data(&self) -> Result<&ArrayD<T>, String> {
        Ok(self.data)
    }

    fn cpu_data_mut(&mut self) -> Result<&mut ArrayD<T>, String> {
        Err("Cannot mutate borrowed data".to_string())
    }

    fn owns_data(&self) -> bool {
        false
    }

    fn clone_storage(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        Err("Cannot clone borrowed storage".to_string())
    }

    fn add(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for addition: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        // Borrowed storage always creates owned result - no zero-copy for operations
        let result = self.data + other_data;
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn sub(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for subtraction: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        let result = self.data - other_data;
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn mul(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for multiplication: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        let result = self.data * other_data;
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn div(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for division: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        let result = self.data / other_data;
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn min(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for min operation: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| if a <= b { a } else { b })
            .collect();

        let result_array = ndarray::Array::from_shape_vec(self.data.raw_dim(), result_data)
            .map_err(|e| format!("Failed to create result tensor: {e}"))?;

        Ok(Box::new(CPUOwnedStorage::new(result_array)))
    }

    fn max(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for max operation: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| if a >= b { a } else { b })
            .collect();

        let result_array = ndarray::Array::from_shape_vec(self.data.raw_dim(), result_data)
            .map_err(|e| format!("Failed to create result tensor: {e}"))?;

        Ok(Box::new(CPUOwnedStorage::new(result_array)))
    }

    fn add_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = self.data + scalar;
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn sub_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = self.data - scalar;
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn mul_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = self.data * scalar;
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn div_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = self.data / scalar;
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn neg(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = self.data.mapv(|x| -x);
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn abs(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.abs()).collect();

        let result_array = ndarray::Array::from_shape_vec(self.data.raw_dim(), result_data)
            .map_err(|e| format!("Failed to create result tensor: {e}"))?;

        Ok(Box::new(CPUOwnedStorage::new(result_array)))
    }

    fn clamp(&self, min_val: T, max_val: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_data = self.data.mapv(|x| {
            if x < min_val {
                min_val
            } else if x > max_val {
                max_val
            } else {
                x
            }
        });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn sqrt(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_data = self.data.mapv(|x| x.sqrt());
        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn greater_equal(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        let storage = self.compare(other, |&a, &b| {
            if a >= b {
                <T as CPUFloat>::one()
            } else {
                <T as CPUFloat>::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn less_equal(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        let storage = self.compare(other, |&a, &b| {
            if a <= b {
                <T as CPUFloat>::one()
            } else {
                <T as CPUFloat>::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn greater(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let storage = self.compare(other, |&a, &b| {
            if a > b {
                <T as CPUFloat>::one()
            } else {
                <T as CPUFloat>::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn less(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let storage = self.compare(other, |&a, &b| {
            if a < b {
                <T as CPUFloat>::one()
            } else {
                <T as CPUFloat>::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn equal(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let storage = self.compare(other, |&a, &b| {
            if a == b {
                <T as CPUFloat>::one()
            } else {
                <T as CPUFloat>::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn logical_not(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Flip 0s to 1s and non-zeros to 0s
        let result_data = self.data.mapv(|x| {
            if x == <T as crate::backend::number::CPUNumber>::zero() {
                <T as crate::backend::number::CPUNumber>::one()
            } else {
                <T as crate::backend::number::CPUNumber>::zero()
            }
        });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn in_range(&self, min_val: T, max_val: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Check if values are in range [min_val, max_val]
        let result_data = self.data.mapv(|x| {
            if x >= min_val && x <= max_val {
                <T as crate::backend::number::CPUNumber>::one()
            } else {
                <T as crate::backend::number::CPUNumber>::zero()
            }
        });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn sign(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Return 1 for positive, -1 for negative, 0 for zero
        let result_data = self.data.mapv(|x| {
            if x > <T as crate::backend::number::CPUNumber>::zero() {
                <T as crate::backend::number::CPUNumber>::one()
            } else if x < <T as crate::backend::number::CPUNumber>::zero() {
                -<T as crate::backend::number::CPUNumber>::one()
            } else {
                <T as crate::backend::number::CPUNumber>::zero()
            }
        });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn matmul(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: Clone + ndarray::LinalgScalar,
    {
        let other_data = other.cpu_data()?;

        if self.data.ndim() != 2 || other_data.ndim() != 2 {
            return Err("Matrix multiplication requires 2D tensors".to_string());
        }

        let a_shape = self.data.shape();
        let b_shape = other_data.shape();

        if a_shape[1] != b_shape[0] {
            return Err(format!(
                "Matrix multiplication shape mismatch: ({}, {}) @ ({}, {})",
                a_shape[0], a_shape[1], b_shape[0], b_shape[1]
            ));
        }

        // Convert to 2D views for matrix multiplication
        let a: ndarray::ArrayView2<T> = self
            .data
            .view()
            .into_dimensionality()
            .map_err(|e| format!("Failed to convert to 2D view: {}", e))?;
        let b: ndarray::ArrayView2<T> = other_data
            .view()
            .into_dimensionality()
            .map_err(|e| format!("Failed to convert to 2D view: {}", e))?;

        // Perform matrix multiplication using ndarray's dot product
        let result = a.dot(&b);

        Ok(Box::new(CPUOwnedStorage::new(result.into_dyn())))
    }

    fn sigmoid(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Sigmoid function: 1 / (1 + exp(-x))
        let result_data = self.data.mapv(|x| {
            let one = <T as crate::backend::number::CPUNumber>::one();
            let neg_x = -x;
            one / (one + neg_x.exp())
        });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn relu(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // ReLU activation: max(0, x)
        let result_data = self.data.mapv(|x| {
            let zero = <T as crate::backend::number::CPUNumber>::zero();
            if x > zero { x } else { zero }
        });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn exp(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Element-wise exponential
        let result_data = self.data.mapv(|x| x.exp());
        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn log(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Element-wise natural logarithm
        let result_data = self.data.mapv(|x| x.ln());
        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn tanh(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Hyperbolic tangent using the same formula as your original
        let result_data = self.data.mapv(|x| {
            let e_x = x.exp();
            let e_neg_x = (-x).exp();
            (e_x - e_neg_x) / (e_x + e_neg_x)
        });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn powf(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for powf: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        // Element-wise power using ndarray's Zip
        let result_data = ndarray::Zip::from(&*self.data)
            .and(&*other_data)
            .map_collect(|&a, &b| a.powf(b));

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn power_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Scalar power operation
        let result_data = self.data.mapv(|x| x.powf(scalar));
        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn sum(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Use reduce_axes with ndarray's sum_axis function
        let result = self.reduce(axes, |array, ax| array.sum_axis(ax))?;
        Ok(Box::new(result) as Box<dyn StorageBackend<T>>)
    }

    fn mean(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        // First compute sum using reduce_axes
        let sum_result = self.sum(axes)?;

        // Calculate the number of elements being averaged over
        let divisor = match axes {
            Some(axes_list) => {
                // Product of dimensions being reduced
                axes_list
                    .iter()
                    .map(|&ax| self.data.shape()[ax])
                    .product::<usize>() as f64
            }
            None => {
                // All elements if no axes specified
                self.data.len() as f64
            }
        };

        // Convert divisor to tensor type and divide
        let divisor_scalar = <T as CPUFloat>::from_f64(1.0 / divisor)
            .ok_or("Failed to convert divisor to tensor type")?;

        sum_result.mul_scalar(divisor_scalar)
    }

    fn max_reduce(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Use reduce_axes with custom max reduction function
        // ndarray doesn't have a direct max_axis function, so we implement our own
        let result = self.reduce(axes, |array, ax| {
            let first = if let Some(f) = array.first() {
                f.clone()
            } else {
                panic!("Array is empty! Cannot reduce over empty array");
            };
            // Fold along the specified axis to find maximum values
            array.fold_axis(ax, first, |&acc, &x| if x > acc { x } else { acc })
        })?;
        Ok(Box::new(result) as Box<dyn StorageBackend<T>>)
    }

    fn min_reduce(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Use reduce_axes with custom min reduction function
        // Similar to max_axes but finding minimum values
        let result = self.reduce(axes, |array, ax| {
            let first = if let Some(f) = array.first() {
                f.clone()
            } else {
                panic!("Array is empty! Cannot reduce over empty array");
            };
            // Fold along the specified axis to find minimum values
            array.fold_axis(ax, first, |&acc, &x| if x < acc { x } else { acc })
        })?;
        Ok(Box::new(result) as Box<dyn StorageBackend<T>>)
    }

    fn broadcast_to(&self, target_shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Borrowed storage always creates owned result for shape operations
        match self.data.broadcast(target_shape) {
            Some(broadcasted) => Ok(Box::new(CPUOwnedStorage::new(broadcasted.to_owned()))),
            None => Err(format!(
                "Cannot broadcast {:?} to {:?}",
                self.data.shape(),
                target_shape
            )),
        }
    }

    fn reshape(&self, new_shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String> {
        let total_elements: usize = self.data.shape().iter().product();
        let new_total_elements: usize = new_shape.iter().product();

        if total_elements != new_total_elements {
            return Err(format!(
                "Cannot reshape tensor with {} elements to shape with {} elements",
                total_elements, new_total_elements
            ));
        }

        // Borrowed data must be cloned for reshape - creates owned storage
        match self.data.clone().into_shape_with_order(IxDyn(new_shape)) {
            Ok(reshaped) => Ok(Box::new(CPUOwnedStorage::new(reshaped))),
            Err(e) => Err(format!("Failed to reshape tensor: {e}")),
        }
    }

    fn transpose(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        match axes {
            Some(axes_order) => {
                if axes_order.len() != self.data.ndim() {
                    return Err(format!(
                        "Axes length {} doesn't match tensor dimensions {}",
                        axes_order.len(),
                        self.data.ndim()
                    ));
                }

                let mut sorted_axes = axes_order.to_vec();
                sorted_axes.sort_unstable();
                let expected: Vec<usize> = (0..self.data.ndim()).collect();
                if sorted_axes != expected {
                    return Err(format!(
                        "Invalid axes permutation: {:?}. Must be a permutation of 0..{}",
                        axes_order,
                        self.data.ndim()
                    ));
                }

                // Clone data for transpose - borrowed storage becomes owned
                let transposed = self.data.clone().permuted_axes(axes_order);
                Ok(Box::new(CPUOwnedStorage::new(transposed)))
            }
            None => {
                match self.data.ndim() {
                    0 | 1 => {
                        // Clone for consistency even though unchanged
                        Ok(Box::new(CPUOwnedStorage::new(self.data.clone())))
                    }
                    2 => {
                        let transposed = self.data.clone().reversed_axes();
                        Ok(Box::new(CPUOwnedStorage::new(transposed)))
                    }
                    _ => {
                        let axes_order: Vec<usize> = (0..self.data.ndim()).rev().collect();
                        let transposed = self.data.clone().permuted_axes(axes_order.as_slice());
                        Ok(Box::new(CPUOwnedStorage::new(transposed)))
                    }
                }
            }
        }
    }

    fn unsqueeze(&self, axis: usize) -> Result<Box<dyn StorageBackend<T>>, String> {
        if axis > self.data.ndim() {
            return Err(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                self.data.ndim()
            ));
        }

        // Borrowed data becomes owned after unsqueeze
        let expanded = self.data.clone().insert_axis(Axis(axis));
        Ok(Box::new(CPUOwnedStorage::new(expanded)))
    }

    fn squeeze(&self, axis: Option<usize>) -> Result<Box<dyn StorageBackend<T>>, String> {
        match axis {
            Some(ax) => {
                if ax >= self.data.ndim() {
                    return Err(format!(
                        "Axis {} out of bounds for tensor with {} dimensions",
                        ax,
                        self.data.ndim()
                    ));
                }

                if self.data.shape()[ax] != 1 {
                    return Err(format!(
                        "Cannot squeeze axis {} with size {}",
                        ax,
                        self.data.shape()[ax]
                    ));
                }

                // Clone and squeeze - borrowed becomes owned
                let squeezed = self.data.clone().remove_axis(Axis(ax));
                Ok(Box::new(CPUOwnedStorage::new(squeezed)))
            }
            None => {
                let mut result = self.data.clone();
                let mut axes_to_remove = Vec::new();

                for (i, &size) in self.data.shape().iter().enumerate() {
                    if size == 1 {
                        axes_to_remove.push(i);
                    }
                }

                // Remove in reverse order
                for &ax in axes_to_remove.iter().rev() {
                    result = result.remove_axis(Axis(ax));
                }

                Ok(Box::new(CPUOwnedStorage::new(result)))
            }
        }
    }

    fn conv2d(
        &self,
        filter: &dyn StorageBackend<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Ensure filter is also CPU storage
        let filter_data = filter.cpu_data()?;

        let input_shape = self.shape();
        let filter_shape = filter.shape();

        // Validate input dimensions for conv2d
        if input_shape.len() != 4 || filter_shape.len() != 4 {
            return Err("Conv2D requires 4D tensors [batch, channels, height, width]".to_string());
        }

        let result = self.conv2d_impl(filter_data, stride, padding)?;
        Ok(Box::new(CPUOwnedStorage::new(result)))
    }

    fn iter_values(&self) -> Result<Vec<T>, String> {
        // Efficient cloning of all values for iteration
        Ok(self.data.iter().cloned().collect())
    }

    fn get_flat(&self, index: usize) -> Result<Option<T>, String> {
        // Flat indexing using ndarray's slice functionality
        match self.data.as_slice() {
            Some(slice) => {
                if index < slice.len() {
                    Ok(Some(slice[index]))
                } else {
                    Ok(None)
                }
            }
            None => Err("Data is not contiguous for flat indexing".to_string()),
        }
    }

    fn get_multi(&self, indices: &[usize]) -> Result<Option<T>, String> {
        // Multi-dimensional indexing using ndarray
        if indices.len() != self.data.ndim() {
            return Ok(None);
        }

        // Check bounds before accessing
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.data.shape()[i] {
                return Ok(None);
            }
        }

        Ok(Some(self.data[ndarray::IxDyn(indices)]))
    }
}

#[cfg(feature = "cuda")]
/// GPU storage - your existing pattern
#[derive(Debug, Clone)]
pub struct GPUOwnedStorage<T: cudarc::driver::DeviceRepr + GPUFloat> {
    pub cuda_data: CudaTensor<T>,
}

#[cfg(feature = "cuda")]
impl<T: DeviceRepr + GPUFloat> GPUOwnedStorage<T> {
    pub fn new(cuda_data: CudaTensor<T>) -> Self {
        Self { cuda_data }
    }
}

#[cfg(feature = "cuda")]
impl<T> StorageBackend<T> for GPUOwnedStorage<T>
where
    T: GPUFloat + cudarc::driver::DeviceRepr + Clone,
{
    fn shape(&self) -> &[usize] {
        self.cuda_data.shape()
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn ndim(&self) -> usize {
        self.cuda_data.shape().len()
    }

    fn size(&self) -> usize {
        self.cuda_data.shape().iter().product()
    }

    fn is_gpu(&self) -> bool {
        true
    }

    fn cpu_data(&self) -> Result<&ArrayD<T>, String> {
        Err("Data is on GPU. Use .to_cpu() to move it first".to_string())
    }

    fn cpu_data_mut(&mut self) -> Result<&mut ArrayD<T>, String> {
        Err("Data is on GPU. Use .to_cpu() to move it first".to_string())
    }

    fn owns_data(&self) -> bool {
        true
    }

    fn clone_storage(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        Ok(Box::new(self.clone()))
    }

    // ELEMENT-WISE OPERATIONS USING CUDA BACKEND

    fn add(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            // Both tensors are on GPU - use CUDA kernels directly
            let other_gpu = other
                .as_any()
                .and_then(|any| any.downcast_ref::<GPUOwnedStorage<T>>())
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for GPU addition: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda = with_cuda_ops(|ops| ops.add(&self.cuda_data, &other_cuda))?;

            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        } else {
            // Mixed GPU-CPU operation: convert CPU to GPU first
            let other_data = other.cpu_data()?;

            if self.shape() != other_data.shape() {
                return Err(format!(
                    "Shape mismatch for mixed addition: {:?} vs {:?}",
                    self.shape(),
                    other_data.shape()
                ));
            }

            // Convert CPU ArrayD to CudaTensor and perform operation
            let other_cuda = with_cuda_context(|ctx| CudaTensor::from_cpu_array(ctx, other_data))?;

            let result_cuda = with_cuda_ops(|ops| ops.add(&self.cuda_data, &other_gpu.cuda_data))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        }
    }

    fn sub(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .and_then(|any| any.downcast_ref::<GPUOwnedStorage<T>>())
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for GPU subtraction: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda = with_cuda_ops(|ops| ops.sub(&self.cuda_data, &other_gpu.cuda_data))?;

            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        } else {
            let other_data = other.cpu_data()?;

            if self.shape() != other_data.shape() {
                return Err(format!(
                    "Shape mismatch for mixed subtraction: {:?} vs {:?}",
                    self.shape(),
                    other_data.shape()
                ));
            }

            let other_cuda = with_cuda_context(|ctx| CudaTensor::from_cpu_array(ctx, other_data))?;
            let result_cuda = with_cuda_ops(|ops| ops.sub(&self.cuda_data, &other_cuda))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        }
    }

    fn mul(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .and_then(|any| any.downcast_ref::<GPUOwnedStorage<T>>())
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for GPU multiplication: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda = with_cuda_ops(|ops| ops.mul(&self.cuda_data, &other_gpu.cuda_data))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        } else {
            let other_data = other.cpu_data()?;

            if self.shape() != other_data.shape() {
                return Err(format!(
                    "Shape mismatch for mixed multiplication: {:?} vs {:?}",
                    self.shape(),
                    other_data.shape()
                ));
            }

            let other_cuda = with_cuda_context(|ctx| CudaTensor::from_cpu_array(ctx, other_data))?;
            let result_cuda = with_cuda_ops(|ops| ops.mul(&self.cuda_data, &other_cuda))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        }
    }

    fn div(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .and_then(|any| any.downcast_ref::<GPUOwnedStorage<T>>())
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for GPU division: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda = with_cuda_ops(|ops| ops.div(&self.cuda_data, &other_gpu.cuda_data))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        } else {
            let other_data = other.cpu_data()?;

            if self.shape() != other_data.shape() {
                return Err(format!(
                    "Shape mismatch for mixed division: {:?} vs {:?}",
                    self.shape(),
                    other_data.shape()
                ));
            }

            let other_cuda = with_cuda_context(|ctx| CudaTensor::from_cpu_array(ctx, other_data))?;
            let result_cuda = with_cuda_ops(|ops| ops.div(&self.cuda_data, &other_cuda))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        }
    }

    fn min(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .and_then(|any| any.downcast_ref::<GPUOwnedStorage<T>>())
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for GPU min operation: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda = with_cuda_ops(|ops| ops.min(&self.cuda_data, &other_gpu.cuda_data))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        } else {
            let other_data = other.cpu_data()?;

            if self.shape() != other_data.shape() {
                return Err(format!(
                    "Shape mismatch for mixed min operation: {:?} vs {:?}",
                    self.shape(),
                    other_data.shape()
                ));
            }

            let other_cuda = with_cuda_context(|ctx| CudaTensor::from_cpu_array(ctx, other_data))?;
            let result_cuda = with_cuda_ops(|ops| ops.min(&self.cuda_data, &other_cuda))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        }
    }

    fn max(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .and_then(|any| any.downcast_ref::<GPUOwnedStorage<T>>())
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for GPU max operation: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda = with_cuda_ops(|ops| ops.max(&self.cuda_data, &other_gpu.cuda_data))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        } else {
            let other_data = other.cpu_data()?;

            if self.shape() != other_data.shape() {
                return Err(format!(
                    "Shape mismatch for mixed max operation: {:?} vs {:?}",
                    self.shape(),
                    other_data.shape()
                ));
            }

            let other_cuda = with_cuda_context(|ctx| CudaTensor::from_cpu_array(ctx, other_data))?;

            let result_cuda = with_cuda_ops(|ops| ops.max(&self.cuda_data, &other_cuda))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        }
    }

    // SCALAR OPERATIONS

    fn add_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|ops| ops.add_scalar(&self.cuda_data, scalar))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn sub_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|ops| ops.sub_scalar(&self.cuda_data, scalar))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn mul_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|ops| ops.mul_scalar(&self.cuda_data, scalar))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn div_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|ops| ops.div_scalar(&self.cuda_data, scalar))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    // UNARY OPERATIONS

    fn neg(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|ops| ops.negate(&self.cuda_data))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn abs(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|ops| ops.abs(&self.cuda_data))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn sqrt(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|ops| ops.sqrt(&self.cuda_data))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn clamp(&self, min_val: T, max_val: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|ops| ops.clamp(&self.cuda_data, min_val, max_val))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn greater(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            // Both tensors are on GPU - use CUDA kernels
            let other_gpu = other
                .as_any()
                .and_then(|any| any.downcast_ref::<GPUOwnedStorage<T>>())
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for greater_equal: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda =
                with_cuda_ops(|ops| ops.greater(&self.cuda_data, &other_gpu.cuda_data))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        } else {
            Err("GPU-CPU mixed operations not supported for comparison operations".to_string())
        }
    }

    fn less(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .and_then(|any| any.downcast_ref::<GPUOwnedStorage<T>>())
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for less_equal: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda = with_cuda_ops(|ops| ops.less(&self.cuda_data, &other_gpu.cuda_data))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        } else {
            Err("GPU-CPU mixed operations not supported for comparison operations".to_string())
        }
    }

    fn greater_equal(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            // Both tensors are on GPU - use CUDA kernels
            let other_gpu = other
                .as_any()
                .and_then(|any| any.downcast_ref::<GPUOwnedStorage<T>>())
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for greater_equal: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda =
                with_cuda_ops(|ops| ops.greater_equal(&self.cuda_data, &other_gpu.cuda_data))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        } else {
            Err("GPU-CPU mixed operations not supported for comparison operations".to_string())
        }
    }

    fn less_equal(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .and_then(|any| any.downcast_ref::<GPUOwnedStorage<T>>())
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for less_equal: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda =
                with_cuda_ops(|ops| ops.less_equal(&self.cuda_data, &other_gpu.cuda_data))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        } else {
            Err("GPU-CPU mixed operations not supported for comparison operations".to_string())
        }
    }

    fn equal(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .and_then(|any| any.downcast_ref::<GPUOwnedStorage<T>>())
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for equal: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda =
                with_cuda_ops(|ops| ops.equal(&self.cuda_data, &other_gpu.cuda_data))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        } else {
            Err("GPU-CPU mixed operations not supported for comparison operations".to_string())
        }
    }

    fn logical_not(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|ops| ops.logical_not(&self.cuda_data))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn in_range(&self, min_val: T, max_val: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|ops| ops.in_range(&self.cuda_data, min_val, max_val))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn sign(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|ops| ops.sign(&self.cuda_data))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn matmul(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .and_then(|any| any.downcast_ref::<GPUOwnedStorage<T>>())
                .ok_or("Failed to cast to GPU storage")?;

            if self.ndim() != 2 || other_gpu.ndim() != 2 {
                return Err("Matrix multiplication requires 2D tensors".to_string());
            }

            let a_shape = self.shape();
            let b_shape = other_gpu.shape();

            if a_shape[1] != b_shape[0] {
                return Err(format!(
                    "Matrix multiplication shape mismatch: ({}, {}) @ ({}, {})",
                    a_shape[0], a_shape[1], b_shape[0], b_shape[1]
                ));
            }

            let result_cuda =
                with_cuda_ops(|ops| ops.matmul(&self.cuda_data, &other_gpu.cuda_data))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        } else {
            Err("GPU-CPU mixed operations not supported for matrix multiplication".to_string())
        }
    }

    fn sigmoid(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|cuda_ops| cuda_ops.sigmoid(&self.cuda_data))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn relu(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|cuda_ops| cuda_ops.relu(&self.cuda_data))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn exp(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|cuda_ops| cuda_ops.exp(&self.cuda_data))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn log(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|cuda_ops| cuda_ops.log(&self.cuda_data))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn tanh(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|cuda_ops| cuda_ops.tanh(&self.cuda_data))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn powf(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .and_then(|any| any.downcast_ref::<GPUOwnedStorage<T>>())
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for powf: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda =
                with_cuda_ops(|cuda_ops| cuda_ops.power(&self.cuda_data, &other_gpu.cuda_data))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        } else {
            Err("GPU-CPU mixed operations not supported for power operations".to_string())
        }
    }

    fn power_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|cuda_ops| cuda_ops.power_scalar(&self.cuda_data, scalar))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn sum(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        match axes {
            Some(axes_list) => {
                if axes_list.is_empty() {
                    return Err("Empty axes list provided".to_string());
                }

                // Validate axes are within bounds
                for &ax in axes_list {
                    if ax >= self.ndim() {
                        return Err(format!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            ax,
                            self.ndim()
                        ));
                    }
                }

                // Use CUDA ops sum_axes method for multiple axes
                let result_cuda =
                    with_cuda_ops(|ops| ops.sum_axes(&self.cuda_data, axes_list, false))?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
            None => {
                // Sum all elements to scalar using CUDA ops
                let result_cuda = with_cuda_ops(|ops| ops.sum_all(&self.cuda_data))?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
        }
    }

    fn mean(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        match axes {
            Some(axes_list) => {
                if axes_list.is_empty() {
                    return Err("Empty axes list provided".to_string());
                }

                // Validate axes are within bounds
                for &ax in axes_list {
                    if ax >= self.ndim() {
                        return Err(format!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            ax,
                            self.ndim()
                        ));
                    }
                }

                // For multiple axes, we compute sum then divide by product of axis sizes
                let sum_result = cuda_ops.sum_axes(&self.cuda_data, axes_list, false)?;

                // Calculate divisor as product of reduced dimensions
                let divisor = axes_list
                    .iter()
                    .map(|&ax| self.shape()[ax])
                    .product::<usize>() as f64;

                let divisor_scalar = <T as GPUFloat>::from_f64(1.0 / divisor)
                    .ok_or("Failed to convert divisor to tensor type")?;

                // Create scalar tensor and divide
                let divisor_tensor = cuda_ops.full(&[], divisor_scalar)?;
                let result_cuda = with_cuda_ops(|ops| ops.div(&sum_result, &divisor_tensor))?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
            None => {
                // Mean of all elements using CUDA ops
                let result_cuda = with_cuda_ops(|ops| ops.mean_all(&self.cuda_data))?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
        }
    }

    fn max_reduce(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        match axes {
            Some(axes_list) => {
                if axes_list.is_empty() {
                    return Err("Empty axes list provided".to_string());
                }

                // Validate axes are within bounds
                for &ax in axes_list {
                    if ax >= self.ndim() {
                        return Err(format!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            ax,
                            self.ndim()
                        ));
                    }
                }

                // Use CUDA ops max_axes method for multiple axes reduction
                let result_cuda =
                    with_cuda_ops(|ops| ops.max_axes(&self.cuda_data, axes_list, false))?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
            None => {
                // Max of all elements to scalar
                let result_cuda = with_cuda_ops(|ops| ops.max_all(&self.cuda_data))?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
        }
    }

    fn min_reduce(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        match axes {
            Some(axes_list) => {
                if axes_list.is_empty() {
                    return Err("Empty axes list provided".to_string());
                }

                // Validate axes are within bounds
                for &ax in axes_list {
                    if ax >= self.ndim() {
                        return Err(format!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            ax,
                            self.ndim()
                        ));
                    }
                }

                // Use CUDA ops min_axes method for multiple axes reduction
                let result_cuda =
                    with_cuda_ops(|ops| ops.min_axes(&self.cuda_data, axes_list, false))?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
            None => {
                // Min of all elements to scalar
                let result_cuda = with_cuda_ops(|ops| ops.min_all(&self.cuda_data))?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
        }
    }

    fn broadcast_to(&self, target_shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|ops| ops.broadcast_to(&self.cuda_data, target_shape))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn reshape(&self, new_shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Validate total elements remain the same
        let total_elements: usize = self.shape().iter().product();
        let new_total_elements: usize = new_shape.iter().product();

        if total_elements != new_total_elements {
            return Err(format!(
                "Cannot reshape tensor with {} elements to shape with {} elements",
                total_elements, new_total_elements
            ));
        }

        // Use CUDA ops for reshape operation

        let result_cuda = with_cuda_ops(|ops| ops.reshape(&self.cuda_data, new_shape))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn transpose(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        match axes {
            Some(axes_order) => {
                // Validate axes specification
                if axes_order.len() != self.ndim() {
                    return Err(format!(
                        "Axes length {} doesn't match tensor dimensions {}",
                        axes_order.len(),
                        self.ndim()
                    ));
                }

                // Use CUDA ops for custom transpose
                let result_cuda =
                    with_cuda_ops(|ops| ops.transpose(&self.cuda_data, Some(axes_order)))?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
            None => {
                // Default transpose using CUDA ops
                let result_cuda = with_cuda_ops(|ops| ops.transpose(&self.cuda_data, None))?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
        }
    }

    fn reshape(&self, new_shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Validate total elements remain the same
        let total_elements: usize = self.shape().iter().product();
        let new_total_elements: usize = new_shape.iter().product();

        if total_elements != new_total_elements {
            return Err(format!(
                "Cannot reshape tensor with {} elements to shape with {} elements",
                total_elements, new_total_elements
            ));
        }

        let result_cuda = with_cuda_ops(|op| op.reshape(&self.cuda_data, new_shape))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn unsqueeze(&self, axis: usize) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Validate axis bounds - can insert at positions 0..ndim (inclusive)
        if axis > self.ndim() {
            return Err(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                self.ndim()
            ));
        }

        let result_cuda = with_cuda_ops(|op| op.unsqueeze(&self.cuda_data, axis))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn squeeze(&self, axis: Option<usize>) -> Result<Box<dyn StorageBackend<T>>, String> {
        match axis {
            Some(ax) => {
                // Squeeze specific axis - validate it exists and has size 1
                if ax >= self.ndim() {
                    return Err(format!(
                        "Axis {} out of bounds for tensor with {} dimensions",
                        ax,
                        self.ndim()
                    ));
                }

                if self.shape()[ax] != 1 {
                    return Err(format!(
                        "Cannot squeeze axis {} with size {}",
                        ax,
                        self.shape()[ax]
                    ));
                }

                // Use CUDA ops for single axis squeeze
                let result_cuda = with_cuda_ops(|ops| ops.squeeze(&self.cuda_data, Some(ax)))?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
            None => {
                // Remove all dimensions of size 1 using CUDA ops
                let result_cuda = with_cuda_ops(|ops| ops.squeeze(&self.cuda_data, None))?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
        }
    }

    fn conv2d(
        &self,
        filter: &dyn StorageBackend<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        if filter.is_gpu() {
            // Both tensors on GPU - use CUDA kernels
            let filter_gpu = filter
                .as_any()
                .and_then(|any| any.downcast_ref::<GPUOwnedStorage<T>>())
                .ok_or("Failed to cast filter to GPU storage")?;

            let result_cuda = with_cuda_ops(|ops| {
                ops.conv2d(&self.cuda_data, &filter_gpu.cuda_data, stride, padding)
            })?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        } else {
            // Mixed GPU/CPU - fall back to CPU computation
            Err(
                "Mixed GPU/CPU convolution not supported - move both tensors to same device"
                    .to_string(),
            )
        }
    }

    // Note that this ops require moving the data to the CPU
    fn iter_values(&self) -> Result<Vec<T>, String> {
        with_cuda_context(|ctx| self.cuda_data.to_vec(ctx))?
    }

    fn get_flat(&self, index: usize) -> Result<Option<T>, String> {
        // GPU flat access requires CPU sync - expensive operation
        if index >= self.size() {
            return Ok(None);
        }

        let values = self.iter_values()?;
        Ok(Some(values[index]))
    }

    fn get_multi(&self, indices: &[usize]) -> Result<Option<T>, String> {
        // GPU multi-dimensional access
        if indices.len() != self.shape().len() {
            return Ok(None);
        }

        // Check bounds
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape()[i] {
                return Ok(None);
            }
        }

        // Convert multi-dim indices to flat index
        let mut flat_index = 0;
        let mut stride = 1;
        for i in (0..indices.len()).rev() {
            flat_index += indices[i] * stride;
            stride *= self.shape()[i];
        }

        self.get_flat(flat_index)
    }
}

#[cfg(feature = "cuda")]
impl<T: cudarc::driver::DeviceRepr + GPUFloat> GPUOwnedStorage<T> {
    pub fn zeros(shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
        T: cudarc::driver::ValidAsZeroBits,
    {
        // Get default CUDA backend from manager

        // Create GPU tensor with zeros directly
        let cuda_tensor = with_cuda_ops(|ops| ops.zeros(shape))?;
        Ok(Box::new(GPUOwnedStorage::new(cuda_tensor)))
    }

    pub fn ones(shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
    {
        // Create GPU tensor with ones directly
        let cuda_tensor = with_cuda_ops(|ops| ops.ones(shape))?;
        Ok(Box::new(GPUOwnedStorage::new(cuda_tensor)))
    }

    pub fn full(shape: &[usize], value: T) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
    {
        // Create GPU tensor filled with value
        let cuda_tensor = with_cuda_ops(|ops| ops.full(shape, value))?;
        Ok(Box::new(GPUOwnedStorage::new(cuda_tensor)))
    }

    pub fn randn(shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
    {
        let cuda_tensor = with_cuda_ops(|ops| ops.randn(shape))?;
        Ok(Box::new(GPUOwnedStorage::new(cuda_tensor)))
    }

    pub fn where_condition(
        condition: &GPUOwnedStorage<T>,
        true_vals: &GPUOwnedStorage<T>,
        false_vals: &GPUOwnedStorage<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        if condition.is_gpu() && true_vals.is_gpu() && false_vals.is_gpu() {
            let cuda_tensor = with_cuda_ops(|ops| {
                ops.where_condition(
                    condition.cuda_data,
                    true_vals.cuda_data,
                    false_vals.cuda_data,
                )
            })?;

            Ok(Box::new(GPUOwnedStorage::new(cuda_tensor)))
        } else {
            Err("Not all provided tensors are on GPU storage".to_string())
        }
    }
}

#[cfg(feature = "cuda")]
fn with_cuda_context<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce(&CudaOps) -> Result<R, String>,
{
    let backend = get_backend();
    let context_manager = backend.cuda_backend().ok_or("CUDA backend not available")?;

    f(&context_manager)
}

// Utility method to get the cuda ops in an idiomatic way
#[cfg(feature = "cuda")]
fn with_cuda_ops<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce(&CudaOps) -> Result<R, String>,
{
    let backend = get_backend();
    let context_manager = backend.cuda_backend().ok_or("CUDA backend not available")?;

    let ops = context_manager.ops();
    f(&ops)
}
