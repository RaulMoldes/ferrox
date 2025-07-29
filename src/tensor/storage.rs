// src/tensor/storage.rs
use ndarray::{ArrayD, IxDyn};
use ndarray::Axis;
use std::borrow::Cow;
use std::fmt::Debug;

#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaTensor;

#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;

use crate::backend::number::{GPUFloat, CPUFloat};

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
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for greater_equal: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        // Use ndarray's Zip for efficient element-wise comparison
        let result_data = ndarray::Zip::from(&self.data)
            .and(other_data)
            .map_collect(|&a, &b| {
                if a >= b {
                    <T as crate::backend::number::CPUNumber>::one()
                } else {
                    <T as crate::backend::number::CPUNumber>::zero()
                }
            });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn less_equal(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for less_equal: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        let result_data = ndarray::Zip::from(&self.data)
            .and(other_data)
            .map_collect(|&a, &b| {
                if a <= b {
                    <T as crate::backend::number::CPUNumber>::one()
                } else {
                    <T as crate::backend::number::CPUNumber>::zero()
                }
            });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn equal(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for equal: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        let result_data = ndarray::Zip::from(&self.data)
            .and(other_data)
            .map_collect(|&a, &b| {
                if a == b {
                    <T as crate::backend::number::CPUNumber>::one()
                } else {
                    <T as crate::backend::number::CPUNumber>::zero()
                }
            });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
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
            // Fold along the specified axis to find maximum values
            array.fold_axis(ax, array.first().unwrap().clone(), |&acc, &x| {
                if x > acc { x } else { acc }
            })
        })?;
        Ok(Box::new(result) as Box<dyn StorageBackend<T>>)
    }

    fn min_reduce(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Use reduce_axes with custom min reduction function
        // Similar to max_axes but finding minimum values
        let result = self.reduce(axes, |array, ax| {
            // Fold along the specified axis to find minimum values
            array.fold_axis(ax, array.first().unwrap().clone(), |&acc, &x| {
                if x < acc { x } else { acc }
            })
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
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for greater_equal: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        // Use ndarray's Zip for efficient element-wise comparison
        let result_data = ndarray::Zip::from(&*self.data)
            .and(other_data)
            .map_collect(|&a, &b| {
                if a >= b {
                    <T as crate::backend::number::CPUNumber>::one()
                } else {
                    <T as crate::backend::number::CPUNumber>::zero()
                }
            });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn less_equal(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for less_equal: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        let result_data = ndarray::Zip::from(&*self.data)
            .and(other_data)
            .map_collect(|&a, &b| {
                if a <= b {
                    <T as crate::backend::number::CPUNumber>::one()
                } else {
                    <T as crate::backend::number::CPUNumber>::zero()
                }
            });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
    }

    fn equal(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.data.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for equal: {:?} vs {:?}",
                self.data.shape(),
                other_data.shape()
            ));
        }

        let result_data = ndarray::Zip::from(&*self.data)
            .and(other_data)
            .map_collect(|&a, &b| {
                if a == b {
                    <T as crate::backend::number::CPUNumber>::one()
                } else {
                    <T as crate::backend::number::CPUNumber>::zero()
                }
            });

        Ok(Box::new(CPUOwnedStorage::new(result_data)))
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
            // Fold along the specified axis to find maximum values
            array.fold_axis(ax, array.first().unwrap().clone(), |&acc, &x| {
                if x > acc { x } else { acc }
            })
        })?;
        Ok(Box::new(result) as Box<dyn StorageBackend<T>>)
    }

    fn min_reduce(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Use reduce_axes with custom min reduction function
        // Similar to max_axes but finding minimum values
        let result = self.reduce(axes, |array, ax| {
            // Fold along the specified axis to find minimum values
            array.fold_axis(ax, array.first().unwrap().clone(), |&acc, &x| {
                if x < acc { x } else { acc }
            })
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
}

#[cfg(feature = "cuda")]
/// GPU storage - your existing pattern
#[derive(Debug, Clone)]
pub struct GPUOwnedStorage<T: cudarc::driver::DeviceRepr> {
    pub cuda_data: CudaTensor<T>,
}

#[cfg(feature = "cuda")]
impl<T: DeviceRepr> GPUOwnedStorage<T> {
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

            // Use CUDA operations backend from the CudaTensor
            let cuda_ops = self
                .cuda_data
                .get_cuda_ops()
                .ok_or("Failed to get CUDA operations backend")?;

            let result_cuda = cuda_ops.add(&self.cuda_data, &other_gpu.cuda_data)?;
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
            let other_cuda = CudaTensor::from_cpu_array(other_data)?;
            let cuda_ops = self
                .cuda_data
                .get_cuda_ops()
                .ok_or("Failed to get CUDA operations backend")?;

            let result_cuda = cuda_ops.add(&self.cuda_data, &other_cuda)?;
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

            let cuda_ops = self
                .cuda_data
                .get_cuda_ops()
                .ok_or("Failed to get CUDA operations backend")?;

            let result_cuda = cuda_ops.sub(&self.cuda_data, &other_gpu.cuda_data)?;
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

            let other_cuda = CudaTensor::from_cpu_array(other_data)?;
            let cuda_ops = self
                .cuda_data
                .get_cuda_ops()
                .ok_or("Failed to get CUDA operations backend")?;

            let result_cuda = cuda_ops.sub(&self.cuda_data, &other_cuda)?;
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

            let cuda_ops = self
                .cuda_data
                .get_cuda_ops()
                .ok_or("Failed to get CUDA operations backend")?;

            let result_cuda = cuda_ops.mul(&self.cuda_data, &other_gpu.cuda_data)?;
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

            let other_cuda = CudaTensor::from_cpu_array(other_data)?;
            let cuda_ops = self
                .cuda_data
                .get_cuda_ops()
                .ok_or("Failed to get CUDA operations backend")?;

            let result_cuda = cuda_ops.mul(&self.cuda_data, &other_cuda)?;
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

            let cuda_ops = self
                .cuda_data
                .get_cuda_ops()
                .ok_or("Failed to get CUDA operations backend")?;

            let result_cuda = cuda_ops.div(&self.cuda_data, &other_gpu.cuda_data)?;
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

            let other_cuda = CudaTensor::from_cpu_array(other_data)?;
            let cuda_ops = self
                .cuda_data
                .get_cuda_ops()
                .ok_or("Failed to get CUDA operations backend")?;

            let result_cuda = cuda_ops.div(&self.cuda_data, &other_cuda)?;
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

            let cuda_ops = self
                .cuda_data
                .get_cuda_ops()
                .ok_or("Failed to get CUDA operations backend")?;

            let result_cuda = cuda_ops.min(&self.cuda_data, &other_gpu.cuda_data)?;
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

            let other_cuda = CudaTensor::from_cpu_array(other_data)?;
            let cuda_ops = self
                .cuda_data
                .get_cuda_ops()
                .ok_or("Failed to get CUDA operations backend")?;

            let result_cuda = cuda_ops.min(&self.cuda_data, &other_cuda)?;
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

            let cuda_ops = self
                .cuda_data
                .get_cuda_ops()
                .ok_or("Failed to get CUDA operations backend")?;

            let result_cuda = cuda_ops.max(&self.cuda_data, &other_gpu.cuda_data)?;
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

            let other_cuda = CudaTensor::from_cpu_array(other_data)?;
            let cuda_ops = self
                .cuda_data
                .get_cuda_ops()
                .ok_or("Failed to get CUDA operations backend")?;

            let result_cuda = cuda_ops.max(&self.cuda_data, &other_cuda)?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        }
    }

    // SCALAR OPERATIONS

    fn add_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let cuda_ops = self
            .cuda_data
            .get_cuda_ops()
            .ok_or("Failed to get CUDA operations backend")?;

        let result_cuda = cuda_ops.add_scalar(&self.cuda_data, scalar)?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn sub_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let cuda_ops = self
            .cuda_data
            .get_cuda_ops()
            .ok_or("Failed to get CUDA operations backend")?;

        let result_cuda = cuda_ops.sub_scalar(&self.cuda_data, scalar)?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn mul_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let cuda_ops = self
            .cuda_data
            .get_cuda_ops()
            .ok_or("Failed to get CUDA operations backend")?;

        let result_cuda = cuda_ops.mul_scalar(&self.cuda_data, scalar)?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn div_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let cuda_ops = self
            .cuda_data
            .get_cuda_ops()
            .ok_or("Failed to get CUDA operations backend")?;

        let result_cuda = cuda_ops.div_scalar(&self.cuda_data, scalar)?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    // UNARY OPERATIONS

    fn neg(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let cuda_ops = self
            .cuda_data
            .get_cuda_ops()
            .ok_or("Failed to get CUDA operations backend")?;

        let result_cuda = cuda_ops.negate(&self.cuda_data)?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn abs(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let cuda_ops = self
            .cuda_data
            .get_cuda_ops()
            .ok_or("Failed to get CUDA operations backend")?;

        let result_cuda = cuda_ops.abs(&self.cuda_data)?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn sqrt(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let cuda_ops = self
            .cuda_data
            .get_cuda_ops()
            .ok_or("Failed to get CUDA operations backend")?;

        let result_cuda = cuda_ops.sqrt(&self.cuda_data)?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn clamp(&self, min_val: T, max_val: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let cuda_ops = self
            .cuda_data
            .get_cuda_ops()
            .ok_or("Failed to get CUDA operations backend")?;

        let result_cuda = cuda_ops.clamp(&self.cuda_data, min_val, max_val)?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
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

            let result_cuda = self.cuda_data.with_cuda_ops(|cuda_ops| {
                cuda_ops.greater_equal(&self.cuda_data, &other_gpu.cuda_data)
            })?;
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

            let result_cuda = self.cuda_data.with_cuda_ops(|cuda_ops| {
                cuda_ops.less_equal(&self.cuda_data, &other_gpu.cuda_data)
            })?;
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

            let result_cuda = self
                .cuda_data
                .with_cuda_ops(|cuda_ops| cuda_ops.equal(&self.cuda_data, &other_gpu.cuda_data))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        } else {
            Err("GPU-CPU mixed operations not supported for comparison operations".to_string())
        }
    }

    fn logical_not(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = self
            .cuda_data
            .with_cuda_ops(|cuda_ops| cuda_ops.logical_not(&self.cuda_data))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn in_range(&self, min_val: T, max_val: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = self
            .cuda_data
            .with_cuda_ops(|cuda_ops| cuda_ops.in_range(&self.cuda_data, min_val, max_val))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn sign(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = self
            .cuda_data
            .with_cuda_ops(|cuda_ops| cuda_ops.sign(&self.cuda_data))?;
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

            let result_cuda = self
                .cuda_data
                .with_cuda_ops(|cuda_ops| cuda_ops.matmul(&self.cuda_data, &other_gpu.cuda_data))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        } else {
            Err("GPU-CPU mixed operations not supported for matrix multiplication".to_string())
        }
    }

    fn sigmoid(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = self
            .cuda_data
            .with_cuda_ops(|cuda_ops| cuda_ops.sigmoid(&self.cuda_data))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn relu(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = self
            .cuda_data
            .with_cuda_ops(|cuda_ops| cuda_ops.relu(&self.cuda_data))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn exp(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = self
            .cuda_data
            .with_cuda_ops(|cuda_ops| cuda_ops.exp(&self.cuda_data))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn log(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = self
            .cuda_data
            .with_cuda_ops(|cuda_ops| cuda_ops.log(&self.cuda_data))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn tanh(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = self
            .cuda_data
            .with_cuda_ops(|cuda_ops| cuda_ops.tanh(&self.cuda_data))?;
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

            let result_cuda = self
                .cuda_data
                .with_cuda_ops(|cuda_ops| cuda_ops.power(&self.cuda_data, &other_gpu.cuda_data))?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        } else {
            Err("GPU-CPU mixed operations not supported for power operations".to_string())
        }
    }

    fn power_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = self
            .cuda_data
            .with_cuda_ops(|cuda_ops| cuda_ops.power_scalar(&self.cuda_data, scalar))?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn sum(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let cuda_ops = self
            .cuda_data
            .get_cuda_ops()
            .ok_or("Failed to get CUDA operations backend")?;

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
                let result_cuda = cuda_ops.sum_axes(&self.cuda_data, axes_list, false)?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
            None => {
                // Sum all elements to scalar using CUDA ops
                let result_cuda = cuda_ops.sum_all(&self.cuda_data)?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
        }
    }

    fn mean(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let cuda_ops = self
            .cuda_data
            .get_cuda_ops()
            .ok_or("Failed to get CUDA operations backend")?;

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

                let divisor_scalar =
                    T::from_f64(1.0 / divisor).ok_or("Failed to convert divisor to tensor type")?;

                // Create scalar tensor and divide
                let divisor_tensor = cuda_ops.full(&[], divisor_scalar)?;
                let result_cuda = cuda_ops.div(&sum_result, &divisor_tensor)?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
            None => {
                // Mean of all elements using CUDA ops
                let result_cuda = cuda_ops.mean_all(&self.cuda_data)?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
        }
    }

    fn max_reduce(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let cuda_ops = self
            .cuda_data
            .get_cuda_ops()
            .ok_or("Failed to get CUDA operations backend")?;

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
                let result_cuda = cuda_ops.max_axes(&self.cuda_data, axes_list, false)?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
            None => {
                // Max of all elements to scalar
                let result_cuda = cuda_ops.max_all(&self.cuda_data)?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
        }
    }

    fn min_reduce(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let cuda_ops = self
            .cuda_data
            .get_cuda_ops()
            .ok_or("Failed to get CUDA operations backend")?;

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
                let result_cuda = cuda_ops.min_axes(&self.cuda_data, axes_list, false)?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
            None => {
                // Min of all elements to scalar
                let result_cuda = cuda_ops.min_all(&self.cuda_data)?;
                Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
            }
        }
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
}
