// src/tensor/storage.rs
use ndarray::ArrayD;
use std::borrow::Cow;
use std::fmt::Debug;

#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaTensor;

#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;

use crate::backend::number::GPUFloat;

/// Trait for different storage ownership patterns
/// This allows us to have different storage implementations without enum overhead
pub trait StorageBackend<T>: Debug
where
    T: crate::backend::number::GPUFloat,
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
    fn greater_equal(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>;

    /// Element-wise less than or equal comparison: self <= other
    /// Returns new storage with 1.0 for true, 0.0 for false
    fn less_equal(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>;

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
    T: crate::backend::number::GPUFloat,
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


     fn greater_equal(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
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

    fn less_equal(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
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

impl<'a, T> StorageBackend<T> for CPUBorrowedStorage<'a, T>
where
    T: crate::backend::number::GPUFloat,
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


     fn greater_equal(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
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

    fn less_equal(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
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
    T: crate::backend::number::GPUFloat + cudarc::driver::DeviceRepr + Clone,
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


    fn greater_equal(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
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

            let cuda_ops = self
                .cuda_data
                .get_cuda_ops()
                .ok_or("Failed to get CUDA operations backend")?;

            let result_cuda = cuda_ops.greater_equal(&self.cuda_data, &other_gpu.cuda_data)?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        } else {
            Err("GPU-CPU mixed operations not supported for comparison operations".to_string())
        }
    }

    fn less_equal(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
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

            let cuda_ops = self
                .cuda_data
                .get_cuda_ops()
                .ok_or("Failed to get CUDA operations backend")?;

            let result_cuda = cuda_ops.less_equal(&self.cuda_data, &other_gpu.cuda_data)?;
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

            let cuda_ops = self
                .cuda_data
                .get_cuda_ops()
                .ok_or("Failed to get CUDA operations backend")?;

            let result_cuda = cuda_ops.equal(&self.cuda_data, &other_gpu.cuda_data)?;
            Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
        } else {
            Err("GPU-CPU mixed operations not supported for comparison operations".to_string())
        }
    }

    fn logical_not(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let cuda_ops = self
            .cuda_data
            .get_cuda_ops()
            .ok_or("Failed to get CUDA operations backend")?;

        let result_cuda = cuda_ops.logical_not(&self.cuda_data)?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn in_range(&self, min_val: T, max_val: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let cuda_ops = self
            .cuda_data
            .get_cuda_ops()
            .ok_or("Failed to get CUDA operations backend")?;

        let result_cuda = cuda_ops.in_range(&self.cuda_data, min_val, max_val)?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }

    fn sign(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let cuda_ops = self
            .cuda_data
            .get_cuda_ops()
            .ok_or("Failed to get CUDA operations backend")?;

        let result_cuda = cuda_ops.sign(&self.cuda_data)?;
        Ok(Box::new(GPUOwnedStorage::new(result_cuda)))
    }


}
