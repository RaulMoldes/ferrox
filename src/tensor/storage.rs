// src/tensor/storage.rs
use ndarray::ArrayD;
use std::borrow::Cow;
use std::fmt::Debug;

#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaTensor;

#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;

use crate::backend::number::GPUNumber;

/// Trait for different storage ownership patterns
/// This allows us to have different storage implementations without enum overhead
pub trait StorageBackend<T>: Debug
where
    T: crate::backend::number::GPUNumber,
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
    T: crate::backend::number::GPUNumber,
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

impl<'a, T> StorageBackend<T> for CPUBorrowedStorage<'a, T>
where
    T: crate::backend::number::GPUNumber,
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
}

// Clone implementation for borrowed storage
impl<'a, T> Clone for CPUBorrowedStorage<'a, T> {
    fn clone(&self) -> Self {
        Self { data: self.data }
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
    T: crate::backend::number::GPUNumber,
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
}
