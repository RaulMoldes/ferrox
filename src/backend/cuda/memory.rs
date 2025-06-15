// src/backend/cuda/memory.rs
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

/// Basic CUDA memory manager for handling GPU memory allocations and transfers
/// 
/// This module provides high-level abstractions for:
/// - Allocating GPU memory buffers
/// - Host-to-device transfers
/// - Device-to-host transfers
/// - Memory copying between GPU buffers
/// - Memory pool management for better performance
pub struct CudaMemoryManager {
    device: Arc<CudaDevice>,
}

impl CudaMemoryManager {
    /// Creates a new CUDA memory manager for the specified device
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }

    /// Allocates zeroed memory on the GPU. In C++ API, you would do this with 
    /// `cudaMalloc` + `cudaMemset`.
    /// This is a convenience method that combines allocation and zeroing.
    /// It is safer, because it ensures the memory is initialized to zero,
    /// avoiding potential issues with uninitialized memory and corrupted data.
    pub fn alloc_zeros<T>(&self, size: usize) -> Result<CudaSlice<T>, String>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    {
        self.device
            .alloc_zeros(size)
            .map_err(|e| format!("Failed to allocate GPU memory: {}", e))
    }

    /// Allocates uninitialized memory on the GPU.
    /// Basic cudaMalloc equivalent.
    pub unsafe fn alloc<T>(&self, size: usize) -> Result<CudaSlice<T>, String>
    where
        T: cudarc::driver::DeviceRepr,
    { 

      // SAFETY: This method does not initialize the memory, so it may contain garbage data.
      // Use with caution, as it may lead to undefined behavior if the data is not initialized.
      // It is the user's responsibility to ensure the memory is initialized before use.
      
        self.device
            .alloc(size)
            .map_err(|e| format!("Failed to allocate GPU memory: {}", e))
    
    }


    // -------- `cudaMemcpy` equivalents for host to device transfers  -------- //
    // ASYNC transfers are not implemented yet, because I don't need them for now.

    /// Copies data from host to device
    /// `htod_copy` is a synchronous operation that blocks until the copy is complete.
    pub fn host_to_device<T>(&self, data: Vec<T>) -> Result<CudaSlice<T>, String>
    where
        T: cudarc::driver::DeviceRepr + std::marker::Unpin, // we need to be able to unpin the data
    {
        self.device
            .htod_copy(data)
            .map_err(|e| format!("Failed to copy host to device: {}", e))
    }

    /// Copies data from device to host synchronously
    pub fn device_to_host<T>(&self, gpu_data: &CudaSlice<T>) -> Result<Vec<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone,
    {
        self.device
            .dtoh_sync_copy(gpu_data)
            .map_err(|e| format!("Failed to copy device to host: {}", e))
    }


    /// Copies data between GPU buffers.
    /// Not used for now but if I scale the project to support multiple GPUs, and parallel execution
    /// will be nice to have
    pub fn device_to_device<T>(&self, src: &CudaSlice<T>, dst: &mut CudaSlice<T>) -> Result<(), String>
    where
        T: cudarc::driver::DeviceRepr,
    {
        self.device
            .dtod_copy(src, dst)
            .map_err(|e| format!("Failed to copy device to device: {}", e))
    }

    /// Returns the total memory available on the device
    pub fn total_memory(&self) -> Result<usize, String> {
        self.device.as_ref()
            .total_memory()
            .map_err(|e| format!("Failed to get total memory: {}", e))
    }


    // Currently cudarc does not allow to access the device free memory directly.
    // There is a crate called ´cust´ that provides this functionality via FFI.
    // If needed, I can implement this later.
    

    /// Synchronizes the device to ensure all operations complete
    pub fn synchronize(&self) -> Result<(), String> {
        self.device
            .synchronize()
            .map_err(|e| format!("Failed to synchronize device: {}", e))
    }

    /// Returns reference to the underlying CUDA device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

/// Convenience struct for managing tensors on GPU
/// 
/// Wraps CudaSlice with additional metadata for tensor operations
pub struct CudaTensor<T> {
    pub data: CudaSlice<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl<T> CudaTensor<T>
where
    T: cudarc::driver::DeviceRepr + Clone,
{
    /// Creates a new CUDA tensor with the given data and shape
    pub fn new(data: CudaSlice<T>, shape: Vec<usize>) -> Self {
        let strides = compute_strides(&shape);
        Self { data, shape, strides }
    }

    /// Creates a new zeroed CUDA tensor with the given shape
    pub fn zeros(memory_manager: &CudaMemoryManager, shape: Vec<usize>) -> Result<Self, String> {
        let size = shape.iter().product();
        let data = memory_manager.alloc_zeros(size)?;
        Ok(Self::new(data, shape))
    }

    /// Returns the total number of elements in the tensor
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the strides of the tensor
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Reshapes the tensor to new dimensions (without copying data)
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<(), String> {
        let new_size: usize = new_shape.iter().product();
        let current_size = self.size();
        
        if new_size != current_size {
            return Err(format!(
                "Cannot reshape tensor of size {} to shape {:?} (size {})",
                current_size, new_shape, new_size
            ));
        }
        
        self.shape = new_shape;
        self.strides = compute_strides(&self.shape);
        Ok(())
    }
}

/// Computes strides for a given shape (row-major order)
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
