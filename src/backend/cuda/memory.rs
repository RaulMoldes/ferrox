// src/backend/cuda/memory.rs
use cudarc::driver::DeviceSlice;
use cudarc::driver::{CudaDevice, CudaSlice};
use std::fmt::Debug;
use std::sync::Arc;
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
        unsafe {
            self.device
                .alloc(size)
                .map_err(|e| format!("Failed to allocate GPU memory: {}", e))
        }
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
    pub fn device_to_device<T>(
        &self,
        src: &CudaSlice<T>,
        dst: &mut CudaSlice<T>,
    ) -> Result<(), String>
    where
        T: cudarc::driver::DeviceRepr,
    {
        self.device
            .dtod_copy(src, dst)
            .map_err(|e| format!("Failed to copy device to device: {}", e))
    }

    // Currently cudarc does not allow to access the device free memory or total memory data directly.
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
    T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    // `ValidAsZeroBits` ensures that the type can be safely zeroed out
    // `Unpin` is required for safe memory operations in cudarc
{
    /// Creates a new CUDA tensor with the given data and shape
    pub fn new(data: CudaSlice<T>, shape: Vec<usize>) -> Self {
        let strides = compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }

    /// Creates a CUDA tensor from host data vector
    /// This is a convenience method that combines memory allocation and data transfer
    pub fn from_vec(
        memory_manager: &CudaMemoryManager,
        data: Vec<T>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        // Validate that data size matches shape
        let expected_size = shape.iter().product::<usize>();
        if data.len() != expected_size {
            return Err(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                expected_size
            ));
        }

        // Transfer data from host to device using the memory manager
        let cuda_data = memory_manager.host_to_device(data)?;

        // Create and return tensor
        Ok(Self::new(cuda_data, shape))
    }

    /// Transfers CUDA tensor data back to CPU as a vector
    /// This method copies data from GPU to host memory using the memory manager
    pub fn to_cpu(&self, memory_manager: &CudaMemoryManager) -> Result<Vec<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone,
    {
        // Use the memory manager to perform the device-to-host transfer
        memory_manager.device_to_host(&self.data)
    }

    // Same as `to_cpu`, but returns a vector of the data
    /// Transfers CUDA tensor data back to CPU as a vector
    pub fn to_vec(&self, memory: &CudaMemoryManager) -> Result<Vec<T>, String> {
        memory.device_to_host(&self.data)
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

    // Safer clone implementation
    /// Clones the tensor data to a new CudaTensor
    pub fn deep_clone(&self, memory_manager: &CudaMemoryManager) -> Result<Self, String> {
        // 1. Allocate new device memory
        let num_elements = self.data.len();
        let mut new_data: CudaSlice<T> = memory_manager
            .alloc_zeros(num_elements) // ENSURE THE MEMORY IS ZEROED
            .map_err(|e| format!("Failed to allocate new GPU memory: {}", e))?;

        // 2. Copy data from old slice to new slice
        memory_manager
            .device_to_device(&self.data, &mut new_data)
            .map_err(|e| format!("Failed to copy data to new GPU memory: {}", e))?;

        // 3. Return a new tensor
        Ok(Self {
            data: new_data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }
}

/// Computes strides for a given shape (row-major order)
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

impl<T> Debug for CudaTensor<T>
where
    T: cudarc::driver::DeviceRepr + Clone + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CudaTensor {{ shape: {:?}, strides: {:?}, data: {:?} }}",
            self.shape, self.strides, self.data
        )
    }
}

// Cloning a CudaTensor requires that the underlying data can be cloned
// This is necessary for operations that may need to duplicate tensors on the GPU
// Here we clone the CudaSlice, which is a handle to the GPU memory.
// It can lead to memory races because the underlying data is not copied, just the handle over the cuda slice, then if one handle is modified, the other will see the changes as well.
// This is a shallow clone, meaning it only copies the handle to the GPU memory.
// I am not sure if it would require additional memory allocation or not.
impl<T> Clone for CudaTensor<T>
where
    T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}
