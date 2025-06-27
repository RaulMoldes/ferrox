// src/backend/cuda/memory.rs
use cudarc::driver::DeviceSlice;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::collections::HashMap;
use std::default::Default;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::Mutex;
///
/// This module provides high-level abstractions for:
/// - Allocating GPU memory buffers
/// - Host-to-device transfers
/// - Device-to-host transfers
/// - Memory copying between GPU buffers
/// - Memory pool management for better performance
/// - Stream management for asynchronous operations
pub struct CudaMemoryManager {
    ctx: Arc<CudaContext>,
    // Uses a HashMap to manage multiple CUDA streams by name
    // This allows for flexible stream management, where each stream can be identified by a unique name
    streams: Arc<Mutex<HashMap<String, Arc<CudaStream>>>>,
    default_stream: Arc<CudaStream>,
}

unsafe impl Send for CudaMemoryManager {}
unsafe impl Sync for CudaMemoryManager {}

impl CudaMemoryManager {
    /// Creates a new CUDA memory manager for the specified device
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self, String> {
        let default_stream = ctx.default_stream();

        Ok(Self {
            ctx,
            streams: Arc::new(Mutex::new(HashMap::new())),
            default_stream,
        })
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
        self.default_stream
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
            self.default_stream
                .alloc(size)
                .map_err(|e| format!("Failed to allocate GPU memory: {}", e))
        }
    }

    /// SYNCHRONOUS memory transfers between host and device
    // -------- `cudaMemcpy` equivalents for host to device transfers  -------- //

    // ASYNC transfers are implemented using streams, which allow overlapping
    // memory transfers with kernel execution, improving performance.

    /// Copies data from host to device
    /// `host_to_device` is a synchronous operation that blocks until the copy is complete.
    pub fn host_to_device<T>(&self, data: Vec<T>) -> Result<CudaSlice<T>, String>
    where
        T: cudarc::driver::DeviceRepr + std::marker::Unpin, // we need to be able to unpin the data
    {
        let mut gpu_mem = unsafe { self.default_stream.alloc::<T>(data.len()) }
            .map_err(|e| format!("Failed to allocate GPU memory: {}", e))?;

        self.default_stream
            .memcpy_htod(&data, &mut gpu_mem)
            .map_err(|e| format!("Failed to copy host to device: {}", e))?;
        Ok(gpu_mem)
    }

    /// Copies data from device to host synchronously
    pub fn device_to_host<T>(&self, gpu_data: &CudaSlice<T>) -> Result<Vec<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + std::default::Default,
    {
        let mut host_data = vec![T::default(); gpu_data.len()];

        self.default_stream
            .memcpy_dtoh(gpu_data, &mut host_data)
            .map_err(|e| format!("Failed to copy device to host: {}", e))?;
        Ok(host_data)
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
        self.default_stream
            .memcpy_dtod(src, dst)
            .map_err(|e| format!("Failed to copy device to device: {}", e))
    }

    // Currently cudarc does not allow to access the device free memory or total memory data directly.
    // There is a crate called ´cust´ that provides this functionality via FFI.
    // If needed, I can implement this later.

    /// Synchronizes the device to ensure all operations complete
    pub fn synchronize(&self) -> Result<(), String> {
        self.ctx
            .synchronize()
            .map_err(|e| format!("Failed to synchronize device: {}", e))
    }

    // -------- ASYNC MEMORY TRANSFERS -------- //

    /// ASYNC operations using streams
    /// These methods allow overlapping memory transfers with kernel execution,
    /// improving performance by utilizing the GPU's capabilities.
    /// NVIDIA docs: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
    /// A stream is basically a GPU-based pipelining of the memory transfers with kernel execution.
    ///
    ///
    //
    // Note: I was going to implement this when I realized that
    // cudarc 0.12.1 doesn't expose async API directly, so I decided to upgrade to 0.16.5
    // which has better async support.
    pub fn host_to_device_async<T>(
        &self,
        data: Vec<T>,
        stream_name: Option<&str>,
    ) -> Result<CudaSlice<T>, String>
    where
        T: cudarc::driver::DeviceRepr + std::marker::Unpin,
    {
        let stream = self
            .get_stream(stream_name.unwrap_or("default"))
            .ok_or_else(|| "Stream not found".to_string())?;

        let mut gpu_mem = unsafe { stream.alloc::<T>(data.len()) }
            .map_err(|e| format!("Failed to allocate GPU memory: {}", e))?;

        stream
            .memcpy_htod(&data, &mut gpu_mem)
            .map_err(|e| format!("Failed stream-aware host to device copy: {}", e))?;
        Ok(gpu_mem)
    }

    /// Stream-aware device to host copy (currently synchronous but stream-scheduled)
    pub fn device_to_host_async<T>(
        &self,
        gpu_data: &CudaSlice<T>,
        stream_name: Option<&str>,
    ) -> Result<Vec<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + std::default::Default,
    {
        let stream = self
            .get_stream(stream_name.unwrap_or("default"))
            .ok_or_else(|| "Stream not found".to_string())?;

        let mut host_data = vec![T::default(); gpu_data.len()];

        stream
            .memcpy_dtoh(gpu_data, &mut host_data)
            .map_err(|e| format!("Failed stream-aware device to host copy: {}", e))?;
        Ok(host_data)
    }

    // -------- STREAM MANAGEMENT -------- //

    /// Check if stream exists (simplified since we can't query CUDA stream status)
    pub fn is_stream_ready(&self, stream_name: &str) -> Result<bool, String> {
        let streams = self
            .streams
            .lock()
            .map_err(|e| format!("Failed to lock streams: {}", e))?;
        if streams.contains_key(stream_name) {
            Ok(true) // Assume ready since we can't query
        } else {
            Err(format!("Stream '{}' not found", stream_name))
        }
    }

    pub fn stream_names(&self) -> Vec<String> {
        let streams = self.streams.lock().unwrap();
        streams.keys().cloned().collect()
    }

    /// Synchronize specific stream
    pub fn sync_stream(&self, stream_name: &str) -> Result<(), String> {
        let streams = self
            .streams
            .lock()
            .map_err(|e| format!("Failed to lock streams: {}", e))?;
        let stream = streams
            .get(stream_name)
            .ok_or_else(|| format!("Stream '{}' not found", stream_name))?;

        stream
            .synchronize()
            .map_err(|e| format!("Failed to sync stream '{}': {}", stream_name, e))
    }

    /// Synchronizes all streams
    pub fn sync_all_streams(&self) -> Result<(), String> {
        let streams = self
            .streams
            .lock()
            .map_err(|e| format!("Failed to lock streams: {}", e))?;
        for (name, stream) in streams.iter() {
            stream
                .synchronize()
                .map_err(|e| format!("Failed to sync stream '{}': {}", name, e))?;
        }
        Ok(())
    }

    /// -------------------------------------------------
    // -------- PARALLEL OPERATION PATTERN -------- //
    /// This method allows you to perform parallel operations on the GPU
    /// using a provided kernel function. It requires the `setup_parallel_streams` to be called before.
    /// ---------------------------------------------------
    pub fn setup_parallel_streams(
        &mut self,
        //   stream_names: &[&str],
    ) -> Result<(), String> {
        let mut streams = self
            .streams
            .lock()
            .map_err(|e| format!("Failed to lock streams: {}", e))?;
        //Default value for testing
        let stream_names = vec![
            "compute",  // Main compute stream
            "copy_h2d", // Host to device copy stream
            "copy_d2h", // Device to host copy stream
        ];
        for &name in stream_names {
            if streams.contains_key(&name) {
                return Err(format!("Stream '{}' already exists", name));
            }
            let stream = self
                .ctx
                .new_stream()
                .map_err(|e| format!("Failed to create stream '{}': {}", name, e))?;
            streams.insert(name.to_string(), Arc::new(stream));
        }
        Ok(())
    }

    /// Parallel transfer and compute pattern
    pub fn parallel_operation<T, F, R>(
        &self,
        input_data: Vec<T>,
        kernel_fn: F,
    ) -> Result<Vec<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + std::marker::Unpin + Default, // `DeviceRepr` is required for CUDA compatibility
        F: FnOnce(&CudaSlice<T>, &CudaStream) -> Result<R, String>,
        R: AsRef<CudaSlice<T>>,
    {
        // Start async H2D transfer
        let gpu_input = self.host_to_device_async(input_data, Some("copy_h2d"))?;

        // Wait for transfer to complete
        self.sync_stream("copy_h2d")?;

        // Launch kernel on compute stream
        let compute_stream = self
            .get_stream("compute")
            .ok_or("Compute stream not found")?;
        let gpu_output = kernel_fn(&gpu_input, &compute_stream)?;

        // Start async D2H transfer
        let result = self.device_to_host_async(gpu_output.as_ref(), Some("copy_d2h"))?;

        // Sync all streams
        self.sync_stream("compute")?;
        self.sync_stream("copy_d2h")?;

        Ok(result)
    }

    /// Get stream reference for kernel launches
    pub fn get_stream(&self, stream_name: &str) -> Option<Arc<CudaStream>> {
        self.streams.lock().unwrap().get(stream_name).cloned()
    }

    /// Returns reference to the underlying CUDA device
    pub fn device(&self) -> &Arc<CudaContext> {
        &self.ctx
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
    T: cudarc::driver::DeviceRepr
        + Clone
        + cudarc::driver::ValidAsZeroBits
        + std::marker::Unpin
        + Default, // `DeviceRepr` is required for CUDA compatibility
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

    // Creates a CUDA tensor from host data using async transfer
    pub fn from_vec_async(
        memory_manager: &CudaMemoryManager,
        data: Vec<T>,
        shape: Vec<usize>,
        stream_name: Option<&str>,
    ) -> Result<Self, String> {
        let expected_size = shape.iter().product::<usize>();
        if data.len() != expected_size {
            return Err(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                expected_size
            ));
        }

        let cuda_data = memory_manager.host_to_device_async(data, stream_name)?;
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

    /// Transfer tensor data back to CPU asynchronously
    pub fn to_cpu_async(
        &self,
        memory_manager: &CudaMemoryManager,
        stream_name: Option<&str>,
    ) -> Result<Vec<T>, String> {
        memory_manager.device_to_host_async(&self.data, stream_name)
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
