// src/backend/cuda/memory.rs
#[allow(unused_imports)]
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
pub struct CudaContextManager {
    ctx: Arc<CudaContext>,
    // Uses a HashMap to manage multiple CUDA streams by name
    // This allows for flexible stream management, where each stream can be identified by a unique name
    streams: Arc<Mutex<HashMap<String, Arc<CudaStream>>>>,
    default_stream: Arc<CudaStream>,
}

unsafe impl Send for CudaContextManager {}
unsafe impl Sync for CudaContextManager {}

impl CudaContextManager {
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
        let stream = if let Some(name) = stream_name {
            let found_stream = self
                .get_stream(name)
                .ok_or_else(|| "Stream not found".to_string())?;
            found_stream
        } else {
            self.default_stream.clone()
        };

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
        let stream = if let Some(name) = stream_name {
            let found_stream = self
                .get_stream(name)
                .ok_or_else(|| "Stream not found".to_string())?;
            found_stream
        } else {
            self.default_stream.clone()
        };

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
        for name in stream_names {
            if streams.contains_key(name) {
                return Err(format!("Stream '{}' already exists", name));
            }
            let stream = self
                .ctx
                .new_stream()
                .map_err(|e| format!("Failed to create stream '{}': {}", name, e))?;

            streams.insert(name.to_string(), stream.clone());
        }
        Ok(())
    }

    pub fn create_stream(&self, stream_name: &str) -> Result<Arc<CudaStream>, String> {
        let mut streams = self
            .streams
            .lock()
            .map_err(|e| format!("Failed to lock streams: {}", e))?;
        if streams.contains_key(stream_name) {
            return Err(format!("Stream '{}' already exists", stream_name));
        }
        let stream = self
            .ctx
            .new_stream()
            .map_err(|e| format!("Failed to create stream '{}': {}", stream_name, e))?;

        streams.insert(stream_name.to_string(), stream.clone());
        Ok(stream)
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
#[repr(C)] // Better memory layout for CUDA compatibility
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
        context_manager: &CudaContextManager,
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
        let cuda_data = context_manager.host_to_device(data)?;

        // Create and return tensor
        Ok(Self::new(cuda_data, shape))
    }

    // Creates a CUDA tensor from host data using async transfer
    pub fn from_vec_async(
        context_manager: &CudaContextManager,
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

        let cuda_data = context_manager.host_to_device_async(data, stream_name)?;
        Ok(Self::new(cuda_data, shape))
    }

    /// Transfers CUDA tensor data back to CPU as a vector
    /// This method copies data from GPU to host memory using the memory manager
    pub fn to_cpu(&self, context_manager: &CudaContextManager) -> Result<Vec<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone,
    {
        // Use the memory manager to perform the device-to-host transfer
        context_manager.device_to_host(&self.data)
    }

    /// Transfer tensor data back to CPU asynchronously
    pub fn to_cpu_async(
        &self,
        context_manager: &CudaContextManager,
        stream_name: Option<&str>,
    ) -> Result<Vec<T>, String> {
        context_manager.device_to_host_async(&self.data, stream_name)
    }

    // Same as `to_cpu`, but returns a vector of the data
    /// Transfers CUDA tensor data back to CPU as a vector
    pub fn to_vec(&self, memory: &CudaContextManager) -> Result<Vec<T>, String> {
        memory.device_to_host(&self.data)
    }

    /// Creates a new zeroed CUDA tensor with the given shape
    pub fn zeros(context_manager: &CudaContextManager, shape: Vec<usize>) -> Result<Self, String> {
        let size = shape.iter().product();
        let data = context_manager.alloc_zeros(size)?;
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
    pub fn deep_clone(&self, context_manager: &CudaContextManager) -> Result<Self, String> {
        // 1. Allocate new device memory
        let num_elements = self.data.len();
        let mut new_data: CudaSlice<T> = context_manager
            .alloc_zeros(num_elements) // ENSURE THE MEMORY IS ZEROED
            .map_err(|e| format!("Failed to allocate new GPU memory: {}", e))?;

        // 2. Copy data from old slice to new slice
        context_manager
            .device_to_device(&self.data, &mut new_data)
            .map_err(|e| format!("Failed to copy data to new GPU memory: {}", e))?;

        // 3. Return a new tensor
        Ok(Self {
            data: new_data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }

    /// Zero-copy memory management
    /// This OPs allow to manipulate the view of the data without copying it
    /// It is useful for broadcasting, slicing, and reshaping tensors on GPU
    /// Zero-copy reshape - only changes shape/strides metadata
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, String> {
        let new_size: usize = new_shape.iter().product();
        let current_size = self.size();

        if new_size != current_size {
            return Err(format!(
                "Cannot reshape tensor of size {} to shape {:?} (size {})",
                current_size, new_shape, new_size
            ));
        }

        // Check if reshape is contiguous (can reuse same data layout)
        // If not contiguous, we cannot reshape without copying
        if !self.is_contiguous() {
            return Err("Cannot reshape non-contiguous tensor without copying".to_string());
        }

        Ok(Self {
            data: self.data.clone(), // Shallow clone - same GPU memory
            shape: new_shape,
            strides: compute_strides(&new_shape),
        })
    }

    /// Zero-copy broadcast - BROADCAST OPERATION (Based on: https://numpy.org/doc/2.1/reference/generated/numpy.broadcast.html)
    /// This operation allows to create a new tensor with a different shape
    /// that shares the same GPU memory as the original tensor.
    /// It uses the can_broadcast function to check if the shapes are compatible.
    /// It returns an error if the shapes cannot be broadcasted.
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Self, String> {
        if !can_broadcast(&self.shape, target_shape) {
            return Err(format!(
                "Cannot broadcast shape {:?} to {:?}",
                self.shape, target_shape
            ));
        }

        let new_strides = broadcast_strides(&self.shape, &self.strides, target_shape);

        Ok(Self {
            data: self.data.clone(), // Same GPU memory
            shape: target_shape.to_vec(),
            strides: new_strides,
        })
    }

    /// Check if tensor memory layout is contiguous
    pub fn is_contiguous(&self) -> bool {
        let expected_strides = compute_strides(&self.shape);
        self.strides == expected_strides
    }

    /// Zero-copy unsqueeze - add dimension of size 1 at specified axis
    pub fn unsqueeze(&self, axis: usize) -> Result<Self, String> {
        if axis > self.shape.len() {
            return Err(format!(
                "Cannot unsqueeze at axis {} for tensor with {} dimensions",
                axis,
                self.shape.len()
            ));
        }

        let mut new_shape = self.shape.clone();
        new_shape.insert(axis, 1);

        let new_strides = unsqueeze_strides(&self.strides, axis);

        Ok(Self {
            data: self.data.clone(), // Zero-copy: same GPU memory
            shape: new_shape,
            strides: new_strides,
        })
    }

    /// Zero-copy squeeze - remove dimensions of size 1
    pub fn squeeze(&self, axis: Option<usize>) -> Result<Self, String> {
        match axis {
            Some(ax) => {
                // Squeeze specific axis
                if ax >= self.shape.len() {
                    return Err(format!("Axis {} out of bounds", ax));
                }
                if self.shape[ax] != 1 {
                    return Err(format!(
                        "Cannot squeeze axis {} with size {}",
                        ax, self.shape[ax]
                    ));
                }

                let mut new_shape = self.shape.clone();
                let mut new_strides = self.strides.clone();
                new_shape.remove(ax);
                new_strides.remove(ax);

                Ok(Self {
                    data: self.data.clone(),
                    shape: new_shape,
                    strides: new_strides,
                })
            }
            None => {
                // Squeeze all dimensions of size 1
                let (new_shape, new_strides) = squeeze_all_dims(&self.shape, &self.strides);

                Ok(Self {
                    data: self.data.clone(),
                    shape: new_shape,
                    strides: new_strides,
                })
            }
        }
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

/// Check if shapes can be broadcasted
/// Follows NumPy broadcasting rules:
/// 1. Start from trailing dimensions and work backward
/// 2. Dimensions are compatible if:
///    - They are equal, OR
///    - One of them is 1, OR
///    - One of them is missing (treat as 1)
/// 3. Missing dimensions are added as size 1 at the beginning
/// 4. Result shape has the maximum size in each dimension
fn can_broadcast_to(source_shape: &[usize], target_shape: &[usize]) -> bool {
    // Source cannot have more dimensions than target
    if source_shape.len() > target_shape.len() {
        return false;
    }

    let offset = target_shape.len() - source_shape.len();

    // Check each dimension from right to left
    for (i, &src_dim) in source_shape.iter().enumerate() {
        let tgt_dim = target_shape[i + offset];

        // Compatible if: equal OR source is 1
        if src_dim != 1 && src_dim != tgt_dim {
            return false;
        }
    }

    true
}

/// Compute broadcast strides for target shape
/// This function calculates the strides for a new shape that is a broadcasted version of the source
/// shape. It follows the broadcasting rules defined in `can_broadcast_to`.
fn broadcast_strides(
    source_shape: &[usize],
    source_strides: &[usize],
    target_shape: &[usize],
) -> Vec<usize> {
    let mut new_strides = vec![0; target_shape.len()];
    let offset = target_shape.len() - source_shape.len();

    // Leading dimensions get stride 0 (broadcasted)
    for i in 0..offset {
        new_strides[i] = 0;
    }

    // Map existing dimensions
    for (i, (&src_dim, &src_stride)) in source_shape.iter().zip(source_strides).enumerate() {
        let tgt_idx = i + offset;
        new_strides[tgt_idx] = if src_dim == 1 { 0 } else { src_stride };
    }

    new_strides
}

/// Utility functions to handle tenso shape and strides calculations
/// Compute strides for unsqueeze operation
fn unsqueeze_strides(strides: &[usize], axis: usize) -> Vec<usize> {
    let mut new_strides = strides.to_vec();

    // Stride for new dimension: use stride of next dimension, or 1 if at end
    let new_stride = if axis < strides.len() {
        strides[axis]
    } else {
        1
    };

    new_strides.insert(axis, new_stride);
    new_strides
}

/// Remove all dimensions of size 1 and their corresponding strides
fn squeeze_all_dims(shape: &[usize], strides: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut new_shape = Vec::new();
    let mut new_strides = Vec::new();

    for (i, &dim_size) in shape.iter().enumerate() {
        if dim_size != 1 {
            new_shape.push(dim_size);
            new_strides.push(strides[i]);
        }
    }

    // Handle edge case: if all dimensions were size 1, keep one
    if new_shape.is_empty() {
        new_shape.push(1);
        new_strides.push(1);
    }

    (new_shape, new_strides)
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
