// src/backend/cuda/context.rs
use super::kernels::{KernelManager, load_all_kernels};
use super::ops::CudaOps;
use super::stream_manager::StreamManager;
use crate::backend::number::CPUNumber;
#[allow(unused_imports)]
use cudarc::driver::DeviceSlice;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::default::Default;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::Mutex;
use crate::GPUFloat;

pub struct CudaContextManager<T>
where
    T: GPUFloat
{
    ctx: Arc<CudaContext>,
    stream_manager: StreamManager,
    ops: Arc<CudaOps<T>>,
}



impl <T> CudaContextManager<T>
where T: GPUFloat
{
    pub fn new() -> Result<Self, String> {
        Self::from_device_id(0)
    }

    /// Creates a new CUDA context manager with full backend capabilities
    pub fn from_device_id(device: usize) -> Result<Self, String> {
        let ctx = CudaContext::new(device).map_err(|e| format!("CUDA init error: {}", e))?;
        let stream_manager = StreamManager::new(&ctx);
        stream_manager.setup_parallel_streams(&ctx)?;

        let kernel_manager = if let Some(compute_stream) = stream_manager.get_stream("compute") {
            // Initialize and load kernels during context creation
            let mut kernels = KernelManager::new(compute_stream);
            load_all_kernels(&mut kernels, &ctx)?;
            kernels
        } else {
            let default_stream = ctx.default_stream();
            let mut kernels = KernelManager::new(default_stream);
            load_all_kernels(&mut kernels, &ctx)?;
            kernels
        };

        let ops = CudaOps::new(kernel_manager);

        Ok(Self {
            ctx,
            stream_manager,
            ops: Arc::new(ops),
        })
    }

    pub fn default_stream(&self) -> Arc<CudaStream> {
        self.ctx.default_stream()
    }

    // ============= GPU MEMORY MANAGEMENT =============

    /// Allocates zeroed memory on the GPU
    pub fn alloc_zeros(&self, size: usize) -> Result<CudaSlice<T>, String>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    {
        let stream = match self.stream_manager.get_stream("memset") {
            Some(memset_stream) => memset_stream,
            None => self.stream_manager.default_stream(),
        };

        stream
            .alloc_zeros(size)
            .map_err(|e| format!("Failed to allocate GPU memory: {}", e))
    }

    /// Allocates uninitialized memory on the GPU
    pub unsafe fn alloc(&self, size: usize) -> Result<CudaSlice<T>, String>
    where
        T: cudarc::driver::DeviceRepr,
    {
        let stream = match self.stream_manager.get_stream("memset") {
            Some(memset_stream) => memset_stream,
            None => self.stream_manager.default_stream(),
        };

        unsafe {
            stream
                .alloc(size)
                .map_err(|e| format!("Failed to allocate GPU memory: {}", e))
        }
    }

    /// Synchronous host to device transfer
    pub fn host_to_device(&self, data: &[T]) -> Result<CudaSlice<T>, String>
    {
        let mut device_buffer = self.alloc_zeros(data.len())?;

        let stream = match self.stream_manager.get_stream("copy_h2d") {
            Some(h2d_stream) => h2d_stream,
            None => self.stream_manager.default_stream(),
        };

        // Copy data from host to device using the correct cudarc API
        unsafe {
            stream
                .memcpy_htod(data, &mut device_buffer) // data is now &[T]
                .map_err(|e| format!("Host to device transfer failed: {}", e))?;
        }

        Ok(device_buffer)
    }

    /// Synchronous device to host transfer
    pub fn device_to_host(&self, data: &CudaSlice<T>) -> Result<Vec<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + Default,
    {
        // Allocate host buffer
        let mut host_buffer = vec![T::default(); data.len()];

        let stream = match self.stream_manager.get_stream("copy_d2h") {
            Some(d2h_stream) => d2h_stream,
            None => self.stream_manager.default_stream(),
        };

        // Copy data from device to host using the correct cudarc API
        unsafe {
            stream
                .memcpy_dtoh(data, &mut host_buffer)
                .map_err(|e| format!("Device to host transfer failed: {}", e))?;
        }

        Ok(host_buffer)
    }

    /// Device to device memory copy
    pub fn device_to_device(
        &self,
        src: &CudaSlice<T>,
        dst: &mut CudaSlice<T>,
    ) -> Result<(), String>
    where
        T: cudarc::driver::DeviceRepr,
    {
        if src.len() != dst.len() {
            return Err(format!(
                "Source and destination sizes don't match: {} vs {}",
                src.len(),
                dst.len()
            ));
        }

        // Copy data from device to device using the correct cudarc API
        unsafe {
            self.stream_manager
                .default_stream() // Use default stream as we cannot go async on device to device transfers.
                .memcpy_dtod(src, dst)
                .map_err(|e| format!("Device to device copy failed: {}", e))?;
        }

        Ok(())
    }

    // ============= BACKEND INTERFACE METHODS =============

    /// Get device ID (compatibility method for tests)
    pub fn id(&self) -> usize {
        self.ctx.ordinal()
    }

    /// Get device name for debugging
    pub fn name(&self) -> String {
        self.ctx.name().expect("Error getting current device name")
    }

    /// Access to operations interface
    pub fn ops(&self) -> Arc<CudaOps<T>> {
        self.ops.clone()
    }

    /// Synchronize all operations
    pub fn synchronize(&self) -> Result<(), String> {
        self.ctx
            .synchronize()
            .map_err(|e| format!("CUDA synchronization failed: {}", e))
    }

    /// Access to underlying CUDA context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Create CUDA tensor from CPU data
    pub fn create_tensor_from_cpu(
        &self,
        data: &[T],
        shape: Vec<usize>,
    ) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr
            + Clone
            + cudarc::driver::ValidAsZeroBits
            + std::marker::Unpin
            + Default,
    {
        let size = shape.iter().product::<usize>();
        if data.len() != size {
            return Err(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                size
            ));
        }

        let cuda_data = self.host_to_device(data)?; // Pass slice reference
        Ok(CudaTensor::new(cuda_data, shape))
    }

    // ============= STREAM MANAGEMENT (DELEGATED TO STREAMMANAGER) =============

    /// Creates or gets a named stream for async operations
    pub fn create_stream(&self, name: &str) -> Result<(), String> {
        self.stream_manager.create_stream(&self.ctx, name)
    }

    /// Get stream reference for kernel launches
    pub fn get_stream(&self, stream_name: &str) -> Option<Arc<CudaStream>> {
        self.stream_manager.get_stream(stream_name)
    }

    /// Check if stream exists and is ready (completed all operations)
    pub fn is_stream_ready(&self, stream_name: &str) -> Result<bool, String> {
        self.stream_manager.is_stream_ready(stream_name)
    }

    /// Get names of all managed streams
    pub fn stream_names(&self) -> Vec<String> {
        self.stream_manager.stream_names()
    }

    /// Synchronize all managed streams
    pub fn sync_all_streams(&self) -> Result<(), String> {
        self.stream_manager.sync_all_streams()
    }

    /// Setup parallel streams commonly used in deep learning
    pub fn setup_parallel_streams(&self) -> Result<(), String> {
        self.stream_manager.setup_parallel_streams(&self.ctx)
    }

    /// Synchronize a specific stream
    pub fn sync_stream(&self, stream_name: &str) -> Result<(), String> {
        self.stream_manager.sync_stream(stream_name)
    }

    /// Get number of managed streams
    pub fn stream_count(&self) -> usize {
        self.stream_manager.stream_count()
    }

    /// Remove a stream by name
    pub fn remove_stream(&self, name: &str) -> Result<(), String> {
        self.stream_manager.remove_stream(name)
    }

    /// Clear all streams
    pub fn clear_streams(&self) {
        self.stream_manager.clear_streams();
    }

    // ============= ASYNC OPERATIONS (USING STREAMMANAGER) =============

    /// Asynchronous host to device transfer using named stream
    pub fn host_to_device_async(
        &self,
        data: &[T], // Changed from Vec<T> to &[T]
        stream_name: Option<&str>,
    ) -> Result<CudaSlice<T>, String>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    {
        self.stream_manager
            .host_to_device_async(&self.ctx, data, stream_name)
    }

    /// Asynchronous device to host transfer using named stream
    pub fn device_to_host_async(
        &self,
        data: &CudaSlice<T>,
        stream_name: Option<&str>,
    ) -> Result<Vec<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + Default,
    {
        self.stream_manager
            .device_to_host_async(&self.ctx, data, stream_name)
    }
}

///
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CudaTensor<T: GPUFloat> {
    pub data: CudaSlice<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl<T> CudaTensor<T>
where
    T: GPUFloat
{
    /// Creates a new CUDA tensor with computed strides
    pub fn new(data: CudaSlice<T>, shape: Vec<usize>) -> Self {
        let strides = compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }

    /// Create tensor from CPU ndarray without taking ownership
    /// This allows borrowing external ndarray data for GPU transfer
    pub fn from_cpu_array(
        context_manager: &CudaContextManager<T>,
        array: &ArrayD<T>,
    ) -> Result<Self, String> {
        let shape = array.shape().to_vec();
        let expected_size = shape.iter().product::<usize>();

        // Get contiguous slice from ndarray
        let data_slice = if array.is_standard_layout() {
            // Array is already contiguous, can use as_slice directly
            array
                .as_slice()
                .ok_or("Failed to get contiguous slice from ndarray")?
        } else {
            // Array is not contiguous, need to make it so
            return Err("Non-contiguous arrays not supported. Use .to_owned() first.".to_string());
        };

        if data_slice.len() != expected_size {
            return Err(format!(
                "Array size {} doesn't match shape {:?} (expected {})",
                data_slice.len(),
                shape,
                expected_size
            ));
        }

        // Transfer ndarray data to GPU - this copies the data to GPU
        let cuda_data = context_manager.host_to_device(data_slice)?;
        Ok(Self::new(cuda_data, shape))
    }

    /// Create tensor from CPU ndarray asynchronously without taking ownership
    /// Useful for overlapping transfers with computation
    pub fn from_cpu_array_async(
        context_manager: &CudaContextManager<T>,
        array: &ArrayD<T>,
        stream_name: Option<&str>,
    ) -> Result<Self, String> {
        let shape = array.shape().to_vec();
        let expected_size = shape.iter().product::<usize>();

        // Get contiguous slice from ndarray
        let data_slice = if array.is_standard_layout() {
            array
                .as_slice()
                .ok_or("Failed to get contiguous slice from ndarray")?
        } else {
            return Err("Non-contiguous arrays not supported. Use .to_owned() first.".to_string());
        };

        if data_slice.len() != expected_size {
            return Err(format!(
                "Array size {} doesn't match shape {:?} (expected {})",
                data_slice.len(),
                shape,
                expected_size
            ));
        }

        // Async transfer ndarray data to GPU
        let cuda_data = context_manager.host_to_device_async(data_slice, stream_name)?;
        Ok(Self::new(cuda_data, shape))
    }

    /// Create tensor from CPU slice asynchronously without taking ownership
    /// Useful for overlapping transfers with computation
    pub fn from_cpu_slice_async(
        context_manager: &CudaContextManager<T>,
        data: &[T],
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

        // Async transfer slice data to GPU
        let cuda_data = context_manager.host_to_device_async(data, stream_name)?;
        Ok(Self::new(cuda_data, shape))
    }

    /// Create tensor from host data using context manager
    pub fn from_vec(
        context_manager: &CudaContextManager<T>,
        data: Vec<T>,
        shape: Vec<usize>,
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

        let cuda_data = context_manager.host_to_device(&data)?; // Pass slice reference
        Ok(Self::new(cuda_data, shape))
    }

    /// Create async tensor from host data
    pub fn from_vec_async(
        context_manager: &CudaContextManager<T>,
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

        let cuda_data = context_manager.host_to_device_async(&data, stream_name)?; // Pass slice reference
        Ok(Self::new(cuda_data, shape))
    }

    /// Transfer tensor data back to CPU
    pub fn to_cpu(&self, context_manager: &CudaContextManager<T>) -> Result<Vec<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone,
    {
        context_manager.device_to_host(&self.data)
    }

    /// Transfer tensor data back to CPU asynchronously
    pub fn to_cpu_async(
        &self,
        context_manager: &CudaContextManager<T>,
        stream_name: Option<&str>,
    ) -> Result<Vec<T>, String> {
        context_manager.device_to_host_async(&self.data, stream_name)
    }

    /// Get tensor data as vector (alias for to_cpu)
    pub fn to_vec(&self, context_manager: &CudaContextManager<T>) -> Result<Vec<T>, String> {
        context_manager.device_to_host(&self.data)
    }

    /// Create zeroed CUDA tensor
    pub fn alloc_init(
        context_manager: &CudaContextManager<T>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        let size = shape.iter().product();
        let data = context_manager.alloc_zeros(size)?;
        Ok(Self::new(data, shape))
    }

    // ============= TENSOR METADATA =============

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Deep clone tensor data
    pub fn deep_clone(&self, context_manager: &CudaContextManager<T>) -> Result<Self, String> {
        let num_elements = self.data.len();
        let mut new_data: CudaSlice<T> = context_manager.alloc_zeros(num_elements)?;

        context_manager.device_to_device(&self.data, &mut new_data)?;

        Ok(Self {
            data: new_data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }

    /// Check if tensor is contiguous
    pub fn is_contiguous(&self) -> bool {
        let expected_strides = compute_strides(&self.shape);
        self.strides == expected_strides
    }

    /// Zero-copy reshape
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<(), String> {
        let new_size: usize = new_shape.iter().product();
        let current_size = self.size();

        if new_size != current_size {
            return Err(format!(
                "Cannot reshape tensor of size {} to shape {:?} (size {})",
                current_size, new_shape, new_size
            ));
        }

        if !self.is_contiguous() {
            return Err("Cannot reshape non-contiguous tensor without copying".to_string());
        }

        self.strides = compute_strides(&new_shape);
        self.shape = new_shape;
        Ok(())
    }

    // ============= IN-PLACE TENSOR OPERATIONS =============

    /// In-place broadcast operation - changes tensor view without copying data
    pub fn broadcast_to(&mut self, target_shape: &[usize]) -> Result<(), String> {
        if !can_broadcast_to(&self.shape, target_shape) {
            return Err(format!(
                "Cannot broadcast shape {:?} to {:?}",
                self.shape, target_shape
            ));
        }

        // Compute new strides for broadcasted tensor
        let mut new_strides = vec![0; target_shape.len()];
        let offset = target_shape.len() - self.shape.len();

        for (i, &dim) in self.shape.iter().enumerate() {
            let target_idx = offset + i;
            if dim == 1 && target_shape[target_idx] != 1 {
                // Broadcasting dimension - stride becomes 0
                new_strides[target_idx] = 0;
            } else {
                // Non-broadcasting dimension - keep original stride
                new_strides[target_idx] = self.strides[i];
            }
        }

        self.shape = target_shape.to_vec();
        self.strides = new_strides;
        Ok(())
    }

    /// In-place unsqueeze - add dimension of size 1 at specified axis
    pub fn unsqueeze(&mut self, axis: usize) -> Result<(), String> {
        if axis > self.shape.len() {
            return Err(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                self.shape.len()
            ));
        }

        // Insert dimension of size 1 at specified axis
        self.shape.insert(axis, 1);

        // Insert stride at specified position
        // New dimension has stride equal to the stride of the dimension it's inserted before
        let new_stride = if axis < self.strides.len() {
            self.strides[axis]
        } else {
            1 // If inserting at the end, stride is 1
        };
        self.strides.insert(axis, new_stride);

        Ok(())
    }

    /// In-place squeeze - remove dimensions of size 1
    pub fn squeeze(&mut self, axis: Option<usize>) -> Result<(), String> {
        match axis {
            Some(ax) => {
                // Squeeze specific axis
                if ax >= self.shape.len() {
                    return Err(format!(
                        "Axis {} out of bounds for tensor with {} dimensions",
                        ax,
                        self.shape.len()
                    ));
                }

                if self.shape[ax] != 1 {
                    return Err(format!(
                        "Cannot squeeze axis {} with size {}",
                        ax, self.shape[ax]
                    ));
                }

                self.shape.remove(ax);
                self.strides.remove(ax);
            }
            None => {
                // Squeeze all dimensions of size 1
                let indices_to_remove: Vec<usize> = self
                    .shape
                    .iter()
                    .enumerate()
                    .filter(|&(_, size)| *size == 1)
                    .map(|(i, _)| i)
                    .collect();

                // Remove in reverse order to maintain correct indices
                for &idx in indices_to_remove.iter().rev() {
                    self.shape.remove(idx);
                    self.strides.remove(idx);
                }
            }
        }

        Ok(())
    }

    /// In-place transpose - permutes tensor axes without moving data
    pub fn transpose(&mut self, axes: Option<&[usize]>) -> Result<(), String> {
        match axes {
            Some(axes_order) => {
                // Validate axes specification - same logic as CPU version
                if axes_order.len() != self.ndim() {
                    return Err(format!(
                        "Axes length {} doesn't match tensor dimensions {}",
                        axes_order.len(),
                        self.ndim()
                    ));
                }

                // Verify axes is valid permutation
                let mut sorted_axes = axes_order.to_vec();
                sorted_axes.sort_unstable();
                let expected: Vec<usize> = (0..self.ndim()).collect();
                if sorted_axes != expected {
                    return Err(format!(
                        "Invalid axes permutation: {:?}. Must be a permutation of 0..{}",
                        axes_order,
                        self.ndim()
                    ));
                }

                // Reorder shape and strides according to axes - reuse existing permute logic
                let old_shape = self.shape.clone();
                let old_strides = self.strides.clone();

                for (new_idx, &old_idx) in axes_order.iter().enumerate() {
                    self.shape[new_idx] = old_shape[old_idx];
                    self.strides[new_idx] = old_strides[old_idx];
                }

                Ok(())
            }
            None => {
                // Default transpose - reverse all axes order like CPU version
                match self.ndim() {
                    0 | 1 => {
                        // 0D and 1D tensors unchanged by transpose
                        Ok(())
                    }
                    2 => {
                        // 2D matrices
                        self.shape.swap(0, 1);
                        self.strides.swap(0, 1);
                        Ok(())
                    }
                    _ => {
                        // Higher dimensions - create reversed axes permutation
                        let axes_order: Vec<usize> = (0..self.ndim()).rev().collect();

                        let old_shape = self.shape.clone();
                        let old_strides = self.strides.clone();

                        for (new_idx, &old_idx) in axes_order.iter().enumerate() {
                            self.shape[new_idx] = old_shape[old_idx];
                            self.strides[new_idx] = old_strides[old_idx];
                        }

                        Ok(())
                    }
                }
            }
        }
    }

    /// In-place permute dimensions
    pub fn permute(&mut self, axes: &[usize]) -> Result<(), String> {
        if axes.len() != self.ndim() {
            return Err(format!(
                "Number of axes {} must match tensor dimensions {}",
                axes.len(),
                self.ndim()
            ));
        }

        // Check that all axes are valid and unique
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_unstable();
        for (i, &axis) in sorted_axes.iter().enumerate() {
            if axis != i {
                return Err(format!(
                    "Invalid or duplicate axis in permutation: {:?}",
                    axes
                ));
            }
        }

        // Reorder shape and strides according to axes
        let old_shape = self.shape.clone();
        let old_strides = self.strides.clone();

        for (new_idx, &old_idx) in axes.iter().enumerate() {
            self.shape[new_idx] = old_shape[old_idx];
            self.strides[new_idx] = old_strides[old_idx];
        }

        Ok(())
    }
}

/// Compute row-major strides for a given shape
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Check if shapes can be broadcasted
pub fn can_broadcast_to(from_shape: &[usize], to_shape: &[usize]) -> bool {
    if from_shape.len() > to_shape.len() {
        return false;
    }

    let offset = to_shape.len() - from_shape.len();
    for (i, &dim) in from_shape.iter().enumerate() {
        let target_dim = to_shape[offset + i];
        if dim != 1 && dim != target_dim {
            return false;
        }
    }
    true
}
