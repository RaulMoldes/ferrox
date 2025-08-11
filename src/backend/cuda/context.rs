// src/backend/cuda/context.rs
use super::kernels::{load_all_kernels, KernelManager};
use super::ops::CudaOps;
use super::stream_manager::StreamManager;
use crate::backend::manager::{alloc_cuda_slice, return_cuda_slice};

use crate::backend::with_cuda_context;
use crate::{FerroxCudaF, FerroxCudaN};
#[allow(unused_imports)]
use cudarc::driver::DeviceSlice;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use ndarray::ArrayD;
use std::default::Default;
use std::fmt::Debug;
use std::sync::Arc;

pub struct CudaContextManager<T>
where
    T: FerroxCudaN,
{
    ctx: Arc<CudaContext>,
    stream_manager: StreamManager,
    ops: Arc<CudaOps<T>>,
}

impl<T> CudaContextManager<T>
where
    T: FerroxCudaN,
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
    pub fn alloc_zeros(&self, size: usize) -> Result<(CudaSlice<T>, u64), String>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    {
        let alloc_result = alloc_cuda_slice::<T>(size)?;
        Ok((alloc_result.data, alloc_result.allocation_id))
    }

    pub fn stream_manager(&self) -> &StreamManager {
        &self.stream_manager
    }

    /// Synchronous host to device transfer
    pub fn host_to_device(&self, data: &[T]) -> Result<(CudaSlice<T>, u64), String> {
        let (mut device_buffer, id) = self.alloc_zeros(data.len())?;

        let stream = match self.stream_manager.get_stream("copy_h2d") {
            Some(h2d_stream) => h2d_stream,
            None => self.stream_manager.default_stream(),
        };

        // Copy data from host to device using the correct cudarc API

        stream
            .memcpy_htod(data, &mut device_buffer) // data is now &[T]
            .map_err(|e| format!("Host to device transfer failed: {}", e))?;

        Ok((device_buffer, id))
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

        stream
            .memcpy_dtoh(data, &mut host_buffer)
            .map_err(|e| format!("Device to host transfer failed: {}", e))?;

        Ok(host_buffer)
    }

    /// Synchronous device to host transfer
    pub fn device_to_device(&self, data: &CudaSlice<T>) -> Result<(CudaSlice<T>, u64), String>
    where
        T: cudarc::driver::DeviceRepr + Clone + Default,
    {
        // Allocate device buffer
        let (mut device_buffer, id) = self.alloc_zeros(data.len())?;

        let stream = match self.stream_manager.get_stream("copy_d2d") {
            Some(d2h_stream) => d2h_stream,
            None => self.stream_manager.default_stream(),
        };

        // Copy data from device to device using the correct cudarc API

        stream
            .memcpy_dtod(data, &mut device_buffer)
            .map_err(|e| format!("Device to host transfer failed: {}", e))?;

        Ok((device_buffer, id))
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
        T: FerroxCudaF,
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

        let (cuda_data, id) = self.host_to_device(data)?; // Pass slice reference
        Ok(CudaTensor::new(cuda_data, shape, id))
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

#[repr(C)]
#[derive(Debug)]
pub struct CudaTensor<T: FerroxCudaN> {
    pub data: Option<CudaSlice<T>>,
    pub allocation_id: u64,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl<T> CudaTensor<T>
where
    T: FerroxCudaN,
{
    /// Creates a new CUDA tensor with computed strides
    pub fn new(data: CudaSlice<T>, shape: Vec<usize>, allocation_id: u64) -> Self {
        let strides = compute_strides(&shape);

        Self {
            data: Some(data),
            allocation_id,
            shape,
            strides,
        }
    }

    pub fn data(&self) -> &CudaSlice<T> {
        self.data
            .as_ref()
            .expect("CudaTensor data was already consumed")
    }

    pub fn data_mut(&mut self) -> &mut CudaSlice<T> {
        self.data
            .as_mut()
            .expect("CudaTensor data was already consumed")
    }

    // Take ownership of the slice (consuming method)
    pub fn take_data(&mut self) -> CudaSlice<T> {
        self.data
            .take()
            .expect("CudaTensor data was already consumed")
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
        let (cuda_data, id) = context_manager.host_to_device(data_slice)?;
        Ok(Self::new(cuda_data, shape, id))
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

        let (cuda_data, id) = context_manager.host_to_device(&data)?; // Pass slice reference
        Ok(Self::new(cuda_data, shape, id))
    }

    /// Transfer tensor data back to CPU
    pub fn to_cpu(self, context_manager: &CudaContextManager<T>) -> Result<Vec<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone,
    {
        let slice = self.data();
        context_manager.device_to_host(slice)
    }

    /// Transfer tensor data back to CPU asynchronously
    pub fn to_cpu_async(
        self,
        context_manager: &CudaContextManager<T>,
        stream_name: Option<&str>,
    ) -> Result<Vec<T>, String> {
        let slice = self.data();
        context_manager.device_to_host_async(slice, stream_name)
    }

    /// Get tensor data as vector
    pub fn to_vec(self, context_manager: &CudaContextManager<T>) -> Result<Vec<T>, String> {
        let slice = self.data();
        let mut full_data = context_manager.device_to_host(slice)?;
        let expected_size = self.size(); // This is shape.iter().product()
        if full_data.len() > expected_size {
            // Memory pool returned larger slice - truncate to logical size
            full_data.truncate(expected_size);
        } else if full_data.len() < expected_size {
            return Err(format!(
                "CudaSlice has {} elements but tensor expects {} - memory corruption!",
                full_data.len(),
                expected_size
            ));
        }

        Ok(full_data)
    }

    /// Create zeroed CUDA tensor
    pub fn alloc_init(
        context_manager: &CudaContextManager<T>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        let size = shape.iter().product();
        let (data, id) = context_manager.alloc_zeros(size)?;
        Ok(Self::new(data, shape, id))
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

        // If shapes are identical, no work needed
        if self.shape == target_shape {
            return Ok(());
        }

        // Compute new strides for broadcasted tensor
        let mut new_strides = vec![0; target_shape.len()];
        let offset = target_shape.len() - self.shape.len();

        for (i, &dim) in self.shape.iter().enumerate() {
            let target_idx = offset + i;
            if dim == 1 && target_shape[target_idx] != 1 {
                // Broadcasting dimension - stride becomes 0 (data will repeat)
                new_strides[target_idx] = 0;
            } else {
                // Non-broadcasting dimension - keep original stride
                new_strides[target_idx] = self.strides[i];
            }
        }

        // Update shape and strides - data stays the same size
        self.shape = target_shape.to_vec();
        self.strides = new_strides;

        // NOTE: self.data remains unchanged - this is key!
        // The physical data size stays small, logical size increases

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


    // Check to see if we need to materialize.
    pub fn needs_materialization(&self) -> bool {
        let logical_size = self.size(); // shape.iter().product()
        let physical_size = if let Some(data) = &self.data {
            data.len() // actual GPU memory allocated
        } else {
            0
        };

        // If logical size > physical size, we have broadcast expansion
        logical_size > physical_size || !self.is_contiguous()
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

impl<T> Drop for CudaTensor<T>
where
    T: FerroxCudaN,
{
    fn drop(&mut self) {
        // Take ownership of the slice without cloning - this is the key fix
        if let Some(slice) = self.data.take() {
            // Move slice to pool - no clone() call means no cuda_free here
            if let Err(e) = return_cuda_slice::<T>(self.allocation_id, slice) {
                eprintln!("Warning: Failed to return CUDA memory to pool: {}", e);
                // slice ownership was transferred to return_cuda_slice
                // If it fails, the slice will be properly dropped inside the pool function
            }
        }
        // If data is None, it was already consumed - nothing to do
    }
}

impl<T> Clone for CudaTensor<T>
where
    T: FerroxCudaN + cudarc::driver::DeviceRepr + Clone,
{
    /// Proper deep clone that allocates new GPU memory with new allocation_id
    /// This prevents double-free errors and memory corruption
    fn clone(&self) -> Self {
        // Copy data from original to new allocation using CUDA memcpy
        let (slice, id) =
            with_cuda_context(|ctx: &CudaContextManager<T>| ctx.device_to_device(self.data()))
                .expect("Failed to copy GPU data during clone");

        // Create new CudaTensor with independent allocation_id
        Self {
            data: Some(slice),
            allocation_id: id,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}
