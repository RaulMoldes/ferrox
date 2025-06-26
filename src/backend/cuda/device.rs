// src/backend/cuda/device.rs
use super::kernels::{CudaKernels, load_all_kernels};
use super::memory::{CudaMemoryManager, CudaTensor};
use super::ops::CudaOps;
use cudarc::driver::CudaContext;
use std::sync::Arc;

/// Main CUDA backend that manages device and kernels
pub struct CudaBackend {
    device: Arc<CudaContext>, // CudaContext::new() returns this type directly
    kernels: CudaKernels,
    device_id: usize,
    memory_manager: CudaMemoryManager,
}

impl CudaBackend {
    /// Creates a new CUDA backend for the specified device
    pub fn new(device_id: usize) -> Result<Self, String> {
        // CudaContext::new() returns Arc<CudaContext> already, not CudaContext
        let device = CudaContext::new(device_id)
            .map_err(|e| format!("Failed to initialize CUDA device {}: {}", device_id, e))?;

        // device is already Arc<CudaContext>, so we can clone it directly
        let mut kernels = CudaKernels::new(device.clone());
        let memory_manager = CudaMemoryManager::new(device.clone());
        // Load all kernels during initialization
        load_all_kernels(&mut kernels)?;

        Ok(Self {
            device,
            kernels,
            device_id,
            memory_manager,
        })
    }

    /// Returns the device ID
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Returns reference to the CUDA device
    pub fn device(&self) -> &Arc<CudaContext> {
        &self.device
    }

    /// Returns reference to the kernel manager
    pub fn kernels(&self) -> &CudaKernels {
        &self.kernels
    }

    /// Returns mutable reference to the kernel manager
    pub fn kernels_mut(&mut self) -> &mut CudaKernels {
        &mut self.kernels
    }

    /// Synchronizes the device (waits for all operations to complete)
    pub fn synchronize(&self) -> Result<(), String> {
        self.device
            .synchronize()
            .map_err(|e| format!("CUDA synchronization failed: {}", e))
    }

    /// Returns device name for debugging
    pub fn name(&self) -> String {
        format!("CUDA Device {}", self.device_id)
    }

    /// Creates a CUDA tensor from CPU data
    /// This method takes host data and copies it to GPU memory
    pub fn create_tensor_from_cpu<T>(
        &self,
        data: Vec<T>,
        shape: Vec<usize>,
    ) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr
            + Clone
            + cudarc::driver::ValidAsZeroBits
            + std::marker::Unpin,
    {
        // Validate that data size matches shape
        let size = shape.iter().product::<usize>();
        if data.len() != size {
            return Err(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                size
            ));
        }

        // Create memory manager for this device
        let memory_manager = self.memory_manager();

        // Transfer data from host to device
        let cuda_data = memory_manager.host_to_device(data)?;

        // Create and return the CUDA tensor
        Ok(CudaTensor::new(cuda_data, shape))
    }

    /// Returns reference to memory manager
    /// This method provides access to the memory manager for external use
    pub fn memory_manager(&self) -> &CudaMemoryManager {
        &self.memory_manager
    }

    /// Returns reference to operations interface
    /// This method provides access to CUDA operations for tensor computations
    pub fn ops(&self) -> CudaOps<'_> {
        let memory = self.memory_manager();
        CudaOps::new(&self.kernels, memory)
    }
}

impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBackend")
            .field("device_id", &self.device_id)
            .field("device_name", &self.name())
            .finish()
    }
}
