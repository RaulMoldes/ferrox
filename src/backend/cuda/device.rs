// src/backend/cuda/device.rs
use super::kernels::{CudaKernels, load_all_kernels};
use cudarc::driver::CudaDevice;
use std::sync::Arc;

/// Main CUDA backend that manages device and kernels
pub struct CudaBackend {
    device: Arc<CudaDevice>, // CudaDevice::new() returns this type directly
    kernels: CudaKernels,
    device_id: usize,
}

impl CudaBackend {
    /// Creates a new CUDA backend for the specified device
    pub fn new(device_id: usize) -> Result<Self, String> {
        // CudaDevice::new() returns Arc<CudaDevice> already, not CudaDevice
        let device = CudaDevice::new(device_id)
            .map_err(|e| format!("Failed to initialize CUDA device {}: {}", device_id, e))?;
        
        // device is already Arc<CudaDevice>, so we can clone it directly
        let mut kernels = CudaKernels::new(device.clone());
        
        // Load all kernels during initialization
        load_all_kernels(&mut kernels)?;

        Ok(Self {
            device,
            kernels,
            device_id,
        })
    }

    /// Returns the device ID
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Returns reference to the CUDA device
    pub fn device(&self) -> &Arc<CudaDevice> {
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
        self.device.synchronize()
            .map_err(|e| format!("CUDA synchronization failed: {}", e))
    }

    /// Returns device name for debugging
    pub fn name(&self) -> String {
        format!("CUDA Device {}", self.device_id)
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