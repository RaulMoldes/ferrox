// src/backend/cuda/device.rs
use super::kernels::{CudaKernels, load_all_kernels};
use cudarc::driver::CudaDevice;
use std::sync::Arc;

/// Main CUDA backend that manages device and kernels
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    kernels: CudaKernels,
    device_id: usize,
}

impl CudaBackend {
    /// Creates a new CUDA backend for the specified device
    pub fn new(device_id: usize) -> Result<Self, String> {
        // Create the device directly as Arc - don't double wrap
        let device = Arc::new(
            CudaDevice::new(device_id)
                .map_err(|e| format!("Failed to initialize CUDA device {}: {}", device_id, e))?
        );
        
        // Pass the Arc directly - no need to clone here since we're moving it
        let mut kernels = CudaKernels::new(device.clone());
        
        // Load all kernels during initialization
        load_all_kernels(&mut kernels)?;

        Ok(Self {
            device, // This is already Arc<CudaDevice>
            kernels,
            device_id,
        })
    }

    // Rest of your methods remain the same...
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub fn kernels(&self) -> &CudaKernels {
        &self.kernels
    }

    pub fn kernels_mut(&mut self) -> &mut CudaKernels {
        &mut self.kernels
    }

    pub fn synchronize(&self) -> Result<(), String> {
        self.device.synchronize()
            .map_err(|e| format!("CUDA synchronization failed: {}", e))
    }

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