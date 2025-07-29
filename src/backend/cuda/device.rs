// CUDA device management
// This module now only handles device initialization, not backend operations
use cudarc::driver::CudaContext;
use std::sync::Arc;

/// CUDA device wrapper
/// Only responsible for CUDA device initialization and context creation
pub struct CudaDevice {
    context: Arc<CudaContext>,
    device_id: usize,
}

impl CudaDevice {
    /// Initialize CUDA device and create context
    pub fn new(device_id: usize) -> Result<Self, String> {
        let context = CudaContext::new(device_id)
            .map_err(|e| format!("Failed to initialize CUDA device {}: {}", device_id, e))?;

        Ok(Self { context, device_id })
    }

    /// Get the CUDA context for use by other components
    pub fn context(&self) -> Arc<CudaContext> {
        self.context.clone()
    }

    /// Get device ID
    pub fn id(&self) -> usize {
        self.device_id
    }

    /// Check if device is available
    pub fn is_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    /// Synchronize device operations
    pub fn synchronize(&self) -> Result<(), String> {
        self.context
            .synchronize()
            .map_err(|e| format!("CUDA synchronization failed: {}", e))
    }
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDevice")
            .field("device_id", &self.device_id)
            .finish()
    }
}
