use serde::{Deserialize, Serialize};
use bincode::{Encode, Decode};
// src/backend/device.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, Encode, Decode)]
pub enum Device {
    #[default]
    CPU,
    #[cfg(feature = "cuda")]
    CUDA(usize), // Device ID for multi-GPU systems
}

impl Device {
    // Check if device is CPU
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::CPU)
    }

    // Check if device is any GPU type
    pub fn is_cuda(&self) -> bool {
        match self {
            Device::CPU => false,
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => true,
        }
    }

    // Get device ID for CUDA devices
    pub fn device_id(&self) -> Option<usize> {
        match self {
            Device::CPU => None,
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => Some(*id),
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::CPU => write!(f, "CPU"),
            #[cfg(feature = "cuda")]
            Device::CUDA(id) => write!(f, "CUDA:{}", id),
        }
    }
}

// Helper functions to maintain compatibility with your existing code
pub fn cpu() -> Device {
    Device::CPU
}

#[cfg(feature = "cuda")]
pub fn cuda(device_id: usize) -> Device {
    Device::CUDA(device_id)
}

// Default device selection - you can modify this logic as needed
pub fn default_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        // Try CUDA first, fall back to CPU if not available
        // You can add actual CUDA detection here if needed
        Device::CUDA(0)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::CPU
    }
}
