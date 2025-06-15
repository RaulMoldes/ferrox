// src/backend/manager.rs
// Backend manager that handles both CPU and CUDA backends.
// Current implementation can be extended to support more backends in the future.
use crate::backend::Device;

#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaBackend;

/// Simple backend manager that coordinates CPU and CUDA operations
pub struct BackendManager {
    #[cfg(feature = "cuda")]
    cuda_backend: Option<CudaBackend>,
}

impl BackendManager {
    /// Create new backend manager
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "cuda")]
            cuda_backend: None,
        }
    }

    /// Initialize with CUDA if available
    pub fn init() -> Self {
        let mut manager = Self::new();

        #[cfg(feature = "cuda")]
        {
            // Try to initialize CUDA backend, but don't fail if unavailable
            if let Ok(cuda_backend) = CudaBackend::new(0) {
                manager.cuda_backend = Some(cuda_backend);
                println!("CUDA backend initialized successfully");
            } else {
                println!("CUDA backend not available, using CPU only");
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            println!("CUDA feature not enabled, using CPU only");
        }

        manager
    }

    /// Check if CUDA is available
    pub fn has_cuda(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.cuda_backend.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Get CUDA backend if available
    #[cfg(feature = "cuda")]
    pub fn cuda_backend(&self) -> Option<&CudaBackend> {
        self.cuda_backend.as_ref()
    }

    /// Get the best device for operations
    pub fn best_device(&self) -> Device {
        if self.has_cuda() {
            #[cfg(feature = "cuda")]
            {
                Device::CUDA(0)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Device::CPU
            }
        } else {
            Device::CPU
        }
    }
}

// Global backend manager instance
// This is actually the way to implement the singleton pattern in Rust.
// The `OnceLock` type ensures that the backend is initialized only once.
// A.k.a the lock is only acquired once, and subsequent calls will return the already initialized instance.
use std::sync::OnceLock;

// static variable to hold the global backend manager instance
static GLOBAL_BACKEND: OnceLock<BackendManager> = OnceLock::new();

/// Get the global backend manager
pub fn get_backend() -> &'static BackendManager {
    GLOBAL_BACKEND.get_or_init(|| BackendManager::init())
}

/// Check if CUDA is available globally
pub fn has_cuda() -> bool {
    get_backend().has_cuda()
}

/// Get the best available device globally
pub fn best_device() -> Device {
    get_backend().best_device()
}
