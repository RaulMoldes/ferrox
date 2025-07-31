// src/backend/cuda/mod.rs
#[cfg(feature = "cuda")]
pub mod context;

#[cfg(feature = "cuda")]
pub mod kernels;
#[cfg(feature = "cuda")]
pub mod ops;
#[cfg(feature = "cuda")]
pub mod stream_manager;

#[cfg(feature = "cuda")]
pub use context::{CudaContextManager, CudaTensor};

#[cfg(feature = "cuda")]
pub use kernels::{KernelManager, load_all_kernels};

// ALIAS para compatibilidad - CudaContextManager es ahora el "backend principal"
#[cfg(feature = "cuda")]
pub use context::CudaContextManager as CudaBackend;

#[cfg(all(test, feature = "cuda"))]
mod tests;

// Dummy implementations when CUDA is not available
#[cfg(not(feature = "cuda"))]
pub struct CudaBackend;

#[cfg(not(feature = "cuda"))]
impl CudaBackend {
    pub fn new(_device_id: usize) -> Result<Self, String> {
        Err("CUDA support not compiled".to_string())
    }
}
