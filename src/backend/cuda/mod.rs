// src/backend/cuda/mod.rs
#[cfg(feature = "cuda")]
pub mod kernels;
#[cfg(feature = "cuda")]
pub mod device;


#[cfg(feature = "cuda")]
pub use device::CudaBackend;
#[cfg(feature = "cuda")]
pub use kernels::{CudaKernels, load_all_kernels};

// Dummy implementations when CUDA is not available
#[cfg(not(feature = "cuda"))]
pub struct CudaBackend;

#[cfg(not(feature = "cuda"))]
impl CudaBackend {
    pub fn new(_device_id: usize) -> Result<Self, String> {
        Err("CUDA support not compiled".to_string())
    }
}