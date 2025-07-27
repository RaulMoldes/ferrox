pub mod device;
pub mod manager;
pub mod number;

#[cfg(feature = "cuda")]
pub mod cuda;
// CUDA backend module

#[cfg(feature = "cuda")]
pub use cuda::CudaContextManager as CudaBackend;

#[cfg(feature = "cuda")]
pub use device::cuda;

mod tests;

pub use device::Device;
pub use device::cpu;
pub use device::default_device;

pub use number::CPUFloat;
pub use number::CPUNumber;
pub use number::GPUFloat;
pub use number::GPUNumber;
