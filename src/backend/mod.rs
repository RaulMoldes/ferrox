pub mod number;
pub mod device;
pub mod manager;

#[cfg(feature = "cuda")]
pub mod cuda;
// CUDA backend module

#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;
#[cfg(feature = "cuda")]
pub use device::cuda;

mod tests;

pub use device::Device;
pub use device::cpu;
pub use device::default_device;


pub use number::GPUFloat;
pub use number::GPUNumber;
pub use number::CPUNumber;
pub use number::CPUFloat;
