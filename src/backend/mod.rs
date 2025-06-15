pub mod device;
pub mod numeric;

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

pub use numeric::Float;
pub use numeric::Numeric;
