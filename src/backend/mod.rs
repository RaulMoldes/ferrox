pub mod device;
pub mod manager;
pub mod memory;
pub mod number;
pub mod storage;
pub mod tensor;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::CudaContextManager;

#[cfg(feature = "cuda")]
pub use device::cuda;


pub use device::Device;
pub use device::cpu;
pub use device::default_device;
pub use manager::{BackendManager, best_device, get_backend, has_cuda};

#[cfg(feature = "cuda")]
pub use manager::{with_cuda_context, with_cuda_ops};

pub use number::FerroxCudaF;
pub use number::FerroxF;
pub use tensor::Tensor;
