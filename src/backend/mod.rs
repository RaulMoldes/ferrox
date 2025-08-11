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

pub use device::cpu;
pub use device::default_device;
pub use device::Device;
pub use manager::{best_device, get_backend, has_cuda, BackendManager};

#[cfg(feature = "cuda")]
pub use manager::{with_cuda_context, with_cuda_ops};

pub use number::FerroxCudaF;
pub use number::FerroxF;
pub use number::FerroxCudaN;
pub use number::FerroxN;
pub use tensor::Tensor;
