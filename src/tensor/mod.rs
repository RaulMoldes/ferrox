#[cfg(not(feature = "cuda"))]
pub mod cputensor;

#[cfg(feature = "cuda")]
pub mod gputensor;

pub mod tests;

#[cfg(not(feature = "cuda"))]
pub use cputensor::CPUTensor;

pub mod storage;

#[cfg(feature = "cuda")]
pub use gputensor::GPUTensor;

#[cfg(feature = "cuda")]
pub type Tensor<T> = GPUTensor<T>;

#[cfg(not(feature = "cuda"))]
pub type Tensor<T> = CPUTensor<T>;
