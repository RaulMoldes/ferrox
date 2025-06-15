// Device abstraction to implement ndarray operations
// Represents a device (e.g., CPU, GPU) for tensor operations.
// Currently the only backend supported is CPU, but this can be extended later.
// Another possible improvement is to start using a generic datatype for the arrays,
// so that we can support different data types like f32, i32, etc.
// This would adhere more to what other libraries like PyTorch or TensorFlow do.
// For now, we will keep it simple and use f64 as the default data type.

use ndarray::{Array, ArrayD, IxDyn};
use rand::Rng;

use super::numeric::Numeric;

/// Device abstraction for tensor operations.
///
/// Represents different computation devices (CPU, GPU) for tensor operations.
/// This enum allows seamless switching between backends while maintaining
/// the same interface for tensor operations
#[derive(Debug, Clone, PartialEq)]
pub enum Device {
    /// Represents the CPU device.
    CPU,

    /// CUDA GPU device
    #[cfg(feature = "cuda")]
    CUDA(usize), // GPU device ID
}

impl Device {
    // Allocate a new array with the given shape and fill it with zeros.
    pub fn zeros<T: Numeric + rand_distr::num_traits::Zero>(&self, shape: &[usize]) -> ArrayD<T> {
        match self {
            Device::CPU => ArrayD::zeros(IxDyn(shape)),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                // For CUDA, we still return CPU arrays here for simplicity
                // The actual GPU operations happen in specialized tensor types
                ArrayD::zeros(IxDyn(shape))
            }
        }
    }

    // Allocate a new array with the given shape and fill it with ones.
    pub fn ones<T: Numeric + rand_distr::num_traits::One>(&self, shape: &[usize]) -> ArrayD<T> {
        match self {
            Device::CPU => ArrayD::ones(IxDyn(shape)),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => ArrayD::ones(IxDyn(shape)),
        }
    }

    // Allocate a new array with the given shape and fill it with random values.
    pub fn randn(&self, shape: &[usize]) -> ArrayD<f64> {
        match self {
            Device::CPU => {
                let mut rng = rand::rng();
                let total_elements: usize = shape.iter().product();
                let data: Vec<f64> = (0..total_elements)
                    .map(|_| rng.random::<f64>() * 2.0 - 1.0) // Simple random between -1 and 1
                    .collect();
                Array::from_shape_vec(IxDyn(shape), data).unwrap()
            }
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                // Generate on CPU then transfer to GPU when needed
                let mut rng = rand::rng();
                let total_elements: usize = shape.iter().product();
                let data: Vec<f64> = (0..total_elements)
                    .map(|_| rng.random::<f64>() * 2.0 - 1.0)
                    .collect();
                Array::from_shape_vec(IxDyn(shape), data).unwrap()
            }
        }
    }

    // Allocate a new array with the given shape and fill it with random integer values.
    pub fn randint(&self, shape: &[usize]) -> ArrayD<i64> {
        match self {
            Device::CPU => {
                let mut rng = rand::rng();
                let total_elements: usize = shape.iter().product();
                let data: Vec<i64> = (0..total_elements)
                    .map(|_| rng.random::<i64>() * 2 - 1) // Simple random between -1 and 1
                    .collect();
                Array::from_shape_vec(IxDyn(shape), data).unwrap()
            }
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                // Generate on CPU then transfer to GPU when needed
                let mut rng = rand::rng();
                let total_elements: usize = shape.iter().product();
                let data: Vec<i64> = (0..total_elements)
                    .map(|_| rng.random::<i64>() * 2 - 1)
                    .collect();
                Array::from_shape_vec(IxDyn(shape), data).unwrap()
            }
        }
    }

    // Allocate a new array with the given shape and fill it with zeroes.
    // This is the same as `zeros`, I would to imitate the numpy.empty function,
    // which does not initialize the array.
    // However ndarray does not have an uninitialized array, so we use zeros for compatibility.
    // If you want to use uninitialized arrays, you can use `Array::uninit` but it is unsafe.
    // For now, we will use `zeros` to keep it safe.
    pub fn empty<T: Numeric + rand_distr::num_traits::Zero>(&self, shape: &[usize]) -> ArrayD<T> {
        match self {
            Device::CPU => ArrayD::zeros(IxDyn(shape)), // ndarray doesn't have uninitialized arrays
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                // For CUDA, we still return CPU arrays here for simplicity
                // The actual GPU operations happen in specialized tensor types
                ArrayD::zeros(IxDyn(shape)) // Same as zeros for now
            }
        }
    }

    // Allocate a new array with the given shape and fill it with a specific value.
    // This is similar to numpy.full.
    pub fn full<T: Numeric>(&self, shape: &[usize], fill_value: T) -> ArrayD<T> {
        match self {
            Device::CPU => ArrayD::from_elem(IxDyn(shape), fill_value),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                // For CUDA, we still return CPU arrays here for simplicity
                // The actual GPU operations happen in specialized tensor types
                ArrayD::from_elem(IxDyn(shape), fill_value)
            }
        }
    }

    pub fn is_cuda(&self) -> bool {
        match self {
            Device::CPU => false,
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => true,
        }
    }

    /// Returns the device ID for CUDA devices.
    #[cfg(feature = "cuda")]
    pub fn cuda_device_id(&self) -> Option<usize> {
        match self {
            Device::CPU => None,
            Device::CUDA(id) => Some(*id),
        }
    }
}

pub fn cpu() -> Device {
    Device::CPU
}

/// Creates a CUDA device with the specified GPU ID.
#[cfg(feature = "cuda")]
pub fn cuda(device_id: usize) -> Device {
    Device::CUDA(device_id)
}

pub fn default_device() -> Device {
    cpu()
}
/// Returns all available devices.

pub fn all_devices() -> Vec<Device> {
    let mut devices = vec![cpu()];

    #[cfg(feature = "cuda")]
    {
        // Try to detect available CUDA devices
        // In a real implementation, you'd query the number of available GPUs
        if let Ok(_) = cudarc::driver::CudaDevice::new(0) {
            devices.push(cuda(0));
        }
    }

    devices
}
