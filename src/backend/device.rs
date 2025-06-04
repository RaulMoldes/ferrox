// Device abstraction to implement ndarray operations
// Represents a device (e.g., CPU, GPU) for tensor operations.
// Currently my only backend is CPU, but this can be extended later.

#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
use ndarray::{Array, ArrayD, IxDyn};
use rand::Rng;

#[derive(Debug, Clone, PartialEq)]
pub enum Device {
    CPU,
}

impl Device {
    pub fn zeros(&self, shape: &[usize]) -> ArrayD<f64> {
        match self {
            Device::CPU => ArrayD::zeros(IxDyn(shape)),
        }
    }
    
    pub fn ones(&self, shape: &[usize]) -> ArrayD<f64> {
        match self {
            Device::CPU => ArrayD::ones(IxDyn(shape)),
        }
    }
    
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
        }
    }
    
    pub fn empty(&self, shape: &[usize]) -> ArrayD<f64> {
        match self {
            Device::CPU => ArrayD::zeros(IxDyn(shape)), // ndarray doesn't have uninitialized arrays
        }
    }
    
    pub fn full(&self, shape: &[usize], fill_value: f64) -> ArrayD<f64> {
        match self {
            Device::CPU => ArrayD::from_elem(IxDyn(shape), fill_value),
        }
    }
}

pub fn cpu() -> Device {
    Device::CPU
}

pub fn default_device() -> Device {
    cpu()
}

pub fn all_devices() -> Vec<Device> {
    vec![cpu()]
}
