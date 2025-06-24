use crate::tensor::Tensor;
use rand::rng;
use rand_distr::{Distribution, Normal, Uniform};

/// Xavier/Glorot uniform initialization
/// Samples from a uniform distribution U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out))
pub fn xavier_uniform(fan_in: usize, fan_out: usize, gain: f64) -> impl Fn() -> f64 {
    let a = gain * (6.0 / (fan_in + fan_out) as f64).sqrt();
    let uniform = Uniform::new(-a, a).unwrap();

    move || {
        let mut rng = rng();
        uniform.sample(&mut rng)
    }
}

/// Xavier/Glorot normal initialization
/// Samples from a normal distribution N(0, std) where std = gain * sqrt(2 / (fan_in + fan_out))
pub fn xavier_normal(fan_in: usize, fan_out: usize, gain: f64) -> impl Fn() -> f64 {
    let std = gain * (2.0 / (fan_in + fan_out) as f64).sqrt();
    let normal = Normal::new(0.0, std).unwrap();

    move || {
        let mut rng = rng();
        normal.sample(&mut rng)
    }
}

/// Kaiming/He uniform initialization
/// Samples from a uniform distribution U(-bound, bound) where bound = sqrt(6 / fan_in)
/// Specifically designed for ReLU activations
pub fn kaiming_uniform(fan_in: usize, _fan_out: usize, nonlinearity: &str) -> impl Fn() -> f64 {
    assert_eq!(nonlinearity, "relu", "Only relu supported currently");

    let bound = (6.0 / fan_in as f64).sqrt();
    let uniform = Uniform::new(-bound, bound).unwrap();

    move || {
        let mut rng = rng();
        uniform.sample(&mut rng)
    }
}

/// Kaiming/He normal initialization
/// Samples from a normal distribution N(0, std) where std = sqrt(2 / fan_in)
/// Specifically designed for ReLU activations
pub fn kaiming_normal(fan_in: usize, _fan_out: usize, nonlinearity: &str) -> impl Fn() -> f64 {
    assert_eq!(nonlinearity, "relu", "Only relu supported currently");

    let std = (2.0 / fan_in as f64).sqrt();
    let normal = Normal::new(0.0, std).unwrap();

    move || {
        let mut rng = rng();
        normal.sample(&mut rng)
    }
}

// Helper functions to initialize complete tensors
pub fn init_tensor_xavier_uniform(shape: &[usize], gain: f64) -> Tensor<f64> {
    let fan_in = if shape.len() >= 2 {
        shape[shape.len() - 2]
    } else {
        1
    };
    let fan_out = if !shape.is_empty() {
        shape[shape.len() - 1]
    } else {
        1
    };
    let total_size: usize = shape.iter().product();

    let initializer = xavier_uniform(fan_in, fan_out, gain);
    let data = (0..total_size).map(|_| initializer()).collect();
    if let Ok(tensor) = Tensor::from_vec(data, shape) {
        tensor
    } else {
        panic!("Failed to create tensor with shape {:?}", shape);
    }
}

pub fn init_tensor_xavier_normal(shape: &[usize], gain: f64) -> Tensor<f64> {
    let fan_in = if shape.len() >= 2 {
        shape[shape.len() - 2]
    } else {
        1
    };
    let fan_out = if !shape.is_empty() {
        shape[shape.len() - 1]
    } else {
        1
    };
    let total_size: usize = shape.iter().product();

    let initializer = xavier_normal(fan_in, fan_out, gain);
    let data = (0..total_size).map(|_| initializer()).collect();
    if let Ok(tensor) = Tensor::from_vec(data, shape) {
        tensor
    } else {
        panic!("Failed to create tensor with shape {:?}", shape);
    }
}

pub fn init_tensor_kaiming_uniform(shape: &[usize], nonlinearity: &str) -> Tensor<f64> {
    let fan_in = if shape.len() >= 2 {
        shape[shape.len() - 2]
    } else {
        1
    };
    let fan_out = if !shape.is_empty() {
        shape[shape.len() - 1]
    } else {
        1
    };
    let total_size: usize = shape.iter().product();

    let initializer = kaiming_uniform(fan_in, fan_out, nonlinearity);
    let data = (0..total_size).map(|_| initializer()).collect();
    if let Ok(tensor) = Tensor::from_vec(data, shape) {
        tensor
    } else {
        panic!("Failed to create tensor with shape {:?}", shape);
    }
}

pub fn init_tensor_kaiming_normal(shape: &[usize], nonlinearity: &str) -> Tensor<f64> {
    let fan_in = if shape.len() >= 2 {
        shape[shape.len() - 2]
    } else {
        1
    };
    let fan_out = if !shape.is_empty() {
        shape[shape.len() - 1]
    } else {
        1
    };
    let total_size: usize = shape.iter().product();

    let initializer = kaiming_normal(fan_in, fan_out, nonlinearity);
    let data = (0..total_size).map(|_| initializer()).collect();
    if let Ok(tensor) = Tensor::from_vec(data, shape) {
        tensor
    } else {
        panic!("Failed to create tensor with shape {:?}", shape);
    }
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_xavier_uniform() {
        let shape = vec![3, 3];
        let tensor = init_tensor_xavier_uniform(&shape, 1.0);
        assert_eq!(tensor.shape(), &[3, 3]);
    }

    #[test]
    fn test_xavier_normal() {
        let shape = vec![3, 3];
        let tensor = init_tensor_xavier_normal(&shape, 1.0);
        assert_eq!(tensor.shape(), &[3, 3]);
    }

    #[test]
    fn test_kaiming_uniform() {
        let shape = vec![3, 3];
        let tensor = init_tensor_kaiming_uniform(&shape, "relu");
        assert_eq!(tensor.shape(), &[3, 3]);
    }

    #[test]
    fn test_kaiming_normal() {
        let shape = vec![3, 3];
        let tensor = init_tensor_kaiming_normal(&shape, "relu");
        assert_eq!(tensor.shape(), &[3, 3]);
    }
    #[test]
    fn test_xavier_uniform_bounds() {
        let fan_in = 100;
        let fan_out = 50;
        let gain = 1.0;
        let expected_bound = gain * (6.0 / (fan_in + fan_out) as f64).sqrt();

        let init = xavier_uniform(fan_in, fan_out, gain);
        for _ in 0..1000 {
            let val = init();
            assert!(val >= -expected_bound && val <= expected_bound);
        }
    }

    #[test]
    fn test_kaiming_uniform_bounds() {
        let fan_in = 784;
        let fan_out = 256;
        let expected_bound = (6.0 / fan_in as f64).sqrt();

        let init = kaiming_uniform(fan_in, fan_out, "relu");
        for _ in 0..1000 {
            let val = init();
            assert!(val >= -expected_bound && val <= expected_bound);
        }
    }

    #[test]
    fn test_tensor_initialization() {
        let shape = [784, 256];
        let weights = init_tensor_xavier_uniform(&shape, 1.0);

        assert_eq!(weights.shape(), &[784, 256]);
    }

    // TESTS FOR THE NEURAL NETWORK MODULE (ADDED ON 2025-06-10)
}
