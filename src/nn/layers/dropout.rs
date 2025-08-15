// src/nn/layers/dropout.rs
// Dropout layer implementation for regularization during training
// Randomly sets input elements to zero with probability p during training

use crate::backend::number::{FerroxCudaF, FerroxN};
use crate::backend::Tensor;
use crate::graph::{AutoFerroxEngine, NodeId};
use crate::nn::parameter::Parameter;
use crate::nn::Module;
use crate::ops::Mul;
use rand::Rng;

/// Dropout layer for regularization
/// During training: randomly sets elements to zero with probability p
/// During evaluation: passes input unchanged (or scaled by 1-p)
/// This prevents overfitting by forcing the network to not rely on specific neurons
#[derive(Debug, Clone)]
pub struct Dropout<T>
where
    T: FerroxCudaF,
{
    /// Probability of setting an element to zero (0.0 to 1.0)
    pub p: f64,
    /// Training mode flag - affects whether dropout is applied
    training: bool,
    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Dropout<T>
where
    T: FerroxCudaF,
{
    /// Create a new Dropout layer with specified dropout probability
    /// p: probability of setting elements to zero (typically 0.1 to 0.5)
    pub fn new(p: f64) -> Self {
        if !(0.0..=1.0).contains(&p) {
            panic!("Dropout probability must be between 0.0 and 1.0, got {}", p);
        }

        Self {
            p,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create dropout with common probability values
    pub fn new_05() -> Self {
        Self::new(0.5)
    }

    pub fn new_02() -> Self {
        Self::new(0.2)
    }

    pub fn new_01() -> Self {
        Self::new(0.1)
    }

    /// Get the dropout probability
    pub fn probability(&self) -> f64 {
        self.p
    }

    /// Create a dropout mask tensor with the same shape as input
    /// Returns a tensor with values 0.0 or 1.0/(1-p)
    fn create_dropout_mask(&self, input_shape: &[usize]) -> Result<Tensor<T>, String> {
        let total_elements: usize = input_shape.iter().product();

        // Generate random mask: 1.0 for keep, 0.0 for drop
        let mut mask_data = Vec::with_capacity(total_elements);

        // Use thread-local RNG for better performance

        let mut rng = rand::rng();

        let keep_prob = 1.0 - self.p;
        let scale_factor = 1.0 / keep_prob; // Inverted dropout scaling

        for _ in 0..total_elements {
            let random_val: f64 = rng.random();
            if random_val < keep_prob {
                // Keep this element and scale it
                mask_data.push(FerroxN::from_f64(scale_factor).unwrap());
            } else {
                // Drop this element
                mask_data.push(FerroxN::from_f64(0.0).unwrap());
            }
        }

        Tensor::from_vec(mask_data, input_shape)
    }
}

impl<T> Default for Dropout<T>
where
    T: FerroxCudaF,
{
    fn default() -> Self {
        Self::new(0.5) // Default 50% dropout
    }
}

impl<T> Module<T> for Dropout<T>
where
    T: FerroxCudaF,
{
    /// Forward pass: apply dropout during training, pass-through during evaluation
    /// During training: multiplies input by random binary mask scaled by 1/(1-p)
    /// During evaluation: returns input unchanged (inverted dropout)
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        // If not training or dropout probability is 0, return input unchanged
        if !self.training || self.p == 0.0 {
            return Ok(input);
        }

        // If dropout probability is 1.0, return zeros
        if self.p == 1.0 {
            let input_tensor = graph
                .get_tensor(input)
                .ok_or("Input tensor not found in graph")?;

            let zeros = Tensor::zeros(input_tensor.shape())?;
            let node = graph.create_variable(zeros, false);
            return Ok(node);
        }

        // Get input tensor to determine shape
        let input_tensor = graph
            .get_tensor(input)
            .ok_or("Input tensor not found in graph")?;

        // Create dropout mask tensor
        let mask_tensor = self.create_dropout_mask(input_tensor.shape())
            .map_err(|e| format!("Failed to create dropout mask: {}", e))?;

        // Create mask node in computational graph
        let mask_node = graph.create_variable(mask_tensor, false);

        // Apply dropout by element-wise multiplication with mask
        let mul_op = Box::new(Mul);
        graph
            .apply_operation(mul_op, vec![input, mask_node])
            .map_err(|e| format!("Dropout mask application failed: {}", e))
    }

    /// Dropout has no parameters
    fn parameters(&self) -> Vec<&Parameter<T>> {
        Vec::new()
    }

    /// Dropout has no mutable parameters
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        Vec::new()
    }

    /// Get training mode - crucial for dropout behavior
    fn training(&self) -> bool {
        self.training
    }

    /// Set training mode - controls dropout application
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl<T> Dropout<T>
where
    T: FerroxCudaF,
{
    /// Set a new dropout probability
    /// Useful for adjusting dropout rate during training
    pub fn set_probability(&mut self, p: f64) {
        if !(0.0..=1.0).contains(&p) {
            panic!("Dropout probability must be between 0.0 and 1.0, got {}", p);
        }
        self.p = p;
    }

    /// Check if dropout is currently active (training mode and p > 0)
    pub fn is_active(&self) -> bool {
        self.training && self.p > 0.0
    }

    /// Calculate the expected number of neurons that will be dropped
    /// given the input size and dropout probability
    pub fn expected_dropped_count(&self, input_size: usize) -> f64 {
        if self.is_active() {
            input_size as f64 * self.p
        } else {
            0.0
        }
    }

    /// Calculate the expected number of neurons that will remain active
    pub fn expected_active_count(&self, input_size: usize) -> f64 {
        if self.is_active() {
            input_size as f64 * (1.0 - self.p)
        } else {
            input_size as f64
        }
    }

    /// Get the scaling factor applied to remaining neurons (inverted dropout)
    /// Returns 1.0 if dropout is not active, otherwise 1.0/(1.0-p)
    pub fn scale_factor(&self) -> f64 {
        if self.is_active() && self.p < 1.0 {
            1.0 / (1.0 - self.p)
        } else {
            1.0
        }
    }
}
