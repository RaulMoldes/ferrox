// src/nn/layers/activation.rs
// Activation function layers for neural networks
// These layers apply non-linear transformations element-wise to tensors

use crate::backend::number::FerroxCudaF;
use crate::graph::{AutoFerroxEngine, NodeId};
use crate::nn::parameter::Parameter;
use crate::nn::Module;
use crate::ops::{ReLU as ReLUOp, Sigmoid as SigmoidOp, Tanh as TanhOp};
use std::collections::HashMap;

/// ReLU activation layer: f(x) = max(0, x)
/// Most commonly used activation function in deep networks
/// Provides non-linearity while being computationally efficient
#[derive(Debug, Clone)]
pub struct ReLU<T>
where
    T: FerroxCudaF,
{
    /// Training mode flag (not used for ReLU but required by Module trait)
    training: bool,
    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T> ReLU<T>
where
    T: FerroxCudaF,
{
    /// Create a new ReLU activation layer
    pub fn new() -> Self {
        Self {
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> Default for ReLU<T>
where
    T: FerroxCudaF,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for ReLU<T>
where
    T: FerroxCudaF,
{
    /// Forward pass: apply ReLU activation element-wise
    /// Input/Output shape: any shape - activation is applied element-wise
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        // Apply ReLU operation using the computational graph
        let relu_op = Box::new(ReLUOp);
        graph
            .apply_operation(relu_op, vec![input])
            .map_err(|e| format!("ReLU activation failed: {}", e))
    }

    /// ReLU has no parameters
    fn parameters(&self) -> Vec<&Parameter<T>> {
        Vec::new()
    }

    /// ReLU has no mutable parameters
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        Vec::new()
    }

    /// Training mode getter
    fn training(&self) -> bool {
        self.training
    }

    /// Set training mode (no effect for ReLU)
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// Sigmoid activation layer: f(x) = 1 / (1 + exp(-x))
/// Squashes input to range (0, 1), useful for binary classification
/// Can suffer from vanishing gradients in deep networks
#[derive(Debug, Clone)]
pub struct Sigmoid<T>
where
    T: FerroxCudaF,
{
    /// Training mode flag
    training: bool,
    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Sigmoid<T>
where
    T: FerroxCudaF,
{
    /// Create a new Sigmoid activation layer
    pub fn new() -> Self {
        Self {
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> Default for Sigmoid<T>
where
    T: FerroxCudaF,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for Sigmoid<T>
where
    T: FerroxCudaF,
{
    /// Forward pass: apply Sigmoid activation element-wise
    /// Input/Output shape: any shape - activation is applied element-wise
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        // Apply Sigmoid operation using the computational graph
        let sigmoid_op = Box::new(SigmoidOp);
        graph
            .apply_operation(sigmoid_op, vec![input])
            .map_err(|e| format!("Sigmoid activation failed: {}", e))
    }

    /// Sigmoid has no parameters
    fn parameters(&self) -> Vec<&Parameter<T>> {
        Vec::new()
    }

    /// Sigmoid has no mutable parameters
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        Vec::new()
    }

    /// Training mode getter
    fn training(&self) -> bool {
        self.training
    }

    /// Set training mode (no effect for Sigmoid)
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// Hyperbolic tangent activation layer: f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
/// Squashes input to range (-1, 1), zero-centered output
/// Better than sigmoid for hidden layers due to zero-centered output
#[derive(Debug, Clone)]
pub struct Tanh<T>
where
    T: FerroxCudaF,
{
    /// Training mode flag
    training: bool,
    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Tanh<T>
where
    T: FerroxCudaF,
{
    /// Create a new Tanh activation layer
    pub fn new() -> Self {
        Self {
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> Default for Tanh<T>
where
    T: FerroxCudaF,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for Tanh<T>
where
    T: FerroxCudaF,
{
    /// Forward pass: apply Tanh activation element-wise
    /// Input/Output shape: any shape - activation is applied element-wise
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        // Apply Tanh operation using the computational graph
        let tanh_op = Box::new(TanhOp);
        graph
            .apply_operation(tanh_op, vec![input])
            .map_err(|e| format!("Tanh activation failed: {}", e))
    }

    /// Tanh has no parameters
    fn parameters(&self) -> Vec<&Parameter<T>> {
        Vec::new()
    }

    /// Tanh has no mutable parameters
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        Vec::new()
    }

    /// Training mode getter
    fn training(&self) -> bool {
        self.training
    }

    /// Set training mode (no effect for Tanh)
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}
