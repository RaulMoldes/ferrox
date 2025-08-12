// src/nn/layers/mod.rs
// Module declaration and basic usage test for neural network layers

pub mod activation;
pub mod linear;

// Re-export commonly used layers for convenience
pub use activation::{ReLU, Sigmoid, Tanh};
pub use linear::Linear;
