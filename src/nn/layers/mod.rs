// src/nn/layers/mod.rs
// Module declaration and basic usage test for neural network layers

pub mod activation;
pub mod conv2d;
pub mod dropout;
pub mod linear;
pub mod norm;
pub mod pooling;
pub mod utils;

// Re-export commonly used layers for convenience
pub use activation::{ReLU, Sigmoid, Tanh};
pub use conv2d::Conv2d;
pub use dropout::Dropout;
pub use linear::Linear;
pub use norm::BatchNorm;
pub use pooling::{Flatten, GlobalAvgPool2d};
