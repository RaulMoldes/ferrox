// src/nn/layers/mod.rs
// Module declaration and basic usage test for neural network layers

pub mod activation;
pub mod linear;
pub mod conv2d;
pub mod dropout;
pub mod pooling;
pub mod norm;

// Re-export commonly used layers for convenience
pub use activation::{ReLU, Sigmoid, Tanh};
pub use linear::Linear;
pub use conv2d::Conv2d;
pub use dropout::Dropout;
pub use pooling::{Flatten, GlobalAvgPool2d, MaxPool2d};
pub use norm::{BatchNorm, LayerNorm, RMSNorm};
