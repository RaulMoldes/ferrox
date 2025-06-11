// Neural Network Module for Ferrox
// This module provides high-level building blocks for constructing neural networks
// on top of the automatic differentiation engine.

pub mod activations;
pub mod layers;
pub mod loss;
pub mod module;
pub mod parameter;
mod tests;
// Re-export the main types and traits for convenience
pub use layers::{Flatten, Identity, Linear, Sequential};
pub use module::{Module, ModuleList};
pub use parameter::{Parameter, ToParameter};
pub use activations::{ReLU, Sigmoid, Tanh, LeakyReLU, ELU};
pub use loss::{MSELoss, Loss, Reduction};

/// Neural network initialization utilities
pub mod init {
    pub use crate::initializers::*;
}
