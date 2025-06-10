// Neural Network Module for Ferrox
// This module provides high-level building blocks for constructing neural networks
// on top of the automatic differentiation engine.

pub mod layers;
pub mod module;
pub mod parameter;
pub mod activations;
pub mod loss;
mod tests;
// Re-export the main types and traits for convenience
pub use layers::{Flatten, Identity, Linear, Sequential};
pub use module::{Module, ModuleList};
pub use parameter::{Parameter, ToParameter};

/// Neural network initialization utilities
pub mod init {
    pub use crate::initializers::*;
}
