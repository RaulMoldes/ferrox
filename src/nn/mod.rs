// Neural Network Module for Ferrox
// This module provides high-level building blocks for constructing neural networks
// on top of the automatic differentiation engine.

pub mod activations;
pub mod layers;
pub mod loss;
pub mod module;
pub mod parameter;
pub mod optim;
pub mod initializers;
mod tests;

// Re-export the main types and traits for convenience
pub use layers::{Flatten, Identity, Linear, Sequential};
pub use module::{Module, ModuleList};
pub use parameter::{Parameter, ToParameter};
pub use activations::{ReLU, Sigmoid, Tanh, LeakyReLU, ELU, Softmax};
pub use loss::{MSELoss, Loss, Reduction};
pub use optim::{Optimizer, SGD, Adam};
pub use initializers::{init_tensor_kaiming_uniform, init_tensor_xavier_uniform, kaiming_uniform, xavier_uniform};

