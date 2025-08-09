// Neural Network Module for Ferrox
// This module provides high-level building blocks for constructing neural networks
// on top of the automatic differentiation engine.

pub mod activations;
pub mod initializers;
pub mod module;
pub mod optim;
pub mod parameter;

pub use activations::ReLU;
pub use initializers::{
    init_tensor_kaiming_uniform, init_tensor_xavier_uniform, kaiming_uniform, xavier_uniform,
};
pub use layers::{Flatten, Identity, Linear, Sequential};
pub use loss::{Loss, MSELoss, Reduction};
pub use module::{Module, ModuleList};
pub use normalization::{BatchNorm1d, LayerNorm};
pub use optim::{Adam, Optimizer, SGD};
pub use parameter::{Parameter, ToParameter};
