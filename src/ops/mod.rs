// op.rs
// To program this i have based the implementation they did on the repo from the
// deep learning systems course by the CMU (repo: https://github.com/dlsyscourse/hw1).
// My implementation is not going to be exactly the same, but i followed a similar approach.
// In rust we do not have the concept of inheritance as in Java or Python so I will handle the operators using a common trait.
use crate::backend::Tensor;
use crate::backend::{FerroxCudaF};
use std::any::type_name;
// All operators in the computational graph implement this trait.
pub trait Operator<T>: std::fmt::Debug
where
    T: FerroxCudaF,
{
    // Defines the interface for operators in the computational graph
    // Compute function computes the output in the computational graph.
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String>;

    // Gradient function computes the gradient of the output with respect to the inputs.
    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String>;

    // Get number of inputs this operator expects
    fn num_inputs(&self) -> usize;

    fn name(&self) -> String {
        let full_name = type_name::<Self>();
        full_name.rsplit("::").next().unwrap_or(full_name).to_string()
    }

    fn clone_op(&self) -> Box<dyn Operator<T>>;

}

pub mod basic;
pub mod comparison;
pub mod elementwise;
pub mod matrix;
pub mod reduction;
pub mod reshape;
pub mod scalar;
pub mod unary;

// Re-export all operations for convenient importing
pub use basic::*;
pub use comparison::*;
pub use elementwise::*;
pub use matrix::*;
pub use reduction::*;
pub use reshape::*;
pub use scalar::*;
pub use unary::*;

/// Module containing all basic arithmetic operations
/// These are the fundamental building blocks for most neural network computations
pub mod arithmetic {
    pub use super::basic::*;
    pub use super::elementwise::{MaxElementwise, MinElementwise};
    pub use super::scalar::*;
}

/// Module containing activation functions and other unary operations
/// Essential for non-linear transformations in neural networks
pub mod activations {
    pub use super::unary::{ReLU, Sigmoid, Tanh};
}

/// Module containing mathematical functions
/// Logarithms, exponentials, powers, and other mathematical operations
pub mod math {
    pub use super::basic::{Add, Div, Mul, Sub};
    pub use super::elementwise::{Reciprocal, Sign};
    pub use super::scalar::PowerScalar;
    pub use super::unary::{Abs, Exp, Log, Neg, Power, Sqrt};
}

/// Module containing linear algebra operations
/// Matrix operations essential for neural network layers
pub mod linalg {
    pub use super::matrix::*;
}

/// Module containing reduction operations
/// Statistical and aggregation operations along tensor dimensions
pub mod reductions {
    pub use super::reduction::*;
}

/// Module containing shape manipulation operations
/// Tensor reshaping, broadcasting, and dimension operations
pub mod shape {
    pub use super::reshape::*;
}

/// Module containing comparison operations
/// Logical comparisons and conditional operations
pub mod comparisons {
    pub use super::comparison::*;
}

/// Module containing utility operations
/// Clamping, clipping, and other utility functions
pub mod utils {
    pub use super::comparison::Clamp;
    pub use super::elementwise::{Reciprocal, Sign};
}
