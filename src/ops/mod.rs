// op.rs
// To program this i have based the implementation they did on the repo from the
// deep learning systems course by the CMU (repo: https://github.com/dlsyscourse/hw1).
// My implementation is not going to be exactly the same, but i followed a similar approach.
// In rust we do not have the concept of inheritance as in Java or Python so I will handle the operators using a common trait.
use crate::backend::manager::best_f32_device;
use crate::backend::FerroxCudaF;
use crate::backend::Tensor;
#[allow(unused_imports)]
use crate::graph::AutoFerroxEngine;
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
        full_name
            .rsplit("::")
            .next()
            .unwrap_or(full_name)
            .to_string()
    }

    fn clone_op(&self) -> Box<dyn Operator<T>>;
}

impl<T> Operator<T> for Box<dyn Operator<T>>
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        // Delegate to the boxed operator
        self.as_ref().compute(inputs)
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        // Delegate to the boxed operator
        self.as_ref().gradient(grad_output, inputs, outputs)
    }

    fn num_inputs(&self) -> usize {
        // Delegate to the boxed operator
        self.as_ref().num_inputs()
    }

    fn name(&self) -> String {
        // Delegate to the boxed operator
        self.as_ref().name()
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        // Delegate to the boxed operator
        self.as_ref().clone_op()
    }
}

pub mod basic;
pub mod batched;
pub mod comparison;
pub mod elementwise;
pub mod matrix;
pub mod reduction;
pub mod reshape;
pub mod scalar;
pub mod unary;

// Re-export all operations for convenient importing
pub use basic::*;
pub use batched::*;
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
    pub use super::batched::Softmax;
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

#[allow(dead_code)]
// TESTING UTILITIES
// Test helper - creates tensors on best available device
fn tensor_1d(data: &[f32]) -> Tensor<f32> {
    let device = best_f32_device();
    Tensor::from_vec_with_device(data.to_vec(), &[data.len()], device)
        .expect("Tensor creation failed")
}

#[allow(dead_code)]
fn tensor_2d(data: &[f32], rows: usize, cols: usize) -> Tensor<f32> {
    let device = best_f32_device();
    Tensor::from_vec_with_device(data.to_vec(), &[rows, cols], device)
        .expect("Tensor creation failed")
}

// Helper to convert tensor to vector for numerical comparison
#[allow(dead_code)]
fn tensor_to_vec(tensor: &Tensor<f32>) -> Vec<f32> {
    tensor.to_vec().unwrap()
}

// Helper to check if two values are approximately equal
#[allow(dead_code)]
fn approx_eq(a: f32, b: f32, tolerance: f32) -> bool {
    (a - b).abs() <= tolerance
}

// Helper to validate tensor values with tolerance
#[allow(dead_code)]
fn validate_tensor_values(actual: &Tensor<f32>, expected: &[f32], tolerance: f32, name: &str) {
    let actual_data = tensor_to_vec(actual);
    assert_eq!(
        actual_data.len(),
        expected.len(),
        "{} output length mismatch",
        name
    );

    for (i, (&actual_val, &expected_val)) in actual_data.iter().zip(expected).enumerate() {
        assert!(
            approx_eq(actual_val, expected_val, tolerance),
            "{} value mismatch at index {}: expected {}, got {}",
            name,
            i,
            expected_val,
            actual_val
        );
    }
}
#[allow(unused_macros)]
macro_rules! test_op_with_values {
    ($op:expr, $inputs:expr, $expected_shape:expr, $expected_values:expr, $tolerance:expr, $name:literal) => {
        let mut graph = AutoFerroxEngine::<f32>::new();
        graph.set_training(true);

        let input_nodes: Vec<_> = $inputs
            .iter()
            .map(|t| graph.create_variable(t.clone(), true))
            .collect();

        // Forward pass
        let output_node = graph
            .apply_operation(Box::new($op), input_nodes.clone())
            .unwrap_or_else(|e| panic!("{} compute failed: {}", $name, e));

        let output = graph
            .get_tensor(output_node)
            .expect("Output tensor missing");

        // Validate shape
        assert_eq!(output.shape(), $expected_shape, "{} shape mismatch", $name);

        // Validate numerical values
        validate_tensor_values(&output, $expected_values, $tolerance, $name);

        // Test backward pass
        let loss_node = graph
            .apply_operation(Box::new(Mean::new()), vec![output_node])
            .unwrap_or_else(|e| panic!("{} loss creation failed: {}", $name, e));

        graph
            .backward(loss_node)
            .unwrap_or_else(|e| panic!("{} backward pass failed: {}", $name, e));

        // Validate gradient shapes
        for (i, (&input_node, input_tensor)) in input_nodes.iter().zip(&$inputs).enumerate() {
            let grad = graph
                .get_gradient(input_node)
                .unwrap_or_else(|| panic!("{} gradient missing for input {}", $name, i));

            assert_eq!(
                grad.shape(),
                input_tensor.shape(),
                "{} gradient shape mismatch for input {}",
                $name,
                i
            );
        }
    };
}

// Macro to test operation compute and gradient functions
// Validates forward pass output shape and values, plus backward pass gradient computation
#[allow(unused_macros)]
macro_rules! test_op_shape {
    ($op:expr, $inputs:expr, $expected_shape:expr, $name:literal) => {
        let mut graph = AutoFerroxEngine::<f32>::new();
        graph.set_training(true); // Enable gradient computation

        // Create variable nodes with gradient tracking
        let input_nodes: Vec<_> = $inputs
            .iter()
            .map(|t| graph.create_variable(t.clone(), true))
            .collect();

        // Forward pass - test compute function
        let output_node = graph
            .apply_operation(Box::new($op), input_nodes.clone())
            .unwrap_or_else(|e| panic!("{} compute failed: {}", $name, e));

        // Validate output shape
        let output = graph
            .get_tensor(output_node)
            .expect("Output tensor missing");
        assert_eq!(output.shape(), $expected_shape, "{} shape mismatch", $name);

        // Backward pass - test gradient function using scalar loss
        let loss_node = graph
            .apply_operation(Box::new(Mean::new()), vec![output_node])
            .unwrap_or_else(|e| panic!("{} loss creation failed: {}", $name, e));

        graph
            .backward(loss_node)
            .unwrap_or_else(|e| panic!("{} gradient computation failed: {}", $name, e));

        // Validate gradients exist and match input shapes
        for (i, (&input_node, input_tensor)) in input_nodes.iter().zip(&$inputs).enumerate() {
            let grad = graph
                .get_gradient(input_node)
                .unwrap_or_else(|| panic!("{} gradient missing for input {}", $name, i));

            assert_eq!(
                grad.shape(),
                input_tensor.shape(),
                "{} gradient shape mismatch for input {}",
                $name,
                i
            );
        }
    };
}

#[cfg(test)]
mod ops_tests {
    use super::*;
    use std::f32::consts::E;

    #[test]
    fn add_operation() {
        let inputs = vec![tensor_1d(&[1.0, 2.0, 3.0]), tensor_1d(&[4.0, 5.0, 6.0])];
        // Expected: [1+4, 2+5, 3+6] = [5, 7, 9]
        test_op_with_values!(Add::default(), inputs, &[3], &[5.0, 7.0, 9.0], 1e-6, "Add");
    }

    #[test]
    fn sub_operation() {
        let inputs = vec![tensor_1d(&[10.0, 8.0, 6.0]), tensor_1d(&[1.0, 2.0, 3.0])];
        // Expected: [10-1, 8-2, 6-3] = [9, 6, 3]
        test_op_with_values!(Sub, inputs, &[3], &[9.0, 6.0, 3.0], 1e-6, "Sub");
    }

    #[test]
    fn mul_operation() {
        let inputs = vec![tensor_1d(&[2.0, 3.0, 4.0]), tensor_1d(&[5.0, 6.0, 7.0])];
        // Expected: [2*5, 3*6, 4*7] = [10, 18, 28]
        test_op_with_values!(Mul, inputs, &[3], &[10.0, 18.0, 28.0], 1e-6, "Mul");
    }

    #[test]
    fn div_operation() {
        let inputs = vec![tensor_1d(&[12.0, 18.0, 24.0]), tensor_1d(&[3.0, 6.0, 8.0])];
        // Expected: [12/3, 18/6, 24/8] = [4, 3, 3]
        test_op_with_values!(Div, inputs, &[3], &[4.0, 3.0, 3.0], 1e-6, "Div");
    }

    #[test]
    fn add_scalar() {
        let inputs = vec![tensor_1d(&[1.0, 2.0, 3.0])];
        // Expected: [1+5, 2+5, 3+5] = [6, 7, 8]
        test_op_with_values!(
            AddScalar::new(5.0),
            inputs,
            &[3],
            &[6.0, 7.0, 8.0],
            1e-6,
            "AddScalar"
        );
    }

    #[test]
    fn sub_scalar() {
        let inputs = vec![tensor_1d(&[10.0, 20.0, 30.0])];
        // Expected: [10-5, 20-5, 30-5] = [5, 15, 25]
        test_op_with_values!(
            SubScalar::new(5.0),
            inputs,
            &[3],
            &[5.0, 15.0, 25.0],
            1e-6,
            "SubScalar"
        );
    }

    #[test]
    fn mul_scalar() {
        let inputs = vec![tensor_1d(&[2.0, 3.0, 4.0])];
        // Expected: [2*2.5, 3*2.5, 4*2.5] = [5, 7.5, 10]
        test_op_with_values!(
            MulScalar::new(2.5),
            inputs,
            &[3],
            &[5.0, 7.5, 10.0],
            1e-6,
            "MulScalar"
        );
    }

    #[test]
    fn div_scalar() {
        let inputs = vec![tensor_1d(&[9.0, 12.0, 15.0])];
        // Expected: [9/3, 12/3, 15/3] = [3, 4, 5]
        test_op_with_values!(
            DivScalar::new(3.0),
            inputs,
            &[3],
            &[3.0, 4.0, 5.0],
            1e-6,
            "DivScalar"
        );
    }

    #[test]
    fn power_scalar() {
        let inputs = vec![tensor_1d(&[2.0, 3.0, 4.0])];
        // Expected: [2^2, 3^2, 4^2] = [4, 9, 16]
        test_op_with_values!(
            PowerScalar::new(2.0),
            inputs,
            &[3],
            &[4.0, 9.0, 16.0],
            1e-6,
            "PowerScalar"
        );
    }

    #[test]
    fn exp_function() {
        let inputs = vec![tensor_1d(&[0.0, 1.0, -1.0])];
        // Expected: [e^0, e^1, e^(-1)] ≈ [1, 2.718, 0.368]
        test_op_with_values!(Exp, inputs, &[3], &[1.0, E, 0.36788], 1e-4, "Exp");
    }

    #[test]
    fn log_function() {
        let inputs = vec![tensor_1d(&[1.0, E, 7.389056])];
        // Expected: [ln(1), ln(e), ln(e^2)] ≈ [0, 1, 2]
        test_op_with_values!(Log, inputs, &[3], &[0.0, 1.0, 2.0], 1e-4, "Log");
    }

    #[test]
    fn sqrt_function() {
        let inputs = vec![tensor_1d(&[4.0, 9.0, 16.0])];
        // Expected: [√4, √9, √16] = [2, 3, 4]
        test_op_with_values!(Sqrt, inputs, &[3], &[2.0, 3.0, 4.0], 1e-6, "Sqrt");
    }

    #[test]
    fn abs_function() {
        let inputs = vec![tensor_1d(&[-2.0, 3.0, -4.0])];
        // Expected: [|-2|, |3|, |-4|] = [2, 3, 4]
        test_op_with_values!(Abs, inputs, &[3], &[2.0, 3.0, 4.0], 1e-6, "Abs");
    }

    #[test]
    fn neg_function() {
        let inputs = vec![tensor_1d(&[1.0, -2.0, 3.0])];
        // Expected: [-1, -(-2), -3] = [-1, 2, -3]
        test_op_with_values!(Neg, inputs, &[3], &[-1.0, 2.0, -3.0], 1e-6, "Neg");
    }

    #[test]
    fn relu_activation() {
        let inputs = vec![tensor_1d(&[-2.0, 0.0, 3.0])];
        // Expected: [max(-2,0), max(0,0), max(3,0)] = [0, 0, 3]
        test_op_with_values!(ReLU, inputs, &[3], &[0.0, 0.0, 3.0], 1e-6, "ReLU");
    }

    #[test]
    fn sigmoid_activation() {
        let inputs = vec![tensor_1d(&[-1.0, 0.0, 1.0])];
        // Expected: sigmoid values ≈ [0.269, 0.5, 0.731]
        test_op_with_values!(
            Sigmoid,
            inputs,
            &[3],
            &[0.26894, 0.5, 0.73106],
            1e-4,
            "Sigmoid"
        );
    }

    #[test]
    fn tanh_activation() {
        let inputs = vec![tensor_1d(&[-1.0, 0.0, 1.0])];
        // Expected: tanh values ≈ [-0.762, 0, 0.762]
        test_op_with_values!(Tanh, inputs, &[3], &[-0.76159, 0.0, 0.76159], 1e-4, "Tanh");
    }

    #[test]
    fn power_operation() {
        let inputs = vec![tensor_1d(&[2.0, 3.0, 4.0]), tensor_1d(&[2.0, 2.0, 2.0])];
        // Expected: [2^2, 3^2, 4^2] = [4, 9, 16]
        test_op_with_values!(Power, inputs, &[3], &[4.0, 9.0, 16.0], 1e-6, "Power");
    }

    #[test]
    fn matrix_multiply() {
        let inputs = vec![
            tensor_2d(&[1.0, 2.0, 3.0, 4.0], 2, 2),
            tensor_2d(&[5.0, 6.0, 7.0, 8.0], 2, 2),
        ];
        test_op_with_values!(
            MatMul,
            inputs,
            &[2, 2],
            &[19.0, 22.0, 43.0, 50.0],
            1e-6,
            "MatMul"
        );
    }

    #[test]
    fn matrix_transpose() {
        let inputs = vec![tensor_2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)];
        test_op_with_values!(
            Transpose { axes: None },
            inputs,
            &[3, 2],
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
            1e-6,
            "Transpose"
        );
    }

    #[test]
    fn sum_all_elements() {
        let inputs = vec![tensor_2d(&[1.0, 2.0, 3.0, 4.0], 2, 2)];
        test_op_with_values!(Sum::new(false), inputs, &[], &[10.0], 1e-6, "Sum");
    }

    #[test]
    fn sum_along_axes() {
        let inputs = vec![tensor_2d(&[1.0, 2.0, 3.0, 4.0], 2, 2)];
        test_op_with_values!(
            Sum::along_axes(vec![0], false),
            inputs,
            &[2],
            &[4.0, 6.0],
            1e-6,
            "SumAxes"
        );
    }

    #[test]
    fn mean_all_elements() {
        let inputs = vec![tensor_2d(&[2.0, 4.0, 6.0, 8.0], 2, 2)];
        test_op_with_values!(Mean::new(), inputs, &[], &[5.0], 1e-6, "Mean");
    }

    #[test]
    fn mean_along_axes() {
        let inputs = vec![tensor_2d(&[2.0, 4.0, 6.0, 8.0], 2, 2)];
        test_op_with_values!(
            Mean::along_axes(vec![1], false),
            inputs,
            &[2],
            &[3.0, 7.0],
            1e-6,
            "MeanAxes"
        );
    }

    #[test]
    fn max_all_elements() {
        let inputs = vec![tensor_2d(&[1.0, 4.0, 2.0, 3.0], 2, 2)];
        test_op_with_values!(Max::new(), inputs, &[], &[4.0], 1e-6, "Max");
    }

    #[test]
    fn max_along_axes() {
        let inputs = vec![tensor_2d(&[1.0, 4.0, 2.0, 3.0], 2, 2)];
        test_op_with_values!(
            Max::along_axes(vec![0], false),
            inputs,
            &[2],
            &[2.0, 4.0],
            1e-6,
            "MaxAxes"
        );
    }

    #[test]
    fn min_all_elements() {
        let inputs = vec![tensor_2d(&[1.0, 4.0, 2.0, 3.0], 2, 2)];
        test_op_with_values!(Min::new(), inputs, &[], &[1.0], 1e-6, "Min");
    }

    #[test]
    fn min_along_axes() {
        let inputs = vec![tensor_2d(&[1.0, 4.0, 2.0, 3.0], 2, 2)];
        test_op_with_values!(
            Min::along_axes(vec![1], false),
            inputs,
            &[2],
            &[1.0, 2.0],
            1e-6,
            "MinAxes"
        );
    }

    #[test]
    fn max_elementwise() {
        let inputs = vec![tensor_1d(&[1.0, 5.0, 2.0]), tensor_1d(&[3.0, 2.0, 4.0])];
        test_op_with_values!(
            MaxElementwise,
            inputs,
            &[3],
            &[3.0, 5.0, 4.0],
            1e-6,
            "MaxElementwise"
        );
    }

    #[test]
    fn min_elementwise() {
        let inputs = vec![tensor_1d(&[1.0, 5.0, 2.0]), tensor_1d(&[3.0, 2.0, 4.0])];
        test_op_with_values!(
            MinElementwise,
            inputs,
            &[3],
            &[1.0, 2.0, 2.0],
            1e-6,
            "MinElementwise"
        );
    }

    #[test]
    fn reciprocal_operation() {
        let inputs = vec![tensor_1d(&[2.0, 4.0, 0.5])];
        test_op_with_values!(
            Reciprocal,
            inputs,
            &[3],
            &[0.5, 0.25, 2.0],
            1e-6,
            "Reciprocal"
        );
    }

    #[test]
    fn reshape_tensor() {
        let inputs = vec![tensor_2d(&[1.0, 2.0, 3.0, 4.0], 2, 2)];
        test_op_shape!(Reshape::new(vec![4]), inputs, &[4], "Reshape");
    }

    #[test]
    fn reshape_to_matrix() {
        let inputs = vec![tensor_1d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])];
        test_op_shape!(Reshape::new(vec![2, 3]), inputs, &[2, 3], "ReshapeToMatrix");
    }

    #[test]
    fn unsqueeze_dimension() {
        let inputs = vec![tensor_1d(&[1.0, 2.0, 3.0])];
        test_op_shape!(Unsqueeze::new(0), inputs, &[1, 3], "Unsqueeze");
    }

    #[test]
    fn unsqueeze_last_dim() {
        let inputs = vec![tensor_1d(&[1.0, 2.0, 3.0])];
        test_op_shape!(Unsqueeze::new(1), inputs, &[3, 1], "UnsqueezeLast");
    }

    #[test]
    fn squeeze_dimension() {
        let inputs = vec![tensor_2d(&[1.0, 2.0, 3.0], 1, 3)];
        test_op_shape!(Squeeze::new(), inputs, &[3], "Squeeze");
    }

    #[test]
    fn squeeze_specific_axis() {
        let inputs = vec![tensor_2d(&[1.0, 2.0, 3.0], 1, 3)];
        test_op_shape!(Squeeze::at_axis(0), inputs, &[3], "SqueezeAxis");
    }

    #[test]
    fn softmax_last_axis() {
        // Batch softmax on 2D tensor (batch_size=2, features=3)
        let inputs = vec![tensor_2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)];
        test_op_with_values!(
            Softmax::new(Some(1)),
            inputs,
            &[2, 3],
            &[0.0900, 0.2447, 0.6652, 0.0900, 0.2447, 0.6652],
            1e-3,
            "Softmax"
        );
    }

    #[test]
    fn softmax_first_axis() {
        // Softmax along axis 0
        let inputs = vec![tensor_2d(&[1.0, 2.0, 3.0, 4.0], 2, 2)];
        test_op_with_values!(
            Softmax::new(Some(0)),
            inputs,
            &[2, 2],
            &[0.1192, 0.1192, 0.8808, 0.8808],
            1e-3,
            "SoftmaxAxis0"
        );
    }
}
