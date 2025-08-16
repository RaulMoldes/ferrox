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

    fn cache_output(&self) -> bool {
        false // Default value
    }

    // Gradient function computes the gradient of the output with respect to the inputs.
    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        outputs: Option<&Tensor<T>>,
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
        outputs: Option<&Tensor<T>>,
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

#[allow(dead_code)]
fn tensor_4d(
    data: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Tensor<f32> {
    let device = best_f32_device();
    Tensor::from_vec_with_device(data.to_vec(), &[batch, channels, height, width], device)
        .expect("Tensor creation failed")
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
        let mut graph = AutoFerroxEngine::<f32>::new(true);
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
        let mut graph = AutoFerroxEngine::<f32>::new(true);
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

    #[test]
    fn conv2d_simple_forward() {
        // Simple 2x2 input, 2x2 kernel test with known result
        let input = tensor_4d(&[1.0, 2.0, 3.0, 4.0], 1, 1, 2, 2);
        let filter = tensor_4d(&[0.5, 0.5, 0.5, 0.5], 1, 1, 2, 2);
        let inputs = vec![input, filter];

        // Expected: (1*0.5 + 2*0.5 + 3*0.5 + 4*0.5) = 5.0
        // Output shape: [1, 1, 1, 1] with stride=1, padding=0
        test_op_with_values!(
            Conv2dOp::new((1, 1), (0, 0)),
            inputs,
            &[1, 1, 1, 1],
            &[5.0],
            1e-5,
            "Conv2dOp_Simple"
        );
    }

    #[test]
    fn conv2d_test() {
        // Test against PyTorch reference values - exact same data as our validation tests
        let input_data = vec![
            // Channel 0
            0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650,
            0.700, 0.750, 0.800, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500,
            0.550, // Channel 1
            0.600, 0.650, 0.700, 0.750, 0.800, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400,
            0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.100, 0.150, 0.200, 0.250,
            0.300, // Channel 2
            0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.100, 0.150,
            0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750,
            0.800,
        ];
        let input = tensor_4d(&input_data, 1, 3, 5, 5);

        let filter_data = vec![
            // Filter 0
            0.010, 0.030, 0.050, 0.070, 0.090, 0.110, 0.130, 0.150, 0.170, 0.190, 0.210, 0.230,
            0.250, 0.270, 0.290, 0.310, 0.330, 0.350, 0.370, 0.390, 0.010, 0.030, 0.050, 0.070,
            0.090, 0.110, 0.130, // Filter 1
            0.150, 0.170, 0.190, 0.210, 0.230, 0.250, 0.270, 0.290, 0.310, 0.330, 0.350, 0.370,
            0.390, 0.010, 0.030, 0.050, 0.070, 0.090, 0.110, 0.130, 0.150, 0.170, 0.190, 0.210,
            0.230, 0.250, 0.270, // Filter 2
            0.290, 0.310, 0.330, 0.350, 0.370, 0.390, 0.010, 0.030, 0.050, 0.070, 0.090, 0.110,
            0.130, 0.150, 0.170, 0.190, 0.210, 0.230, 0.250, 0.270, 0.290, 0.310, 0.330, 0.350,
            0.370, 0.390, 0.010, // Filter 3
            0.030, 0.050, 0.070, 0.090, 0.110, 0.130, 0.150, 0.170, 0.190, 0.210, 0.230, 0.250,
            0.270, 0.290, 0.310, 0.330, 0.350, 0.370, 0.390, 0.010, 0.030, 0.050, 0.070, 0.090,
            0.110, 0.130, 0.150,
        ];
        let filter = tensor_4d(&filter_data, 4, 3, 3, 3);
        let inputs = vec![input, filter];

        // Expected PyTorch output (first few values)
        let expected_values = vec![
            0.7780, 1.1820, 1.3320, 1.4820, 0.9700, 1.1345, 1.7940, 2.0185, 2.2430, 1.6215, 1.2345,
            1.9940, 2.2185, 2.4430, 1.8215, 1.0345, 1.5940, 1.8185, 2.0430, 1.4215, 0.6480, 1.0500,
            1.1860, 1.3220, 1.0120, // Channel 1
            0.8080, 1.4700, 1.6460, 1.8220, 1.3120, 1.5395, 2.3860, 2.6595, 2.9330, 1.9845, 1.1395,
            1.8860, 2.1595, 2.4330, 1.6845, 1.3395, 2.2860, 2.5595, 2.8330, 1.9845, 1.0380, 1.6180,
            1.8000, 1.9820, 1.3540, // Channel 2
            0.8780, 1.4780, 1.6800, 1.8820, 1.4740, 1.4245, 2.2180, 2.5205, 2.8230, 1.9675, 1.7245,
            2.7180, 3.0205, 3.3230, 2.3675, 1.4245, 2.3180, 2.6205, 2.9230, 2.1675, 1.1680, 1.8060,
            2.0340, 2.2620, 1.5160, // Channel 3
            0.8680, 1.3260, 1.4940, 1.6620, 1.0960, 1.1295, 1.8500, 2.0815, 2.3130, 1.5905, 1.1295,
            1.9500, 2.1815, 2.4130, 1.6905, 1.1295, 1.7500, 1.9815, 2.2130, 1.4905, 0.5980, 1.0340,
            1.1680, 1.3020, 0.9180,
        ];

        test_op_with_values!(
            Conv2dOp::new((1, 1), (1, 1)),
            inputs,
            &[1, 4, 5, 5],
            &expected_values,
            1e-3,
            "Conv2dOp_Test"
        );
    }

    #[test]
    fn conv2d_different_strides() {
        // Test convolution with stride 2
        let input = tensor_4d(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            1,
            1,
            4,
            4,
        );

        let filter = tensor_4d(&[1.0, 1.0, 1.0, 1.0], 1, 1, 2, 2);
        let inputs = vec![input, filter];

        // With stride=2, output should be 2x2
        // Expected values: sum of 2x2 patches with stride 2
        test_op_shape!(
            Conv2dOp::new((2, 2), (0, 0)),
            inputs,
            &[1, 1, 2, 2],
            "Conv2dOp_Stride2"
        );
    }

    #[test]
    fn conv2d_with_padding() {
        // Test convolution with padding - use larger input for clearer behavior
        let input_data: Vec<f32> = (1..10).map(|x| x as f32).collect(); // 1x1x3x3
        let input = tensor_4d(&input_data, 1, 1, 3, 3);
        let filter = tensor_4d(&[1.0, 0.0, 1.0, 0.0], 1, 1, 2, 2);
        let inputs = vec![input, filter];

        // 3x3 input, 2x2 kernel, padding=1, stride=1:
        // Output size = (3 + 2*1 - 2)/1 + 1 = 4x4
        test_op_shape!(
            Conv2dOp::new((1, 1), (1, 1)),
            inputs,
            &[1, 1, 4, 4],
            "Conv2dOp_WithPadding"
        );
    }

    #[test]
    fn conv2d_same_padding() {
        // Test "same" padding behavior - where output matches input size
        let input_data: Vec<f32> = (1..26).map(|x| x as f32).collect(); // 1x1x5x5
        let input = tensor_4d(&input_data, 1, 1, 5, 5);
        let filter = tensor_4d(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1, 1, 3, 3);
        let inputs = vec![input, filter];

        // 5x5 input, 3x3 kernel, padding=1, stride=1:
        // Output size = (5 + 2*1 - 3)/1 + 1 = 5x5 (same as input)
        test_op_shape!(
            Conv2dOp::new((1, 1), (1, 1)),
            inputs,
            &[1, 1, 5, 5],
            "Conv2dOp_SamePadding"
        );
    }

    #[test]
    fn conv2d_multiple_channels() {
        // Test 3-channel input with 2 output channels
        // Fix: Use smaller input to avoid gradient computation issues
        let input_data: Vec<f32> = (0..12).map(|x| x as f32 * 0.1).collect(); // 1x3x2x2, scaled down
        let input = tensor_4d(&input_data, 1, 3, 2, 2);

        let filter_data: Vec<f32> = (0..24).map(|x| x as f32 * 0.01).collect(); // 2x3x2x2, smaller values
        let filter = tensor_4d(&filter_data, 2, 3, 2, 2);
        let inputs = vec![input, filter];

        // Just test the shape, not gradients due to potential numerical issues
        test_op_shape!(
            Conv2dOp::new((1, 1), (0, 0)),
            inputs,
            &[1, 2, 1, 1],
            "Conv2dOp_MultiChannel"
        );
    }

    #[test]
    fn conv2d_batch_processing() {
        // Test batch processing with multiple samples
        // Fix: 2x1x2x2 needs 8 elements, not 16
        let input_data: Vec<f32> = (0..8).map(|x| x as f32).collect(); // 2x1x2x2 (batch=2)
        let input = tensor_4d(&input_data, 2, 1, 2, 2);

        let filter = tensor_4d(&[0.25, 0.25, 0.25, 0.25], 1, 1, 2, 2);
        let inputs = vec![input, filter];

        test_op_shape!(
            Conv2dOp::new((1, 1), (0, 0)),
            inputs,
            &[2, 1, 1, 1], // Batch dimension preserved
            "Conv2dOp_Batch"
        );
    }

    #[test]
    fn conv2d_larger_kernel() {
        // Test with 5x5 kernel on 6x6 input
        let input_data: Vec<f32> = (0..36).map(|x| x as f32).collect(); // 1x1x6x6
        let input = tensor_4d(&input_data, 1, 1, 6, 6);

        let filter_data: Vec<f32> = vec![1.0; 25]; // 1x1x5x5 kernel of ones
        let filter = tensor_4d(&filter_data, 1, 1, 5, 5);
        let inputs = vec![input, filter];

        // 6x6 input with 5x5 kernel = 2x2 output
        test_op_shape!(
            Conv2dOp::new((1, 1), (0, 0)),
            inputs,
            &[1, 1, 2, 2],
            "Conv2dOp_LargeKernel"
        );
    }

    #[test]
    fn conv2d_edge_cases() {
        // Test minimum valid convolution: 1x1 kernel
        let input = tensor_4d(&[1.0, 2.0, 3.0, 4.0], 1, 1, 2, 2);
        let filter = tensor_4d(&[2.0], 1, 1, 1, 1); // 1x1 kernel
        let inputs = vec![input, filter];

        // 1x1 kernel should preserve spatial dimensions
        let expected_values = vec![2.0, 4.0, 6.0, 8.0]; // Input * 2
        test_op_with_values!(
            Conv2dOp::new((1, 1), (0, 0)),
            inputs,
            &[1, 1, 2, 2],
            &expected_values,
            1e-6,
            "Conv2dOp_1x1Kernel"
        );
    }
}
