// elementwise_ops.rs
// Additional element-wise operations for the computational graph
// These operations provide element-wise min/max and other utility functions

use crate::backend::{FerroxCudaF, FerroxF, Tensor};
use crate::ops::Operator;

/// Element-wise minimum: output = min(input1, input2)
/// Compares corresponding elements and takes the smaller value
#[derive(Debug, Clone)]
pub struct MinElementwise;

impl<T> Operator<T> for MinElementwise
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("MinElementwise operation requires exactly 2 inputs".to_string());
        }

        inputs[0].min(inputs[1])
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 2 {
            return Err("MinElementwise operation requires exactly 2 inputs".to_string());
        }

        // For element-wise min: gradient flows to the element that was selected (the minimum)
        // Create masks indicating which input had the minimum value

        // mask1 = (input1 <= input2), mask2 = (input2 < input1)
        let mask1 = inputs[0].less_equal(inputs[1])?;
        let mask2 = inputs[1].less(inputs[0])?;

        // Apply masks to gradient
        let grad_input1 = grad_output.mul(&mask1)?;
        let grad_input2 = grad_output.mul(&mask2)?;

        Ok(vec![grad_input1, grad_input2])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        2
    }
}

/// Element-wise maximum: output = max(input1, input2)
/// Compares corresponding elements and takes the larger value
#[derive(Debug, Clone)]
pub struct MaxElementwise;

impl<T> Operator<T> for MaxElementwise
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("MaxElementwise operation requires exactly 2 inputs".to_string());
        }

        inputs[0].max(inputs[1])
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 2 {
            return Err("MaxElementwise operation requires exactly 2 inputs".to_string());
        }

        // mask1 = (input1 >= input2), mask2 = (input2 > input1)
        let mask1 = inputs[0].greater_equal(inputs[1])?;
        let mask2 = inputs[1].greater(inputs[0])?;

        // Apply masks to gradient
        let grad_input1 = grad_output.mul(&mask1)?;
        let grad_input2 = grad_output.mul(&mask2)?;

        Ok(vec![grad_input1, grad_input2])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        2
    }
}

/// Reciprocal operation: output = 1 / input
/// Computes element-wise reciprocal (multiplicative inverse)
#[derive(Debug, Clone)]
pub struct Reciprocal;

impl<T> Operator<T> for Reciprocal
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Reciprocal operation requires exactly 1 input".to_string());
        }

        inputs[0].reciprocal()
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Reciprocal operation requires exactly 1 input".to_string());
        }

        // For reciprocal: d/dx(1/x) = -1/x²

        let reciprocal = inputs[0].reciprocal()?;
        let reciprocal_squared = reciprocal.mul(&reciprocal)?;
        let negative_reciprocal_squared = reciprocal_squared.neg()?;
        let grad_input = grad_output.mul(&negative_reciprocal_squared)?;

        Ok(vec![grad_input])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

/// Sign operation: output = sign(input)
/// Returns -1 for negative, 0 for zero, +1 for positive
#[derive(Debug, Clone)]
pub struct Sign;

impl<T> Operator<T> for Sign
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Sign operation requires exactly 1 input".to_string());
        }

        inputs[0].sign()
    }

    fn gradient(
        &self,
        _grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Sign operation requires exactly 1 input".to_string());
        }

        // For sign function: gradient is 0 everywhere (undefined at 0, but we use 0)
        let zero = <T as FerroxF>::zero();
        let zero_grad = Tensor::full(inputs[0].shape(), zero)?;

        Ok(vec![zero_grad])
    }
    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}
