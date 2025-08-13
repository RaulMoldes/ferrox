// basic_ops.rs
// Basic arithmetic operations for the computational graph
// These operators wrap the existing tensor API methods to enable automatic differentiation

use crate::backend::{FerroxCudaF, Tensor};
use crate::ops::Operator;

/// Helper to reduce gradients when broadcasting was used in forward pass
/// This ensures gradient shapes match the original input shapes
pub fn reduce_gradient_for_broadcasting<T>(
    grad: &mut Tensor<T>,
    target_shape: &[usize],
) -> Result<(), String>
where
    T: FerroxCudaF,
{
    let grad_shape = grad.shape();

    // If shapes match, no reduction needed
    if grad_shape == target_shape {
        return Ok(());
    }

    // Find axes that were broadcasted (size 1 in target, size > 1 in grad)
    let mut axes_to_reduce = Vec::new();
    let mut target_idx = target_shape.len();

    // Handle case where grad has more dimensions than target
    for i in (0..grad_shape.len()).rev() {
        if target_idx == 0 {
            // Extra leading dimensions in grad - sum them out
            axes_to_reduce.push(i);
        } else {
            target_idx -= 1;
            if target_shape[target_idx] == 1 && grad_shape[i] > 1 {
                // This dimension was broadcasted - sum it out
                axes_to_reduce.push(i);
            }
        }
    }

    // Reduce along identified axes
    if !axes_to_reduce.is_empty() {
        *grad = grad.sum(Some(&axes_to_reduce), false)?;
    }

    // Reshape to match target if needed (handles trailing/leading 1s)
    if grad.shape() != target_shape {
        grad.reshape(target_shape)?;
    }

    Ok(())
}

/// Element-wise addition: output = input1 + input2
/// Supports broadcasting as per PyTorch semantics
#[derive(Debug, Clone)]
pub struct Add<T: FerroxCudaF> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for Add<T>
where
    T: FerroxCudaF,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Add<T>
where
    T: FerroxCudaF,
{
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> Operator<T> for Add<T>
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("Add operation requires exactly 2 inputs".to_string());
        }

        // Use the tensor API to perform addition with broadcasting support
        inputs[0].add(inputs[1])
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 2 {
            return Err("Add operation requires exactly 2 inputs".to_string());
        }

        // For addition: d/dx(x + y) = 1, d/dy(x + y) = 1
        // Gradients are the same as grad_output, but may need broadcasting reduction
        let mut grad_input1 = grad_output.clone(); // Clone for first input
        let mut grad_input2 = grad_output; // Move for second input

        // Reduce gradients to match input shapes if broadcasting occurred
        reduce_gradient_for_broadcasting(&mut grad_input1, inputs[0].shape())?;
        reduce_gradient_for_broadcasting(&mut grad_input2, inputs[1].shape())?;

        Ok(vec![grad_input1, grad_input2])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        2
    }
}

/// Element-wise subtraction: output = input1 - input2
#[derive(Debug, Clone)]
pub struct Sub;

impl<T> Operator<T> for Sub
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("Sub operation requires exactly 2 inputs".to_string());
        }
        inputs[0].sub(inputs[1])
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 2 {
            return Err("Sub operation requires exactly 2 inputs".to_string());
        }

        // For subtraction: d/dx(x - y) = 1, d/dy(x - y) = -1
        let mut grad_input2 = grad_output.neg()?;
        let mut grad_input1 = grad_output;

        // Handle broadcasting reductions
        reduce_gradient_for_broadcasting(&mut grad_input1, inputs[0].shape())?;
        reduce_gradient_for_broadcasting(&mut grad_input2, inputs[1].shape())?;

        Ok(vec![grad_input1, grad_input2])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        2
    }
}

/// Element-wise multiplication: output = input1 * input2
#[derive(Debug, Clone)]
pub struct Mul;

impl<T> Operator<T> for Mul
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("Mul operation requires exactly 2 inputs".to_string());
        }

        inputs[0].mul(inputs[1])
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 2 {
            return Err("Mul operation requires exactly 2 inputs".to_string());
        }

        // For multiplication: d/dx(x * y) = y, d/dy(x * y) = x
        let mut grad_input1 = grad_output.mul(inputs[1])?;
        let mut grad_input2 = grad_output.mul(inputs[0])?;

        // Handle broadcasting reductions
        reduce_gradient_for_broadcasting(&mut grad_input1, inputs[0].shape())?;
        reduce_gradient_for_broadcasting(&mut grad_input2, inputs[1].shape())?;

        Ok(vec![grad_input1, grad_input2])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        2
    }
}

/// Element-wise division: output = input1 / input2
#[derive(Debug, Clone)]
pub struct Div;

impl<T> Operator<T> for Div
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("Div operation requires exactly 2 inputs".to_string());
        }

        inputs[0].div(inputs[1])
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 2 {
            return Err("Div operation requires exactly 2 inputs".to_string());
        }

        // For division: d/dx(x / y) = 1/y, d/dy(x / y) = -x/y²
        let mut grad_input1 = grad_output.div(inputs[1])?;
        reduce_gradient_for_broadcasting(&mut grad_input1, inputs[0].shape())?;

        let neg_output = outputs.div(inputs[1])?.neg()?;
        let mut grad_input2 = grad_output.mul(&neg_output)?;

        // Handle broadcasting reductions
        reduce_gradient_for_broadcasting(&mut grad_input2, inputs[1].shape())?;

        Ok(vec![grad_input1, grad_input2])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        2
    }
}
