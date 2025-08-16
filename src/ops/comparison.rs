use crate::backend::{FerroxCudaF, Tensor};
use crate::ops::Operator;
use crate::FerroxF;

/// Element-wise greater than: output = input1 > input2
/// Returns 1.0 for true, 0.0 for false following ML conventions
#[derive(Debug, Clone)]
pub struct Greater;

impl<T> Operator<T> for Greater
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("Greater operation requires exactly 2 inputs".to_string());
        }

        inputs[0].greater(inputs[1])
    }

    fn gradient(
        &self,
        _grad_output: Tensor<T>,
        _inputs: &mut [&Tensor<T>],
        _outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        // Comparison operations have zero gradients everywhere
        // They produce step functions which are not differentiable
        let zero = FerroxF::zero();
        let grad_shape = _inputs[0].shape();

        let zero_grad1 = Tensor::full(grad_shape, zero)?;
        let zero_grad2 = Tensor::full(_inputs[1].shape(), zero)?;

        Ok(vec![zero_grad1, zero_grad2])
    }
    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        2
    }
}

/// Element-wise greater than or equal: output = input1 >= input2
#[derive(Debug, Clone)]
pub struct GreaterEqual;

impl<T> Operator<T> for GreaterEqual
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("GreaterEqual operation requires exactly 2 inputs".to_string());
        }

        inputs[0].greater_equal(inputs[1])
    }

    fn gradient(
        &self,
        _grad_output: Tensor<T>,
        _inputs: &mut [&Tensor<T>],
        _outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        // Comparison operations have zero gradients
        let zero = FerroxF::zero();
        let zero_grad1 = Tensor::full(_inputs[0].shape(), zero)?;
        let zero_grad2 = Tensor::full(_inputs[1].shape(), zero)?;

        Ok(vec![zero_grad1, zero_grad2])
    }
    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        2
    }
}

/// Element-wise less than: output = input1 < input2
#[derive(Debug, Clone)]
pub struct Less;

impl<T> Operator<T> for Less
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("Less operation requires exactly 2 inputs".to_string());
        }

        inputs[0].less(inputs[1])
    }

    fn gradient(
        &self,
        _grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        // Comparison operations have zero gradients
        let zero = FerroxF::zero();
        let zero_grad1 = Tensor::full(inputs[0].shape(), zero)?;
        let zero_grad2 = Tensor::full(inputs[1].shape(), zero)?;

        Ok(vec![zero_grad1, zero_grad2])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        2
    }
}

/// Element-wise less than or equal: output = input1 <= input2
#[derive(Debug, Clone)]
pub struct LessEqual;

impl<T> Operator<T> for LessEqual
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("LessEqual operation requires exactly 2 inputs".to_string());
        }

        inputs[0].less_equal(inputs[1])
    }

    fn gradient(
        &self,
        _grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        // Comparison operations have zero gradients
        let zero = FerroxF::zero();
        let zero_grad1 = Tensor::full(inputs[0].shape(), zero)?;
        let zero_grad2 = Tensor::full(inputs[1].shape(), zero)?;

        Ok(vec![zero_grad1, zero_grad2])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        2
    }
}

/// Element-wise equality: output = input1 == input2
/// Note: For floating point, this is exact equality - use with caution
#[derive(Debug, Clone)]
pub struct Equal;

impl<T> Operator<T> for Equal
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("Equal operation requires exactly 2 inputs".to_string());
        }

        inputs[0].equal(inputs[1])
    }

    fn gradient(
        &self,
        _grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        // Comparison operations have zero gradients
        let zero = FerroxF::zero();
        let zero_grad1 = Tensor::full(inputs[0].shape(), zero)?;
        let zero_grad2 = Tensor::full(inputs[1].shape(), zero)?;

        Ok(vec![zero_grad1, zero_grad2])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        2
    }
}

/// Scalar comparison operations for efficiency
/// Greater than scalar: output = input > scalar
#[derive(Debug, Clone)]
pub struct GreaterThanScalar<T> {
    pub scalar: T,
}

impl<T> Operator<T> for GreaterThanScalar<T>
where
    T: FerroxCudaF + Clone,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("GreaterThanScalar operation requires exactly 1 input".to_string());
        }

        inputs[0].greater_scalar(self.scalar)
    }

    fn gradient(
        &self,
        _grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        // Comparison operations have zero gradients
        let zero = FerroxF::zero();
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

impl<T> GreaterThanScalar<T> {
    pub fn new(scalar: T) -> Self {
        Self { scalar }
    }
}

/// Clamp operation: output = clamp(input, min_val, max_val)
/// Constrains values to the range [min_val, max_val]
#[derive(Debug, Clone)]
pub struct Clamp<T> {
    pub min_val: T,
    pub max_val: T,
}

impl<T> Operator<T> for Clamp<T>
where
    T: FerroxCudaF + Clone + PartialOrd,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Clamp operation requires exactly 1 input".to_string());
        }

        inputs[0].clamp(self.min_val, self.max_val)
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Clamp operation requires exactly 1 input".to_string());
        }

        // mask = (input >= min_val) & (input <= max_val)
        let min_mask = inputs[0].greater_equal_scalar(self.min_val)?;
        let max_mask = inputs[0].less_equal_scalar(self.max_val)?;
        let combined_mask = min_mask.mul(&max_mask)?;

        // Apply mask to gradient
        let grad_input = grad_output.mul(&combined_mask)?;

        Ok(vec![grad_input])
    }
    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

impl<T> Clamp<T> {
    pub fn new(min_val: T, max_val: T) -> Self {
        Self { min_val, max_val }
    }
}
