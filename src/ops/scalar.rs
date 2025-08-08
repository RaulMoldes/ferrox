use crate::backend::{FerroxCudaF, Tensor};
use crate::ops::Operator;

/// Scalar addition: output = input + scalar
#[derive(Debug, Clone)]
pub struct AddScalar<T> {
    /// The scalar value to add
    pub scalar: T,
}

impl<T> Operator<T> for AddScalar<T>
where
    T: FerroxCudaF + Clone,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("AddScalar operation requires exactly 1 input".to_string());
        }

        inputs[0].add_scalar(self.scalar)
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        _inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        // For scalar addition: d/dx(x + c) = 1
        // Gradient w.r.t. input is just grad_output unchanged
        Ok(vec![grad_output])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

impl<T> AddScalar<T> {
    /// Create a scalar addition operation
    pub fn new(scalar: T) -> Self {
        Self { scalar }
    }
}

/// Scalar subtraction: output = input - scalar
#[derive(Debug, Clone)]
pub struct SubScalar<T> {
    /// The scalar value to subtract
    pub scalar: T,
}

impl<T> Operator<T> for SubScalar<T>
where
    T: FerroxCudaF + Clone,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("SubScalar operation requires exactly 1 input".to_string());
        }

        inputs[0].sub_scalar(self.scalar)
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        _inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        // For scalar subtraction: d/dx(x - c) = 1
        // Gradient w.r.t. input is just grad_output unchanged
        Ok(vec![grad_output])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

impl<T> SubScalar<T> {
    /// Create a scalar subtraction operation
    pub fn new(scalar: T) -> Self {
        Self { scalar }
    }
}

/// Scalar multiplication: output = input * scalar
#[derive(Debug, Clone)]
pub struct MulScalar<T> {
    /// The scalar value to multiply by
    pub scalar: T,
}

impl<T> Operator<T> for MulScalar<T>
where
    T: FerroxCudaF + Clone,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("MulScalar operation requires exactly 1 input".to_string());
        }

        inputs[0].mul_scalar(self.scalar)
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        _inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        // For scalar multiplication: d/dx(x * c) = c
        // Gradient w.r.t. input is grad_output * scalar
        let grad_input = grad_output.mul_scalar(self.scalar)?;
        Ok(vec![grad_input])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

impl<T> MulScalar<T> {
    /// Create a scalar multiplication operation
    pub fn new(scalar: T) -> Self {
        Self { scalar }
    }
}

/// Scalar division: output = input / scalar
#[derive(Debug, Clone)]
pub struct DivScalar<T> {
    /// The scalar value to divide by
    pub scalar: T,
}

impl<T> Operator<T> for DivScalar<T>
where
    T: FerroxCudaF + Clone,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("DivScalar operation requires exactly 1 input".to_string());
        }

        inputs[0].div_scalar(self.scalar)
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        _inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        // For scalar division: d/dx(x / c) = 1/c
        // Gradient w.r.t. input is grad_output / scalar
        let grad_input = grad_output.div_scalar(self.scalar)?;
        Ok(vec![grad_input])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

impl<T> DivScalar<T> {
    /// Create a scalar division operation
    pub fn new(scalar: T) -> Self {
        Self { scalar }
    }
}

/// Scalar power: output = input ^ scalar
#[derive(Debug, Clone)]
pub struct PowerScalar<T> {
    /// The scalar exponent
    pub exponent: T,
}

impl<T> Operator<T> for PowerScalar<T>
where
    T: FerroxCudaF + Clone,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("PowerScalar operation requires exactly 1 input".to_string());
        }

        inputs[0].power_scalar(self.exponent)
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("PowerScalar operation requires exactly 1 input".to_string());
        }

        let grad = grad_output.div_scalar(self.exponent)?;

        Ok(vec![grad])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

impl<T> PowerScalar<T> {
    /// Create a scalar power operation
    pub fn new(exponent: T) -> Self {
        Self { exponent }
    }
}
