// unary_ops.rs
// Unary operations for the computational graph
// These wrap tensor API methods to enable automatic differentiation

use crate::backend::{FerroxCudaF, Tensor};
use crate::ops::Operator;
use crate::FerroxF;

/// Element-wise exponential: output = exp(input)
#[derive(Debug, Clone)]
pub struct Exp;

impl<T> Operator<T> for Exp
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Exp operation requires exactly 1 input".to_string());
        }

        inputs[0].exp()
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Exp operation requires exactly 1 input".to_string());
        }

        // For exp: d/dx(exp(x)) = exp(x)
        // The output of forward pass is exp(input), so we can recompute or cache it
        let result = grad_output.mul(outputs)?;

        Ok(vec![result])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

/// Element-wise natural logarithm: output = log(input)
#[derive(Debug, Clone)]
pub struct Log;

impl<T> Operator<T> for Log
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Log operation requires exactly 1 input".to_string());
        }

        inputs[0].log()
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Log operation requires exactly 1 input".to_string());
        }
        // For log: d/dx(log(x)) = 1/x
        let result = grad_output.div(inputs[0])?;

        Ok(vec![result])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

/// Element-wise square root: output = sqrt(input)
#[derive(Debug, Clone)]
pub struct Sqrt;

impl<T> Operator<T> for Sqrt
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Sqrt operation requires exactly 1 input".to_string());
        }

        inputs[0].sqrt()
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Sqrt operation requires exactly 1 input".to_string());
        }

        // For sqrt: d/dx(sqrt(x)) = 1/(2*sqrt(x))

        let two = FerroxF::from_f64(2.0).ok_or("Failed to convert 2.0 to tensor type")?;
        let denominator = outputs.mul_scalar(two)?;
        let result = grad_output.div(&denominator)?;

        Ok(vec![result])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

/// Element-wise absolute value: output = abs(input)
#[derive(Debug, Clone)]
pub struct Abs;

impl<T> Operator<T> for Abs
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Abs operation requires exactly 1 input".to_string());
        }

        inputs[0].abs()
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Abs operation requires exactly 1 input".to_string());
        }

        // For abs: d/dx(abs(x)) = sign(x)
        // Note: gradient at x=0 is undefined, we use 0 as convention
        let sign = inputs[0].sign()?;
        let grad_input = grad_output.mul(&sign)?;

        Ok(vec![grad_input])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

/// Element-wise negation: output = -input
#[derive(Debug, Clone)]
pub struct Neg;

impl<T> Operator<T> for Neg
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Neg operation requires exactly 1 input".to_string());
        }

        inputs[0].neg()
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Neg operation requires exactly 1 input".to_string());
        }

        // For negation: d/dx(-x) = -1
        let result = grad_output.neg()?;

        Ok(vec![result])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

/// Sigmoid activation: output = 1 / (1 + exp(-input))
#[derive(Debug, Clone)]
pub struct Sigmoid;

impl<T> Operator<T> for Sigmoid
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Sigmoid operation requires exactly 1 input".to_string());
        }

        inputs[0].sigmoid()
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Sigmoid operation requires exactly 1 input".to_string());
        }

        // For sigmoid: d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))

        let one = FerroxF::one();
        let one_minus_sigmoid = outputs.sub_scalar(one)?;
        let local_grad = outputs.mul(&one_minus_sigmoid)?;
        let result = grad_output.mul(&local_grad)?;

        Ok(vec![result])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

/// Hyperbolic tangent activation: output = tanh(input)
#[derive(Debug, Clone)]
pub struct Tanh;

impl<T> Operator<T> for Tanh
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Tanh operation requires exactly 1 input".to_string());
        }

        inputs[0].tanh()
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Tanh operation requires exactly 1 input".to_string());
        }

        // For tanh: d/dx(tanh(x)) = 1 - tanh²(x)

        let tanh_squared = outputs.mul(outputs)?;
        let one = FerroxF::one();
        let local_grad = tanh_squared.sub_scalar(one)?.neg()?; // 1 - tanh²(x) = -(tanh²(x) - 1)
        let result = grad_output.mul(&local_grad)?;

        Ok(vec![result])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

/// ReLU activation: output = max(0, input)
#[derive(Debug, Clone)]
pub struct ReLU;

impl<T> Operator<T> for ReLU
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("ReLU operation requires exactly 1 input".to_string());
        }

        inputs[0].relu()
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("ReLU operation requires exactly 1 input".to_string());
        }

        // For ReLU: d/dx(max(0, x)) = 1 if x > 0, else 0
        // Create a mask where input > 0
        let zero = FerroxF::zero();
        let mask = inputs[0].greater_scalar(zero)?;
        let result = grad_output.mul(&mask)?;

        Ok(vec![result])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

/// Element-wise power: output = input1 ^ input2
#[derive(Debug, Clone)]
pub struct Power;

impl<T> Operator<T> for Power
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("Power operation requires exactly 2 inputs".to_string());
        }

        inputs[0].powf(inputs[1])
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 2 {
            return Err("Power operation requires exactly 2 inputs".to_string());
        }

        // For power: d/dx(x^y) = y * x^(y-1), d/dy(x^y) = x^y * ln(x)
        // Gradient w.r.t. base: y * x^(y-1)
        // Optimized: x^(y-1) = x^y / x = outputs / x
        let grad_base = grad_output
            .mul(inputs[1])? // * y
            .mul(&outputs.div(inputs[0])?)?; // * (outputs / x) = * x^(y-1)

        // Gradient w.r.t. exponent: x^y * ln(x)
        // Optimized: x^y is already computed in outputs!
        let log_base = inputs[0].log()?;
        let grad_exponent = grad_output
            .mul(outputs)? // * x^y (reuse outputs)
            .mul(&log_base)?; // * ln(x)

        Ok(vec![grad_base, grad_exponent])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        2
    }
}
