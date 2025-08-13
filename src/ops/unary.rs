// unary_ops.rs
// Unary operations for the computational graph
// These wrap tensor API methods to enable automatic differentiation

use crate::backend::{FerroxCudaF, Tensor};
use crate::ops::Operator;
use crate::{FerroxF, FerroxN};

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

        let two = <T as FerroxN>::from_f64(2.0).ok_or("Failed to convert 2.0 to tensor type")?;
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

/// Softmax activation: output = 1 / (1 + exp(-input))
#[derive(Debug, Clone)]
pub struct Softmax {
    axis: Option<usize>, // None means apply to entire tensor (current behavior)
}

impl Softmax {
    pub fn new(axis: Option<usize>) -> Self {
        Self { axis }
    }

    // Default constructor for backward compatibility
    pub fn default_axis() -> Self {
        Self { axis: None }
    }
}

impl<T> Operator<T> for Softmax
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("SoftmaxAxis operation requires exactly 1 input".to_string());
        }

        let input_tensor = inputs[0];

        match self.axis {
            None => {
                // Fallback to existing softmax implementation (entire tensor)
                input_tensor.softmax()
            }
            Some(axis) => {
                // Check if axis is valid
                if axis >= input_tensor.ndim() {
                    return Err(format!(
                        "Softmax axis {} out of bounds for tensor with {} dimensions",
                        axis,
                        input_tensor.ndim()
                    ));
                }

                // Get axis size to determine number of partitions
                let axis_size = input_tensor.shape()[axis];

                // Partition tensor along the specified axis
                let partitions = input_tensor.partition(axis, axis_size)?;

                // Apply softmax to each partition (slice along axis)
                let mut softmax_partitions = Vec::with_capacity(partitions.len());
                for partition in partitions {
                    let softmax_result = partition.softmax()?;
                    softmax_partitions.push(softmax_result);
                }

                // Concatenate results back along the same axis
                Tensor::from_partitions(softmax_partitions, axis)
            }
        }
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        _inputs: &mut [&Tensor<T>],
        outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if outputs.shape() != grad_output.shape() {
            return Err("SoftmaxAxis gradient: shape mismatch".to_string());
        }

        match self.axis {
            None => {
                // Use existing gradient computation for entire tensor
                self.compute_global_gradient(grad_output, outputs)
            }
            Some(axis) => {
                // Partition both grad_output and outputs along axis
                let grad_partitions = grad_output.partition(axis, grad_output.shape()[axis])?;
                let output_partitions = outputs.partition(axis, outputs.shape()[axis])?;

                // Compute gradient for each partition
                let mut grad_input_partitions = Vec::with_capacity(grad_partitions.len());
                for (grad_part, out_part) in grad_partitions.iter().zip(output_partitions.iter()) {
                    let grad_input_part = self.compute_partition_gradient::<T>(grad_part, out_part)?;
                    grad_input_partitions.push(grad_input_part);
                }

                // Concatenate gradient partitions back
                let grad_input = Tensor::from_partitions(grad_input_partitions, axis)?;
                Ok(vec![grad_input])
            }
        }
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

impl Softmax
{
    // Helper method for global gradient computation (existing logic)
    fn compute_global_gradient<T>(
        &self,
        grad_output: Tensor<T>,
        outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String>
    where T: FerroxCudaF
    {
        // Your existing softmax gradient logic
        let element_wise_prod = grad_output.mul(outputs)?;
        let mut sum_prod = element_wise_prod.sum(None)?;
        sum_prod.broadcast_to(outputs.shape())?;
        let grad_diff = grad_output.sub(&sum_prod)?;
        let grad_input = outputs.mul(&grad_diff)?;
        Ok(vec![grad_input])
    }

    // Helper method for partition-specific gradient computation
    fn compute_partition_gradient<T>(
        &self,
        grad_partition: &Tensor<T>,
        output_partition: &Tensor<T>,
    ) -> Result<Tensor<T>, String>
    where T: FerroxCudaF,
    {
        // Apply softmax gradient formula to individual partition
        let element_wise_prod = grad_partition.mul(output_partition)?;
        let mut sum_prod = element_wise_prod.sum(None)?;
        sum_prod.broadcast_to(output_partition.shape())?;
        let grad_diff = grad_partition.sub(&sum_prod)?;
        output_partition.mul(&grad_diff)
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

        let one = <T as FerroxF>::one();
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
        let one = <T as FerroxF>::one();
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
        let zero = <T as FerroxF>::zero();
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
