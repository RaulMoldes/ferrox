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

/// Sigmoid activation: output = 1 / (1 + exp(-input))
#[derive(Debug, Clone)]
pub struct Softmax;

impl<T> Operator<T> for Softmax
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Softmax operation requires exactly 1 input".to_string());
        }

        inputs[0].softmax()
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        _inputs: &mut [&Tensor<T>],
        outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if outputs.shape() != grad_output.shape() {
            return Err("Softmax gradient: shape mismatch".to_string());
        }

        // FIXED: Correct softmax gradient for batched operations
        // For softmax: grad_input = softmax_output * (grad_output - sum(grad_output * softmax_output))

        // Step 1: Element-wise multiplication of grad_output and softmax output
        let element_wise_prod = grad_output
            .mul(outputs)
            .map_err(|e| format!("Softmax gradient element-wise product failed: {}", e))?;

        // Step 2: Sum along the class dimension (last dimension) keeping batch dimension
        // This is critical - we need to sum per sample, not globally
        let shape = outputs.shape();
        let batch_size = if shape.len() > 1 { shape[0] } else { 1 };


        // Reshape for proper reduction if needed
        let mut sum_per_sample = if shape.len() > 1 {
            // For batched data: sum across classes, keep batch dimension
            element_wise_prod.sum(Some(&[1]))?
        } else {
            // For single sample: sum all elements to scalar
            element_wise_prod.sum(None)?
        };

        // Step 3: Broadcast the sum back to original shape for subtraction
        if shape.len() > 1 {
            // Reshape sum to [batch_size, 1] then broadcast to [batch_size, num_classes]
            sum_per_sample.reshape(&[batch_size, 1])?;
            sum_per_sample.broadcast_to(shape)?

        } else {
            // For single sample, broadcast scalar to original shape
            sum_per_sample.broadcast_to(shape)?
        };

        // Step 4: Compute final gradient: softmax * (grad_output - broadcasted_sum)
        let grad_diff = grad_output
            .sub(&sum_per_sample)
            .map_err(|e| format!("Softmax gradient subtraction failed: {}", e))?;

        let grad_input = outputs
            .mul(&grad_diff)
            .map_err(|e| format!("Softmax gradient final multiplication failed: {}", e))?;

        Ok(vec![grad_input])
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




#[cfg(test)]
mod softmax_tests {
    use crate::backend::manager::best_f32_device;
    use crate::backend::Tensor;
    use crate::ops::unary::Softmax;
    use crate::ops::Operator;
    use crate::FerroxN;
    use std::f32::consts::E;

    #[test]
    fn test_softmax_forward() {
        let device = best_f32_device();

        // Test simple 2-class logits that should produce known probabilities
        let logits = Tensor::from_vec_with_device(
            vec![2.0, 1.0], // exp(2)/sum = exp(2)/(exp(2)+exp(1)) ≈ 0.731, exp(1)/sum ≈ 0.269
            &[2],
            device,
        ).unwrap();

        let softmax_op = Softmax;
        let mut inputs = [&logits];
        let result = softmax_op.compute(&mut inputs).unwrap();

        // Verify probabilities sum to 1
        let sum: f32 = result.as_slice().unwrap().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Softmax probabilities should sum to 1, got: {}", sum);

        // Verify expected approximate values
        let probs = result.as_slice().unwrap();
        let expected_first = E.powi(2) / (E.powi(2) + E); // ≈ 0.731
        assert!((probs[0] - expected_first).abs() < 0.01,
            "First probability should be ~0.731, got: {}", probs[0]);

        println!("✓ Softmax forward pass: input [{}, {}] -> output [{}, {}]",
                 2.0, 1.0, probs[0], probs[1]);
    }



    #[test]
    fn test_softmax_gradient() {
        let device = best_f32_device();

        // Simple 2-class case for manual verification
        let logits = Tensor::from_vec_with_device(vec![1.0, 2.0], &[2], device).unwrap();

        let softmax_op = Softmax;
        let mut inputs = [&logits];

        // Compute forward pass
        let softmax_output = softmax_op.compute(&mut inputs).unwrap();
        let prob_data = softmax_output.clone().into_data().unwrap();
        let probs  = prob_data.as_slice().unwrap();
        println!("Softmax output: [{:.4}, {:.4}]", probs[0], probs[1]);

        // Gradient from next layer (simulating CCE loss gradient)
        let grad_output = Tensor::from_vec_with_device(
            vec![-1.0, 1.0], // Typical CCE gradient pattern
            &[2],
            device
        ).unwrap();

        // Compute gradient
        let grad_result = softmax_op.gradient(grad_output, &mut inputs, &softmax_output).unwrap();
        let grad_data = grad_result[0].clone().into_data().unwrap();
        let grad_input = grad_data.as_slice().unwrap();

        // Manual verification: for 2-class softmax gradient
        // grad[0] = p0 * (grad_out[0] - (p0*grad_out[0] + p1*grad_out[1]))
        // grad[1] = p1 * (grad_out[1] - (p0*grad_out[0] + p1*grad_out[1]))
        let dot_product = -probs[0] + probs[1] * 1.0;
        let expected_grad0 = probs[0] * (-1.0 - dot_product);
        let expected_grad1 = probs[1] * (1.0 - dot_product);

        println!("Expected gradient: [{:.4}, {:.4}]", expected_grad0, expected_grad1);
        println!("Computed gradient: [{:.4}, {:.4}]", grad_input[0], grad_input[1]);

        // Verify gradients match expected values
        assert!(FerroxN::abs(grad_input[0] - expected_grad0) < 1e-5,
            "Gradient[0] mismatch: expected {:.6}, got {:.6}", expected_grad0, grad_input[0]);
        assert!(FerroxN::abs(grad_input[1] - expected_grad1) < 1e-5,
            "Gradient[1] mismatch: expected {:.6}, got {:.6}", expected_grad1, grad_input[1]);

        // Critical property: gradient should sum to zero for softmax
        let grad_sum: f32 = grad_input.iter().sum();
        assert!(grad_sum.abs() < 1e-5,
            "Softmax gradient should sum to ~0, got: {:.6}", grad_sum);
    }
}
