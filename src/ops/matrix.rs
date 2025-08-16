// matrix_ops.rs
// Matrix operations for the computational graph
// These wrap tensor API methods to enable automatic differentiation

use crate::backend::{FerroxCudaF, Tensor};
use crate::ops::Operator;

/// Matrix multiplication: output = input1 @ input2
/// Standard linear algebra operation essential for neural networks
#[derive(Debug, Clone)]
pub struct MatMul;

impl<T> Operator<T> for MatMul
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("MatMul operation requires exactly 2 inputs".to_string());
        }

        // Ensure both inputs are 2D matrices
        if inputs[0].ndim() != 2 || inputs[1].ndim() != 2 {
            return Err("MatMul requires 2D tensors (matrices)".to_string());
        }

        inputs[0].matmul(inputs[1])
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 2 {
            return Err("MatMul operation requires exactly 2 inputs".to_string());
        }

        // For matrix multiplication: C = A @ B
        // dA = grad_output @ B^T
        // dB = A^T @ grad_output

        // Transpose B for gradient w.r.t. A: grad_A = grad_output @ B^T
        let mut b_transposed = inputs[1].clone();
        b_transposed.transpose(None)?;
        let grad_a = grad_output.matmul(&b_transposed)?;

        // Transpose A for gradient w.r.t. B: grad_B = A^T @ grad_output
        let mut a_transposed = inputs[0].clone();
        a_transposed.transpose(None)?;
        let grad_b = a_transposed.matmul(&grad_output)?;

        Ok(vec![grad_a, grad_b])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        2
    }
}

/// Transpose operation: output = input^T
/// Swaps the dimensions of a 2D tensor (matrix)
#[derive(Debug, Clone)]
pub struct Transpose {
    /// Optional axes permutation. If None, reverses all axes (standard transpose)
    pub axes: Option<Vec<usize>>,
}

impl<T> Operator<T> for Transpose
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Transpose operation requires exactly 1 input".to_string());
        }

        let mut result = inputs[0].clone();
        result.transpose(self.axes.as_deref())?;
        Ok(result)
    }

    fn gradient(
        &self,
        mut grad_output: Tensor<T>,
        _inputs: &mut [&Tensor<T>],
        _outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        // For transpose: gradient is just transpose of the grad_output
        // If we transposed with custom axes, we need to invert the permutation

        if let Some(ref axes) = self.axes {
            // Invert the permutation to get back to original order
            let mut inverse_axes = vec![0; axes.len()];
            for (new_pos, &old_pos) in axes.iter().enumerate() {
                inverse_axes[old_pos] = new_pos;
            }
            grad_output.transpose(Some(&inverse_axes))?;
        } else {
            // Standard transpose - just transpose again to get back to original
            grad_output.transpose(None)?;
        }

        Ok(vec![grad_output])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

impl Transpose {
    /// Create a standard transpose operation (reverses all axes)
    pub fn new() -> Self {
        Self { axes: None }
    }

    /// Create a transpose with custom axis permutation
    pub fn with_axes(axes: Vec<usize>) -> Self {
        Self { axes: Some(axes) }
    }
}

impl Default for Transpose {
    fn default() -> Self {
        Self::new()
    }
}
