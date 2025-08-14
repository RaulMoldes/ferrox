use crate::backend::{FerroxCudaF, Tensor};
use crate::ops::Operator;

#[derive(Debug, Clone)]
pub struct Softmax {
    axis: Option<usize>, // None means apply to entire tensor (current behavior)
}

impl<T> Operator<T> for Softmax
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Softmax operation requires exactly 1 input".to_string());
        }

        let input_tensor = inputs[0];

        match self.axis {
            None => {
                // Fallback to existing whole-tensor softmax implementation
                input_tensor.softmax()
            }
            Some(axis) => {
                // Use optimized batch-aware kernel instead of partitioning
                if axis >= input_tensor.ndim() {
                    return Err(format!(
                        "Softmax axis {} out of bounds for tensor with {} dimensions",
                        axis,
                        input_tensor.ndim()
                    ));
                }

                // Call the batch-aware softmax method
                input_tensor.softmax_batched(axis)
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
            return Err("Softmax gradient: shape mismatch".to_string());
        }

        match self.axis {
            None => {
                // Use existing gradient computation for entire tensor
                self.compute_global_gradient(grad_output, outputs)
            }
            Some(axis) => {
                // Use optimized batch-aware gradient computation
                self.compute_batch_gradient(grad_output, outputs, axis)
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

impl Softmax {
    pub fn new(axis: Option<usize>) -> Self {
        Self { axis }
    }

    // Helper method for global gradient computation
    fn compute_global_gradient<T>(
        &self,
        grad_output: Tensor<T>,
        outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String>
    where
        T: FerroxCudaF,
    {
        let element_wise_prod = grad_output.mul(outputs)?;
        let mut sum_prod = element_wise_prod.sum(None, false)?;
        sum_prod.broadcast_to(outputs.shape())?;
        let grad_diff = grad_output.sub(&sum_prod)?;
        let grad_input = outputs.mul(&grad_diff)?;
        Ok(vec![grad_input])
    }

    /// Optimized gradient computation for batch softmax
    /// Uses the mathematical property: ∇softmax = softmax * (∇L - Σ(softmax * ∇L))
    /// where the sum is computed along the softmax axis
    fn compute_batch_gradient<T>(
        &self,
        grad_output: Tensor<T>,
        outputs: &Tensor<T>,
        axis: usize,
    ) -> Result<Vec<Tensor<T>>, String>
    where
        T: FerroxCudaF,
    {
        // Element-wise product: softmax * grad_output
        let element_wise_prod = grad_output.mul(outputs)?;

        // Sum along the softmax axis, keeping dimensions
        let mut sum_prod = element_wise_prod.sum(Some(&[axis]), true)?;

        // Broadcast sum back to original shape for subtraction
        sum_prod.broadcast_to(outputs.shape())?;

        // Compute: grad_output - broadcasted_sum
        let grad_diff = grad_output.sub(&sum_prod)?;

        // Final gradient: softmax * (grad_output - sum)
        let grad_input = outputs.mul(&grad_diff)?;

        Ok(vec![grad_input])
    }
}
