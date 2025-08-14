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


/// 2D Convolution operation for batched inputs
/// Input shape: [batch_size, in_channels, height, width]
/// Filter shape: [out_channels, in_channels, kernel_height, kernel_width]
/// Output shape: [batch_size, out_channels, out_height, out_width]
#[derive(Debug, Clone)]
pub struct Conv2d {
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl Conv2d {
    pub fn new(stride: (usize, usize), padding: (usize, usize)) -> Self {
        Self { stride, padding }
    }
}

impl<T> Operator<T> for Conv2d
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("Conv2d operation requires exactly 2 inputs (input, filter)".to_string());
        }

        let input = inputs[0];
        let filter = inputs[1];

        // Validate input dimensions for batched convolution
        if input.ndim() != 4 {
            return Err("Conv2d input must be 4D [batch_size, in_channels, height, width]".to_string());
        }

        if filter.ndim() != 4 {
            return Err("Conv2d filter must be 4D [out_channels, in_channels, kernel_height, kernel_width]".to_string());
        }

        // Validate channel compatibility
        let input_channels = input.shape()[1];
        let filter_channels = filter.shape()[1];
        if input_channels != filter_channels {
            return Err(format!(
                "Input channels {} don't match filter channels {}",
                input_channels, filter_channels
            ));
        }

        // Perform batched convolution using existing backend
        input.conv2d(filter, self.stride, self.padding)
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 2 {
            return Err("Conv2d operation requires exactly 2 inputs".to_string());
        }

        let input = inputs[0];   // [batch, in_channels, in_h, in_w]
        let filter = inputs[1];  // [out_channels, in_channels, kernel_h, kernel_w]

        // grad_output: [batch, out_channels, out_h, out_w]

        // 1. Gradient w.r.t. input (data gradient) using deconvolution
        let grad_input = grad_output.deconv2d(input, filter, self.stride, self.padding)?;

        // 2. Gradient w.r.t. filter (weight gradient) using cross-correlation
        let grad_filter = input.cross_correlation(&grad_output, filter.shape(), self.stride, self.padding)?;

        Ok(vec![grad_input, grad_filter])
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }
}
