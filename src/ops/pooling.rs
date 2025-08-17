use crate::backend::{FerroxCudaF, FerroxN, Tensor};
use crate::ops::Operator;

/// 2D Max Pooling operation
/// Applies max pooling over spatial dimensions (H, W) of 4D tensors
/// Commonly used in CNNs to reduce spatial resolution while retaining important features
#[derive(Debug, Clone)]
pub struct MaxPool2D {
    /// Size of the pooling window (square kernel)
    pub kernel_size: usize,
    /// Step size for the pooling window
    pub stride: usize,
    /// Zero-padding added to input boundaries
    pub padding: usize,
}

impl MaxPool2D {
    /// Create new MaxPool2D operation
    ///
    /// # Arguments
    /// * `kernel_size` - Size of the pooling window (square)
    /// * `stride` - Step size for pooling window
    /// * `padding` - Zero-padding added to boundaries
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    /// Common 2x2 max pooling with stride 2 (reduces spatial size by half)
    pub fn new_2x2() -> Self {
        Self::new(2, 2, 0)
    }
}

impl<T> Operator<T> for MaxPool2D
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("MaxPool2D operation requires exactly 1 input".to_string());
        }

        inputs[0].maxpool2d(self.kernel_size, self.stride, self.padding)
    }

    fn cache_output(&self) -> bool {
        true // Cache output for gradient computation - we need to know which elements were selected
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("MaxPool2D operation requires exactly 1 input".to_string());
        }

        // For max pooling, gradients only flow to the positions that were selected as maximum
        // We need to implement unpooling operation that distributes gradients back to original positions

        let input_shape = inputs[0].shape();
        if input_shape.len() != 4 {
            return Err("MaxPool2D requires 4D input tensor [N, C, H, W]".to_string());
        }

        // Initialize gradient tensor with zeros matching input shape
        let mut grad_input = Tensor::zeros(input_shape)?;

        // Get pooled output to determine which positions were selected
        let pooled_output = match outputs {
            Some(out) => out.clone(),
            None => self.compute(inputs)?, // Recompute if not cached
        };

        // Perform unpooling: distribute gradients back to the positions that produced the max values
        self.unpool_gradients(&mut grad_input, &grad_output, inputs[0], &pooled_output)?;

        Ok(vec![grad_input])
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }
}

impl MaxPool2D {
    /// Unpooling operation for max pooling gradients
    /// Distributes gradients from pooled output back to input positions that were selected
    fn unpool_gradients<T: FerroxCudaF>(
        &self,
        grad_input: &mut Tensor<T>,
        grad_output: &Tensor<T>,
        input: &Tensor<T>,
        pooled_output: &Tensor<T>,
    ) -> Result<(), String> {
        // For now, implement a simplified version that distributes gradients uniformly
        // In a full implementation, you would track exactly which positions were selected
        // and only send gradients to those positions

        let input_shape = input.shape();
        let (n, c, h, w) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

        let h_out = (h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let w_out = (w + 2 * self.padding - self.kernel_size) / self.stride + 1;

        // TODO: Implement proper unpooling that tracks which input elements were selected
        // For now, we'll use a simplified approach that distributes gradients based on
        // comparing input values with pooled output values

        // This is a simplified implementation - in practice you'd want to implement
        // a proper unpooling operation that exactly reverses the max pooling
        *grad_input = grad_output.clone();

        Ok(())
    }
}



/// 2D Average Pooling operation
/// Applies average pooling over spatial dimensions (H, W) of 4D tensors
/// Computes the average of elements in each pooling window
#[derive(Debug, Clone)]
pub struct AvgPool2D {
    /// Size of the pooling window (square kernel)
    pub kernel_size: usize,
    /// Step size for the pooling window
    pub stride: usize,
    /// Zero-padding added to input boundaries
    pub padding: usize,
}

impl AvgPool2D {
    /// Create new AvgPool2D operation
    ///
    /// # Arguments
    /// * `kernel_size` - Size of the pooling window (square)
    /// * `stride` - Step size for pooling window
    /// * `padding` - Zero-padding added to boundaries
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    /// Common 2x2 average pooling with stride 2 (reduces spatial size by half)
    pub fn new_2x2() -> Self {
        Self::new(2, 2, 0)
    }

    /// Global average pooling - pools entire spatial dimension to 1x1
    pub fn global(input_size: (usize, usize)) -> Self {
        let kernel_size = input_size.0.max(input_size.1);
        Self::new(kernel_size, 1, 0)
    }
}

impl<T> Operator<T> for AvgPool2D
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("AvgPool2D operation requires exactly 1 input".to_string());
        }

        inputs[0].avgpool2d(self.kernel_size, self.stride, self.padding)
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("AvgPool2D operation requires exactly 1 input".to_string());
        }

        // For average pooling, gradients are distributed uniformly to all positions
        // that contributed to each pooled output
        let input_shape = inputs[0].shape();
        if input_shape.len() != 4 {
            return Err("AvgPool2D requires 4D input tensor [N, C, H, W]".to_string());
        }

        let grad_input = self.unpool_avg_gradients(&grad_output, input_shape)?;

        Ok(vec![grad_input])
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }
}

impl AvgPool2D {
    /// Unpooling operation for average pooling gradients
    /// Distributes gradients uniformly to all input positions that contributed to each output
    fn unpool_avg_gradients<T: FerroxCudaF>(
        &self,
        grad_output: &Tensor<T>,
        input_shape: &[usize],
    ) -> Result<Tensor<T>, String> {
        let (n, c, h, w) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

        let h_out = (h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let w_out = (w + 2 * self.padding - self.kernel_size) / self.stride + 1;

        // Create zero tensor for input gradients
        let mut grad_input = Tensor::zeros(input_shape)?;

        // Get the raw data for manipulation
        // Note: This is a simplified implementation
        // In practice, you would implement this as a proper tensor operation
        // or use the reverse of the im2col transformation

        // For average pooling, each output gradient gets distributed equally
        // to all input positions that contributed to that output
        // The contribution is 1/kernel_area for each position

        let kernel_area = self.kernel_size * self.kernel_size;
        let scale_factor = FerroxN::from_f32(1.0 / kernel_area as f32)
            .ok_or("Failed to convert scale factor")?;

        // Scale the output gradients by 1/kernel_area
        let scaled_grad = grad_output.mul_scalar(scale_factor)?;

        // TODO: Implement proper unpooling that distributes scaled gradients
        // back to all contributing input positions
        // For now, return a simplified version
        grad_input = scaled_grad;

        Ok(grad_input)
    }
}



/// 1D Max Pooling operation
/// Applies max pooling over temporal/sequential dimension (L) of 3D tensors
/// Commonly used in sequence modeling and 1D CNNs
#[derive(Debug, Clone)]
pub struct MaxPool1D {
    /// Size of the pooling window
    pub kernel_size: usize,
    /// Step size for the pooling window
    pub stride: usize,
    /// Zero-padding added to input boundaries
    pub padding: usize,
}

impl MaxPool1D {
    /// Create new MaxPool1D operation
    ///
    /// # Arguments
    /// * `kernel_size` - Size of the pooling window
    /// * `stride` - Step size for pooling window
    /// * `padding` - Zero-padding added to boundaries
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    /// Common configuration: kernel size 2, stride 2 (reduces sequence length by half)
    pub fn new_2x2() -> Self {
        Self::new(2, 2, 0)
    }

    /// Common configuration: kernel size 3, stride 1, padding 1 (preserves sequence length)
    pub fn new_3x1() -> Self {
        Self::new(3, 1, 1)
    }
}

impl<T> Operator<T> for MaxPool1D
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("MaxPool1D operation requires exactly 1 input".to_string());
        }

        inputs[0].maxpool1d(self.kernel_size, self.stride, self.padding)
    }

    fn cache_output(&self) -> bool {
        true // Cache output for gradient computation - we need to know which elements were selected
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("MaxPool1D operation requires exactly 1 input".to_string());
        }

        let input_shape = inputs[0].shape();
        if input_shape.len() != 3 {
            return Err("MaxPool1D requires 3D input tensor [N, C, L]".to_string());
        }

        // Initialize gradient tensor with zeros matching input shape
        let mut grad_input = Tensor::zeros(input_shape)?;

        // Get pooled output to determine which positions were selected
        let pooled_output = match outputs {
            Some(out) => out.clone(),
            None => self.compute(inputs)?, // Recompute if not cached
        };

        // Perform unpooling: distribute gradients back to the positions that produced the max values
        self.unpool_gradients(&mut grad_input, &grad_output, inputs[0], &pooled_output)?;

        Ok(vec![grad_input])
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }
}

impl MaxPool1D {
    /// Unpooling operation for 1D max pooling gradients
    /// Distributes gradients from pooled output back to input positions that were selected
    fn unpool_gradients<T: FerroxCudaF>(
        &self,
        grad_input: &mut Tensor<T>,
        grad_output: &Tensor<T>,
        input: &Tensor<T>,
        pooled_output: &Tensor<T>,
    ) -> Result<(), String> {
        let input_shape = input.shape();
        let (n, c, l) = (input_shape[0], input_shape[1], input_shape[2]);

        let l_out = (l + 2 * self.padding - self.kernel_size) / self.stride + 1;

        // TODO: Implement proper unpooling that tracks which input elements were selected
        // For now, we'll use a simplified approach

        // This is a simplified implementation - in practice you'd want to implement
        // a proper unpooling operation that exactly reverses the max pooling
        *grad_input = grad_output.clone();

        Ok(())
    }
}



/// 1D Average Pooling operation
/// Applies average pooling over temporal/sequential dimension (L) of 3D tensors
/// Computes the average of elements in each pooling window
#[derive(Debug, Clone)]
pub struct AvgPool1D {
    /// Size of the pooling window
    pub kernel_size: usize,
    /// Step size for the pooling window
    pub stride: usize,
    /// Zero-padding added to input boundaries
    pub padding: usize,
}

impl AvgPool1D {
    /// Create new AvgPool1D operation
    ///
    /// # Arguments
    /// * `kernel_size` - Size of the pooling window
    /// * `stride` - Step size for pooling window
    /// * `padding` - Zero-padding added to boundaries
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    /// Common configuration: kernel size 2, stride 2 (reduces sequence length by half)
    pub fn new_2x2() -> Self {
        Self::new(2, 2, 0)
    }

    /// Common configuration: kernel size 3, stride 1, padding 1 (preserves sequence length)
    pub fn new_3x1() -> Self {
        Self::new(3, 1, 1)
    }

    /// Global average pooling - pools entire sequence dimension
    pub fn global(sequence_length: usize) -> Self {
        Self::new(sequence_length, 1, 0)
    }
}

impl<T> Operator<T> for AvgPool1D
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("AvgPool1D operation requires exactly 1 input".to_string());
        }

        inputs[0].avgpool1d(self.kernel_size, self.stride, self.padding)
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("AvgPool1D operation requires exactly 1 input".to_string());
        }

        let input_shape = inputs[0].shape();
        if input_shape.len() != 3 {
            return Err("AvgPool1D requires 3D input tensor [N, C, L]".to_string());
        }

        // For average pooling, gradients are distributed uniformly to all positions
        // that contributed to each pooled output
        let grad_input = self.unpool_avg_gradients(&grad_output, input_shape)?;

        Ok(vec![grad_input])
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }
}

impl AvgPool1D {
    /// Unpooling operation for 1D average pooling gradients
    /// Distributes gradients uniformly to all input positions that contributed to each output
    fn unpool_avg_gradients<T: FerroxCudaF>(
        &self,
        grad_output: &Tensor<T>,
        input_shape: &[usize],
    ) -> Result<Tensor<T>, String> {
        let (n, c, l) = (input_shape[0], input_shape[1], input_shape[2]);

        let l_out = (l + 2 * self.padding - self.kernel_size) / self.stride + 1;

        // Create zero tensor for input gradients
        let mut grad_input = Tensor::zeros(input_shape)?;

        // For average pooling, each output gradient gets distributed equally
        // to all input positions that contributed to that output
        // The contribution is 1/kernel_size for each position

        let scale_factor = FerroxN::from_f32(1.0 / self.kernel_size as f32)
            .ok_or("Failed to convert scale factor")?;

        // Scale the output gradients by 1/kernel_size
        let scaled_grad = grad_output.mul_scalar(scale_factor)?;

        // TODO: Implement proper unpooling that distributes scaled gradients
        // back to all contributing input positions
        // For now, return a simplified version
        grad_input = scaled_grad;

        Ok(grad_input)
    }
}



/// Global Average Pooling 2D operation
/// Pools each channel to a single value by averaging all spatial locations
/// Commonly used to replace fully connected layers in modern CNN architectures
/// Input: [N, C, H, W] -> Output: [N, C, 1, 1]
#[derive(Debug, Clone)]
pub struct GlobalAvgPool2D;

impl GlobalAvgPool2D {
    /// Create new GlobalAvgPool2D operation
    pub fn new() -> Self {
        Self
    }
}

impl Default for GlobalAvgPool2D {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Operator<T> for GlobalAvgPool2D
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("GlobalAvgPool2D operation requires exactly 1 input".to_string());
        }

        let input_shape = inputs[0].shape();
        if input_shape.len() != 4 {
            return Err("GlobalAvgPool2D requires 4D input tensor [N, C, H, W]".to_string());
        }

        // Use global_avgpool2d method from tensor API
        inputs[0].global_avgpool2d()
    }

    fn gradient(
        &self,
        mut grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("GlobalAvgPool2D operation requires exactly 1 input".to_string());
        }

        let input_shape = inputs[0].shape();
        if input_shape.len() != 4 {
            return Err("GlobalAvgPool2D requires 4D input tensor [N, C, H, W]".to_string());
        }

        let (n, c, h, w) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

        // For global average pooling, the gradient is distributed uniformly
        // across all spatial positions for each channel
        // Each spatial position receives grad_output / (H * W)

        let spatial_size = h * w;
        let scale_factor = FerroxN::from_f32(1.0 / spatial_size as f32)
            .ok_or("Failed to convert scale factor")?;

        // Broadcast the output gradient back to input shape
        // grad_output shape: [N, C, 1, 1]
        // We need to expand it to [N, C, H, W] and scale by 1/(H*W)

        // First, broadcast to match input shape
       grad_output.broadcast_to(input_shape)?;

        // Scale by 1/(H*W) since each output value was the average of H*W input values
        let grad_input = grad_output.mul_scalar(scale_factor)?;

        Ok(vec![grad_input])
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }
}


/// Adaptive Average Pooling 2D operation
/// Pools to a specific output size regardless of input size
/// Useful for handling variable input sizes in networks
#[derive(Debug, Clone)]
pub struct AdaptiveAvgPool2D {
    /// Target output size [height, width]
    pub output_size: (usize, usize),
}

impl AdaptiveAvgPool2D {
    /// Create new AdaptiveAvgPool2D operation
    ///
    /// # Arguments
    /// * `output_size` - Target spatial dimensions (height, width)
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }

    /// Create adaptive pooling that outputs 1x1 (equivalent to global average pooling)
    pub fn global() -> Self {
        Self::new((1, 1))
    }
}

impl<T> Operator<T> for AdaptiveAvgPool2D
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("AdaptiveAvgPool2D operation requires exactly 1 input".to_string());
        }

        // Use adaptive_avgpool2d method from tensor API
        inputs[0].adaptive_avgpool2d(&[self.output_size.0, self.output_size.1])
    }

    fn gradient(
        &self,
        mut grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("AdaptiveAvgPool2D operation requires exactly 1 input".to_string());
        }

        let input_shape = inputs[0].shape();
        if input_shape.len() != 4 {
            return Err("AdaptiveAvgPool2D requires 4D input tensor [N, C, H, W]".to_string());
        }

        let (n, c, in_h, in_w) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
        let (out_h, out_w) = self.output_size;

        // Calculate the scale factors for gradient distribution
        let h_scale = in_h as f32 / out_h as f32;
        let w_scale = in_w as f32 / out_w as f32;
        let area_scale = h_scale * w_scale;

        let scale_factor = FerroxN::from_f32(1.0 / area_scale)
            .ok_or("Failed to convert scale factor")?;

        // Broadcast the output gradient back to input shape and scale appropriately
         grad_output.broadcast_to(input_shape)?;
        let grad_input = grad_output.mul_scalar(scale_factor)?;

        Ok(vec![grad_input])
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }
}
