use crate::backend::{FerroxCudaF,  FerroxN, Tensor};
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

        let input_shape = inputs[0].shape();
        if input_shape.len() != 4 {
            return Err("MaxPool2D requires 4D input tensor [N, C, H, W]".to_string());
        }

        // Get pooled output to determine which positions were selected
        let pooled_output = match outputs {
            Some(out) => out.clone(),
            None => self.compute(inputs)?, // Recompute if not cached
        };

        // Perform unpooling using Tensor API
        let grad_input = self.unpool_gradients(&grad_output, inputs[0], &pooled_output)?;

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
    /// Proper unpooling operation for max pooling gradients using Tensor API
    fn unpool_gradients<T: FerroxCudaF>(
        &self,
        grad_output: &Tensor<T>,
        input: &Tensor<T>,
        pooled_output: &Tensor<T>,
    ) -> Result<Tensor<T>, String> {
        // Use the new Tensor API method for max unpooling
        grad_output.max_unpool2d(
            input,
            pooled_output,
            self.kernel_size,
            self.stride,
            self.padding,
        )
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
        outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("AvgPool2D operation requires exactly 1 input".to_string());
        }



        let input_shape = inputs[0].shape();
        if input_shape.len() != 4 {
            return Err("AvgPool2D requires 4D input tensor [N, C, H, W]".to_string());
        }

        // For average pooling, we need the pooled output to properly compute gradients
        let pooled_output = match outputs {
            Some(out) => out,
            None => &self.compute(inputs)?, // Recompute if not cached
        };

        let grad_input = self.unpool_gradients(&grad_output, inputs[0], pooled_output)?;

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
    /// Proper unpooling operation for average pooling gradients using Tensor API
    fn unpool_gradients<T: FerroxCudaF>(
        &self,
        grad_output: &Tensor<T>,
        original_input: &Tensor<T>,
        pooled_output: &Tensor<T>,
    ) -> Result<Tensor<T>, String> {
        // Use the new Tensor API method for average unpooling
        grad_output.avg_unpool2d(
            original_input,
            pooled_output,
            self.kernel_size,
            self.stride,
            self.padding,
        )
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

        // Get pooled output to determine which positions were selected
        let pooled_output = match outputs {
            Some(out) => out,
            None => &self.compute(inputs)?, // Recompute if not cached
        };

        // Perform unpooling using Tensor API
        let grad_input = self.unpool_gradients(&grad_output, inputs[0], pooled_output)?;

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
        grad_output: &Tensor<T>,
        input: &Tensor<T>,
        pooled_output: &Tensor<T>,
    ) -> Result<Tensor<T>, String> {
        // Use the new Tensor API method for 1D max unpooling
        grad_output.max_unpool1d(
            input,
            pooled_output,
            self.kernel_size,
            self.stride,
            self.padding,
        )
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
        outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("AvgPool1D operation requires exactly 1 input".to_string());
        }

        let input_shape = inputs[0].shape();
        if input_shape.len() != 3 {
            return Err("AvgPool1D requires 3D input tensor [N, C, L]".to_string());
        }

        // For average pooling, we need the pooled output to properly compute gradients
        let pooled_output = match outputs {
            Some(out) => out,
            None => &self.compute(inputs)?, // Recompute if not cached
        };

        let grad_input = self.unpool_gradients(&grad_output, inputs[0], pooled_output)?;

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
     fn unpool_gradients<T: FerroxCudaF>(
        &self,
        grad_output: &Tensor<T>,
        original_input: &Tensor<T>,
        pooled_output: &Tensor<T>,
    ) -> Result<Tensor<T>, String> {
        // Use the new Tensor API method for 1D average unpooling
        grad_output.avg_unpool1d(
            original_input,
            pooled_output,
            self.kernel_size,
            self.stride,
            self.padding,
        )
    }
}



#[derive(Debug, Clone)]
pub struct AdaptiveAvgPool1D {
    /// Target output size
    pub output_size: usize,
}

impl AdaptiveAvgPool1D {
    /// Create new AdaptiveAvgPool1D operation
    ///
    /// # Arguments
    /// * `output_size` - Target sequence length
    pub fn new(output_size: usize) -> Self {
        Self { output_size }
    }

    /// Create adaptive pooling that outputs length 1 (equivalent to global average pooling)
    pub fn global() -> Self {
        Self::new(1)
    }
}

impl<T> Operator<T> for AdaptiveAvgPool1D
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("AdaptiveAvgPool1D operation requires exactly 1 input".to_string());
        }

        let input_shape = inputs[0].shape();
        if input_shape.len() != 3 {
            return Err("AdaptiveAvgPool1D requires 3D input tensor [N, C, L]".to_string());
        }

        let (_, _, l) = (input_shape[0], input_shape[1], input_shape[2]);
        let l_out = self.output_size;

        if l_out == 1 {
            // Global average pooling - use existing global method
            return inputs[0].global_avgpool1d();
        }

        // Calculate adaptive kernel size and stride
        let kernel_size = l.div_ceil(l_out); // Ceiling division
        let stride = l / l_out;

        // Use regular average pooling with calculated parameters
        inputs[0].avgpool1d(kernel_size, stride, 0)
    }

    fn gradient(
        &self,
        mut grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: Option<&Tensor<T>>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("AdaptiveAvgPool1D operation requires exactly 1 input".to_string());
        }

        let input_shape = inputs[0].shape();
        if input_shape.len() != 3 {
            return Err("AdaptiveAvgPool1D requires 3D input tensor [N, C, L]".to_string());
        }

        let (_, _, l) = (input_shape[0], input_shape[1], input_shape[2]);
        let l_out = self.output_size;

        if l_out == 1 {
            // Global average pooling gradient - distribute uniformly
            let scale_factor = FerroxN::from_f32(1.0 / l as f32)
                .ok_or("Failed to convert scale factor")?;

            // Broadcast the output gradient back to input shape and scale appropriately
            grad_output.broadcast_to(input_shape)?;
            let grad_input = grad_output.mul_scalar(scale_factor)?;

            return Ok(vec![grad_input]);
        }

        // For general adaptive pooling, calculate scale factor
        let scale_factor = FerroxN::from_f32(l_out as f32 / l as f32)
            .ok_or("Failed to convert scale factor")?;

        // Broadcast and scale
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

        let (_, _, h, w) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

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
