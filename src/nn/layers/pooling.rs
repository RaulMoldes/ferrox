// src/nn/layers/pooling.rs
// Pooling and utility layers for neural networks
// Includes Flatten, MaxPool2d, and GlobalAvgPool2d implementations

use crate::backend::number::FerroxCudaF;
use crate::graph::{AutoFerroxEngine, NodeId};
use crate::nn::parameter::Parameter;
use crate::nn::Module;
use crate::ops::pooling::*;
use crate::ops::reduction::Mean;
use crate::ops::reshape::Reshape;
use std::marker::PhantomData;
/// Flatten layer: reshapes input tensor to 1D while preserving batch dimension
/// Converts tensor from [batch_size, ...] to [batch_size, flattened_features]
/// Commonly used between convolutional and linear layers
#[derive(Debug, Clone)]
pub struct Flatten<T>
where
    T: FerroxCudaF,
{
    /// Starting dimension for flattening (default: 1 to preserve batch dimension)
    start_dim: usize,
    /// Ending dimension for flattening (default: -1 for all remaining dimensions)
    end_dim: Option<usize>,
    /// Training mode flag
    training: bool,
    /// Phantom data for type parameter
    _phantom: PhantomData<T>,
}

impl<T> Flatten<T>
where
    T: FerroxCudaF,
{
    /// Create a new Flatten layer
    /// start_dim: dimension to start flattening from (default: 1 to preserve batch)
    /// end_dim: dimension to end flattening at (None means flatten to the end)
    pub fn new(start_dim: usize, end_dim: Option<usize>) -> Self {
        Self {
            start_dim,
            end_dim,
            training: true,
            _phantom: PhantomData,
        }
    }

    /// Create flatten layer that preserves batch dimension (most common usage)
    /// Flattens from dimension 1 onwards: [N, C, H, W] -> [N, C*H*W]
    pub fn new_batch_first() -> Self {
        Self::new(1, None)
    }

    /// Create flatten layer that flattens everything to 1D
    /// [N, C, H, W] -> [N*C*H*W]
    pub fn new_all() -> Self {
        Self::new(0, None)
    }

    /// Calculate the flattened shape given input shape
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let ndim = input_shape.len();
        let start = self.start_dim;
        let end = self.end_dim.unwrap_or(ndim);

        // Validate dimensions
        if start >= ndim || end > ndim || start >= end {
            panic!(
                "Invalid flatten dimensions: start={}, end={}, input_ndim={}",
                start, end, ndim
            );
        }

        let mut output_shape = Vec::new();

        // Keep dimensions before start_dim unchanged
        output_shape.extend_from_slice(&input_shape[..start]);

        // Calculate flattened size for dimensions [start_dim, end_dim)
        let flattened_size: usize = input_shape[start..end].iter().product();
        output_shape.push(flattened_size);

        // Keep dimensions after end_dim unchanged
        if end < ndim {
            output_shape.extend_from_slice(&input_shape[end..]);
        }

        output_shape
    }
}

impl<T> Default for Flatten<T>
where
    T: FerroxCudaF,
{
    fn default() -> Self {
        Self::new_batch_first() // Default preserves batch dimension
    }
}

impl<T> Module<T> for Flatten<T>
where
    T: FerroxCudaF,
{
    /// Forward pass: flatten tensor using reshape operation
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        // Get input tensor to determine output shape
        let input_tensor = graph
            .get_tensor(input)
            .ok_or("Input tensor not found in graph")?;

        let input_shape = input_tensor.shape();
        let output_shape = self.calculate_output_shape(input_shape);

        // Apply reshape operation
        let reshape_op = Box::new(Reshape::new(output_shape));
        graph
            .apply_operation(reshape_op, vec![input])
            .map_err(|e| format!("Flatten operation failed: {}", e))
    }

    /// Flatten has no parameters
    fn parameters(&self) -> Vec<&Parameter<T>> {
        Vec::new()
    }

    /// Flatten has no mutable parameters
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        Vec::new()
    }

    /// Get training mode
    fn training(&self) -> bool {
        self.training
    }

    /// Set training mode (no effect for Flatten)
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}


/// 2D Max Pooling layer
/// Applies 2D max pooling over input tensor
/// Input: [batch, channels, height, width]
/// Output: [batch, channels, height_out, width_out]
#[derive(Debug, Clone)]
pub struct MaxPool2d<T>
where
    T: FerroxCudaF,
{
    /// Pooling kernel size (square kernel)
    pub kernel_size: usize,
    /// Stride for pooling window movement
    pub stride: usize,
    /// Zero padding applied to input
    pub padding: usize,
    /// Training mode flag
    training: bool,
    /// Phantom data for type parameter
    _phantom: PhantomData<T>,
}

impl<T> MaxPool2d<T>
where
    T: FerroxCudaF,
{
    /// Create new MaxPool2d layer
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            training: true,
            _phantom: PhantomData,
        }
    }

    /// Create MaxPool2d with default stride equal to kernel size (non-overlapping)
    pub fn new_default_stride(kernel_size: usize, padding: usize) -> Self {
        Self::new(kernel_size, kernel_size, padding)
    }

    /// Common 2x2 max pooling with stride 2 (typical downsampling)
    pub fn new_2x2() -> Self {
        Self::new(2, 2, 0)
    }

    /// Common 3x3 max pooling with stride 2 and padding 1
    pub fn new_3x3() -> Self {
        Self::new(3, 2, 1)
    }
}

impl<T> Module<T> for MaxPool2d<T>
where
    T: FerroxCudaF,
{
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        // Validate input dimensions
        let input_tensor = graph
            .get_tensor(input)
            .ok_or("Input tensor not found")?;
        let input_shape = input_tensor.shape();

        if input_shape.len() != 4 {
            return Err("MaxPool2d requires 4D input [batch, channels, height, width]".to_string());
        }

        // Apply max pooling operation
        let maxpool_op = Box::new(MaxPool2D::new(
            self.kernel_size,
            self.stride,
            self.padding,
        ));

        graph
            .apply_operation(maxpool_op, vec![input])
            .map_err(|e| format!("MaxPool2d operation failed: {}", e))
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        Vec::new() // Pooling layers have no parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        Vec::new()
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// 2D Average Pooling layer
/// Applies 2D average pooling over input tensor
/// Input: [batch, channels, height, width]
/// Output: [batch, channels, height_out, width_out]
#[derive(Debug, Clone)]
pub struct AvgPool2d<T>
where
    T: FerroxCudaF,
{
    /// Pooling kernel size (square kernel)
    pub kernel_size: usize,
    /// Stride for pooling window movement
    pub stride: usize,
    /// Zero padding applied to input
    pub padding: usize,
    /// Training mode flag
    training: bool,
    /// Phantom data for type parameter
    _phantom: PhantomData<T>,
}

impl<T> AvgPool2d<T>
where
    T: FerroxCudaF,
{
    /// Create new AvgPool2d layer
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            training: true,
            _phantom: PhantomData,
        }
    }

    /// Create AvgPool2d with default stride equal to kernel size
    pub fn new_default_stride(kernel_size: usize, padding: usize) -> Self {
        Self::new(kernel_size, kernel_size, padding)
    }

    /// Common 2x2 average pooling
    pub fn new_2x2() -> Self {
        Self::new(2, 2, 0)
    }

    /// Common 3x3 average pooling with stride 2 and padding 1
    pub fn new_3x3() -> Self {
        Self::new(3, 2, 1)
    }
}

impl<T> Module<T> for AvgPool2d<T>
where
    T: FerroxCudaF,
{
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        // Validate input dimensions
        let input_tensor = graph
            .get_tensor(input)
            .ok_or("Input tensor not found")?;
        let input_shape = input_tensor.shape();

        if input_shape.len() != 4 {
            return Err("AvgPool2d requires 4D input [batch, channels, height, width]".to_string());
        }

        // Apply average pooling operation
        let avgpool_op = Box::new(AvgPool2D::new(
            self.kernel_size,
            self.stride,
            self.padding,
        ));

        graph
            .apply_operation(avgpool_op, vec![input])
            .map_err(|e| format!("AvgPool2d operation failed: {}", e))
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        Vec::new()
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// 1D Max Pooling layer
/// Applies 1D max pooling over input tensor along the last dimension
/// Input: [batch, channels, length]
/// Output: [batch, channels, length_out]
#[derive(Debug, Clone)]
pub struct MaxPool1d<T>
where
    T: FerroxCudaF,
{
    /// Pooling kernel size
    pub kernel_size: usize,
    /// Stride for pooling window movement
    pub stride: usize,
    /// Zero padding applied to input
    pub padding: usize,
    /// Training mode flag
    training: bool,
    /// Phantom data for type parameter
    _phantom: PhantomData<T>,
}

impl<T> MaxPool1d<T>
where
    T: FerroxCudaF,
{
    /// Create new MaxPool1d layer
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            training: true,
            _phantom: PhantomData,
        }
    }

    /// Create MaxPool1d with default stride equal to kernel size
    pub fn new_default_stride(kernel_size: usize, padding: usize) -> Self {
        Self::new(kernel_size, kernel_size, padding)
    }

    /// Common kernel size 2 max pooling
    pub fn new_kernel2() -> Self {
        Self::new(2, 2, 0)
    }

    /// Common kernel size 3 max pooling with stride 2
    pub fn new_kernel3() -> Self {
        Self::new(3, 2, 1)
    }
}

impl<T> Module<T> for MaxPool1d<T>
where
    T: FerroxCudaF,
{
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        // Validate input dimensions
        let input_tensor = graph
            .get_tensor(input)
            .ok_or("Input tensor not found")?;
        let input_shape = input_tensor.shape();

        if input_shape.len() != 3 {
            return Err("MaxPool1d requires 3D input [batch, channels, length]".to_string());
        }

        // Apply 1D max pooling operation
        let maxpool_op = Box::new(MaxPool1D::new(
            self.kernel_size,
            self.stride,
            self.padding,
        ));

        graph
            .apply_operation(maxpool_op, vec![input])
            .map_err(|e| format!("MaxPool1d operation failed: {}", e))
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        Vec::new()
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// 1D Average Pooling layer
/// Applies 1D average pooling over input tensor along the last dimension
/// Input: [batch, channels, length]
/// Output: [batch, channels, length_out]
#[derive(Debug, Clone)]
pub struct AvgPool1d<T>
where
    T: FerroxCudaF,
{
    /// Pooling kernel size
    pub kernel_size: usize,
    /// Stride for pooling window movement
    pub stride: usize,
    /// Zero padding applied to input
    pub padding: usize,
    /// Training mode flag
    training: bool,
    /// Phantom data for type parameter
    _phantom: PhantomData<T>,
}

impl<T> AvgPool1d<T>
where
    T: FerroxCudaF,
{
    /// Create new AvgPool1d layer
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            training: true,
            _phantom: PhantomData,
        }
    }

    /// Create AvgPool1d with default stride equal to kernel size
    pub fn new_default_stride(kernel_size: usize, padding: usize) -> Self {
        Self::new(kernel_size, kernel_size, padding)
    }

    /// Common kernel size 2 average pooling
    pub fn new_kernel2() -> Self {
        Self::new(2, 2, 0)
    }

    /// Common kernel size 3 average pooling with stride 2
    pub fn new_kernel3() -> Self {
        Self::new(3, 2, 1)
    }
}

impl<T> Module<T> for AvgPool1d<T>
where
    T: FerroxCudaF,
{
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        // Validate input dimensions
        let input_tensor = graph
            .get_tensor(input)
            .ok_or("Input tensor not found")?;
        let input_shape = input_tensor.shape();

        if input_shape.len() != 3 {
            return Err("AvgPool1d requires 3D input [batch, channels, length]".to_string());
        }

        // Apply 1D average pooling operation
        let avgpool_op = Box::new(AvgPool1D::new(
            self.kernel_size,
            self.stride,
            self.padding,
        ));

        graph
            .apply_operation(avgpool_op, vec![input])
            .map_err(|e| format!("AvgPool1d operation failed: {}", e))
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        Vec::new()
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}




/// Global Average Pooling 2D layer: averages over all spatial dimensions
/// Reduces spatial dimensions to 1x1 by computing mean over height and width
/// Input shape: [batch_size, channels, height, width]
/// Output shape: [batch_size, channels, 1, 1] or [batch_size, channels] if flatten=true
#[derive(Debug, Clone)]
pub struct GlobalAvgPool2d<T>
where
    T: FerroxCudaF,
{
    /// Whether to flatten output to 2D [batch_size, channels] or keep 4D [batch_size, channels, 1, 1]
    flatten: bool,
    /// Training mode flag
    training: bool,
    /// Phantom data for type parameter
    _phantom: PhantomData<T>,
}

impl<T> GlobalAvgPool2d<T>
where
    T: FerroxCudaF,
{
    /// Create a new GlobalAvgPool2d layer
    /// flatten: if true, output shape is [batch_size, channels]
    ///         if false, output shape is [batch_size, channels, 1, 1]
    pub fn new(flatten: bool) -> Self {
        Self {
            flatten,
            training: true,
            _phantom: PhantomData,
        }
    }

    /// Create GlobalAvgPool2d that keeps 4D output shape [batch, channels, 1, 1]
    pub fn new_keep_dims() -> Self {
        Self::new(false)
    }

    /// Create GlobalAvgPool2d that flattens to 2D output shape [batch, channels]
    pub fn new_flatten() -> Self {
        Self::new(true)
    }

    /// Calculate output shape based on flatten setting
    #[allow(dead_code)]
    fn calculate_output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>, String> {
        if input_shape.len() != 4 {
            return Err("GlobalAvgPool2d requires 4D input [batch, channels, height, width]".to_string());
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];

        if self.flatten {
            Ok(vec![batch_size, channels])
        } else {
            Ok(vec![batch_size, channels, 1, 1])
        }
    }
}

impl<T> Default for GlobalAvgPool2d<T>
where
    T: FerroxCudaF,
{
    fn default() -> Self {
        Self::new_flatten() // Default flattens output
    }
}

impl<T> Module<T> for GlobalAvgPool2d<T>
where
    T: FerroxCudaF,
{
    /// Forward pass: apply global average pooling using mean reduction
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        // Validate input shape
        let input_tensor = graph
            .get_tensor(input)
            .ok_or("Input tensor not found in graph")?;
        let input_shape = input_tensor.shape();

        if input_shape.len() != 4 {
            return Err("GlobalAvgPool2d requires 4D input [batch, channels, height, width]".to_string());
        }

        // Apply mean reduction along spatial dimensions (axes 2 and 3)
        // keep_dims = !flatten to control output shape
        let mean_op = Box::new(Mean::along_axes(vec![2, 3], !self.flatten));
        let pooled_result = graph
            .apply_operation(mean_op, vec![input])
            .map_err(|e| format!("Global average pooling failed: {}", e))?;

        Ok(pooled_result)
    }

    /// GlobalAvgPool2d has no parameters
    fn parameters(&self) -> Vec<&Parameter<T>> {
        Vec::new()
    }

    /// GlobalAvgPool2d has no mutable parameters
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        Vec::new()
    }

    /// Get training mode
    fn training(&self) -> bool {
        self.training
    }

    /// Set training mode (no effect for GlobalAvgPool2d)
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl<T> GlobalAvgPool2d<T>
where
    T: FerroxCudaF,
{
    /// Get the flatten setting
    pub fn is_flatten(&self) -> bool {
        self.flatten
    }

    /// Set the flatten behavior
    pub fn set_flatten(&mut self, flatten: bool) {
        self.flatten = flatten;
    }

    /// Calculate the reduction in spatial dimensions
    /// Returns the number of elements averaged (height * width)
    pub fn calculate_reduction_size(&self, input_shape: &[usize]) -> Result<usize, String> {
        if input_shape.len() != 4 {
            return Err("Input must be 4D".to_string());
        }

        let height = input_shape[2];
        let width = input_shape[3];
        Ok(height * width)
    }
}
