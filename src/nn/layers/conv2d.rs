// src/nn/layers/conv2d.rs
// 2D Convolutional layer implementation using the computational graph engine
// This is a fundamental building block for convolutional neural networks

use crate::backend::number::FerroxCudaF;
use crate::backend::{Device, Tensor};
use crate::graph::{AutoFerroxEngine, NodeId};
use crate::nn::parameter::Parameter;
use crate::nn::Module;

/// 2D Convolutional layer: applies convolution over input tensor
/// Implements standard convolution operation with configurable kernel size, stride, and padding
/// Weight tensor has shape [out_channels, in_channels, kernel_height, kernel_width]
#[derive(Debug)]
pub struct Conv2d<T>
where
    T: FerroxCudaF,
{
    /// Weight tensor [out_channels, in_channels, kernel_height, kernel_width]
    pub weight: Parameter<T>,
    /// Optional bias vector [out_channels]
    pub bias: Option<Parameter<T>>,
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output channels
    pub out_channels: usize,
    /// Kernel size (height, width)
    pub kernel_size: (usize, usize),
    /// Stride (height, width)
    pub stride: (usize, usize),
    /// Padding (height, width)
    pub padding: (usize, usize),
    /// Training mode flag
    training: bool,
    parameter_maps: std::cell::RefCell<Option<std::collections::HashMap<String, NodeId>>>,
}

impl<T> Conv2d<T>
where
    T: FerroxCudaF,
{
    /// Create a new 2D convolutional layer
    /// Initializes weights with Kaiming uniform distribution for ReLU activations
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
        device: Device,
    ) -> Self {
        // Weight shape: [out_channels, in_channels, kernel_height, kernel_width]
        let weight_shape = [out_channels, in_channels, kernel_size.0, kernel_size.1];
        let mut weight = Parameter::kaiming_uniform_with_device(&weight_shape, device);
        weight.set_name("weight".to_string());

        // Initialize bias to zeros if requested
        let bias_param = if bias {
            let mut b = Parameter::zeros_with_device(&[out_channels], device);
            b.set_name("bias".to_string());
            Some(b)
        } else {
            None
        };

        Self {
            weight,
            bias: bias_param,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            training: true,
            parameter_maps: std::cell::RefCell::new(None),
        }
    }

    /// Create convolutional layer with square kernel and default parameters
    pub fn new_square(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        device: Device,
    ) -> Self {
        Self::new(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            (1, 1), // default stride
            (0, 0), // default padding
            true,   // default bias
            device,
        )
    }

    /// Get weight tensor reference
    pub fn weight_tensor(&self) -> &Tensor<T> {
        &self.weight.data
    }

    /// Get bias tensor reference (if exists)
    pub fn bias_tensor(&self) -> Option<&Tensor<T>> {
        self.bias.as_ref().map(|b| &b.data)
    }

    /// Check if layer has bias
    pub fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    // Helper method to get parameter node IDs (cached from create_parameters_in_graph)
    fn get_parameter_node(&self, param_name: &str) -> Result<NodeId, String> {
        let nodes_ref = self.parameter_maps.borrow();
        match nodes_ref.as_ref() {
            Some(node_map) => node_map
                .get(param_name)
                .copied()
                .ok_or_else(|| format!("Parameter '{}' not found in node mapping", param_name)),
            None => Err(
                "Parameters not yet created in graph. Call create_parameters_in_graph() first."
                    .to_string(),
            ),
        }
    }
}

impl<T> Module<T> for Conv2d<T>
where
    T: FerroxCudaF + rand_distr::num_traits::FromPrimitive,
{
    /// Create parameter nodes in the computational graph and return mapping
    fn create_parameters_in_graph(
        &self,
        engine: &mut AutoFerroxEngine<T>,
    ) -> std::collections::HashMap<String, NodeId> {
        let mut param_map = std::collections::HashMap::new();
        println!("Initializing Conv2d parameters in graph!");

        // Create weight node
        let weight_node = engine.create_variable(self.weight.data.clone(), self.weight.requires_grad);
        param_map.insert("weight".to_string(), weight_node);

        // Create bias node if present
        if let Some(ref bias_param) = self.bias {
            let bias_node = engine.create_variable(bias_param.data.clone(), bias_param.requires_grad);
            param_map.insert("bias".to_string(), bias_node);
        }

        // Store the node mappings for use in forward()
        *self.parameter_maps.borrow_mut() = Some(param_map.clone());

        param_map
    }

    /// Forward pass: apply 2D convolution to input tensor
    /// Input shape: [batch_size, in_channels, height, width]
    /// Output shape: [batch_size, out_channels, out_height, out_width]
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        // Get input tensor to validate dimensions
        let input_tensor = graph
            .get_tensor(input)
            .ok_or("Input tensor not found in graph")?;

        // Validate input dimensions for conv2d
        let input_shape = input_tensor.shape();
        if input_shape.len() != 4 {
            return Err(format!(
                "Conv2d requires 4D input [batch, channels, height, width], got shape {:?}",
                input_shape
            ));
        }

        // Validate input channels match layer configuration
        let input_channels = input_shape[1];
        if input_channels != self.in_channels {
            return Err(format!(
                "Input channels {} don't match layer channels {}",
                input_channels, self.in_channels
            ));
        }

        // Get weight node from cached parameter mappings
        let weight_node = self.get_parameter_node("weight")?;

        // Apply convolution operation using the existing Conv2d operation
        let conv_op = Box::new(crate::ops::batched::Conv2d::new(self.stride, self.padding));
        let conv_result = graph
            .apply_operation(conv_op, vec![input, weight_node])
            .map_err(|e| format!("Convolution operation failed: {}", e))?;

        // Add bias if present
        if self.bias.is_some() {
            // Get bias node from cached parameter mappings
            let bias_node = self.get_parameter_node("bias")?;

            // Add bias using element-wise addition (bias will be broadcasted automatically)
            let add_op = Box::new(crate::ops::basic::Add::new());
            let final_result = graph
                .apply_operation(add_op, vec![conv_result, bias_node])
                .map_err(|e| format!("Bias addition failed: {}", e))?;

            Ok(final_result)
        } else {
            Ok(conv_result)
        }
    }

    /// Collect all parameters (weight and optional bias)
    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    /// Mutable access to parameters
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }

    /// Get training mode
    fn training(&self) -> bool {
        self.training
    }

    /// Set training mode
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl<T> Conv2d<T>
where
    T: FerroxCudaF,
{
    /// Calculate output dimensions given input dimensions
    pub fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>, String> {
        if input_shape.len() != 4 {
            return Err("Input must be 4D [batch, channels, height, width]".to_string());
        }

        let batch_size = input_shape[0];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        // Calculate output dimensions using conv2d formula
        let output_height = (input_height + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let output_width = (input_width + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;

        Ok(vec![batch_size, self.out_channels, output_height, output_width])
    }

    /// Get number of parameters in this layer
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.out_channels * self.in_channels * self.kernel_size.0 * self.kernel_size.1;
        let bias_params = if self.bias.is_some() { self.out_channels } else { 0 };
        weight_params + bias_params
    }

    /// Create a Conv2d layer with same padding to preserve spatial dimensions
    pub fn new_same_padding(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        device: Device,
    ) -> Self {
        // Calculate padding for "same" padding behavior
        let padding = ((kernel_size.0 - 1) / 2, (kernel_size.1 - 1) / 2);

        Self::new(
            in_channels,
            out_channels,
            kernel_size,
            (1, 1), // stride = 1 for same padding
            padding,
            true,   // bias = true
            device,
        )
    }
}
