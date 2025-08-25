// src/nn/layers/conv1d.rs
// 1D Convolutional layer implementation using the computational graph engine
// This is the 1D analog to Conv2d, suitable for sequence processing and time series

use crate::backend::number::FerroxCudaF;
use crate::backend::{Device, Tensor};
use crate::graph::{AutoFerroxEngine, NodeId};
use crate::nn::parameter::Parameter;
use crate::nn::Module;
use crate::ops::batched::Conv1dOp;
use crate::ops::basic::Add;
use std::cell::RefCell;
use std::collections::HashMap;

/// 1D Convolutional layer: applies convolution over 1D input tensor
/// Implements standard 1D convolution operation with configurable kernel size, stride, and padding
/// Weight tensor has shape [out_channels, in_channels, kernel_size]
#[derive(Debug)]
pub struct Conv1d<T>
where
    T: FerroxCudaF,
{
    /// Weight tensor [out_channels, in_channels, kernel_size]
    pub weight: Parameter<T>,
    /// Optional bias vector [out_channels]
    pub bias: Option<Parameter<T>>,
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output channels
    pub out_channels: usize,
    /// Kernel size (length)
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
    /// Training mode flag
    training: bool,
    parameter_maps: RefCell<Option<HashMap<String, NodeId>>>,
}

impl<T> Conv1d<T>
where
    T: FerroxCudaF,
{
    /// Create a new 1D convolutional layer
    /// Initializes weights with Kaiming uniform distribution for ReLU activations
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
        device: Device,
    ) -> Self {
        // Weight shape: [out_channels, in_channels, kernel_size]
        let weight_shape = [out_channels, in_channels, kernel_size];
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
            parameter_maps: RefCell::new(None),
        }
    }

    /// Create 1D convolutional layer with default parameters
    /// Uses stride=1, padding=0, and bias=true
    pub fn new_simple(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        device: Device,
    ) -> Self {
        Self::new(
            in_channels,
            out_channels,
            kernel_size,
            1,    // default stride
            0,    // default padding
            true, // default bias
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

impl<T> Conv1d<T>
where
    T: FerroxCudaF,
{
    /// Calculate output length given input length
    pub fn output_length(
        input_length: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> usize {
        (input_length + 2 * padding - kernel_size) / stride + 1
    }

    /// Calculate output shape given input shape
    pub fn output_shape(
        input_shape: &[usize],
        kernel_size: usize,
        stride: usize,
        padding: usize,
        out_channels: usize,
    ) -> Result<Vec<usize>, String> {
        match input_shape.len() {
            2 => {
                // [in_channels, length] -> [out_channels, out_length]
                let input_length = input_shape[1];
                let out_length = Self::output_length(input_length, kernel_size, stride, padding);
                Ok(vec![out_channels, out_length])
            }
            3 => {
                // [batch, in_channels, length] -> [batch, out_channels, out_length]
                let batch_size = input_shape[0];
                let input_length = input_shape[2];
                let out_length = Self::output_length(input_length, kernel_size, stride, padding);
                Ok(vec![batch_size, out_channels, out_length])
            }
            _ => Err(format!(
                "Input must be 2D [channels, length] or 3D [batch, channels, length], got {:?}",
                input_shape
            )),
        }
    }
}

impl<T> Module<T> for Conv1d<T>
where
    T: FerroxCudaF,
{
    /// Create parameter nodes in the computational graph and return mapping
    fn create_parameters_in_graph(
        &self,
        engine: &mut AutoFerroxEngine<T>,
    ) -> HashMap<String, NodeId> {
        let mut param_map = HashMap::new();
        println!("Initializing Conv1d parameters in graph!");

        // Create weight node
        let weight_node =
            engine.create_variable(self.weight.data.clone(), self.weight.requires_grad);
        param_map.insert("weight".to_string(), weight_node);

        // Create bias node if present
        if let Some(ref bias_param) = self.bias {
            let bias_node =
                engine.create_variable(bias_param.data.clone(), bias_param.requires_grad);
            param_map.insert("bias".to_string(), bias_node);
        }

        // Store the node mappings for use in forward()
        *self.parameter_maps.borrow_mut() = Some(param_map.clone());

        param_map
    }

    /// Forward pass: apply 1D convolution to input tensor
    /// Input shape: [batch_size, in_channels, length] or [in_channels, length]
    /// Output shape: [batch_size, out_channels, out_length] or [out_channels, out_length]
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        // Get input tensor to validate dimensions
        let input_tensor = graph
            .get_tensor(input)
            .ok_or("Input tensor not found in graph")?;

        // Validate input dimensions for conv1d
        let input_shape = input_tensor.shape();

        if input_shape.len() < 2 || input_shape.len() > 3 {
            return Err(format!(
                "Conv1d requires 2D [channels, length] or 3D [batch, channels, length] input, got shape {:?}",
                input_shape
            ));
        }

        // Handle both 2D and 3D inputs
        let (_batch_size, input_channels, input_length) = match input_shape.len() {
            2 => (1, input_shape[0], input_shape[1]), // [channels, length] -> [1, channels, length]
            3 => (input_shape[0], input_shape[1], input_shape[2]), // [batch, channels, length]
            _ => unreachable!(),
        };

        // Validate input channels match layer configuration
        if input_channels != self.in_channels {
            return Err(format!(
                "Input channels {} don't match layer channels {}",
                input_channels, self.in_channels
            ));
        }

        // Calculate output length
        let out_length = (input_length + 2 * self.padding - self.kernel_size) / self.stride + 1;
        if out_length == 0 {
            return Err("Invalid convolution parameters result in zero output length".to_string());
        }

        // Get weight node from cached parameter mappings
        let weight_node = self.get_parameter_node("weight")?;

        // Apply convolution operation using the existing Conv1d operation
        let conv_op = Box::new(Conv1dOp);
        let conv_result = graph
            .apply_operation(conv_op, vec![input, weight_node])
            .map_err(|e| format!("Convolution operation failed: {}", e))?;

        // Add bias if present
        if let Ok(bias_node) = self.get_parameter_node("bias") {
            let add_op = Box::new(Add::new());
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
