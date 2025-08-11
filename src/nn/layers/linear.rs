// src/nn/layers/linear.rs
// Linear layer implementation using the computational graph engine
// This is the fundamental building block for feedforward neural networks

use crate::backend::number::FerroxCudaF;
use crate::backend::{Device, Tensor};
use crate::graph::{AutoFerroxEngine, NodeId};
use crate::nn::parameter::Parameter;
use crate::nn::Module;
use crate::ops::{MatMul, Add, BroadcastTo};
use crate::ops::Transpose;


/// Linear transformation layer: y = x * W^T + b
/// This implements the basic feedforward layer found in most neural networks
/// Weight matrix is stored as [out_features, in_features] to match PyTorch convention
#[derive(Debug)]
pub struct Linear<T>
where
    T: FerroxCudaF,
{
    /// Weight matrix [out_features, in_features] - transposed for efficiency
    pub weight: Parameter<T>,
    /// Optional bias vector [out_features]
    pub bias: Option<Parameter<T>>,
    /// Number of input features
    pub in_features: usize,
    /// Number of output features
    pub out_features: usize,
    /// Training mode flag
    training: bool,
}

impl<T> Linear<T>
where
    T: FerroxCudaF,
{
    /// Create a new linear layer with specified input and output features
    /// Initializes weights with Xavier uniform distribution and bias with zeros
    pub fn new(in_features: usize, out_features: usize, bias: bool, device: Device) -> Self {
        Self::new_with_device(in_features, out_features, bias, device)
    }

    /// Create a new linear layer with specified device
    pub fn new_with_device(
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: Device,
    ) -> Self {
        // Weight matrix shape: [out_features, in_features] for efficient computation
        // This allows input @ weight.T + bias without needing transpose in forward pass
        let mut weight = Parameter::xavier_uniform_with_device(&[out_features, in_features], device);


        weight.set_name("weight".to_string());

        // Initialize bias to zeros if requested
        let bias_param = if bias {
            let mut b = Parameter::zeros_with_device(&[out_features], device);
            b.set_name("bias".to_string());
            Some(b)
        } else {
            None
        };

        Self {
            weight,
            bias: bias_param,
            in_features,
            out_features,
            training: true,
        }
    }

    /// Create linear layer with custom weight and bias tensors
    pub fn from_tensors(
        weight: Tensor<T>,
        bias: Option<Tensor<T>>,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self, String> {
        // Validate weight shape
        if weight.shape() != [out_features, in_features] {
            return Err(format!(
                "Weight shape {:?} doesn't match expected [out_features={}, in_features={}]",
                weight.shape(),
                out_features,
                in_features
            ));
        }

        // Validate bias shape if provided
        if let Some(ref bias_tensor) = bias {
            if bias_tensor.shape() != [out_features] {
                return Err(format!(
                    "Bias shape {:?} doesn't match expected [out_features={}]",
                    bias_tensor.shape(),
                    out_features
                ));
            }
        }

        let mut weight_param = Parameter::new(weight);
        weight_param.set_name("weight".to_string());
        let mut bias_param = bias.map(|b| Parameter::new(b));
        if let Some(ref mut bias) = bias_param {
            bias.set_name("bias".to_string());
        }
        Ok(Self {
            weight: weight_param,
            bias: bias_param,
            in_features,
            out_features,
            training: true,
        })
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
}

impl<T> Module<T> for Linear<T>
where
    T: FerroxCudaF  + rand_distr::num_traits::FromPrimitive,
{
      /// Forward pass: y = x @ W^T + b
    /// Input shape: [batch_size, in_features] or [in_features]
    /// Output shape: [batch_size, out_features] or [out_features]
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        // Get input tensor to validate dimensions
        let input_tensor = graph.get_tensor(input)
            .ok_or("Input tensor not found in graph")?;

        // Validate input dimensions
        let input_shape = input_tensor.shape();
        let expected_last_dim = self.in_features;

        // Support both 1D [in_features] and 2D [batch_size, in_features] inputs
        match input_shape.len() {
            1 => {
                if input_shape[0] != expected_last_dim {
                    return Err(format!(
                        "Input shape {:?} doesn't match expected last dimension {}",
                        input_shape, expected_last_dim
                    ));
                }
            }
            2 => {
                if input_shape[1] != expected_last_dim {
                    return Err(format!(
                        "Input shape {:?} doesn't match expected last dimension {}",
                        input_shape, expected_last_dim
                    ));
                }
            }
            _ => {
                return Err(format!(
                    "Linear layer only supports 1D or 2D inputs, got shape {:?}",
                    input_shape
                ));
            }
        }

        // Create weight node in computational graph
        let weight_node = graph.create_variable(self.weight.data.clone(), self.weight.requires_grad);


// Apply linear transformation: input @ weight^T
// Since our weight is stored as [out_features, in_features], we need to transpose it
// Use Transpose operation through the computational graph for proper gradient flow
let transpose_op = Box::new(Transpose::new());
let weight_t_node = graph
    .apply_operation(transpose_op, vec![weight_node])
    .map_err(|e| format!("Weight transpose failed: {}", e))?;

        // Perform matrix multiplication: output = input @ weight^T
        let matmul_op = Box::new(MatMul);
        let linear_result = graph
            .apply_operation(matmul_op, vec![input, weight_t_node])
            .map_err(|e| format!("Linear transformation failed: {}", e))?;

        // Add bias if present
        if let Some(ref bias_param) = self.bias {
            // Extract shape information first to avoid borrow checker issues
            let result_shape = {
                let linear_result_tensor = graph.get_tensor(linear_result)
                    .ok_or("Linear result tensor not found in graph")?;
                linear_result_tensor.shape().to_vec() // Copy shape to owned Vec
            };

            let bias_shape = bias_param.data.shape();

            // Create bias node in computational graph
            let bias_node = graph.create_variable(bias_param.data.clone(), bias_param.requires_grad);

            // Check if we need to broadcast bias to match result shape
            if bias_shape != result_shape.as_slice() {
                // Broadcast bias to match the result shape
                // bias shape [out_features] -> [batch_size, out_features] (or [out_features])
                let broadcast_op = Box::new(BroadcastTo::new(result_shape));
                let broadcasted_bias = graph
                    .apply_operation(broadcast_op, vec![bias_node])
                    .map_err(|e| format!("Bias broadcasting failed: {}", e))?;

                // Now perform element-wise addition with matching shapes
                let add_op = Box::new(Add::new());
                let final_result = graph
                    .apply_operation(add_op, vec![linear_result, broadcasted_bias])
                    .map_err(|e| format!("Bias addition failed: {}", e))?;

                Ok(final_result)
            } else {
                // Shapes already match, direct addition
                let add_op = Box::new(Add::new());
                let final_result = graph
                    .apply_operation(add_op, vec![linear_result, bias_node])
                    .map_err(|e| format!("Bias addition failed: {}", e))?;

                Ok(final_result)
            }
        } else {
            Ok(linear_result)
        }
    }


    /// Collect all parameters (weight and bias)
    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    /// Collect mutable parameter references
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }

    /// Training mode getter
    fn training(&self) -> bool {
        self.training
    }

    /// Set training mode
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}
