// src/nn/layers/normalization.rs
// Normalization layers implemented using existing graph operations
// BatchNorm, LayerNorm, and RMSNorm for stable training and better convergence

use crate::backend::number::FerroxCudaF;
use crate::backend::{Device, Tensor};
use crate::graph::{AutoFerroxEngine, NodeId};
use crate::nn::parameter::Parameter;
use crate::nn::Module;
use crate::ops::{
    basic::{Add, Div, Mul, Sub},
    reduction::Mean,
    scalar::{AddScalar},
    unary::Sqrt,

};
use crate::FerroxN;
use std::cell::RefCell;
use std::collections::HashMap;
/// ---------------------------------------------------
/// BATCH NORM
/// ---------------------------------------------------
///
/// Batch Normalization layer: normalizes across the batch dimension
/// Normalizes input using batch statistics: (x - batch_mean) / sqrt(batch_var + eps)
/// Learnable scale (gamma) and shift (beta) parameters
/// Maintains running statistics for inference mode
/// Input shape: [batch_size, features, ...] or [batch_size, channels, height, width]
#[derive(Debug)]
pub struct BatchNorm<T>
where
    T: FerroxCudaF,
{

    /// Small epsilon for numerical stability
    eps: f64,
    /// Momentum for running mean/var updates (default: 0.1)
    momentum: f64,
    /// Whether to track running statistics (default: true)
    track_running_stats: bool,
    /// Learnable scale parameter (gamma)
    pub weight: Parameter<T>,
    /// Learnable shift parameter (beta)
    pub bias: Parameter<T>,
    /// Running mean for inference (not learnable)
    pub running_mean: RefCell<Tensor<T>>,
    /// Running variance for inference (not learnable)
    pub running_var: RefCell<Tensor<T>>,
    /// Number of batches tracked (for momentum adjustment)
    num_batches_tracked: RefCell<usize>,
    /// Training mode flag
    training: bool,
    parameter_maps: RefCell<Option<HashMap<String, NodeId>>>,
}

impl<T> BatchNorm<T>
where
    T: FerroxCudaF,
{
    /// Create a new BatchNorm layer
    pub fn new(num_features:usize, eps: f64, momentum: f64, track_running_stats: bool, device: Device) -> Self {
        // Initialize gamma (weight) to ones
        let mut weight = Parameter::ones_with_device(&[num_features], device);
        weight.set_name("weight".to_string());

        // Initialize beta (bias) to zeros
        let mut bias = Parameter::zeros_with_device(&[num_features], device);
        bias.set_name("bias".to_string());

        // Initialize running statistics
        let running_mean = RefCell::new(
            Tensor::zeros_with_device(&[num_features], device)
                .expect("Failed to create running_mean tensor")
        );
        let running_var = RefCell::new(
            Tensor::ones_with_device(&[num_features], device)
                .expect("Failed to create running_var tensor")
        );

        Self {

            eps,
            momentum,
            track_running_stats,
            weight,
            bias,
            running_mean,
            running_var,
            num_batches_tracked: RefCell::new(0),
            training: true,
            parameter_maps: RefCell::new(None),
        }
    }

    /// Create BatchNorm with default parameters
    pub fn new_default(num_features: usize, device: Device) -> Self {
        Self::new(num_features, 1e-5, 0.1, true, device)
    }

    /// Create BatchNorm without tracking running statistics (training only)
    pub fn new_no_stats(num_features: usize, device: Device) -> Self {
        Self::new(num_features, 1e-5, 0.1, false, device)
    }

    // Helper method to get parameter node IDs
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

    /// Update running statistics using exponential moving average
    fn update_running_stats(&self, batch_mean: &Tensor<T>, batch_var: &Tensor<T>) -> Result<(), String> {
        if !self.track_running_stats || !self.training {
            return Ok(());
        }

        let momentum = FerroxN::from_f64(self.momentum).unwrap();
        let one_minus_momentum = FerroxN::from_f64(1.0 - self.momentum).unwrap();

        // Increment batch counter
        {
            let mut num_batches = self.num_batches_tracked.borrow_mut();
            *num_batches += 1;
        }

        // Update running mean: running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        {
            let mut running_mean = self.running_mean.borrow_mut();
            let scaled_running = running_mean.mul_scalar(one_minus_momentum)?;
            let scaled_batch = batch_mean.mul_scalar(momentum)?;
            *running_mean = scaled_running.add(&scaled_batch)?;
        }

        // Update running variance: running_var = (1 - momentum) * running_var + momentum * batch_var
        {
            let mut running_var = self.running_var.borrow_mut();
            let scaled_running = running_var.mul_scalar(one_minus_momentum)?;
            let scaled_batch = batch_var.mul_scalar(momentum)?;
            *running_var = scaled_running.add(&scaled_batch)?;
        }

        Ok(())
    }

    /// Apply batch normalization using existing operations
    fn apply_batch_norm(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        // Get input tensor for shape validation
        let input_tensor = graph
            .get_tensor(input)
            .ok_or("Input tensor not found in graph")?;
        let input_shape = input_tensor.shape();

        if input_shape.len() < 2 {
            return Err("BatchNorm requires at least 2D input [batch_size, features]".to_string());
        }

        if self.training {
            // Training mode: use batch statistics
            self.apply_training_batch_norm(graph, input)
        } else {
            // Inference mode: use running statistics
            self.apply_inference_batch_norm(graph, input)
        }
    }

    /// Apply batch norm in training mode using batch statistics
    fn apply_training_batch_norm(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        let input_tensor = graph
            .get_tensor(input)
            .ok_or("Input tensor not found in graph")?;
        let input_shape = input_tensor.shape();

        // For BatchNorm, we normalize across batch dimension (axis 0)
        // and optionally spatial dimensions for 4D tensors
        let norm_axes = if input_shape.len() == 4 {
            vec![0, 2, 3] // Batch, Height, Width for [N, C, H, W]
        } else {
            vec![0] // Just batch dimension for [N, F]
        };

        // Step 1: Compute batch mean
        let mean_op = Box::new(Mean::along_axes(norm_axes.clone(), true));
        let batch_mean = graph
            .apply_operation(mean_op, vec![input])
            .map_err(|e| format!("BatchNorm mean computation failed: {}", e))?;

        // Step 2: Compute (x - mean)
        let sub_op = Box::new(Sub);
        let centered = graph
            .apply_operation(sub_op, vec![input, batch_mean])
            .map_err(|e| format!("BatchNorm centering failed: {}", e))?;

        // Step 3: Compute variance = mean((x - mean)^2)
        let mul_op = Box::new(Mul);
        let squared_diff = graph
            .apply_operation(mul_op, vec![centered, centered])
            .map_err(|e| format!("BatchNorm variance computation failed: {}", e))?;

        let var_mean_op = Box::new(Mean::along_axes(norm_axes, true));
        let variance = graph
            .apply_operation(var_mean_op, vec![squared_diff])
            .map_err(|e| format!("BatchNorm variance mean failed: {}", e))?;

        // Update running statistics (this happens outside the graph)
        if self.track_running_stats {
            // Extract tensors from graph for running stats update
            let batch_mean_tensor = graph.get_tensor(batch_mean)
                .ok_or("Could not get batch mean tensor")?;
            let variance_tensor = graph.get_tensor(variance)
                .ok_or("Could not get variance tensor")?;

            // Update running statistics
            self.update_running_stats(batch_mean_tensor, variance_tensor)
                .map_err(|e| format!("Failed to update running stats: {}", e))?;
        }

        // Continue with normalization
        self.finish_normalization(graph, centered, variance)
    }

    /// Apply batch norm in inference mode using running statistics
    fn apply_inference_batch_norm(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        if !self.track_running_stats {
            // If no running stats, fall back to batch statistics
            return self.apply_training_batch_norm(graph, input);
        }

        // Use running statistics
        let running_mean = self.running_mean.borrow();
        let running_var = self.running_var.borrow();

        // Create nodes for running statistics
        let running_mean_node = graph.create_variable(running_mean.clone(), false);
        let running_var_node = graph.create_variable(running_var.clone(), false);

        // Step 1: Compute (x - running_mean)
        let sub_op = Box::new(Sub);
        let centered = graph
            .apply_operation(sub_op, vec![input, running_mean_node])
            .map_err(|e| format!("BatchNorm inference centering failed: {}", e))?;

        // Continue with normalization using running variance
        self.finish_normalization(graph, centered, running_var_node)
    }

    /// Complete the normalization process (common for training and inference)
    fn finish_normalization(&self, graph: &mut AutoFerroxEngine<T>, centered: NodeId, variance: NodeId) -> Result<NodeId, String> {

        // Step 1: Add epsilon for numerical stability
        let eps_scalar = FerroxN::from_f64(self.eps).unwrap();
        let add_eps_op = Box::new(AddScalar::new(eps_scalar));
        let var_plus_eps = graph
            .apply_operation(add_eps_op, vec![variance])
            .map_err(|e| format!("BatchNorm epsilon addition failed: {}", e))?;

        // Step 2: Compute sqrt(var + eps)
        let sqrt_op = Box::new(Sqrt);
        let std_dev = graph
            .apply_operation(sqrt_op, vec![var_plus_eps])
            .map_err(|e| format!("BatchNorm sqrt computation failed: {}", e))?;

        // Step 3: Normalize: (x - mean) / std
        let div_op = Box::new(Div);
        let normalized = graph
            .apply_operation(div_op, vec![centered, std_dev])
            .map_err(|e| format!("BatchNorm normalization failed: {}", e))?;

        // Step 4: Apply learnable parameters: gamma * normalized + beta
        let weight_node = self.get_parameter_node("weight")?;
        let bias_node = self.get_parameter_node("bias")?;

        // Scale with gamma (weight)
        let scale_op = Box::new(Mul);
        let scaled = graph
            .apply_operation(scale_op, vec![normalized, weight_node])
            .map_err(|e| format!("BatchNorm scaling failed: {}", e))?;

        // Shift with beta (bias)
        let shift_op = Box::new(Add::new());
        let output = graph
            .apply_operation(shift_op, vec![scaled, bias_node])
            .map_err(|e| format!("BatchNorm bias addition failed: {}", e))?;

        Ok(output)
    }

    /// Reset running statistics
    pub fn reset_running_stats(&self) -> Result<(), String> {
        {
            let mut running_mean = self.running_mean.borrow_mut();
            *running_mean = Tensor::zeros(running_mean.shape())?;
        }
        {
            let mut running_var = self.running_var.borrow_mut();
            *running_var = Tensor::ones(running_var.shape())?;
        }
        {
            let mut num_batches = self.num_batches_tracked.borrow_mut();
            *num_batches = 0;
        }
        Ok(())
    }

    /// Get the number of batches tracked
    pub fn num_batches_tracked(&self) -> usize {
        *self.num_batches_tracked.borrow()
    }

    /// Get running mean (for inspection/debugging)
    pub fn get_running_mean(&self) -> Tensor<T> {
        self.running_mean.borrow().clone()
    }

    /// Get running variance (for inspection/debugging)
    pub fn get_running_var(&self) -> Tensor<T> {
        self.running_var.borrow().clone()
    }
    /// Get the momentum value
    pub fn momentum(&self) -> f64 {
        self.momentum
    }

    /// Set the momentum value
    pub fn set_momentum(&mut self, momentum: f64) {
        self.momentum = momentum;
    }

    /// Get the epsilon value
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Whether running statistics are being tracked
    pub fn is_tracking_running_stats(&self) -> bool {
        self.track_running_stats
    }

    /// Enable or disable running statistics tracking
    pub fn set_track_running_stats(&mut self, track: bool) {
        self.track_running_stats = track;
    }
}



impl<T> Module<T> for BatchNorm<T>
where
    T: FerroxCudaF,
{
    /// Create parameter nodes in the computational graph
    fn create_parameters_in_graph(
        &self,
        engine: &mut AutoFerroxEngine<T>,
    ) -> HashMap<String, NodeId> {
        let mut param_map = HashMap::new();

        // Create weight (gamma) parameter node
        let weight_node = engine.create_variable(self.weight.data.clone(), self.weight.requires_grad);
        param_map.insert("weight".to_string(), weight_node);

        // Create bias (beta) parameter node
        let bias_node = engine.create_variable(self.bias.data.clone(), self.bias.requires_grad);
        param_map.insert("bias".to_string(), bias_node);

        // Store the node mappings for use in forward()
        *self.parameter_maps.borrow_mut() = Some(param_map.clone());

        param_map
    }

    /// Forward pass: apply batch normalization
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        self.apply_batch_norm(graph, input)
    }

    /// Collect all parameters (weight and bias)
    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![&self.weight, &self.bias]
    }

    /// Mutable access to parameters
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![&mut self.weight, &mut self.bias]
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

/// ---------------------------------------------------
/// LAYER NORM
/// ---------------------------------------------------
/// Layer Normalization: normalizes across feature dimensions for each sample
/// Normalizes input using layer statistics: (x - layer_mean) / sqrt(layer_var + eps)
/// Learnable scale (gamma) and shift (beta) parameters
/// Input shape: [..., normalized_shape] where normalized_shape is the dimensions to normalize
#[derive(Debug)]
pub struct LayerNorm<T>
where
    T: FerroxCudaF,
{
    /// Shape of the dimensions to normalize (last N dimensions)
    normalized_shape: Vec<usize>,
    /// Small epsilon for numerical stability
    eps: f64,
    /// Learnable scale parameter (gamma)
    pub weight: Parameter<T>,
    /// Learnable shift parameter (beta)
    pub bias: Parameter<T>,
    /// Training mode flag
    training: bool,
    parameter_maps: std::cell::RefCell<Option<std::collections::HashMap<String, NodeId>>>,
}

impl<T> LayerNorm<T>
where
    T: FerroxCudaF,
{
    /// Create a new LayerNorm layer
    pub fn new(normalized_shape: Vec<usize>, eps: f64, device: Device) -> Self {
        // Initialize gamma (weight) to ones
        let mut weight = Parameter::ones_with_device(&normalized_shape, device);
        weight.set_name("weight".to_string());

        // Initialize beta (bias) to zeros
        let mut bias = Parameter::zeros_with_device(&normalized_shape, device);
        bias.set_name("bias".to_string());

        Self {
            normalized_shape,
            eps,
            weight,
            bias,
            training: true,
            parameter_maps: std::cell::RefCell::new(None),
        }
    }

    /// Create LayerNorm with default epsilon
    pub fn new_default(normalized_shape: Vec<usize>, device: Device) -> Self {
        Self::new(normalized_shape, 1e-5, device)
    }

    /// Create LayerNorm for transformer-style inputs [batch, seq_len, hidden_dim]
    pub fn new_transformer(hidden_dim: usize, device: Device) -> Self {
        Self::new_default(vec![hidden_dim], device)
    }

    // Helper method to get parameter node IDs
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

    /// Apply layer normalization using existing operations
    fn apply_layer_norm(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        // Get input tensor for shape validation
        let input_tensor = graph
            .get_tensor(input)
            .ok_or("Input tensor not found in graph")?;
        let input_shape = input_tensor.shape();

        // Calculate which axes to normalize (last N dimensions)
        let ndim = input_shape.len();
        let norm_dims = self.normalized_shape.len();

        if norm_dims > ndim {
            return Err("Normalized shape has more dimensions than input".to_string());
        }

        let norm_axes: Vec<usize> = (ndim - norm_dims..ndim).collect();

        // Step 1: Compute layer mean
        let mean_op = Box::new(Mean::along_axes(norm_axes.clone(), true));
        let layer_mean = graph
            .apply_operation(mean_op, vec![input])
            .map_err(|e| format!("LayerNorm mean computation failed: {}", e))?;

        // Step 2: Compute (x - mean)
        let sub_op = Box::new(Sub);
        let centered = graph
            .apply_operation(sub_op, vec![input, layer_mean])
            .map_err(|e| format!("LayerNorm centering failed: {}", e))?;

        // Step 3: Compute variance = mean((x - mean)^2)
        let mul_op = Box::new(Mul);
        let squared_diff = graph
            .apply_operation(mul_op, vec![centered, centered])
            .map_err(|e| format!("LayerNorm variance computation failed: {}", e))?;

        let var_mean_op = Box::new(Mean::along_axes(norm_axes, true));
        let variance = graph
            .apply_operation(var_mean_op, vec![squared_diff])
            .map_err(|e| format!("LayerNorm variance mean failed: {}", e))?;

        // Step 4: Add epsilon for numerical stability
        let eps_scalar = FerroxN::from_f64(self.eps).unwrap();
        let add_eps_op = Box::new(AddScalar::new(eps_scalar));
        let var_plus_eps = graph
            .apply_operation(add_eps_op, vec![variance])
            .map_err(|e| format!("LayerNorm epsilon addition failed: {}", e))?;

        // Step 5: Compute sqrt(var + eps)
        let sqrt_op = Box::new(Sqrt);
        let std_dev = graph
            .apply_operation(sqrt_op, vec![var_plus_eps])
            .map_err(|e| format!("LayerNorm sqrt computation failed: {}", e))?;

        // Step 6: Normalize: (x - mean) / std
        let div_op = Box::new(Div);
        let normalized = graph
            .apply_operation(div_op, vec![centered, std_dev])
            .map_err(|e| format!("LayerNorm normalization failed: {}", e))?;

        // Step 7: Apply learnable parameters: gamma * normalized + beta
        let weight_node = self.get_parameter_node("weight")?;
        let bias_node = self.get_parameter_node("bias")?;

        // Scale with gamma (weight)
        let scale_op = Box::new(Mul);
        let scaled = graph
            .apply_operation(scale_op, vec![normalized, weight_node])
            .map_err(|e| format!("LayerNorm scaling failed: {}", e))?;

        // Shift with beta (bias)
        let shift_op = Box::new(Add::new());
        let output = graph
            .apply_operation(shift_op, vec![scaled, bias_node])
            .map_err(|e| format!("LayerNorm bias addition failed: {}", e))?;

        Ok(output)
    }
}

impl<T> Module<T> for LayerNorm<T>
where
    T: FerroxCudaF,
{
    /// Create parameter nodes in the computational graph
    fn create_parameters_in_graph(
        &self,
        engine: &mut AutoFerroxEngine<T>,
    ) -> HashMap<String, NodeId> {
        let mut param_map = HashMap::new();

        // Create weight (gamma) parameter node
        let weight_node = engine.create_variable(self.weight.data.clone(), self.weight.requires_grad);
        param_map.insert("weight".to_string(), weight_node);

        // Create bias (beta) parameter node
        let bias_node = engine.create_variable(self.bias.data.clone(), self.bias.requires_grad);
        param_map.insert("bias".to_string(), bias_node);

        // Store the node mappings for use in forward()
        *self.parameter_maps.borrow_mut() = Some(param_map.clone());

        param_map
    }

    /// Forward pass: apply layer normalization
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        self.apply_layer_norm(graph, input)
    }

    /// Collect all parameters (weight and bias)
    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![&self.weight, &self.bias]
    }

    /// Mutable access to parameters
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![&mut self.weight, &mut self.bias]
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


///---------------------------------------------------------
/// RMSNorm
/// --------------------------------------------------------
/// RMS Normalization: simpler variant of LayerNorm without mean centering
/// Normalizes using RMS: x / sqrt(mean(x^2) + eps) * gamma
/// Only has scale (gamma) parameter, no bias term
/// Popular in modern transformer architectures (LLaMA, etc.)
#[derive(Debug)]
pub struct RMSNorm<T>
where
    T: FerroxCudaF,
{
    /// Shape of the dimensions to normalize (last N dimensions)
    normalized_shape: Vec<usize>,
    /// Small epsilon for numerical stability
    eps: f64,
    /// Learnable scale parameter (gamma)
    pub weight: Parameter<T>,
    /// Training mode flag
    training: bool,
    parameter_maps: std::cell::RefCell<Option<std::collections::HashMap<String, NodeId>>>,
}

impl<T> RMSNorm<T>
where
    T: FerroxCudaF,
{
    /// Create a new RMSNorm layer
    pub fn new(normalized_shape: Vec<usize>, eps: f64, device: Device) -> Self {
        // Initialize gamma (weight) to ones
        let mut weight = Parameter::ones_with_device(&normalized_shape, device);
        weight.set_name("weight".to_string());

        Self {
            normalized_shape,
            eps,
            weight,
            training: true,
            parameter_maps: std::cell::RefCell::new(None),
        }
    }

    /// Create RMSNorm with default epsilon
    pub fn new_default(normalized_shape: Vec<usize>, device: Device) -> Self {
        Self::new(normalized_shape, 1e-6, device) // Smaller epsilon is typical for RMSNorm
    }

    /// Create RMSNorm for transformer-style inputs [batch, seq_len, hidden_dim]
    pub fn new_transformer(hidden_dim: usize, device: Device) -> Self {
        Self::new_default(vec![hidden_dim], device)
    }

    // Helper method to get parameter node IDs
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

    /// Apply RMS normalization using existing operations
    fn apply_rms_norm(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        // Get input tensor for shape validation
        let input_tensor = graph
            .get_tensor(input)
            .ok_or("Input tensor not found in graph")?;
        let input_shape = input_tensor.shape();

        // Calculate which axes to normalize (last N dimensions)
        let ndim = input_shape.len();
        let norm_dims = self.normalized_shape.len();

        if norm_dims > ndim {
            return Err("Normalized shape has more dimensions than input".to_string());
        }

        let norm_axes: Vec<usize> = (ndim - norm_dims..ndim).collect();

        // Step 1: Compute x^2
        let mul_op = Box::new(Mul);
        let squared = graph
            .apply_operation(mul_op, vec![input, input])
            .map_err(|e| format!("RMSNorm squaring failed: {}", e))?;

        // Step 2: Compute mean(x^2)
        let mean_op = Box::new(Mean::along_axes(norm_axes, true));
        let mean_squared = graph
            .apply_operation(mean_op, vec![squared])
            .map_err(|e| format!("RMSNorm mean squared computation failed: {}", e))?;

        // Step 3: Add epsilon for numerical stability
        let eps_scalar = FerroxN::from_f64(self.eps).unwrap();
        let add_eps_op = Box::new(AddScalar::new(eps_scalar));
        let mean_sq_plus_eps = graph
            .apply_operation(add_eps_op, vec![mean_squared])
            .map_err(|e| format!("RMSNorm epsilon addition failed: {}", e))?;

        // Step 4: Compute sqrt(mean(x^2) + eps) = RMS
        let sqrt_op = Box::new(Sqrt);
        let rms = graph
            .apply_operation(sqrt_op, vec![mean_sq_plus_eps])
            .map_err(|e| format!("RMSNorm sqrt computation failed: {}", e))?;

        // Step 5: Normalize: x / RMS
        let div_op = Box::new(Div);
        let normalized = graph
            .apply_operation(div_op, vec![input, rms])
            .map_err(|e| format!("RMSNorm normalization failed: {}", e))?;

        // Step 6: Apply learnable scale parameter: gamma * normalized
        let weight_node = self.get_parameter_node("weight")?;

        let scale_op = Box::new(Mul);
        let output = graph
            .apply_operation(scale_op, vec![normalized, weight_node])
            .map_err(|e| format!("RMSNorm scaling failed: {}", e))?;

        Ok(output)
    }
}

impl<T> Module<T> for RMSNorm<T>
where
    T: FerroxCudaF,
{
    /// Create parameter nodes in the computational graph
    fn create_parameters_in_graph(
        &self,
        engine: &mut AutoFerroxEngine<T>,
    ) -> HashMap<String, NodeId> {
        let mut param_map = HashMap::new();

        // Create weight (gamma) parameter node
        let weight_node = engine.create_variable(self.weight.data.clone(), self.weight.requires_grad);
        param_map.insert("weight".to_string(), weight_node);

        // Store the node mappings for use in forward()
        *self.parameter_maps.borrow_mut() = Some(param_map.clone());

        param_map
    }

    /// Forward pass: apply RMS normalization
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        self.apply_rms_norm(graph, input)
    }

    /// Collect all parameters (only weight for RMSNorm)
    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![&self.weight]
    }

    /// Mutable access to parameters
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![&mut self.weight]
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

impl<T> RMSNorm<T>
where
    T: FerroxCudaF,
{
    /// Get the normalized shape
    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }

    /// Get the epsilon value
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Check if this is the same configuration as another RMSNorm
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.normalized_shape == other.normalized_shape &&
        (self.eps - other.eps).abs() < 1e-10
    }
}
