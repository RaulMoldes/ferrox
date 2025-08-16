// src/nn/layers/normalization.rs
// Normalization layers implemented using existing graph operations
// BatchNorm, LayerNorm, and RMSNorm for stable training and better convergence

use crate::backend::number::FerroxCudaF;
use crate::backend::{Device, Tensor};
use crate::graph::{AutoFerroxEngine, NodeId};
use crate::nn::parameter::Parameter;
use crate::nn::Module;
use crate::ops::BroadcastTo;
use crate::ops::{
    basic::{Add, Mul, Sub},
    reduction::Mean,
    scalar::AddScalar,
    shape::Reshape,
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
    pub fn new(
        num_features: usize,
        eps: f64,
        momentum: f64,
        track_running_stats: bool,
        device: Device,
    ) -> Self {
        // Initialize gamma (weight) to ones
        let mut weight = Parameter::ones_with_device(&[num_features], device);
        weight.set_name("weight".to_string());

        // Initialize beta (bias) to zeros
        let mut bias = Parameter::zeros_with_device(&[num_features], device);
        bias.set_name("bias".to_string());

        // Initialize running statistics
        let running_mean = RefCell::new(
            Tensor::zeros_with_device(&[num_features], device)
                .expect("Failed to create running_mean tensor"),
        );
        let running_var = RefCell::new(
            Tensor::ones_with_device(&[num_features], device)
                .expect("Failed to create running_var tensor"),
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
    fn update_running_stats(
        &self,
        batch_mean: &Tensor<T>,
        batch_var: &Tensor<T>,
    ) -> Result<(), String> {
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
    fn apply_batch_norm(
        &self,
        graph: &mut AutoFerroxEngine<T>,
        input: NodeId,
    ) -> Result<NodeId, String> {
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
    fn apply_training_batch_norm(
        &self,
        graph: &mut AutoFerroxEngine<T>,
        input: NodeId,
    ) -> Result<NodeId, String> {
        let norm_axes = {
            let input_shape = graph
                .get_node_shape(&input)
                .ok_or("Input tensor not found in graph")?;

            // For BatchNorm, we normalize across batch dimension (axis 0)
            // and optionally spatial dimensions for 4D tensors
            let axes: Vec<usize> = if input_shape.len() == 4 {
                vec![0, 2, 3] // Batch, Height, Width for [N, C, H, W]
            } else {
                vec![0] // Just batch dimension for [N, F]
            };

            axes
        };

        // Step 1: Compute batch mean
        let mean_op = Box::new(Mean::along_axes(norm_axes.clone(), false));
        let batch_mean = graph
            .apply_operation(mean_op, vec![input])
            .map_err(|e| format!("BatchNorm mean computation failed: {}", e))?;

        let mean_broadcasted = reshape_and_broadcast(batch_mean, input, graph)?;

        // Step 2: Compute (x - mean)
        let sub_op = Box::new(Sub);
        let centered = graph
            .apply_operation(sub_op, vec![input, mean_broadcasted])
            .map_err(|e| format!("BatchNorm centering failed: {}", e))?;

        // Step 3: Compute variance = mean((x - mean)^2)
        let mul_op = Box::new(Mul);
        let squared_diff = graph
            .apply_operation(mul_op, vec![centered, centered])
            .map_err(|e| format!("BatchNorm variance computation failed: {}", e))?;

        let var_mean_op = Box::new(Mean::along_axes(norm_axes.clone(), false));
        let variance = graph
            .apply_operation(var_mean_op, vec![squared_diff])
            .map_err(|e| format!("BatchNorm variance mean failed: {}", e))?;

        // Reshape variance from [C] to [1, C, 1, 1] for proper broadcasting in finish_normalization

        let bc_variance = reshape_and_broadcast(variance, input, graph)?;

        // Update running statistics (this happens outside the graph)
        if self.track_running_stats {
            // Extract tensors from graph for running stats update
            let batch_mean_tensor = graph
                .get_tensor(batch_mean)
                .ok_or("Could not get batch mean tensor")?;
            let variance_tensor = graph
                .get_tensor(variance)
                .ok_or("Could not get variance tensor")?;

            // Update running statistics
            self.update_running_stats(batch_mean_tensor, variance_tensor)
                .map_err(|e| format!("Failed to update running stats: {}", e))?;
        }

        // Continue with normalization using reshaped variance
        self.finish_normalization(graph, centered, bc_variance)
    }
    /// Apply batch norm in inference mode using running statistics
    fn apply_inference_batch_norm(
        &self,
        graph: &mut AutoFerroxEngine<T>,
        input: NodeId,
    ) -> Result<NodeId, String> {
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


        let reshaped_running_avg = reshape_and_broadcast(running_mean_node, input, graph)?;

        // Step 1: Compute (x - running_mean)
        let sub_op = Box::new(Sub);
        let centered = graph
            .apply_operation(sub_op, vec![input, reshaped_running_avg])
            .map_err(|e| format!("BatchNorm inference centering failed: {}", e))?;

        // Continue with normalization using running variance
        self.finish_normalization(graph, centered, running_var_node)
    }

    /// Complete the normalization process (common for training and inference)
    fn finish_normalization(
        &self,
        graph: &mut AutoFerroxEngine<T>,
        centered: NodeId,
        variance: NodeId,
    ) -> Result<NodeId, String> {
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

        let normalized = reshape_and_broadcast(std_dev, centered, graph)?;
        // Step 4: Apply learnable parameters: gamma * normalized + beta
        let w_node = self.get_parameter_node("weight")?;

        let broadcasted_w = reshape_and_broadcast(w_node, normalized, graph)?;
        // Scale with gamma (weight)
        let scale_op = Box::new(Mul);
        let scaled = graph
            .apply_operation(scale_op, vec![normalized, broadcasted_w])
            .map_err(|e| format!("BatchNorm scaling failed: {}", e))?;

        let b_node = self.get_parameter_node("bias")?;
        let broadcasted_b = reshape_and_broadcast(b_node, normalized, graph)?;
        // Shift with beta (bias)
        let shift_op = Box::new(Add::new());
        let output = graph
            .apply_operation(shift_op, vec![scaled, broadcasted_b])
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
        let weight_node =
            engine.create_variable(self.weight.data.clone(), self.weight.requires_grad);
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

fn reshape_and_broadcast<T>(
    input: NodeId,
    target: NodeId,
    graph: &mut AutoFerroxEngine<T>,
) -> Result<NodeId, String>
where
    T: FerroxCudaF,
{
    let target_shape = graph
        .get_node_shape(&target)
        .expect("Failed to get target shape for broadcasting")
        .to_vec();

    let input_shape = graph
        .get_node_shape(&input)
        .expect("Failed to get input shape for broadcasting")
        .to_vec();
    let res = if input_shape.len() == 1 {
        let new_shape = if target_shape.len() == 4 {
            vec![1, input_shape[0], 1, 1]
        } else {
            vec![1, input_shape[0]]
        };

        let reshape = Box::new(Reshape::new(new_shape));
        graph.apply_operation(reshape, vec![input])?
    } else {
        input
    };

    if input_shape != target_shape {
        let bcop = Box::new(BroadcastTo::<T>::new(target_shape));
        graph.apply_operation(bcop, vec![res])
    } else {
        Ok(res)
    }
}
