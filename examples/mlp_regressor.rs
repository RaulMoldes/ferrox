// examples/mlp_regressor_comparison.rs
// Complete MLP regressor example comparing different backends with loss functions
// Demonstrates the neural network library capabilities for regression tasks

use ferrox::backend::{best_device, Device, FerroxCudaF, Tensor};
use ferrox::graph::AutoFerroxEngine;
use ferrox::nn::{
    layers::{Linear, ReLU},
    losses::{L1Loss, Loss, MSELoss, ReductionType},
    optim::{Optimizer, SGD},
    Module,
};
use ferrox::FerroxF;
use ferrox::FerroxN;
use rand_distr::{Distribution, Normal};
use std::time::Instant;

/// Multi-Layer Perceptron for regression tasks
/// Demonstrates composition of neural network layers using the Module trait
#[derive(Debug)]
pub struct MLPRegressor<T>
where
    T: FerroxCudaF + rand_distr::num_traits::FromPrimitive,
{
    /// First hidden layer: input_size -> hidden_size
    hidden1: Linear<T>,
    /// First activation function
    activation1: ReLU<T>,
    /// Second hidden layer: hidden_size -> hidden_size
    hidden2: Linear<T>,
    /// Second activation function
    activation2: ReLU<T>,
    /// Output layer: hidden_size -> output_size
    output: Linear<T>,
    /// Training mode flag
    training: bool,
}

impl<T> MLPRegressor<T>
where
    T: FerroxCudaF + rand_distr::num_traits::FromPrimitive,
{
    /// Create new MLP regressor with specified architecture
    /// Uses proper Xavier initialization and conservative architecture
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, device: Device) -> Self {
        Self {
            // CRITICAL FIX: Use new_with_device instead of new, and smaller hidden layer
            hidden1: Linear::new_with_device(input_size, hidden_size, true, device),
            activation1: ReLU::new(),
            hidden2: Linear::new_with_device(hidden_size, hidden_size, true, device), // Add bias to hidden2
            activation2: ReLU::new(),
            output: Linear::new_with_device(hidden_size, output_size, false, device), // No bias on output
            training: true,
        }
    }

    /// Get all trainable parameters for optimizer registration
    pub fn get_parameters(&self) -> Vec<&ferrox::nn::parameter::Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.hidden1.parameters());
        params.extend(self.hidden2.parameters());
        params.extend(self.output.parameters());
        params
    }
}

impl<T> Module<T> for MLPRegressor<T>
where
    T: FerroxCudaF + rand_distr::num_traits::FromPrimitive,
{
    /// Forward pass through the network
    fn forward(
        &self,
        graph: &mut AutoFerroxEngine<T>,
        input: ferrox::graph::NodeId,
    ) -> Result<ferrox::graph::NodeId, String> {
        // Layer 1: Linear -> ReLU
        let hidden1_out = self.hidden1.forward(graph, input)?;
        let activated1 = self.activation1.forward(graph, hidden1_out)?;

        // Layer 2: Linear -> ReLU
        let hidden2_out = self.hidden2.forward(graph, activated1)?;
        let activated2 = self.activation2.forward(graph, hidden2_out)?;

        // Output layer (no activation for regression)
        let output = self.output.forward(graph, activated2)?;
        Ok(output)
    }

    /// CRITICAL FIX: Override create_parameters_in_graph to properly handle nested layers
    fn create_parameters_in_graph(
        &self,
        engine: &mut AutoFerroxEngine<T>,
    ) -> std::collections::HashMap<String, ferrox::graph::NodeId> {
        let mut param_map = std::collections::HashMap::new();

        // Create parameters for each layer individually
        // This ensures each Linear layer gets its parameter nodes created properly

        // Hidden layer 1 parameters
        let hidden1_params = self.hidden1.create_parameters_in_graph(engine);
        for (param_name, node_id) in hidden1_params {
            param_map.insert(format!("hidden1_{}", param_name), node_id);
        }

        // Hidden layer 2 parameters
        let hidden2_params = self.hidden2.create_parameters_in_graph(engine);
        for (param_name, node_id) in hidden2_params {
            param_map.insert(format!("hidden2_{}", param_name), node_id);
        }

        // Output layer parameters
        let output_params = self.output.create_parameters_in_graph(engine);
        for (param_name, node_id) in output_params {
            param_map.insert(format!("output_{}", param_name), node_id);
        }

        param_map
    }

    fn parameters(&self) -> Vec<&ferrox::nn::parameter::Parameter<T>> {
        self.get_parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut ferrox::nn::parameter::Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.hidden1.parameters_mut());
        params.extend(self.hidden2.parameters_mut());
        params.extend(self.output.parameters_mut());
        params
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.hidden1.set_training(training);
        self.activation1.set_training(training);
        self.hidden2.set_training(training);
        self.activation2.set_training(training);
        self.output.set_training(training);
    }
}

/// Training configuration for the MLP regressor
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub num_epochs: usize,
    pub learning_rate: f32,
    pub print_every: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_epochs: 100,
            learning_rate: 0.0001, // Even smaller learning rate for stability
            print_every: 1,
        }
    }
}

/// Generate synthetic regression dataset for testing
/// Creates a simple linear relationship to avoid complexity: y = 0.3*x1 + 0.1*x2 + small_noise
fn generate_regression_data<T>(
    num_samples: usize,
    input_size: usize,
    device: Device,
) -> Result<(Tensor<T>, Tensor<T>), String>
where
    T: FerroxCudaF + rand_distr::num_traits::FromPrimitive,
{
    let mut rng = rand::rng();
    let normal = Normal::new(0.0, 0.5).unwrap(); // Reduced variance for stability

    // Generate random input features with smaller range
    let mut input_data = Vec::with_capacity(num_samples * input_size);
    for _ in 0..(num_samples * input_size) {
        let value = normal.sample(&mut rng) as f64;
        input_data.push(<T as FerroxN>::from_f64(value).ok_or("Failed to convert input data")?);
    }

    // Generate corresponding targets with simple linear relationship
    let mut target_data = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let x1 = <T as FerroxN>::to_f64(input_data[i * input_size]);
        let x2 = if input_size > 1 {
            <T as FerroxN>::to_f64(input_data[i * input_size + 1])
        } else {
            0.0
        };

        // Simple linear relationship with minimal noise
        let noise = normal.sample(&mut rng) * 0.05; // Very small noise
        let target = 0.3 * x1 + 0.1 * x2 + noise;
        target_data.push(<T as FerroxN>::from_f64(target).ok_or("Failed to convert target data")?);
    }

    let inputs = Tensor::from_vec_with_device(input_data, &[num_samples, input_size], device)?;
    let targets = Tensor::from_vec_with_device(target_data, &[num_samples, 1], device)?;

    Ok((inputs, targets))
}

/// Train the MLP regressor with specified loss function and gradient clipping
fn train_mlp_with_loss<T, L>(
    model: &mut MLPRegressor<T>,
    loss_fn: &L,
    inputs: &Tensor<T>,
    targets: &Tensor<T>,
    config: &TrainingConfig,
    device_name: &str,
) -> Result<Vec<T>, String>
where
    T: FerroxCudaF + rand_distr::num_traits::FromPrimitive,
    L: Loss<T>,
{
    let mut graph = AutoFerroxEngine::new();
    let mut optimizer = SGD::new(
        <T as FerroxN>::from_f32(config.learning_rate).unwrap(),
        <T as FerroxN>::from_f32(0.0).unwrap(),   // No momentum for stability
        <T as FerroxN>::from_f32(0.0001).unwrap(), // Small weight decay
        false,
    );

    // Register model parameters with optimizer
    let param_map = model.create_parameters_in_graph(&mut graph);
    for (_, param_node) in param_map {
        optimizer.add_param(0, param_node);
    }

    let mut loss_history: Vec<T> = Vec::new();
    println!(
        "Training MLP on {} with {} loss (lr: {}):",
        device_name,
        match loss_fn.reduction() {
            ReductionType::Mean => "MSE",
            _ => "L1",
        },
        config.learning_rate
    );

    for epoch in 0..config.num_epochs {
        let start_time = Instant::now();

        // Create input and target nodes in computational graph
        let input_node = graph.create_variable(inputs.clone(), false);
        let target_node = graph.create_variable(targets.clone(), false);

        // Forward pass through model
        let predictions = model.forward(&mut graph, input_node)?;

        // Compute loss using the loss function
        let loss_node = loss_fn.forward(&mut graph, predictions, target_node)?;

        // Get loss value for tracking
        let loss_tensor = graph.get_tensor(loss_node).ok_or("Loss tensor not found")?;
        let loss_value = loss_tensor.clone().first()?;

        // Check for NaN or inf early and stop training if detected
        if loss_value.is_nan() || loss_value.is_infinite() {
            println!("WARNING: Training became unstable at epoch {}. Stopping.", epoch + 1);
            break;
        }

        loss_history.push(loss_value);

        // Backward pass - compute gradients
        graph.backward(loss_node)?;

        // Update parameters using optimizer
        optimizer.step(&mut graph).map_err(|e| e.to_string())?;

        // Clear gradients for next iteration
        graph.zero_gradients();

        let epoch_time = start_time.elapsed();

        // Print progress
        if epoch % config.print_every == 0 || epoch == config.num_epochs - 1 {
            println!(
                "Epoch {}/{}: Loss = {:.6}, Time = {:.2}ms",
                epoch + 1,
                config.num_epochs,
                <T as FerroxN>::to_f32(loss_value),
                epoch_time.as_millis()
            );
        }
    }

    Ok(loss_history)
}

/// Compare MLP performance across different backends and loss functions - FIXED VERSION
fn compare_backends_and_losses() -> Result<(), String> {
    println!("=== MLP Regressor Backend & Loss Comparison ===\n");

    let input_size = 4;
    let hidden_size = 8; // Reduced from 16 for stability
    let output_size = 1;
    let num_samples = 500; // Reduced dataset size

    // Test different devices with conservative learning rates
    let devices = vec![
        (Device::CPU, "CPU", 0.0001f32),
        (best_device::<f32>(), "Best Available", 0.0001f32),
    ];

    for (device, device_name, learning_rate) in devices {
        println!("--- Testing on {} ---", device_name);

        // Conservative configuration for both devices
        let config = TrainingConfig {
            batch_size: 32,
            num_epochs: 30, // Reduced epochs
            learning_rate,
            print_every: 5,
        };

        // Generate dataset for this device
        let (train_inputs, train_targets) =
            generate_regression_data::<f32>(num_samples, input_size, device)?;

        // Test MSE Loss
        {
            let mut model = MLPRegressor::<f32>::new(input_size, hidden_size, output_size, device);
            let mse_loss = MSELoss::<f32>::default();

            let start_time = Instant::now();
            let mse_history = train_mlp_with_loss(
                &mut model,
                &mse_loss,
                &train_inputs,
                &train_targets,
                &config,
                device_name,
            )?;
            let total_time = start_time.elapsed();

            println!("MSE Training completed in {:.2}s", total_time.as_secs_f64());

            // FIXED: Proper type handling for final loss check
            if let Some(&final_loss) = mse_history.last() {
                if final_loss.is_nan() {
                    println!("WARNING: Final MSE Loss is NaN! Training unstable.");
                } else {
                    println!("Final MSE Loss: {:.6}", final_loss);
                }
            } else {
                println!("No loss history recorded for MSE training.");
            }
        }

        // Test L1 Loss
        {
            let mut model = MLPRegressor::<f32>::new(input_size, hidden_size, output_size, device);
            let l1_loss = L1Loss::<f32>::default();

            let start_time = Instant::now();
            let l1_history = train_mlp_with_loss(
                &mut model,
                &l1_loss,
                &train_inputs,
                &train_targets,
                &config,
                device_name,
            )?;
            let total_time = start_time.elapsed();

            println!("L1 Training completed in {:.2}s", total_time.as_secs_f64());

            // FIXED: Proper type handling for final loss check
            if let Some(&final_loss) = l1_history.last() {
                if final_loss.is_nan() {
                    println!("WARNING: Final L1 Loss is NaN! Training unstable.");
                } else {
                    println!("Final L1 Loss: {:.6}", final_loss);
                }
            } else {
                println!("No loss history recorded for L1 training.");
            }
        }

        println!();
    }

    println!("=== Comparison Complete ===");
    Ok(())
}

/// Demonstrate model evaluation on test data - FIXED VERSION
fn evaluate_model<T>(
    model: &MLPRegressor<T>,
    test_inputs: &Tensor<T>,
    test_targets: &Tensor<T>,
) -> Result<(), String>
where
    T: FerroxCudaF + rand_distr::num_traits::FromPrimitive,
{
    let mut graph = AutoFerroxEngine::new();

    // CRITICAL FIX: Initialize model parameters in the new graph
    let _param_map = model.create_parameters_in_graph(&mut graph);

    // Create input and target nodes
    let input_node = graph.create_variable(test_inputs.clone(), false);
    let target_node = graph.create_variable(test_targets.clone(), false);

    // Forward pass
    let predictions = model.forward(&mut graph, input_node)?;

    // Store prediction info before computing losses
    let (pred_shape, num_samples) = {
        let pred_tensor = graph
            .get_tensor(predictions)
            .ok_or("Predictions tensor not found")?;
        (pred_tensor.shape().to_vec(), test_inputs.shape()[0])
    };

    // Compute MSE loss
    let mse_loss = MSELoss::default();
    let mse_loss_node = mse_loss.forward(&mut graph, predictions, target_node)?;

    // Get MSE value and store it
    let mse_value = {
        let mse_tensor = graph
            .get_tensor(mse_loss_node)
            .ok_or("MSE loss tensor not found")?;
        mse_tensor.clone().first()?
    };

    // For L1 loss, we need fresh nodes since the previous ones were consumed
    let input_node_2 = graph.create_variable(test_inputs.clone(), false);
    let target_node_2 = graph.create_variable(test_targets.clone(), false);
    let predictions_2 = model.forward(&mut graph, input_node_2)?;

    // Calculate L1 loss
    let l1_loss = L1Loss::default();
    let l1_loss_node = l1_loss.forward(&mut graph, predictions_2, target_node_2)?;

    let l1_value = {
        let l1_tensor = graph
            .get_tensor(l1_loss_node)
            .ok_or("L1 loss tensor not found")?;
        l1_tensor.clone().first()?
    };

    // Print evaluation results
    println!("=== Model Evaluation Results ===");
    println!("Evaluated on {} samples", num_samples);
    println!("Prediction shape: {:?}", pred_shape);
    println!("MSE Loss: {:.6}", <T as FerroxN>::to_f64(mse_value));
    println!("L1 Loss (MAE): {:.6}", <T as FerroxN>::to_f64(l1_value));

    // Additional metrics for regression evaluation
    let rmse = <T as FerroxN>::to_f64(mse_value).sqrt();
    println!("RMSE: {:.6}", rmse);

    Ok(())
}

fn main() -> Result<(), String> {
    println!("Starting MLP Regressor with Loss Functions Example");

    // Run the backend and loss comparison
    compare_backends_and_losses()?;

    // Demonstrate evaluation
    println!("\n--- Model Evaluation Demo ---");
    let device = best_device::<f32>();

    println!("Best device is : {}", device);
    let (test_inputs, test_targets) = generate_regression_data::<f32>(100, 4, device)?;
    let model = MLPRegressor::new(4, 8, 1, device);

    evaluate_model(&model, &test_inputs, &test_targets)?;

    println!("Example completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_creation() {
        let device = Device::CPU;
        let model = MLPRegressor::<f32>::new(4, 8, 1, device);

        // Verify model structure - should have 4 parameters now (3 weights + 1 bias)
        assert_eq!(model.parameters().len(), 4);
    }

    #[test]
    fn test_data_generation() {
        let device = Device::CPU;
        let (inputs, targets) = generate_regression_data::<f32>(100, 4, device).unwrap();

        assert_eq!(inputs.shape(), &[100, 4]);
        assert_eq!(targets.shape(), &[100, 1]);
    }

    #[test]
    fn test_forward_pass() {
        let device = Device::CPU;
        let mut graph = AutoFerroxEngine::new();
        let model = MLPRegressor::<f32>::new(4, 8, 1, device);

        // CRITICAL: Initialize parameters in graph for test
        let _param_map = model.create_parameters_in_graph(&mut graph);

        // Create dummy input
        let input_data = vec![0.1f32; 4]; // Smaller values for stability
        let input_tensor = Tensor::from_vec_with_device(input_data, &[1, 4], device).unwrap();
        let input_node = graph.create_variable(input_tensor, false);

        // Test forward pass
        let output = model.forward(&mut graph, input_node).unwrap();
        let output_tensor = graph.get_tensor(output).unwrap();

        assert_eq!(output_tensor.shape(), &[1, 1]);
    }
}
