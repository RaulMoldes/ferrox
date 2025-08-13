// examples/mlp_regressor_comparison.rs
// Complete MLP regressor example comparing different backends with loss functions
// Demonstrates the neural network library capabilities for regression tasks

use ferrox::backend::manager::best_f32_device;
use ferrox::backend::{Device, FerroxCudaF, Tensor};
use ferrox::dataset::{BatchedDataset, Dataset, TensorDataset};
use ferrox::graph::{AutoFerroxEngine, NodeId};
use ferrox::nn::{
    layers::{Linear, ReLU},
    losses::{L1Loss, Loss, MSELoss, ReductionType},
    optim::{Adam, Optim, Optimizer, SGD},
    Module,
};
use ferrox::FerroxN;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::time::Instant;

/// Multi-Layer Perceptron for regression tasks
/// Demonstrates composition of neural network layers using the Module trait
#[derive(Debug)]
pub struct MLPRegressor<T>
where
    T: FerroxCudaF,
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
    T: FerroxCudaF,
{
    /// Create new MLP regressor with specified architecture
    /// Uses proper Xavier initialization and conservative architecture
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, device: Device) -> Self {
        Self {
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

    /// Override create_parameters_in_graph to properly handle nested layers
    fn create_parameters_in_graph(
        &self,
        engine: &mut AutoFerroxEngine<T>,
    ) -> HashMap<String, ferrox::graph::NodeId> {
        let mut param_map = HashMap::new();

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

/// Training configuration supporting multiple optimizers with common fields
/// Designed to work with pattern matching for optimizer creation
#[derive(Debug, Clone)]
pub struct TrainingConfig<T>
where
    T: FerroxCudaF,
{
    // Common training parameters shared across all optimizers
    pub batch_size: usize,
    pub num_epochs: usize,
    pub learning_rate: Option<T>,
    pub print_every: usize,

    // Optimizer selection and parameters
    pub optimizer: &'static str,

    // SGD-specific parameters
    pub momentum: Option<T>,
    pub nesterov: bool,
    pub decay: Option<T>,

    // Adam-specific parameters
    pub beta1: Option<T>,
    pub beta2: Option<T>,
    pub eps: Option<T>,
    pub amsgrad: bool,
}

impl<T> Default for TrainingConfig<T>
where
    T: FerroxCudaF + FerroxN,
{
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_epochs: 100,
            learning_rate: Some(FerroxN::from_f32(0.0001).unwrap()), // Small learning rate for stability
            print_every: 1,
            optimizer: "SGD",
            // SGD defaults
            momentum: Some(FerroxN::from_f32(0.0).unwrap()),
            nesterov: false,
            decay: Some(FerroxN::from_f32(0.0001).unwrap()),
            // Adam defaults
            beta1: Some(FerroxN::from_f32(0.9).unwrap()),
            beta2: Some(FerroxN::from_f32(0.999).unwrap()),
            eps: Some(FerroxN::from_f32(1e-8).unwrap()),
            amsgrad: false,
        }
    }
}

/// Generate a synthetic regression dataset for testing
/// Creates a simple linear relationship to avoid complexity: y = 0.3*x1 + 0.1*x2 + small_noise
fn generate_data<T>(
    num_samples: usize,
    input_size: usize,
    device: Device,
) -> Result<TensorDataset<T>, String>
where
    T: FerroxCudaF,
{
    let mut rng = rand::rng();
    let normal = Normal::new(0.0, 0.5).unwrap(); // Normal distribution to generate white noise.

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
    // Initialize the data. It is recommended to initialize the data on CPU as it will be moved later by the batched dataset to the gpu.
    let inputs = Tensor::from_vec_with_device(input_data, &[num_samples, input_size], device)?;
    let targets = Tensor::from_vec_with_device(target_data, &[num_samples, 1], device)?;

    TensorDataset::from_tensor(inputs, targets)
}

fn with_benchmark<F, R>(func: F) -> impl FnOnce() -> R
where
    F: FnOnce() -> R,
{
    move || {
        let start_time = Instant::now();
        let result = func(); // Execute the wrapped function with its own captured parameters
        let elapsed = start_time.elapsed();

        println!("[INFO] Execution time: {:?}", elapsed);
        result
    }
}

fn initialize_model<M, T, O>(
    model: &M,
    graph: &mut AutoFerroxEngine<T>,
    opt: &mut O,
) -> Result<(), String>
where
    T: FerroxCudaF,
    O: Optimizer<T>,
    M: Module<T>,
{
    let param_map = model.create_parameters_in_graph(graph);
    for (_, p) in param_map {
        opt.add_param(0, p);
    }
    Ok(())
}
/// Run forward pass and return predictions, prediction shape, and number of samples
fn run_forward<M, T>(
    model: &M,
    graph: &mut AutoFerroxEngine<T>,
    inputs: &Tensor<T>,
) -> Result<(NodeId, Vec<usize>, usize), String>
where
    M: Module<T>,
    T: FerroxCudaF,
{
    let input_node = graph.create_variable(inputs.clone(), false);
    let predictions = model.forward(graph, input_node)?;

    let pred_tensor = graph
        .get_tensor(predictions)
        .ok_or("Predictions tensor not found")?;
    let pred_shape = pred_tensor.shape().to_vec();
    let num_samples = inputs.shape()[0];

    Ok((predictions, pred_shape, num_samples))
}

/// Compute a loss value (MSE, etc.)
fn compute_loss<T, L>(
    graph: &mut AutoFerroxEngine<T>,
    loss_fn: L,
    predictions: NodeId,
    targets: &Tensor<T>,
) -> Result<T, String>
where
    T: FerroxCudaF,
    L: Loss<T>,
{
    let target_node = graph.create_variable(targets.clone(), false);
    let loss_node = loss_fn.forward(graph, predictions, target_node)?;

    let loss_tensor = graph.get_tensor(loss_node).ok_or("Loss tensor not found")?;
    loss_tensor.clone().first()
}

#[allow(dead_code)]
struct MLPConfig {
    input_size: u32,
    hidden_size: u32,
    output_size: u8,
    num_samples: u64,
}
#[allow(dead_code)]
impl MLPConfig {
    fn new(input_size: u32, hidden_size: u32, output_size: u8, num_samples: u64) -> Self {
        Self {
            input_size,
            hidden_size, // Reduced from 16 for stability
            output_size,
            num_samples,
        }
    }
}

impl Default for MLPConfig {
    fn default() -> Self {
        Self {
            input_size: 4,
            hidden_size: 8, // Reduced from 16 for stability
            output_size: 1,
            num_samples: 500,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn epoch<T, M, O, L>(
    graph: &mut AutoFerroxEngine<T>,
    optimizer: &mut O,
    model: &mut M,
    loss_fn: &L,
    input_node: NodeId,
    target_node: NodeId,
    loss_history: &mut Vec<T>,
    epoch: usize,
) -> Result<(), String>
where
    T: FerroxCudaF,
    M: Module<T>,
    O: Optimizer<T>,
    L: Loss<T>,
{
    // Forward pass through model
    let predictions = model.forward(graph, input_node)?;

    // Compute loss using the loss function
    let loss_node = loss_fn.forward(graph, predictions, target_node)?;

    // Get loss value for tracking
    let loss_tensor = graph.get_tensor(loss_node).ok_or("Loss tensor not found")?;
    let loss_value = loss_tensor.clone().first()?;

    // Check for NaN or inf early and stop training if detected
    if loss_value.is_nan() || loss_value.is_infinite() {
        println!("[WARNING]: Training became unstable at epoch {}", epoch + 1);
    }

    loss_history.push(loss_value);

    // Backward pass - compute gradients
    graph.backward(loss_node)?;

    // Update parameters using optimizer
    optimizer.step(graph).map_err(|e| e.to_string())?;

    // Clear gradients for next iteration
    graph.zero_gradients();
    Ok(())
}

fn train<T, M, L>(
    model: &mut M,
    loss_fn: &L,
    data: BatchedDataset<T>,
    config: &TrainingConfig<T>,
) -> Result<AutoFerroxEngine<T>, String>
where
    T: FerroxCudaF,
    M: Module<T>,
    L: Loss<T>,
{
    // Create new computational graph for training
    let mut graph = AutoFerroxEngine::new();

    // Initialize optimizer based on configuration
    let mut optimizer = match config.optimizer {
        "SGD" => Optim::SGD(SGD::new(
            config.learning_rate.expect("Lr not provided"),
            config.momentum.expect("Momentum not provided"),
            config.decay.expect("Decay not provided"),
            config.nesterov,
        )),
        "Adam" => Optim::Adam(Adam::new(
            config.learning_rate.expect("Learning rate not provided"),
            config.beta1.expect("Beta1 not provided"),
            config.beta2.expect("Beta2 not provided"),
            config.eps.expect("Eps not provided"),
            config.decay.expect("Weight decay not provided"),
            config.amsgrad,
        )),
        _ => panic!("Invalid optimizer name! Must be one of Adam or SGD"),
    };

    // Initialize model parameters in the computational graph
    initialize_model(model, &mut graph, &mut optimizer)?;

    // Pre-allocate loss history for better performance
    let mut loss_history: Vec<T> = Vec::with_capacity(config.num_epochs);

    // Training loop - iterate through epochs
    for ep in 0..config.num_epochs {
        for (inputs, targets) in &data {
            // Create variable nodes for inputs and targets (no gradients needed for data)
            let input_node = graph.create_variable(inputs.clone(), false);
            let target_node = graph.create_variable(targets.clone(), false);

            // Execute single epoch with benchmarking wrapper
            with_benchmark(|| {
                epoch(
                    &mut graph,
                    &mut optimizer,
                    model,
                    loss_fn,
                    input_node,
                    target_node,
                    &mut loss_history,
                    ep,
                )
            })()?;

            // Print progress at specified intervals
            if ep % config.print_every == 0 {
                if let Some(&current_loss) = loss_history.last() {
                    println!(
                        "[EPOCH {}] Loss: {:.6}",
                        ep,
                        <T as FerroxN>::to_f64(current_loss)
                    );
                }
            }
        }
    }

    Ok(graph)
}

fn eval<T, M>(
    model: &mut M,
    graph: &mut AutoFerroxEngine<T>,
    data: TensorDataset<T>,
) -> Result<(), String>
where
    T: FerroxCudaF,
    M: Module<T>,
{
    // Switch model to evaluation mode (disables dropout, batch norm updates, etc.)
    model.eval();
    let (test_inputs, test_targets) = data.get_item(0)?;
    // Run forward pass through the model

    let (predictions, pred_shape, num_samples) = run_forward(model, graph, &test_inputs)?;

    // Compute L1 loss (Mean Absolute Error)
    let l1_loss = L1Loss::new(ReductionType::Mean);
    let l1_value = compute_loss(graph, l1_loss, predictions, &test_targets)?;

    // Compute MSE loss (Mean Squared Error)
    let mse_loss = MSELoss::new(ReductionType::Mean);
    let mse_value = compute_loss(graph, mse_loss, predictions, &test_targets)?;

    // Display comprehensive evaluation results
    println!("=== Model Evaluation Results ===");
    println!("Evaluated on {} samples", num_samples);
    println!("Prediction shape: {:?}", pred_shape);
    println!("MSE Loss: {:.6}", <T as FerroxN>::to_f64(mse_value));
    println!("L1 Loss (MAE): {:.6}", <T as FerroxN>::to_f64(l1_value));

    Ok(())
}

fn main() -> Result<(), String> {
    println!("--- Model Training Demo ---");
    let config = MLPConfig::default();
    let training_config = TrainingConfig::default();
    let device = best_f32_device();
    let mut model = MLPRegressor::<f32>::new(
        config.input_size as usize,
        config.hidden_size as usize,
        config.output_size as usize,
        device,
    );
    let loss_fn = L1Loss::<f32>::new(ReductionType::Mean);

    let train_data = generate_data::<f32>(10000, 4, Device::CPU)?; // Train data is generated on cpu and automatically transferred after batching

    let test_data = generate_data::<f32>(100, 4, device)?;

    let mut trained_autograd = train(
        &mut model,
        &loss_fn,
        train_data.into_batches(1000, false)?,
        &training_config,
    )?;

    println!("Model trained finished");
    eval(&mut model, &mut trained_autograd, test_data)?;
    println!("Example completed successfully!");
    Ok(())
}
