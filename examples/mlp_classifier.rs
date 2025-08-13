// examples/mlp_classifier.rs
// Complete MLP classifier example comparing different backends with loss functions
// Demonstrates the neural network library capabilities for classification tasks

use ferrox::backend::manager::best_f32_device;
use ferrox::backend::{Device, FerroxCudaF, Tensor};
use ferrox::dataset::{BatchedDataset, Dataset, TensorDataset};
use ferrox::graph::{AutoFerroxEngine, NodeId};
use ferrox::nn::{
    layers::{Linear, ReLU},
    losses::{BCELoss, CCELoss, Loss, ReductionType},
    optim::{Adam, Optim, Optimizer, SGD},
    Module,
};
use ferrox::ops::unary::Sigmoid;
use ferrox::FerroxN;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::time::Instant;

/// Multi-Layer Perceptron for classification tasks
/// Similar structure to MLPRegressor but designed for classification
#[derive(Debug)]
pub struct MLPClassifier<T>
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
    /// Output layer: hidden_size -> num_classes (no activation for logits)
    output: Linear<T>,
    /// Training mode flag
    training: bool,
}

impl<T> MLPClassifier<T>
where
    T: FerroxCudaF,
{
    /// Create new MLP classifier with specified architecture
    /// num_classes: number of output classes (1 for binary, >1 for multiclass)
    pub fn new(input_size: usize, hidden_size: usize, num_classes: usize, device: Device) -> Self {
        Self {
            hidden1: Linear::new_with_device(input_size, hidden_size, true, device),
            activation1: ReLU::new(),
            hidden2: Linear::new_with_device(hidden_size, hidden_size, true, device),
            activation2: ReLU::new(),
            // Output layer produces logits (no bias for stability)
            output: Linear::new_with_device(hidden_size, num_classes, false, device),
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

impl<T> Module<T> for MLPClassifier<T>
where
    T: FerroxCudaF + rand_distr::num_traits::FromPrimitive,
{
    /// Forward pass through the network - outputs logits for classification
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
        // Output layer: Linear (produces logits, no activation)
        let logits = self.output.forward(graph, activated2)?;

        Ok(logits)
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

    fn set_training(&mut self, training: bool) {
        self.training = training
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
            learning_rate: Some(FerroxN::from_f32(0.00001).unwrap()), // Much lower learning rate for stability
            print_every: 1,
            optimizer: "Adam", // Adam works better for classification
            // SGD defaults
            momentum: Some(FerroxN::from_f32(0.9).unwrap()),
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

/// Generate synthetic binary classification dataset for BCE loss
/// Creates simple linearly separable binary classification: y = sign(0.5*x1 + 0.3*x2 - threshold)
fn generate_binary_data<T>(
    num_samples: usize,
    input_size: usize,
    device: Device,
) -> Result<TensorDataset<T>, String>
where
    T: FerroxCudaF,
{
    let mut rng = rand::rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate random input features
    let mut input_data = Vec::with_capacity(num_samples * input_size);
    for _ in 0..(num_samples * input_size) {
        let value = normal.sample(&mut rng) as f64;
        input_data.push(<T as FerroxN>::from_f64(value).ok_or("Failed to convert input data")?);
    }

    // Generate binary targets using simple decision boundary
    let mut target_data = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let x1 = <T as FerroxN>::to_f64(input_data[i * input_size]);
        let x2 = if input_size > 1 {
            <T as FerroxN>::to_f64(input_data[i * input_size + 1])
        } else {
            0.0
        };

        // Simple binary classification boundary: 0.5*x1 + 0.3*x2 > 0
        let decision_value = 0.5 * x1 + 0.3 * x2;
        let target = if decision_value > 0.0 { 1.0 } else { 0.0 };
        target_data.push(<T as FerroxN>::from_f64(target).ok_or("Failed to convert target data")?);
    }

    let inputs = Tensor::from_vec_with_device(input_data, &[num_samples, input_size], device)?;
    let targets = Tensor::from_vec_with_device(target_data, &[num_samples, 1], device)?;

    TensorDataset::from_tensor(inputs, targets)
}

/// Generate synthetic multiclass classification dataset for CCE loss
/// Creates simple multiclass problem with one-hot encoded targets
fn generate_multiclass_data<T>(
    num_samples: usize,
    input_size: usize,
    num_classes: usize,
    device: Device,
) -> Result<TensorDataset<T>, String>
where
    T: FerroxCudaF,
{
    let mut rng = rand::rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate random input features
    let mut input_data = Vec::with_capacity(num_samples * input_size);
    for _ in 0..(num_samples * input_size) {
        let value = normal.sample(&mut rng) as f64;
        input_data.push(<T as FerroxN>::from_f64(value).ok_or("Failed to convert input data")?);
    }

    // Generate multiclass targets using simple decision regions
    let mut target_data = Vec::with_capacity(num_samples * num_classes);
    for i in 0..num_samples {
        let x1 = <T as FerroxN>::to_f64(input_data[i * input_size]);
        let x2 = if input_size > 1 {
            <T as FerroxN>::to_f64(input_data[i * input_size + 1])
        } else {
            0.0
        };

        // Simple multiclass decision based on input sum modulo num_classes
        let sum = x1 + x2;
        let class_idx = ((sum.abs() * 10.0) as usize) % num_classes;

        // Create one-hot encoding
        for j in 0..num_classes {
            let target_value = if j == class_idx { 1.0 } else { 0.0 };
            target_data.push(
                <T as FerroxN>::from_f64(target_value).ok_or("Failed to convert target data")?,
            );
        }
    }

    let inputs = Tensor::from_vec_with_device(input_data, &[num_samples, input_size], device)?;
    let targets = Tensor::from_vec_with_device(target_data, &[num_samples, num_classes], device)?;

    TensorDataset::from_tensor(inputs, targets)
}

fn with_benchmark<F, R>(func: F) -> impl FnOnce() -> R
where
    F: FnOnce() -> R,
{
    move || {
        let start_time = Instant::now();
        let result = func();
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

/// Compute a loss value using any loss function
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
            hidden_size,
            output_size,
            num_samples,
        }
    }
}

impl Default for MLPConfig {
    fn default() -> Self {
        Self {
            input_size: 4,
            hidden_size: 8,
            output_size: 3, // 3 classes for multiclass demo
            num_samples: 1000,
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

fn eval_classifier<T, M>(
    model: &mut M,
    graph: &mut AutoFerroxEngine<T>,
    data: TensorDataset<T>,
) -> Result<(), String>
where
    T: FerroxCudaF,
    M: Module<T>,
{
    // Switch model to evaluation mode
    model.eval();
    let (test_inputs, test_targets) = data.get_item(0)?;

    // Run forward pass through the model
    let (predictions, pred_shape, num_samples) = run_forward(model, graph, &test_inputs)?;

    // Compute classification losses based on target shape
    let target_shape = test_targets.shape();

    if target_shape[1] == 1 {
        // Binary classification - use BCE loss
        println!("=== Binary Classification Evaluation ===");

        // For BCE, need to apply sigmoid to logits first
        let sigmoid_op = Box::new(Sigmoid);
        let probabilities = graph.apply_operation(sigmoid_op, vec![predictions])?;

        let bce_loss = BCELoss::new(ReductionType::Mean);
        let bce_value = compute_loss(graph, bce_loss, probabilities, &test_targets)?;

        println!("Evaluated on {} samples", num_samples);
        println!("Prediction shape: {:?}", pred_shape);
        println!("BCE Loss: {:.6}", <T as FerroxN>::to_f64(bce_value));
    } else {
        // Multiclass classification - use CCE loss
        println!("=== Multiclass Classification Evaluation ===");

        let cce_loss = CCELoss::from_logits(ReductionType::Mean);
        let cce_value = compute_loss(graph, cce_loss, predictions, &test_targets)?;

        println!("Evaluated on {} samples", num_samples);
        println!("Prediction shape: {:?}", pred_shape);
        println!("Target shape: {:?}", target_shape);
        println!("CCE Loss: {:.6}", <T as FerroxN>::to_f64(cce_value));
    }

    Ok(())
}

fn main() -> Result<(), String> {
    println!("--- Classification Model Training Demo ---");
    let config = MLPConfig::default();
    let training_config = TrainingConfig::default();
    let device = best_f32_device();

    // Demo 1: Multiclass Classification with CCE Loss
    println!("\n=== Training Multiclass Classifier ===");
    let mut multiclass_model = MLPClassifier::<f32>::new(
        config.input_size as usize,
        config.hidden_size as usize,
        config.output_size as usize, // 3 classes
        device,
    );

    let cce_loss = CCELoss::<f32>::from_logits(ReductionType::Mean);

    let multiclass_train_data = generate_multiclass_data::<f32>(
        config.num_samples as usize,
        config.input_size as usize,
        config.output_size as usize,
        Device::CPU,
    )?;

    let multiclass_test_data = generate_multiclass_data::<f32>(
        100,
        config.input_size as usize,
        config.output_size as usize,
        device,
    )?;

    let mut trained_multiclass_graph = train(
        &mut multiclass_model,
        &cce_loss,
        multiclass_train_data.into_batches(100, false)?,
        &training_config,
    )?;

    eval_classifier(
        &mut multiclass_model,
        &mut trained_multiclass_graph,
        multiclass_test_data,
    )?;

    // Demo 2: Binary Classification with BCE Loss
    println!("\n=== Training Binary Classifier ===");
    let mut binary_model = MLPClassifier::<f32>::new(
        config.input_size as usize,
        config.hidden_size as usize,
        1, // binary classification
        device,
    );

    let bce_loss = BCELoss::<f32>::new(ReductionType::Mean);

    let binary_train_data = generate_binary_data::<f32>(
        config.num_samples as usize,
        config.input_size as usize,
        Device::CPU,
    )?;

    let binary_test_data = generate_binary_data::<f32>(100, config.input_size as usize, device)?;

    let mut trained_binary_graph = train(
        &mut binary_model,
        &bce_loss,
        binary_train_data.into_batches(100, false)?,
        &training_config,
    )?;

    eval_classifier(
        &mut binary_model,
        &mut trained_binary_graph,
        binary_test_data,
    )?;

    println!("\n--- Classification Demo Completed Successfully! ---");
    Ok(())
}
