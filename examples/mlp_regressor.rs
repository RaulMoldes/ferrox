mod common;

use common::{
    compute_loss, generate_regression_data, run_forward, train, MLPConfig, MLPRegressor,
    TrainingConfig,
};
use ferrox::backend::manager::best_f32_device;
use ferrox::backend::Device;
use ferrox::nn::losses::{L1Loss, MSELoss, ReductionType};
use ferrox::nn::Module;

/// Evaluate regression model on test data
fn evaluate_regression<T>(
    model: &mut MLPRegressor<T>,
    graph: &mut ferrox::graph::AutoFerroxEngine<T>,
    test_data: ferrox::dataset::TensorDataset<T>,
) -> Result<(), String>
where
    T: ferrox::backend::FerroxCudaF,
{
    // Switch to evaluation mode
    model.set_training(false);
    let (test_inputs, test_targets) = test_data.into_data();

    // Run forward pass
    let (predictions, pred_shape, num_samples) = run_forward(model, graph, &test_inputs)?;

    // Compute different metrics
    let l1_loss = L1Loss::new(ReductionType::Mean);
    let l1_value = compute_loss(graph, l1_loss, predictions, &test_targets)?;

    let mse_loss = MSELoss::new(ReductionType::Mean);
    let mse_value = compute_loss(graph, mse_loss, predictions, &test_targets)?;

    // Display results
    println!("=== Regression Model Evaluation ===");
    println!("Evaluated on {} samples", num_samples);
    println!("Prediction shape: {:?}", pred_shape);
    println!("MSE Loss: {:.6}", ferrox::FerroxN::to_f64(mse_value));
    println!("L1 Loss (MAE): {:.6}", ferrox::FerroxN::to_f64(l1_value));

    Ok(())
}

fn main() -> Result<(), String> {
    println!("--- MLP Regression Demo ---");

    // Configuration
    let model_config = MLPConfig::regression();
    let training_config = TrainingConfig::default();
    let device = best_f32_device();

    // Create model
    let mut model = MLPRegressor::<f32>::new(
        model_config.input_size,
        model_config.hidden_size,
        model_config.output_size,
        device,
    );

    // Generate data
    let train_data = generate_regression_data::<f32>(
        model_config.num_samples,
        model_config.input_size,
        Device::CPU, // Generate on CPU, transfer automatically during batching
    )?;

    let test_data = generate_regression_data::<f32>(100, model_config.input_size, device)?;

    // Create loss function
    let loss_fn = L1Loss::<f32>::new(ReductionType::Mean);

    // Train model
    println!("Starting training...");
    let mut trained_graph = train(
        &mut model,
        &loss_fn,
        train_data.into_batches(training_config.batch_size, false)?,
        &training_config,
    )?;

    // Evaluate model
    println!("Training completed, evaluating...");
    evaluate_regression(&mut model, &mut trained_graph, test_data)?;

    println!("Regression example completed successfully!");
    Ok(())
}
