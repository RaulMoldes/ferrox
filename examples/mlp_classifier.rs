mod common;

use common::{
    compute_loss, generate_binary_classification_data, generate_multiclass_classification_data,
    run_forward, train, MLPClassifier, MLPConfig, TrainingConfig,
};
use ferrox::backend::manager::best_f32_device;
use ferrox::backend::Device;
use ferrox::nn::losses::{BCELoss, CCELoss, ReductionType};
use ferrox::nn::Module;
use ferrox::ops::unary::Sigmoid;

/// Evaluate classification model on test data
fn evaluate_classification<T>(
    model: &mut MLPClassifier<T>,
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

    // Determine classification type based on target shape
    let target_shape = test_targets.shape();

    if target_shape[1] == 1 {
        // Binary classification - use BCE loss
        println!("=== Binary Classification Evaluation ===");

        // Apply sigmoid to logits for BCE
        let sigmoid_op = Box::new(Sigmoid);
        let probabilities = graph.apply_operation(sigmoid_op, vec![predictions])?;

        let bce_loss = BCELoss::new(ReductionType::Mean);
        let bce_value = compute_loss(graph, bce_loss, probabilities, &test_targets)?;

        println!("Evaluated on {} samples", num_samples);
        println!("Prediction shape: {:?}", pred_shape);
        println!("BCE Loss: {:.6}", ferrox::FerroxN::to_f64(bce_value));
    } else {
        // Multiclass classification - use CCE loss
        println!("=== Multiclass Classification Evaluation ===");

        let cce_loss = CCELoss::from_logits(ReductionType::Mean, Some(1));
        let cce_value = compute_loss(graph, cce_loss, predictions, &test_targets)?;

        println!("Evaluated on {} samples", num_samples);
        println!("Prediction shape: {:?}", pred_shape);
        println!("Target shape: {:?}", target_shape);
        println!("CCE Loss: {:.6}", ferrox::FerroxN::to_f64(cce_value));
    }

    Ok(())
}

fn main() -> Result<(), String> {
    println!("--- MLP Classification Demo ---");

    let device = best_f32_device();
    let training_config = TrainingConfig::fast(); // Use fast config for demo

    // Demo 1: Binary Classification
    println!("\n=== Binary Classification Demo ===");
    {
        let config = MLPConfig::binary_classification();
        let mut model = MLPClassifier::<f32>::new(
            config.input_size,
            config.hidden_size,
            config.output_size, // 1 for binary
            device,
        );

        let train_data = generate_binary_classification_data::<f32>(
            config.num_samples,
            config.input_size,
            Device::CPU,
        )?;

        let test_data = generate_binary_classification_data::<f32>(200, config.input_size, device)?;

        let loss_fn = BCELoss::<f32>::new(ReductionType::Mean);

        println!("Training binary classifier...");
        let mut trained_graph = train(
            &mut model,
            &loss_fn,
            train_data.into_batches(training_config.batch_size, false)?,
            &training_config,
        )?;
        trained_graph.print_stats();
        evaluate_classification(&mut model, &mut trained_graph, test_data)?;
    }

    // Demo 2: Multiclass Classification
    println!("\n=== Multiclass Classification Demo ===");
    {
        let config = MLPConfig::multiclass_classification(3); // 3 classes
        let mut model = MLPClassifier::<f32>::new(
            config.input_size,
            config.hidden_size,
            config.output_size, // 3 for multiclass
            device,
        );

        let train_data = generate_multiclass_classification_data::<f32>(
            config.num_samples,
            config.input_size,
            config.output_size,
            Device::CPU,
        )?;

        let test_data = generate_multiclass_classification_data::<f32>(
            200,
            config.input_size,
            config.output_size,
            device,
        )?;

        let loss_fn = CCELoss::<f32>::from_logits(ReductionType::Mean, Some(1));

        println!("Training multiclass classifier...");
        let mut trained_graph = train(
            &mut model,
            &loss_fn,
            train_data.into_batches(training_config.batch_size, false)?,
            &training_config,
        )?;

        trained_graph.print_stats();

        evaluate_classification(&mut model, &mut trained_graph, test_data)?;
    }

    println!("\nClassification examples completed successfully!");
    Ok(())
}
