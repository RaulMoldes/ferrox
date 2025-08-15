// examples/cnn_classifier.rs
// CNN image classifier example using synthetic image data

mod common;

use common::{
    compute_loss, generate_synthetic_image_data, run_forward, train, CNNClassifier, TrainingConfig,
};
use ferrox::backend::manager::{best_f32_device};
use ferrox::backend::Device;
use ferrox::backend::FerroxCudaF;
use ferrox::nn::losses::{CCELoss, ReductionType};
use ferrox::nn::Module;

/// Evaluate CNN classification model on test data
fn evaluate_classification<T>(
    model: &mut CNNClassifier<T>,
    graph: &mut ferrox::graph::AutoFerroxEngine<T>,
    test_data: ferrox::dataset::TensorDataset<T>,
) -> Result<(), String>
where
    T: FerroxCudaF,
{
    // Switch to evaluation mode
    model.set_training(false);
    let (test_inputs, test_targets) = test_data.into_data();

    // Run forward pass
    let (predictions, pred_shape, num_samples) = run_forward(model, graph, &test_inputs)?;

    // Compute CCE loss for multiclass classification
    let cce_loss = CCELoss::from_logits(ReductionType::Mean, Some(1));
    let cce_value = compute_loss(graph, cce_loss, predictions, &test_targets)?;

    // Display results
    println!("=== CNN Classification Evaluation ===");
    println!("Evaluated on {} samples", num_samples);
    println!("Prediction shape: {:?}", pred_shape);
    println!("Target shape: {:?}", test_targets.shape());
    println!("CCE Loss: {:.6}", ferrox::FerroxN::to_f64(cce_value));

    Ok(())
}
fn main() -> Result<(), String> {
    println!("--- CNN Image Classification Demo ---");

    let device = best_f32_device();
    let training_config = TrainingConfig::fast(); // Use fast config for demo

    // Configuration for image classification
    let image_size = 16;
    let channels = 1; // Grayscale images
    let num_classes = 3; // Three pattern classes
    let num_samples = 100;
    println!("\n=== CNN Image Classification Demo ===");

    println!("Creating CNN classifier...");
    let mut model = CNNClassifier::<f32>::new(channels, num_classes, image_size, device);

    println!("Generating synthetic image data...");
    let train_data = generate_synthetic_image_data::<f32>(
        num_samples,
        num_classes,
        image_size,
        channels,
        Device::CPU, // Generate on CPU, transfer automatically during batching
    )?;

    let test_data =
        generate_synthetic_image_data::<f32>(20, num_classes, image_size, channels, device)?;

    // Loss function for multiclass classification
    let loss_fn = CCELoss::<f32>::from_logits(ReductionType::Mean, Some(1));

    println!("Training CNN classifier...");
    println!(
        "Image size: {}x{}, Channels: {}, Classes: {}",
        image_size, image_size, channels, num_classes
    );

    let mut trained_graph = train(
        &mut model,
        &loss_fn,
        train_data.into_batches(training_config.batch_size, false)?,
        &training_config,
    )?;

    println!("\nEvaluating model on test data...");
    evaluate_classification(&mut model, &mut trained_graph, test_data)?;

    println!("\nCNN image classification example completed successfully!");
    Ok(())
}
