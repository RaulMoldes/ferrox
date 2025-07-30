// examples/classification_with_softmax.rs

use ferrox::graph::Engine;
use ferrox::nn::loss::{CCELoss, Loss};
use ferrox::nn::optim::{Adam, Optimizer};
use ferrox::nn::{Linear, Module, ReLU, Sequential, Softmax};

fn clf_network() -> Result<(), Box<dyn std::error::Error>> {
    println!("Building a classification network with Softmax activation...");

    // Create computation graph
    let mut engine = Engine::new();

    // Network architecture: 4 input features -> 8 hidden -> 3 classes
    let mut model = Sequential::new();
    model.add(Box::new(Linear::new(4, 8, true))); // Input layer
    model.add(Box::new(ReLU::new())); // Hidden activation
    model.add(Box::new(Linear::new(8, 3, true))); // Output layer
    model.add(Box::new(Softmax::new(1))); // Softmax for probabilities

    println!("Model architecture:");
    println!("  Input: 4 features");
    println!("  Hidden: 8 units with ReLU");
    println!("  Output: 3 classes with Softmax");

    // Create dummy training data (batch_size=4, features=4)
    let input_data = vec![
        // Sample 1: class 0
        0.1, 0.2, 0.3, 0.4, // Sample 2: class 1
        0.5, 0.6, 0.7, 0.8, // Sample 3: class 2
        0.9, 1.0, 1.1, 1.2, // Sample 4: class 0
        0.2, 0.3, 0.4, 0.5,
    ];

    let input = engine.tensor_from_vec(input_data, &[4, 4], true)?;

    // Create one-hot encoded targets
    let target_data = vec![
        // Sample 1: class 0 -> [1, 0, 0]
        1.0, 0.0, 0.0, // Sample 2: class 1 -> [0, 1, 0]
        0.0, 1.0, 0.0, // Sample 3: class 2 -> [0, 0, 1]
        0.0, 0.0, 1.0, // Sample 4: class 0 -> [1, 0, 0]
        1.0, 0.0, 0.0,
    ];

    let targets = engine.tensor_from_vec(target_data, &[4, 3], false)?;

    // Create optimizer
    let mut optimizer = Adam::with_defaults(0.001);
    optimizer.add_param_group(&model, &mut engine);

    // Create loss function
    let loss_fn = CCELoss::new();

    println!("\nStarting training loop...");

    // Training loop
    for epoch in 0..100 {
        // Forward pass
        let predictions = model.forward(&mut engine, input)?;

        // Compute loss
        let loss = loss_fn.compute_loss(&mut engine, predictions, targets)?;

        // Get loss value for printing
        let loss_value = engine.get_data(loss);

        if epoch % 20 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss_value[0]);

            // Print some predictions for debugging
            let pred_data = engine.get_data(predictions);
            println!("  Sample predictions:");
            for sample in 0..2 {
                // Show first 2 samples
                let start_idx = sample * 3;
                println!(
                    "    Sample {}: [{:.3}, {:.3}, {:.3}]",
                    sample,
                    pred_data[start_idx],
                    pred_data[start_idx + 1],
                    pred_data[start_idx + 2]
                );

                // Verify probabilities sum to 1
                let sum =
                    pred_data[start_idx] + pred_data[start_idx + 1] + pred_data[start_idx + 2];
                println!("      (sum = {:.6})", sum);
            }
        }

        // Backward pass
        engine.backward(loss)?;

        // Update parameters
        optimizer.step(&mut engine);

        // Clear gradients
        optimizer.reset_grad(&mut engine);
    }

    println!("\nTraining completed!");

    // Final evaluation
    let final_predictions = model.forward(&mut engine, input)?;
    let final_pred_data = engine.get_data(final_predictions);

    println!("\nFinal predictions vs targets:");
    for sample in 0..4 {
        let pred_start = sample * 3;
        let target_start = sample * 3;

        // Find predicted class (argmax)
        let mut pred_class = 0;
        let mut max_prob = final_pred_data[pred_start];
        for class in 1..3 {
            if final_pred_data[pred_start + class] > max_prob {
                max_prob = final_pred_data[pred_start + class];
                pred_class = class;
            }
        }

        // Find target class
        let target_data_ref = engine.get_data(targets);
        let mut target_class = 0;
        for class in 0..3 {
            if target_data_ref[target_start + class] > 0.5 {
                target_class = class;
                break;
            }
        }

        println!(
            "Sample {}: Predicted class {} (prob={:.3}), Target class {}",
            sample, pred_class, max_prob, target_class
        );

        // Show full probability distribution
        println!(
            "  Probabilities: [{:.3}, {:.3}, {:.3}]",
            final_pred_data[pred_start],
            final_pred_data[pred_start + 1],
            final_pred_data[pred_start + 2]
        );
    }

    Ok(())
}

// Helper function to demonstrate softmax properties
fn demonstrate_softmax_properties() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Demonstrating Softmax Properties ===");

    let mut engine = Engine::new();
    let softmax = Softmax::new(1);

    // Test 1: Probability distribution property
    println!("\n1. Probability Distribution Property:");
    let logits1 = engine.tensor_from_vec(vec![1.0, 2.0, 3.0], &[1, 3], true)?;
    let probs1 = softmax.forward(&mut engine, logits1)?;
    let prob_data1 = engine.get_data(probs1);

    let sum: f64 = prob_data1.sum(None)?.first()?;
    println!("  Input logits: [1.0, 2.0, 3.0]");
    println!(
        "  Softmax output: [{:.3}, {:.3}, {:.3}]",
        prob_data1[0], prob_data1[1], prob_data1[2]
    );
    println!("  Sum of probabilities: {:.6}", sum);

    // Test 2: numerical stability with large values
    println!("\n2. numerical stability:");
    let large_logits = engine.tensor_from_vec(vec![1000.0, 1001.0, 1002.0], &[1, 3], true)?;
    let stable_probs = softmax.forward(&mut engine, large_logits)?;
    let stable_data = engine.get_data(stable_probs);

    let stable_sum: f64 = stable_data.sum(None)?.first()?;
    println!("  Large input logits: [1000.0, 1001.0, 1002.0]");
    println!(
        "  Stable softmax output: [{:.3}, {:.3}, {:.3}]",
        stable_data[0], stable_data[1], stable_data[2]
    );
    println!("  Sum of probabilities: {:.6}", stable_sum);

    // Test 3: Translation invariance
    println!("\n3. Translation Invariance (softmax(x+c) = softmax(x)):");
    let original = engine.tensor_from_vec(vec![1.0, 2.0, 3.0], &[1, 3], true)?;
    let shifted = engine.tensor_from_vec(vec![11.0, 12.0, 13.0], &[1, 3], true)?;

    let probs_orig = softmax.forward(&mut engine, original)?;
    let probs_shift = softmax.forward(&mut engine, shifted)?;

    let orig_data = engine.get_data(probs_orig);
    let shift_data = engine.get_data(probs_shift);

    println!("  Original logits: [1.0, 2.0, 3.0]");
    println!("  Shifted logits (+10): [11.0, 12.0, 13.0]");
    println!(
        "  Original softmax: [{:.6}, {:.6}, {:.6}]",
        orig_data[0], orig_data[1], orig_data[2]
    );
    println!(
        "  Shifted softmax:  [{:.6}, {:.6}, {:.6}]",
        shift_data[0], shift_data[1], shift_data[2]
    );

    let approx_eq = orig_data.zip_all(&shift_data, |a, b| {
            let diff = if a > b { a - b } else { b - a };
            diff <= 1e-6
        }).expect("Error comparing tensors");
    assert!(approx_eq);
    println!("Approx equal comparison {:?}", approx_eq);

    Ok(())
}

// Run both examples
fn main() {
    if let Err(e) = clf_network() {
        eprintln!("Error in main example: {}", e);
    }

    if let Err(e) = demonstrate_softmax_properties() {
        eprintln!("Error in properties demonstration: {}", e);
    }
}
