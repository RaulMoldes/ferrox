// examples/simple_regression.rs
//! Simple Linear Regression Example.
//! This example demonstrates how to use Ferrox's neural network module to train and test a simple linear regresion model.
// This example generates synthetic data for a linear function, trains a linear regression model,
use ferrox::{
    graph::Engine,
    nn::{Adam, Linear, Loss, MSELoss, Module, Optimizer},
};

/// Generate synthetic linear data: y = 2*x + 1 + noise
fn generate_data(n_samples: usize) -> (Vec<f64>, Vec<f64>) {
    use rand::Rng;
    let mut rng = rand::rng();

    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for _ in 0..n_samples {
        let x = rng.random_range(-1.0..1.0);
        let noise = rng.random_range(-0.05..0.05);
        let y = 2.0 * x + 1.0 + noise;

        x_data.push(x);
        y_data.push(y);
    }

    (x_data, y_data)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Working Linear Regression ===\n");

    // Generate data
    let (train_x, train_y) = generate_data(500);
    let (test_x, test_y) = generate_data(100);

    println!("Generated {} training samples", train_x.len());
    println!("Generated {} test samples\n", test_x.len());

    // Create single engine
    let mut engine = Engine::new();

    // Create model - simple linear model for debugging
    let model = Linear::new(1, 1, true); // Just y = w*x + b

    println!("Model: Linear(1 -> 1) with bias");

    // Set up training
    let mut optimizer = Adam::with_defaults(0.01);
    let loss_fn = MSELoss::new();

    println!("\n1. Initializing parameter nodes...");

    // CRITICAL: Initialize cached parameter nodes with a dummy forward pass
    let dummy_input = engine.tensor_from_vec(vec![0.0], &[1, 1], true)?;
    let _dummy_output = model.forward(&mut engine, dummy_input)?;

    println!(
        "   Parameters initialized: {}",
        model.parameters_initialized()
    );

    // Get the cached parameter nodes and add them to optimizer
    println!("\n2. Adding cached parameter nodes to optimizer...");
    let cached_nodes = model.get_cached_parameter_nodes();

    let mut param_count = 0;
    for (_, node_opt) in cached_nodes.iter().enumerate() {
        if let Some(node_id) = node_opt {
            optimizer.add_param(param_count, *node_id);
            println!("   Added parameter {} (NodeId: {})", param_count, node_id);
            param_count += 1;
        }
    }

    println!("   Total parameters registered: {}", param_count);

    // Training loop
    println!("\n3. Training...");
    let epochs = 50;

    for epoch in 0..epochs {
        // Full batch training (no mini-batches due to NodeId consistency)
        let input = engine.tensor_from_vec(train_x.clone(), &[train_x.len(), 1], true)?;
        let target = engine.tensor_from_vec(train_y.clone(), &[train_y.len(), 1], false)?;

        // Clear gradients
        optimizer.reset_grad(&mut engine);

        // Forward pass
        let prediction = model.forward(&mut engine, input)?;

        // Compute loss
        let loss = loss_fn.compute_loss(&mut engine, prediction, target)?;
        let loss_value = engine.get_data(loss)[0];

        // Backward pass
        engine.backward(loss)?;

        // Check if gradients exist for our parameters (debugging)
        if epoch == 0 {
            println!("\n   Checking gradients after first backward pass:");
            for (i, node_opt) in cached_nodes.iter().enumerate() {
                if let Some(node_id) = node_opt {
                    if let Some(_grad) = engine.get_gradient(*node_id) {
                        println!("   Parameter {} has gradient", i);
                    } else {
                        println!("   Parameter {} has NO gradient", i);
                    }
                }
            }
            println!();
        }

        // Update parameters
        optimizer.step(&mut engine);

        // Print progress
        if epoch % 10 == 0 || epoch == epochs - 1 {
            // Test prediction on a simple input to see if it's changing
            let test_input = engine.tensor_from_vec(vec![0.0], &[1, 1], false)?;
            let test_pred = model.forward(&mut engine, test_input)?;
            let pred_at_zero = engine.get_data(test_pred)[0];

            println!(
                "   Epoch {}: Loss = {:.6}, f(0) = {:.6}",
                epoch + 1,
                loss_value,
                pred_at_zero
            );
        }
    }

    println!("\nTraining completed!\n");

    // Test the learned function
    println!("Testing learned function:");
    println!("(Target: y = 2*x + 1)");
    println!("Input | Predicted | Expected | Error");
    println!("------|-----------|----------|-------");

    for &test_val in test_y.iter() {
        let input = engine.tensor_from_vec(vec![test_val], &[1, 1], false)?;
        let prediction = model.forward(&mut engine, input)?;
        let predicted = engine.get_data(prediction)[0];
        let expected = 2.0 * test_val + 1.0;
        let error = (predicted - expected).abs();

        println!(
            "{:5.1} | {:8.3} | {:7.1} | {:.3}",
            test_val, predicted, expected, error
        );
    }

    println!("\n=== Example completed! ===");

    Ok(())
}
