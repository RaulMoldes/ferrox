// examples/simple_regression.rs
//! Simple Linear Regression Example
//! 
//! This example demonstrates basic usage of Ferrox for a simple regression task.
//! Key insight: We must use a SINGLE engine throughout training to maintain
//! consistent NodeIds for the optimizer.

use ferrox::{
    nn::{
        Linear, ReLU, Sequential, MSELoss, Loss, Adam, Optimizer, Module
    },
    graph::Engine,
    
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
    println!("=== Simple Linear Regression ===\n");
    
    // 1. Generate data
    let (train_x, train_y) = generate_data(500);
    let (test_x, test_y) = generate_data(100);
    
    println!("Generated {} training samples", train_x.len());
    println!("Generated {} test samples\n", test_x.len());
    
    // 2. Create model
    let mut model = Sequential::new();
    model.add(Box::new(Linear::new(1, 8, true)));   // 1 input -> 8 hidden
    model.add(Box::new(ReLU::new()));
    model.add(Box::new(Linear::new(8, 1, true)));   // 8 hidden -> 1 output
    
    println!("Model: Input(1) -> Linear(8) -> ReLU -> Linear(1) -> Output");
    println!("Parameters: {}\n", model.num_parameters());
    
    // 3. Create SINGLE engine for entire training process
    let mut engine = Engine::new();
    
    // 4. Setup training
    let mut optimizer = Adam::with_defaults(0.01);
    let loss_fn = MSELoss::new();
    
    // Initialize optimizer with model parameters (do this ONCE)
    optimizer.add_param_group(&model, &mut engine);
    
    let epochs = 100;
    
    println!("Training for {} epochs...", epochs);
    println!("Using SINGLE engine (no mini-batches for now)");
    
    // 5. Training loop - process ALL data at once as we currently cannot do mini-batches
    // We cannot use batching because we require consistent NodeIds for the optimizer,
    // which is not possible with multiple engines or mini-batches in this example.
    // In a real-world scenario, you would implement mini-batch training with consistent NodeIds.
    // Maybe we could handle this by cloning the Engine?
    for epoch in 0..epochs {
        // Create input tensor for entire training set
        let input = engine.tensor_from_vec(train_x.clone(), &[train_x.len(), 1], true)?;
        let target = engine.tensor_from_vec(train_y.clone(), &[train_y.len(), 1], false)?;
        
        // Clear gradients
        optimizer.reset_grad(&mut engine);
        
        // Forward pass
        let prediction = model.forward(&mut engine, input)?;
        
        // Compute loss
        let loss = loss_fn.compute_loss(&mut engine, prediction, target)?;

        let loss_value = engine.get_data(loss);
        
        let loss_value = loss_value[0];
        println!("Epoch {}: Loss = {:.6}", epoch + 1, loss_value);
        // Backward pass
        engine.backward(loss)?;
        
        // Update parameters
        optimizer.step(&mut engine);
        
        // Print progress
        if epoch % 20 == 0 || epoch == epochs - 1 {
            println!("Epoch {}: Loss = {:.6}", epoch + 1, loss_value);
        }
    }
    
    println!("\nTraining completed!\n");
    
    // 6. Test the model on individual samples
    println!("Testing on sample inputs:");
    println!("(Remember: true function is y = 2*x + 1)");
    println!("Input | Predicted | Expected | Error");
    println!("------|-----------|----------|-------");
    
    for &test_val in &[-0.5, 0.0, 0.5, 1.0] {
        let input = engine.tensor_from_vec(vec![test_val], &[1, 1], false)?;
        let prediction = model.forward(&mut engine, input)?;
        let predicted = engine.get_data(prediction)[0];
        let expected = 2.0 * test_val + 1.0;
        let error = (predicted - expected).abs();
        
        println!("{:5.1} | {:8.3} | {:7.1} | {:.3}", 
                 test_val, predicted, expected, error);
    }
    
    // 7. Evaluate on test set
    println!("\nEvaluating on test set...");
    let test_input = engine.tensor_from_vec(test_x, &[test_y.len(), 1], false)?;
    let test_target = engine.tensor_from_vec(test_y.clone(), &[test_y.len(), 1], false)?;
    
    let test_prediction = model.forward(&mut engine, test_input)?;
    let test_loss = loss_fn.compute_loss(&mut engine, test_prediction, test_target)?;
    let test_loss_value = engine.get_data(test_loss)[0];
    
    println!("Test Loss: {:.6}", test_loss_value);
    
    // 8. Show some test predictions
    println!("\nSample test predictions:");
    println!("Input | Predicted | Actual  | Error");
    println!("------|-----------|---------|-------");
    
    let test_predictions = engine.get_data(test_prediction);
    // Show first 5 test predictions
    let test_inputs = engine.get_data(test_input);
    let test_targets = engine.get_data(test_target);
    for i in 0..5.min(test_predictions.len()) {
        let input = test_inputs[i];
        let pred = test_predictions[i];
        let actual = test_targets[i];
        let error = (pred - actual).abs();
        println!("{:5.2} | {:8.3} | {:6.3} | {:.3}", input, pred, actual, error);
    }
    
    println!("\n=== Example completed! ===");
    println!("Note: This example processes all data at once due to NodeId consistency requirements.");
    println!("Mini-batch training would require architectural changes to the framework.");
    
    Ok(())
}