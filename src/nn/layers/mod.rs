// src/nn/layers/mod.rs
// Module declaration and basic usage test for neural network layers

pub mod activation;
pub mod linear;

// Re-export commonly used layers for convenience
pub use activation::{ReLU, Sigmoid, Tanh};
pub use linear::Linear;

#[cfg(test)]
mod layer_tests {
    use super::*;
    use crate::backend::manager::best_f32_device;
    use crate::backend::Tensor;
    use crate::graph::AutoFerroxEngine;
    use crate::nn::Module;

    /// Test basic linear layer functionality
    #[test]
    fn test_linear_layer_forward() {
        let device = best_f32_device();
        let mut engine = AutoFerroxEngine::new();

        // Create a simple linear layer: 3 input features -> 2 output features
        let layer = Linear::<f32>::new_with_device(3, 2, true, device);

        // Create test input: batch of 2 samples, each with 3 features
        let input_data =
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device)
                .expect("Failed to create input tensor");

        let input_node = engine.create_variable(input_data, false);

        // Forward pass through linear layer
        let output_node = layer
            .forward(&mut engine, input_node)
            .expect("Linear layer forward pass failed");

        // Check output tensor exists and has correct shape
        let output_tensor = engine
            .get_tensor(output_node)
            .expect("Output tensor not found");

        assert_eq!(output_tensor.shape(), &[2, 2]); // [batch_size, out_features]
        println!(
            "Linear layer test passed - output shape: {:?}",
            output_tensor.shape()
        );
    }

    /// Test ReLU activation layer
    #[test]
    fn test_relu_activation() {
        let device = best_f32_device();
        let mut engine = AutoFerroxEngine::new();

        // Create ReLU activation layer
        let relu = ReLU::<f32>::new();

        // Create test input with both positive and negative values
        let input_data =
            Tensor::from_vec_with_device(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5], device)
                .expect("Failed to create input tensor");

        let input_node = engine.create_variable(input_data, false);

        // Forward pass through ReLU
        let output_node = relu
            .forward(&mut engine, input_node)
            .expect("ReLU forward pass failed");

        // Check output tensor exists and has correct shape
        let output_tensor = engine
            .get_tensor(output_node)
            .expect("Output tensor not found");

        assert_eq!(output_tensor.shape(), &[5]);
        println!(
            "ReLU activation test passed - output shape: {:?}",
            output_tensor.shape()
        );

        // Verify ReLU behavior: negative values should become 0
        let output_vec = output_tensor
            .clone()
            .into_data()
            .expect("Failed to convert to  Array");
        assert_eq!(output_vec[0], 0.0); // -2.0 -> 0.0
        assert_eq!(output_vec[1], 0.0); // -1.0 -> 0.0
        assert_eq!(output_vec[2], 0.0); // 0.0 -> 0.0
        assert_eq!(output_vec[3], 1.0); // 1.0 -> 1.0
        assert_eq!(output_vec[4], 2.0); // 2.0 -> 2.0
    }

    /// Test complete layer composition: Linear + ReLU
    #[test]
    fn test_linear_relu_composition() {
        let device = best_f32_device();
        let mut engine = AutoFerroxEngine::new();

        // Create layers
        let linear = Linear::<f32>::new_with_device(2, 3, true, device);
        let relu = ReLU::<f32>::new();

        // Create test input
        let input_data = Tensor::from_vec_with_device(
            vec![1.0, -1.0, 2.0, -2.0],
            &[2, 2], // 2 samples, 2 features each
            device,
        )
        .expect("Failed to create input tensor");

        let input_node = engine.create_variable(input_data, false);

        // Forward pass: input -> linear -> relu
        let linear_output = linear
            .forward(&mut engine, input_node)
            .expect("Linear forward failed");

        let final_output = relu
            .forward(&mut engine, linear_output)
            .expect("ReLU forward failed");

        // Check final output has correct shape
        let output_tensor = engine
            .get_tensor(final_output)
            .expect("Final output tensor not found");

        assert_eq!(output_tensor.shape(), &[2, 3]); // [batch_size, linear_out_features]
        println!(
            "Linear + ReLU composition test passed - final shape: {:?}",
            output_tensor.shape()
        );
    }

    /// Test parameter collection from layers
    #[test]
    fn test_parameter_collection() {
        let device = best_f32_device();

        // Create linear layer with bias
        let linear_with_bias = Linear::<f32>::new_with_device(3, 2, true, device);
        assert_eq!(linear_with_bias.parameters().len(), 2); // weight + bias

        // Create linear layer without bias
        let linear_no_bias = Linear::<f32>::new_with_device(3, 2, false, device);
        assert_eq!(linear_no_bias.parameters().len(), 1); // weight only

        // Create activation layers (no parameters)
        let relu = ReLU::<f32>::new();
        let sigmoid = Sigmoid::<f32>::new();
        let tanh = Tanh::<f32>::new();

        assert_eq!(relu.parameters().len(), 0);
        assert_eq!(sigmoid.parameters().len(), 0);
        assert_eq!(tanh.parameters().len(), 0);

        println!("Parameter collection test passed");
    }

    /// Test layer training mode functionality
    #[test]
    fn test_training_mode() {
        let device = best_f32_device();

        let mut linear = Linear::<f32>::new_with_device(2, 2, true, device);
        let mut relu = ReLU::<f32>::new();

        // Initially in training mode
        assert!(linear.training());
        assert!(relu.training());

        // Switch to evaluation mode
        linear.eval();
        relu.eval();
        assert!(!linear.training());
        assert!(!relu.training());

        // Switch back to training mode
        linear.train();
        relu.train();
        assert!(linear.training());
        assert!(relu.training());

        println!("Training mode test passed");
    }
}
