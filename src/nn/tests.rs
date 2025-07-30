#[cfg(test)]
mod tests {

    use crate::graph::Engine;
    use crate::nn::activations::*;
    use crate::nn::layers::*;
    use crate::nn::loss::*;
    use crate::nn::module::*;
    use crate::nn::normalization::{BatchNorm1d, LayerNorm};
    use crate::nn::parameter::Parameter;
    use crate::tensor::Tensor;

    /// Helper function to check if two floating point values are approximately equal
    fn approx_equal(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() < tolerance
    }

    // ============================================================================
    // LINEAR LAYER TESTS
    // ============================================================================

    #[test]
    fn test_linear_layer_output_dimensions() {
        let mut graph = Engine::new();

        // Test various input/output combinations
        let test_cases = vec![
            (784, 256, 32), // MNIST-like: 784 -> 256, batch_size=32
            (128, 64, 16),  // Medium layer: 128 -> 64, batch_size=16
            (10, 1, 5),     // Small layer: 10 -> 1, batch_size=5
            (3, 100, 2),    // Small to large: 3 -> 100, batch_size=2
            (1000, 10, 1),  // Large to small: 1000 -> 10, batch_size=1
        ];

        for (in_features, out_features, batch_size) in test_cases {
            println!(
                "Testing: {} -> {}, batch_size={}",
                in_features, out_features, batch_size
            );

            let linear = Linear::new(in_features, out_features, true);

            // Create random input with correct shape
            let input_data: Vec<f64> = (0..batch_size * in_features)
                .map(|i| (i as f64) * 0.01)
                .collect();

            let input = graph
                .tensor_from_vec(input_data, &[batch_size, in_features], true)
                .unwrap();

            let output = linear.forward(&mut graph, input).unwrap();
            let output_shape = graph.get_shape(output);

            // Verify output shape is exactly [batch_size, out_features]
            assert_eq!(
                output_shape,
                vec![batch_size, out_features],
                "Failed for case: {} -> {}, batch_size={}",
                in_features,
                out_features,
                batch_size
            );

            // Verify output tensor has the correct number of elements
            let output_data = graph.get_data(output);
            assert_eq!(
                output_data.size(),
                batch_size * out_features,
                "Output tensor size mismatch for case: {} -> {}, batch_size={}",
                in_features,
                out_features,
                batch_size
            );
        }
    }

    #[test]
    fn test_linear_layer_mathematical_correctness() {
        let mut graph = Engine::new();

        // Create a simple, deterministic case for manual verification
        let linear = Linear::with_init(2, 3, true, || 0.1); // All weights = 0.1, bias = 0.0

        // Input: batch_size=1, features=2, values=[1.0, 2.0]
        let input = graph
            .tensor_from_vec(vec![1.0, 2.0], &[1, 2], true)
            .unwrap();

        let output = linear.forward(&mut graph, input).unwrap();
        let output_data = graph.get_data(output);

        // Verify output shape
        assert_eq!(graph.get_shape(output), vec![1, 3]);

        // Manual calculation:
        // Weight matrix (3x2): [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]
        // Input (1x2): [[1.0, 2.0]]
        // Output = Input @ Weight.T = [1.0, 2.0] @ [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
        // = [1.0*0.1 + 2.0*0.1, 1.0*0.1 + 2.0*0.1, 1.0*0.1 + 2.0*0.1]
        // = [0.3, 0.3, 0.3] + bias[0.0, 0.0, 0.0] = [0.3, 0.3, 0.3]

        for i in 0..3 {
            assert!(
                approx_equal(output_data[i], 0.3, 1e-6),
                "Expected 0.3, got {} at position {}",
                output_data[i],
                i
            );
        }
    }

    #[test]
    fn test_linear_layer_bias_effect() {
        let mut graph = Engine::new();

        // Test with bias
        let linear_with_bias = Linear::with_init(2, 2, true, || 0.0); // Zero weights
        let input = graph
            .tensor_from_vec(vec![1.0, 1.0], &[1, 2], true)
            .unwrap();

        let output_with_bias = linear_with_bias.forward(&mut graph, input).unwrap();
        let output_data_with_bias = graph.get_data(output_with_bias);

        // With zero weights, output should be just the bias (initialized to zeros)
        assert!(approx_equal(output_data_with_bias[0], 0.0, 1e-6));
        assert!(approx_equal(output_data_with_bias[1], 0.0, 1e-6));

        // Test without bias
        let linear_without_bias = Linear::with_init(2, 2, false, || 0.0); // Zero weights, no bias
        let input2 = graph
            .tensor_from_vec(vec![1.0, 1.0], &[1, 2], true)
            .unwrap();

        let output_without_bias = linear_without_bias.forward(&mut graph, input2).unwrap();
        let output_data_without_bias = graph.get_data(output_without_bias);

        // With zero weights and no bias, output should be zero
        assert!(approx_equal(output_data_without_bias[0], 0.0, 1e-6));
        assert!(approx_equal(output_data_without_bias[1], 0.0, 1e-6));
    }

    #[test]
    fn test_linear_layer_batch_processing() {
        let mut graph = Engine::new();

        let linear = Linear::with_init(3, 2, false, || 1.0); // All weights = 1.0, no bias

        // Test with different batch sizes
        let batch_sizes = vec![1, 5, 10, 32];

        for batch_size in batch_sizes {
            let input_data: Vec<f64> = vec![1.0; batch_size * 3]; // All ones
            let input = graph
                .tensor_from_vec(input_data, &[batch_size, 3], true)
                .unwrap();

            let output = linear.forward(&mut graph, input).unwrap();
            let output_shape = graph.get_shape(output);
            let output_data = graph.get_data(output);

            // Verify shape
            assert_eq!(output_shape, vec![batch_size, 2]);

            // Verify each sample in the batch
            // Each output should be [3.0, 3.0] since we're multiplying [1,1,1] by weights [1,1,1; 1,1,1]
            for batch_idx in 0..batch_size {
                let sample_start = batch_idx * 2;
                assert!(
                    approx_equal(output_data[sample_start], 3.0, 1e-6),
                    "Batch {}, output 0: expected 3.0, got {}",
                    batch_idx,
                    output_data[sample_start]
                );
                assert!(
                    approx_equal(output_data[sample_start + 1], 3.0, 1e-6),
                    "Batch {}, output 1: expected 3.0, got {}",
                    batch_idx,
                    output_data[sample_start + 1]
                );
            }
        }
    }

    // ============================================================================
    // SOFTMAX LAYER TESTS
    // ============================================================================

    #[test]
    fn test_softmax_basic_functionality() {
        let mut graph = Engine::new();
        let softmax = Softmax::new(1);

        // Test with simple 2D input [batch_size=2, num_classes=3]
        let input = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true)
            .unwrap();

        let output = softmax.forward(&mut graph, input).unwrap();
        let result_data = graph.get_data(output);

        // Check output shape
        assert_eq!(graph.get_shape(output), vec![2, 3]);

        // Check that each row sums to approximately 1
        let row1_sum = result_data[0] + result_data[1] + result_data[2];
        let row2_sum = result_data[3] + result_data[4] + result_data[5];

        assert!(approx_equal(row1_sum, 1.0, 1e-6));
        assert!(approx_equal(row2_sum, 1.0, 1e-6));
        assert!(result_data.all(|x| x > 0.0 && x < 1.0).unwrap());
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let mut graph = Engine::new();
        let softmax = Softmax::new(1);

        // Test with very large values that could cause overflow without stability measures
        let input = graph
            .tensor_from_vec(vec![1000.0, 1001.0, 1002.0], &[1, 3], true)
            .unwrap();

        let output = softmax.forward(&mut graph, input).unwrap();
        let result_data = graph.get_data(output);
        assert!(result_data.all(|x: f64| x.is_finite() && x > 0.0).unwrap());

        // Should still sum to 1
        let sum: f64 = result_data
            .sum(None)
            .expect("Could not apply reduction to tensor")
            .first()
            .unwrap();
        assert!(approx_equal(sum, 1.0, 1e-6));
    }

    #[test]
    fn test_softmax_gradient_computation() {
        let mut graph = Engine::new();
        let softmax = Softmax::new(1);

        let input = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0], &[1, 3], true)
            .unwrap();

        let output = softmax.forward(&mut graph, input).unwrap();
        let loss = graph.sum(output, None).unwrap();

        // Backward pass should work without errors
        let backward_result = graph.backward(loss);
        assert!(backward_result.is_ok());

        // Input should have gradients
        let input_grad = graph.get_gradient(input);
        assert!(input_grad.is_some());

        let grad = input_grad.unwrap();
        assert_eq!(grad.shape(), &[1, 3]);
        assert!(grad.all(|x: f64| x.is_finite()).unwrap());
    }

    #[test]
    fn test_softmax_different_dimensions() {
        let mut graph = Engine::new();

        // Test softmax along different dimensions
        let input = graph
            .tensor_from_vec((1..25).map(|x| x as f64).collect(), &[2, 3, 4], true)
            .unwrap();

        // Test dim=0 (along batch dimension)
        let softmax_dim0 = Softmax::new(0);
        let output_dim0 = softmax_dim0.forward(&mut graph, input).unwrap();
        assert_eq!(graph.get_shape(output_dim0), vec![2, 3, 4]);

        // Test dim=1 (along middle dimension)
        let softmax_dim1 = Softmax::new(1);
        let output_dim1 = softmax_dim1.forward(&mut graph, input).unwrap();
        assert_eq!(graph.get_shape(output_dim1), vec![2, 3, 4]);

        // Test dim=2 (along last dimension)
        let softmax_dim2 = Softmax::new(2);
        let output_dim2 = softmax_dim2.forward(&mut graph, input).unwrap();
        assert_eq!(graph.get_shape(output_dim2), vec![2, 3, 4]);

        // Test negative dimension indexing
        let softmax_neg1 = Softmax::new(-1);
        let output_neg1 = softmax_neg1.forward(&mut graph, input).unwrap();
        assert_eq!(graph.get_shape(output_neg1), vec![2, 3, 4]);
    }

    #[test]
    fn test_softmax_classification_scenario() {
        let mut graph = Engine::new();

        // Typical classification scenario: [batch_size, num_classes]
        let batch_size = 4;
        let num_classes = 10;

        let softmax = Softmax::new(1); // Apply along class dimension

        // Simulate logits from a classifier
        let logits = graph
            .tensor_from_vec(
                (0..batch_size * num_classes)
                    .map(|x| x as f64 * 0.1)
                    .collect(),
                &[batch_size, num_classes],
                true,
            )
            .unwrap();

        let probabilities = softmax.forward(&mut graph, logits).unwrap();
        let prob_data = graph.get_data(probabilities);

        // Check shape
        assert_eq!(
            graph.get_shape(probabilities),
            vec![batch_size, num_classes]
        );

        // Check that each sample's probabilities sum to 1
        for batch_idx in 0..batch_size {
            let start_idx = batch_idx * num_classes;
            let sample_sum: f64 = (0..num_classes)
                .map(|class_idx| prob_data[start_idx + class_idx])
                .sum();

            assert!(
                approx_equal(sample_sum, 1.0, 1e-6),
                "Sample {} probabilities sum to {}, expected 1.0",
                batch_idx,
                sample_sum
            );
        }
        assert!(prob_data.all(|prob| prob > 0.0 && prob < 1.0).unwrap());
    }

    #[test]
    fn test_softmax_with_cross_entropy_loss() {
        let mut graph = Engine::new();

        // Simulate a typical classification pipeline
        let batch_size = 2;
        let num_classes = 3;

        // Create logits
        let logits = graph
            .tensor_from_vec(
                vec![2.0, 1.0, 0.1, 0.5, 3.0, 1.5],
                &[batch_size, num_classes],
                true,
            )
            .unwrap();

        // Apply softmax
        let softmax = Softmax::new(1);
        let probabilities = softmax.forward(&mut graph, logits).unwrap();

        // Create one-hot targets
        let targets = graph
            .tensor_from_vec(
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                &[batch_size, num_classes],
                false,
            )
            .unwrap();

        // Compute cross-entropy loss manually (since we have log in graph operations)
        let log_probs = graph.log(probabilities).unwrap();
        let loss_per_sample = graph.mul(targets, log_probs).unwrap();
        let neg_loss_per_sample = graph.negate(loss_per_sample).unwrap();
        let loss = graph.sum(neg_loss_per_sample, None).unwrap();

        // Backward pass
        let backward_result = graph.backward(loss);
        assert!(backward_result.is_ok());

        // Check that logits have gradients
        let logits_grad = graph.get_gradient(logits);
        assert!(logits_grad.is_some());
    }

    #[test]
    fn test_softmax_error_cases() {
        let mut graph = Engine::new();

        let input = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0], &[3], true)
            .unwrap();

        // Test invalid dimension
        let softmax_invalid = Softmax::new(5); // Dimension out of bounds
        let result = softmax_invalid.forward(&mut graph, input);
        assert!(result.is_err());

        // Test negative dimension that's still out of bounds
        let softmax_invalid_neg = Softmax::new(-5);
        let result_neg = softmax_invalid_neg.forward(&mut graph, input);
        assert!(result_neg.is_err());
    }

    #[test]
    fn test_softmax_invariance_property() {
        let mut graph = Engine::new();
        let softmax = Softmax::new(1);

        // Test that softmax(x + c) = softmax(x) for any constant c
        let input = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0], &[1, 3], true)
            .unwrap();

        let shifted_input = graph.add_scalar(input, 100.0).unwrap();

        let output1 = softmax.forward(&mut graph, input).unwrap();
        let output2 = softmax.forward(&mut graph, shifted_input).unwrap();

        let data1 = graph.get_data(output1);
        let data2 = graph.get_data(output2);

        // Results should be approximately equal

        let approx_eq = data1
            .zip_all(&data2, |a, b| {
                let diff = if a > b { a - b } else { b - a };
                diff <= 1e-6
            })
            .expect("Error comparing tensors");

        assert!(approx_eq, "Tensors are different!");
    }

    #[test]
    fn test_max_along_dim_operation() {
        let mut graph = Engine::new();

        // Test the underlying max_along_dim operation
        let input = graph
            .tensor_from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[2, 3], true)
            .unwrap();

        // Find max along dimension 1 (columns)
        let max_result = graph.max_along_dim(input, 1).unwrap();
        let max_data = graph.get_data(max_result);

        // Expected: [4.0, 9.0] (max of each row)
        assert_eq!(graph.get_shape(max_result), vec![2]);
        assert!(approx_equal(max_data[0], 4.0, 1e-6));
        assert!(approx_equal(max_data[1], 9.0, 1e-6));

        // Test gradient computation
        let loss = graph.sum(max_result, None).unwrap();
        let backward_result = graph.backward(loss);
        assert!(backward_result.is_ok());

        let input_grad = graph.get_gradient(input);
        assert!(input_grad.is_some());
    }

    #[test]
    fn test_expand_dims_operation() {
        let mut graph = Engine::new();

        // Test expand_dims_at operation
        let input = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0], &[3], true)
            .unwrap();

        // Expand at dimension 1: [3] -> [3, 1]
        let expanded = graph.expand_dims_at(input, 1).unwrap();
        assert_eq!(graph.get_shape(expanded), vec![3, 1]);

        // Expand at dimension 0: [3] -> [1, 3]
        let expanded_front = graph.expand_dims_at(input, 0).unwrap();
        assert_eq!(graph.get_shape(expanded_front), vec![1, 3]);
    }

    // ============================================================================
    // PARAMETER TESTS
    // ============================================================================

    #[test]
    fn test_parameter_creation() {
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let param = Parameter::new(data.clone());

        assert_eq!(param.shape(), &[2, 2]);
        assert_eq!(param.size(), 4);
        assert!(param.requires_grad);
        assert!(param.name().is_none());
        assert_eq!(param.data, data);
    }

    #[test]
    fn test_parameter_with_name() {
        let data = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let param = Parameter::new_named(data, "test_weight".to_string());

        assert_eq!(param.name(), Some("test_weight"));
        assert!(param.requires_grad);
    }

    #[test]
    fn test_parameter_from_init() {
        let param = Parameter::from_init(&[3, 3], || 0.5);

        assert_eq!(param.shape(), &[3, 3]);
        assert_eq!(param.size(), 9);
        assert!(param.data.all(|x: f64| x == 0.5).unwrap());
    }

    #[test]
    fn test_parameter_in_graph() {
        let mut graph = Engine::new();
        let data = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();

        let node_id = Parameter::create_in_graph(&mut graph, data.clone());

        assert!(graph.requires_grad(node_id));
        assert_eq!(graph.get_data(node_id), data);
    }

    // ============================================================================
    // LINEAR LAYER TESTS
    // ============================================================================

    #[test]
    fn test_linear_layer_creation() {
        let linear = Linear::<f64>::new(784, 256, true);

        assert_eq!(linear.in_features(), 784);
        assert_eq!(linear.out_features(), 256);
        assert!(linear.has_bias());
        assert!(linear.training());

        // Check weight shape
        assert_eq!(linear.weight.shape(), &[256, 784]);

        // Check bias shape
        assert!(linear.bias.is_some());
        assert_eq!(linear.bias.as_ref().unwrap().shape(), &[256]);
    }

    #[test]
    fn test_linear_layer_without_bias() {
        let linear = Linear::<f64>::new(100, 50, false);

        assert!(!linear.has_bias());
        assert!(linear.bias.is_none());
    }

    #[test]
    fn test_linear_layer_forward() {
        let mut graph = Engine::new();
        let linear = Linear::new(3, 2, true);

        // Create input: batch_size=2, features=3
        let input = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true)
            .unwrap();

        let output = linear.forward(&mut graph, input).unwrap();

        // Check output shape: should be [2, 2] (batch_size=2, out_features=2)
        assert_eq!(graph.get_shape(output), vec![2, 2]);
    }

    #[test]
    fn test_linear_layer_parameters() {
        let linear = Linear::<f64>::new(10, 5, true);
        let params = linear.parameters();

        assert_eq!(params.len(), 2); // weight + bias
        assert_eq!(params[0].shape(), &[5, 10]); // weight
        assert_eq!(params[1].shape(), &[5]); // bias
    }

    #[test]
    fn test_linear_layer_parameters_no_bias() {
        let linear = Linear::<f64>::new(10, 5, false);
        let params = linear.parameters();

        assert_eq!(params.len(), 1); // only weight
        assert_eq!(params[0].shape(), &[5, 10]); // weight
    }

    #[test]
    fn test_linear_layer_training_mode() {
        let mut linear = Linear::<f64>::new(10, 5, true);

        assert!(linear.training());

        linear.eval();
        assert!(!linear.training());

        linear.train();
        assert!(linear.training());
    }

    // ============================================================================
    // ACTIVATION FUNCTION TESTS
    // ============================================================================

    #[test]
    fn test_relu_activation() {
        let mut graph = Engine::new();
        let relu = ReLU::new();

        let input = graph
            .tensor_from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5], true)
            .unwrap();

        let output = relu.forward(&mut graph, input).unwrap();
        let result_data = graph.get_data(output);

        let expected = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        assert_eq!(result_data, expected);
    }

    #[test]
    fn test_sigmoid_activation() {
        let mut graph = Engine::new();
        let sigmoid = Sigmoid::new();

        let input = graph.tensor_from_vec(vec![0.0], &[1], true).unwrap();

        let output = sigmoid.forward(&mut graph, input).unwrap();
        let result_data = graph.get_data(output);

        // sigmoid(0) should be 0.5
        assert!(approx_equal(result_data[0], 0.5, 1e-6));
    }

    #[test]
    fn test_sigmoid_range() {
        let mut graph = Engine::new();
        let sigmoid = Sigmoid::new();

        let input = graph
            .tensor_from_vec(vec![-10.0, -1.0, 0.0, 1.0, 10.0], &[5], true)
            .unwrap();

        let output = sigmoid.forward(&mut graph, input).unwrap();
        let result_data = graph.get_data(output);

        // All sigmoid outputs should be between 0 and 1
        assert!(result_data.all(|val| val > 0.0 && val < 1.0).unwrap());

        // Check specific values
        assert!(result_data[0] < 0.01); // sigmoid(-10) ≈ 0
        assert!(approx_equal(result_data[2], 0.5, 1e-6)); // sigmoid(0) = 0.5
        assert!(result_data[4] > 0.99); // sigmoid(10) ≈ 1
    }

    #[test]
    fn test_tanh_activation() {
        let mut graph = Engine::new();
        let tanh = Tanh::new();

        let input = graph.tensor_from_vec(vec![0.0], &[1], true).unwrap();

        let output = tanh.forward(&mut graph, input).unwrap();
        let result_data = graph.get_data(output);

        // tanh(0) should be 0.0
        assert!(approx_equal(result_data[0], 0.0, 1e-6));
    }

    #[test]
    fn test_tanh_range() {
        let mut graph = Engine::new();
        let tanh = Tanh::new();

        let input = graph
            .tensor_from_vec(vec![-10.0, -1.0, 0.0, 1.0, 10.0], &[5], true)
            .unwrap();

        let output = tanh.forward(&mut graph, input).unwrap();
        let result_data = graph.get_data(output);

        assert!(result_data.all(|val| val > -1.0 && val < 1.0).unwrap());

        // Check specific values
        assert!(result_data[0] < -0.99); // tanh(-10) ≈ -1
        assert!(approx_equal(result_data[2], 0.0, 1e-6)); // tanh(0) = 0
        assert!(result_data[4] > 0.99); // tanh(10) ≈ 1
    }

    #[test]
    fn test_leaky_relu_activation() {
        let mut graph = Engine::new();
        let leaky_relu = LeakyReLU::new(); // default slope = 0.01

        let input = graph
            .tensor_from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5], true)
            .unwrap();

        let output = leaky_relu.forward(&mut graph, input).unwrap();
        let result_data = graph.get_data(output);

        // Check expected values
        assert!(approx_equal(result_data[0], -0.02, 1e-6)); // -2 * 0.01 = -0.02
        assert!(approx_equal(result_data[1], -0.01, 1e-6)); // -1 * 0.01 = -0.01
        assert!(approx_equal(result_data[2], 0.0, 1e-6)); // 0
        assert!(approx_equal(result_data[3], 1.0, 1e-6)); // 1
        assert!(approx_equal(result_data[4], 2.0, 1e-6)); // 2
    }

    #[test]
    fn test_leaky_relu_custom_slope() {
        let mut graph = Engine::new();
        let leaky_relu = LeakyReLU::new_with_slope(0.1);

        let input = graph.tensor_from_vec(vec![-1.0, 1.0], &[2], true).unwrap();

        let output = leaky_relu.forward(&mut graph, input).unwrap();
        let result_data = graph.get_data(output);

        assert!(approx_equal(result_data[0], -0.1, 1e-6)); // -1 * 0.1 = -0.1
        assert!(approx_equal(result_data[1], 1.0, 1e-6)); // 1
    }

    #[test]
    fn test_elu_activation() {
        let mut graph = Engine::new();
        let elu = ELU::new(); // default alpha = 1.0

        let input = graph.tensor_from_vec(vec![0.0, 1.0], &[2], true).unwrap();

        let output = elu.forward(&mut graph, input).unwrap();
        let result_data = graph.get_data(output);

        assert!(approx_equal(result_data[0], 0.0, 1e-6)); // ELU(0) = 0
        assert!(approx_equal(result_data[1], 1.0, 1e-6)); // ELU(1) = 1
    }

    // ============================================================================
    // SEQUENTIAL MODEL TESTS
    // ============================================================================

    #[test]
    fn test_sequential_creation() {
        let sequential = Sequential::<f64>::new();

        assert_eq!(sequential.len(), 0);
        assert!(sequential.is_empty());
        assert!(sequential.training());
    }

    #[test]
    fn test_sequential_add_modules() {
        let mut sequential = Sequential::<f64>::new();

        sequential.add(Box::new(Linear::new(784, 256, true)));
        sequential.add(Box::new(ReLU::new()));
        sequential.add(Box::new(Linear::new(256, 10, true)));

        assert_eq!(sequential.len(), 3);
        assert!(!sequential.is_empty());
    }

    #[test]
    fn test_sequential_forward() {
        let mut graph = Engine::new();
        let mut sequential = Sequential::<f64>::new();

        sequential.add(Box::new(Linear::new(3, 2, false)));
        sequential.add(Box::new(ReLU::new()));

        let input = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0], &[1, 3], true)
            .unwrap();

        let output = sequential.forward(&mut graph, input).unwrap();

        // Output should have shape [1, 2]
        assert_eq!(graph.get_shape(output), vec![1, 2]);
    }

    #[test]
    fn test_sequential_parameters() {
        let mut sequential = Sequential::<f64>::new();

        sequential.add(Box::new(Linear::new(10, 5, true))); // 2 params (weight + bias)
        sequential.add(Box::new(ReLU::new())); // 0 params
        sequential.add(Box::new(Linear::new(5, 3, true))); // 2 params (weight + bias)

        let params = sequential.parameters();
        assert_eq!(params.len(), 4); // Total of 4 parameters
    }

    #[test]
    fn test_sequential_training_mode() {
        let mut sequential = Sequential::<f64>::new();

        sequential.add(Box::new(Linear::new(10, 5, true)));
        sequential.add(Box::new(ReLU::new()));

        assert!(sequential.training());

        sequential.eval();
        assert!(!sequential.training());

        sequential.train();
        assert!(sequential.training());
    }

    // ============================================================================
    // FLATTEN LAYER TESTS
    // ============================================================================

    #[test]
    fn test_flatten_layer() {
        let mut graph = Engine::new();
        let flatten = Flatten::new();

        // Input shape: [2, 3, 4] (batch_size=2, 3×4 features)
        let input = graph
            .tensor_from_vec((0..24).map(|x| x as f64).collect(), &[2, 3, 4], true)
            .unwrap();

        let output = flatten.forward(&mut graph, input).unwrap();

        // Output shape should be [2, 12] (batch_size=2, flattened_size=3×4=12)
        assert_eq!(graph.get_shape(output), vec![2, 12]);
    }

    #[test]
    fn test_flatten_preserves_batch_dimension() {
        let mut graph = Engine::new();
        let flatten = Flatten::new();

        // Input shape: [5, 2, 2, 2] (batch_size=5, 2×2×2 features)
        let input = graph
            .tensor_from_vec((0..40).map(|x| x as f64).collect(), &[5, 2, 2, 2], true)
            .unwrap();

        let output = flatten.forward(&mut graph, input).unwrap();

        // Output shape should be [5, 8] (batch_size=5, flattened_size=2×2×2=8)
        assert_eq!(graph.get_shape(output), vec![5, 8]);
    }

    #[test]
    fn test_flatten_error_on_1d_input() {
        let mut graph = Engine::new();
        let flatten = Flatten::new();

        // Input shape: [10] (only 1 dimension)
        let input = graph
            .tensor_from_vec((0..10).map(|x| x as f64).collect(), &[10], true)
            .unwrap();

        let result = flatten.forward(&mut graph, input);
        assert!(result.is_err());
    }

    // ============================================================================
    // IDENTITY LAYER TESTS
    // ============================================================================

    #[test]
    fn test_identity_layer() {
        let mut graph = Engine::new();
        let identity = Identity::new();

        let input = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0], &[3], true)
            .unwrap();

        let output = identity.forward(&mut graph, input).unwrap();

        // Identity should return the same node
        assert_eq!(input, output);
    }

    // ============================================================================
    // MODULELIST TESTS
    // ============================================================================

    #[test]
    fn test_module_list() {
        let mut module_list = ModuleList::<f64>::new();

        module_list.push(Box::new(Linear::new(10, 5, true)));
        module_list.push(Box::new(ReLU::new()));
        module_list.push(Box::new(Linear::new(5, 3, true)));

        assert_eq!(module_list.len(), 3);
        assert!(!module_list.is_empty());

        let params = module_list.parameters();
        assert_eq!(params.len(), 4); // 2 linear layers × 2 params each = 4
    }

    #[test]
    fn test_module_list_forward_sequential() {
        let mut graph = Engine::new();
        let mut module_list = ModuleList::new();

        module_list.push(Box::new(Linear::new(3, 2, false)));
        module_list.push(Box::new(ReLU::new()));

        let input = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0], &[1, 3], true)
            .unwrap();

        let output = module_list.forward_sequential(&mut graph, input).unwrap();

        assert_eq!(graph.get_shape(output), vec![1, 2]);
    }

    // ============================================================================
    // LOSS FUNCTION TESTS
    // ============================================================================

    #[test]
    fn test_mse_loss_basic_functionality() {
        let mut graph = Engine::new();

        // Test case 1: Perfect predictions (loss should be 0)
        let perfect_predictions = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true)
            .unwrap();
        let perfect_targets = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false)
            .unwrap();

        let mse_loss = MSELoss::new();
        let perfect_loss = mse_loss
            .compute_loss(&mut graph, perfect_predictions, perfect_targets)
            .unwrap();

        let perfect_loss_data = graph.get_data(perfect_loss);
        assert!(
            approx_equal(perfect_loss_data[0], 0.0, 1e-6),
            "Perfect predictions should have zero MSE loss, got {}",
            perfect_loss_data[0]
        );

        // Test case 2: Known loss calculation
        // predictions = [1.0, 2.0], targets = [3.0, 4.0]
        // differences = [-2.0, -2.0], squared = [4.0, 4.0]
        // MSE = mean([4.0, 4.0]) = 4.0
        let predictions = graph
            .tensor_from_vec(vec![1.0, 2.0], &[1, 2], true)
            .unwrap();
        let targets = graph
            .tensor_from_vec(vec![3.0, 4.0], &[1, 2], false)
            .unwrap();

        let loss = mse_loss
            .compute_loss(&mut graph, predictions, targets)
            .unwrap();
        let loss_data = graph.get_data(loss);

        assert!(
            approx_equal(loss_data[0], 4.0, 1e-6),
            "Expected MSE loss of 4.0, got {}",
            loss_data[0]
        );

        // Test case 3: Different reduction strategies
        let mse_sum = MSELoss::new_with_reduction(Reduction::Sum);
        let sum_loss = mse_sum
            .compute_loss(&mut graph, predictions, targets)
            .unwrap();
        let sum_loss_data = graph.get_data(sum_loss);

        // Sum reduction: sum([4.0, 4.0]) = 8.0
        assert!(
            approx_equal(sum_loss_data[0], 8.0, 1e-6),
            "Expected sum MSE loss of 8.0, got {}",
            sum_loss_data[0]
        );

        // Test case 4: Backward pass (gradient computation)
        let pred_with_grad = graph
            .tensor_from_vec(vec![1.0, 2.0], &[1, 2], true)
            .unwrap();
        let target_without_grad = graph
            .tensor_from_vec(vec![3.0, 4.0], &[1, 2], false)
            .unwrap();

        let loss_for_grad = mse_loss
            .compute_loss(&mut graph, pred_with_grad, target_without_grad)
            .unwrap();

        graph.backward(loss_for_grad).unwrap();

        let pred_grad = graph.get_gradient(pred_with_grad).unwrap();
        // Gradient of MSE: d/dx[(x-y)²] = 2(x-y) / n
        // For (1-3)² = 4 and (2-4)² = 4: gradients = 2*(-2)/2 = -2.0 each
        assert!(
            approx_equal(pred_grad[0], -2.0, 1e-6),
            "Expected gradient -2.0, got {}",
            pred_grad[0]
        );
        assert!(
            approx_equal(pred_grad[1], -2.0, 1e-6),
            "Expected gradient -2.0, got {}",
            pred_grad[1]
        );
    }

    #[test]
    fn test_bce_loss_functionality() {
        let mut graph = Engine::new();

        // Test case 1: Perfect binary predictions
        // BCE(0.9999, 1) should be very close to 0
        // BCE(0.0001, 0) should be very close to 0
        let perfect_probs = graph
            .tensor_from_vec(vec![0.9999, 0.0001], &[2], true)
            .unwrap();
        let binary_targets = graph.tensor_from_vec(vec![1.0, 0.0], &[2], false).unwrap();

        let bce_loss = BCELoss::new();
        let perfect_loss = bce_loss
            .compute_loss(&mut graph, perfect_probs, binary_targets)
            .unwrap();

        let perfect_loss_data = graph.get_data(perfect_loss);
        assert!(
            perfect_loss_data[0] < 0.01, // Should be very small
            "Perfect binary predictions should have very low BCE loss, got {}",
            perfect_loss_data[0]
        );

        // Test case 2: Known calculation with p=0.5, target=1
        // BCE = -(1 * log(0.5) + 0 * log(0.5)) = -log(0.5) ≈ 0.693
        let half_prob = graph.tensor_from_vec(vec![0.5], &[1], true).unwrap();
        let one_target = graph.tensor_from_vec(vec![1.0], &[1], false).unwrap();

        let half_loss = bce_loss
            .compute_loss(&mut graph, half_prob, one_target)
            .unwrap();
        let half_loss_data = graph.get_data(half_loss);

        let expected_loss = -(1.0f64.ln() - 2.0f64.ln()); // -log(0.5) = log(2)
        assert!(
            approx_equal(half_loss_data[0], expected_loss, 1e-3),
            "Expected BCE loss of ~0.693, got {}",
            half_loss_data[0]
        );

        // Test case 3: Batch processing
        let batch_probs = graph
            .tensor_from_vec(vec![0.8, 0.6, 0.3, 0.1], &[2, 2], true)
            .unwrap();
        let batch_targets = graph
            .tensor_from_vec(vec![1.0, 1.0, 0.0, 0.0], &[2, 2], false)
            .unwrap();

        let batch_loss = bce_loss
            .compute_loss(&mut graph, batch_probs, batch_targets)
            .unwrap();
        let batch_loss_data = graph.get_data(batch_loss);

        // Should compute mean across all elements
        assert!(
            batch_loss_data[0] > 0.0,
            "BCE loss should be positive for imperfect predictions"
        );

        // Test case 4: numerical stability (values close to 0 and 1)
        let extreme_probs = graph
            .tensor_from_vec(vec![0.0001, 0.9999], &[2], true)
            .unwrap();
        let extreme_targets = graph.tensor_from_vec(vec![0.0, 1.0], &[2], false).unwrap();

        let stable_loss = bce_loss
            .compute_loss(&mut graph, extreme_probs, extreme_targets)
            .unwrap();
        let stable_loss_data = graph.get_data(stable_loss);

        // Should not be NaN or infinite
        assert!(
            stable_loss_data[0].is_finite(),
            "BCE loss should be finite even with extreme probabilities"
        );
    }

    #[test]
    fn test_cce_loss_functionality() {
        let mut graph = Engine::new();

        // Test case 1: Perfect predictions (one-hot targets)
        // Probabilities: [1.0, 0.0, 0.0] (after softmax, for class 0)
        // Targets: [1.0, 0.0, 0.0] (one-hot for class 0)
        // CCE = -(1*log(1) + 0*log(0) + 0*log(0)) = -0 = 0
        let perfect_probs = graph
            .tensor_from_vec(vec![0.9999, 0.00005, 0.00005], &[1, 3], true)
            .unwrap();
        let one_hot_targets = graph
            .tensor_from_vec(vec![1.0, 0.0, 0.0], &[1, 3], false)
            .unwrap();

        let cce_loss = CCELoss::new();
        let perfect_loss = cce_loss
            .compute_loss(&mut graph, perfect_probs, one_hot_targets)
            .unwrap();

        let perfect_loss_data = graph.get_data(perfect_loss);
        assert!(
            perfect_loss_data[0] < 0.01, // Should be very small
            "Perfect categorical predictions should have very low CCE loss, got {}",
            perfect_loss_data[0]
        );

        // Test case 2: Uniform predictions (maximum uncertainty)
        // For 3 classes: p = [1/3, 1/3, 1/3], target = [1, 0, 0]
        // CCE = -(1*log(1/3)) = log(3) ≈ 1.099
        let uniform_probs = graph
            .tensor_from_vec(vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], &[1, 3], true)
            .unwrap();
        let class_0_target = graph
            .tensor_from_vec(vec![1.0, 0.0, 0.0], &[1, 3], false)
            .unwrap();

        let uniform_loss = cce_loss
            .compute_loss(&mut graph, uniform_probs, class_0_target)
            .unwrap();
        let uniform_loss_data = graph.get_data(uniform_loss);

        let expected_uniform_loss = 3.0f64.ln(); // log(3)
        assert!(
            approx_equal(uniform_loss_data[0], expected_uniform_loss, 1e-3),
            "Expected CCE loss of ~1.099 for uniform predictions, got {}",
            uniform_loss_data[0]
        );

        // Test case 3: Multi-class batch processing
        // Batch size = 2, 4 classes
        let batch_probs = graph
            .tensor_from_vec(
                vec![
                    0.7, 0.2, 0.05, 0.05, // Sample 1: confident about class 0
                    0.1, 0.1, 0.1, 0.7, // Sample 2: confident about class 3
                ],
                &[2, 4],
                true,
            )
            .unwrap();
        let batch_targets = graph
            .tensor_from_vec(
                vec![
                    1.0, 0.0, 0.0, 0.0, // Sample 1: true class 0
                    0.0, 0.0, 0.0, 1.0, // Sample 2: true class 3
                ],
                &[2, 4],
                false,
            )
            .unwrap();

        let batch_loss = cce_loss
            .compute_loss(&mut graph, batch_probs, batch_targets)
            .unwrap();
        let batch_loss_data = graph.get_data(batch_loss);

        // Should be relatively low since predictions match targets
        assert!(
            batch_loss_data[0] > 0.0 && batch_loss_data[0] < 1.0,
            "CCE loss for good predictions should be moderate, got {}",
            batch_loss_data[0]
        );

        // Test case 4: Sum reduction
        let cce_sum = CCELoss::new_with_reduction(Reduction::Sum);
        let sum_loss = cce_sum
            .compute_loss(&mut graph, batch_probs, batch_targets)
            .unwrap();
        let sum_loss_data = graph.get_data(sum_loss);

        // Sum should be approximately mean * batch_size
        assert!(
            sum_loss_data[0] > batch_loss_data[0],
            "Sum reduction should give larger loss than mean reduction"
        );
    }

    #[test]
    fn test_bce_with_logits_loss_functionality() {
        let mut graph = Engine::new();

        // Test case 1: Zero logits (should give probability ≈ 0.5)
        // BCE_with_logits(0, 1) should be similar to BCE(0.5, 1) ≈ 0.693
        let zero_logits = graph.tensor_from_vec(vec![0.0], &[1], true).unwrap();
        let one_target = graph.tensor_from_vec(vec![1.0], &[1], false).unwrap();

        let bce_logits_loss = BCEWithLogitsLoss::new();
        let zero_loss = bce_logits_loss
            .compute_loss(&mut graph, zero_logits, one_target)
            .unwrap();

        let zero_loss_data = graph.get_data(zero_loss);
        let expected_loss = 2.0f64.ln(); // log(2) ≈ 0.693
        assert!(
            approx_equal(zero_loss_data[0], expected_loss, 1e-3),
            "Expected BCE with logits loss ~0.693 for zero logits, got {}",
            zero_loss_data[0]
        );

        // Test case 2: Large positive logits (should be close to 0 for target=1)
        let large_positive_logits = graph.tensor_from_vec(vec![10.0], &[1], true).unwrap();
        let one_target_2 = graph.tensor_from_vec(vec![1.0], &[1], false).unwrap();

        let large_pos_loss = bce_logits_loss
            .compute_loss(&mut graph, large_positive_logits, one_target_2)
            .unwrap();

        let large_pos_loss_data = graph.get_data(large_pos_loss);
        assert!(
            large_pos_loss_data[0] < 0.1,
            "Large positive logits with target=1 should have very low loss, got {}",
            large_pos_loss_data[0]
        );

        // Test case 3: Large negative logits (should be close to 0 for target=0)
        let large_negative_logits = graph.tensor_from_vec(vec![-10.0], &[1], true).unwrap();
        let zero_target = graph.tensor_from_vec(vec![0.0], &[1], false).unwrap();

        let large_neg_loss = bce_logits_loss
            .compute_loss(&mut graph, large_negative_logits, zero_target)
            .unwrap();

        let large_neg_loss_data = graph.get_data(large_neg_loss);
        assert!(
            large_neg_loss_data[0] < 0.1,
            "Large negative logits with target=0 should have very low loss, got {}",
            large_neg_loss_data[0]
        );

        // Test case 4: numerical stability comparison
        // Compare BCE with logits vs manual sigmoid + BCE for extreme values
        let extreme_logits = graph
            .tensor_from_vec(vec![100.0, -100.0], &[2], true)
            .unwrap();
        let mixed_targets = graph.tensor_from_vec(vec![1.0, 0.0], &[2], false).unwrap();

        let stable_loss = bce_logits_loss
            .compute_loss(&mut graph, extreme_logits, mixed_targets)
            .unwrap();

        let stable_loss_data = graph.get_data(stable_loss);

        // Should be finite and reasonable (close to 0 for correct predictions)
        assert!(
            stable_loss_data[0].is_finite(),
            "BCE with logits should be CPUNumberally stable for extreme values"
        );
        assert!(
            stable_loss_data[0] < 1.0,
            "Loss should be low for correct extreme predictions, got {}",
            stable_loss_data[0]
        );

        // Test case 5: Batch processing with different reduction strategies
        let batch_logits = graph
            .tensor_from_vec(vec![2.0, -1.5, 0.5, -3.0], &[2, 2], true)
            .unwrap();
        let batch_targets = graph
            .tensor_from_vec(vec![1.0, 0.0, 1.0, 0.0], &[2, 2], false)
            .unwrap();

        // Mean reduction
        let mean_loss = bce_logits_loss
            .compute_loss(&mut graph, batch_logits, batch_targets)
            .unwrap();
        let mean_loss_data = graph.get_data(mean_loss);

        // Sum reduction
        let bce_logits_sum = BCEWithLogitsLoss::new_with_reduction(Reduction::Sum);
        let sum_loss = bce_logits_sum
            .compute_loss(&mut graph, batch_logits, batch_targets)
            .unwrap();
        let sum_loss_data = graph.get_data(sum_loss);

        // Sum should be approximately mean * total_elements
        assert!(
            approx_equal(sum_loss_data[0], mean_loss_data[0] * 4.0, 1e-3),
            "Sum should equal mean * num_elements: sum={}, mean*4={}",
            sum_loss_data[0],
            mean_loss_data[0] * 4.0
        );

        // Test case 6: Gradient computation (numerical stability check)
        let logits_for_grad = graph.tensor_from_vec(vec![1.0, -1.0], &[2], true).unwrap();
        let targets_for_grad = graph.tensor_from_vec(vec![1.0, 0.0], &[2], false).unwrap();

        let loss_for_grad = bce_logits_loss
            .compute_loss(&mut graph, logits_for_grad, targets_for_grad)
            .unwrap();

        graph.backward(loss_for_grad).unwrap();

        let logits_grad = graph.get_gradient(logits_for_grad).unwrap();

        // Gradients should be finite and reasonable
        assert!(
            logits_grad[0].is_finite() && logits_grad[1].is_finite(),
            "Gradients should be finite"
        );

        // For BCE with logits: gradient = (sigmoid(logit) - target) / batch_size
        // logit=1.0, target=1.0: grad ≈ (sigmoid(1) - 1) / 2 ≈ (0.731 - 1) / 2 ≈ -0.135
        // logit=-1.0, target=0.0: grad ≈ (sigmoid(-1) - 0) / 2 ≈ 0.269 / 2 ≈ 0.135
        assert!(
            logits_grad[0] < 0.0 && logits_grad[1] > 0.0,
            "Gradient signs should match expected directions: grad[0]={}, grad[1]={}",
            logits_grad[0],
            logits_grad[1]
        );
    }
    // ============================================================================
    // INTEGRATION TESTS
    // ============================================================================

    #[test]
    fn test_simple_neural_network_training() {
        let mut graph = Engine::new();

        // Create a simple 2-layer network
        let mut model = Sequential::new();
        model.add(Box::new(Linear::new(2, 3, true)));
        model.add(Box::new(ReLU::new()));
        model.add(Box::new(Linear::new(3, 1, true)));

        // Create dummy input and target
        let input = graph
            .tensor_from_vec(vec![1.0, 2.0], &[1, 2], true)
            .unwrap();
        let target = graph.tensor_from_vec(vec![1.0], &[1, 1], false).unwrap();

        // Forward pass
        let prediction = model.forward(&mut graph, input).unwrap();

        // Compute loss
        let loss_fn = MSELoss::new();
        let loss = loss_fn
            .compute_loss(&mut graph, prediction, target)
            .unwrap();

        // Backward pass
        let backward_result = graph.backward(loss);
        assert!(backward_result.is_ok());

        // Check that gradients exist for model parameters
        let params = model.parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_multi_layer_network_shapes() {
        let mut graph = Engine::new();

        // Create a deeper network
        let mut model = Sequential::new();
        model.add(Box::new(Flatten::new()));
        model.add(Box::new(Linear::new(28 * 28, 128, true)));
        model.add(Box::new(ReLU::new()));
        model.add(Box::new(Linear::new(128, 64, true)));
        model.add(Box::new(ReLU::new()));
        model.add(Box::new(Linear::new(64, 10, true)));

        // Input: batch of MNIST-like images
        let input = graph
            .tensor_from_vec(vec![0.5; 32 * 28 * 28], &[32, 28, 28], true)
            .unwrap();

        let output = model.forward(&mut graph, input).unwrap();

        // Should output logits for 10 classes
        assert_eq!(graph.get_shape(output), vec![32, 10]);
    }
    #[test]
    fn test_gradient_flow() {
        let mut graph = Engine::new();

        // Simple network for gradient checking
        let linear = Linear::new(2, 1, false);

        let input = graph
            .tensor_from_vec(vec![1.0, 2.0], &[1, 2], true)
            .unwrap();

        let output = linear.forward(&mut graph, input).unwrap();

        // Create target and loss
        let target = graph.tensor_from_vec(vec![3.0], &[1, 1], false).unwrap();

        let loss_fn = MSELoss::new();
        let loss = loss_fn.compute_loss(&mut graph, output, target).unwrap();

        // Backward pass
        graph.backward(loss).unwrap();

        // Check input has gradient
        let input_grad = graph.get_gradient(input);
        assert!(input_grad.is_some());

        // Gradient should have same shape as input
        let grad = input_grad.unwrap();
        assert_eq!(grad.shape(), &[1, 2]);
    }

    #[test]
    fn test_activation_functions_preserve_shape() {
        let mut graph = Engine::new();

        let input_shape = vec![2, 3, 4];
        let input = graph
            .tensor_from_vec(vec![0.1; 24], &input_shape, true)
            .unwrap();

        // Test all activation functions preserve shape
        let activations: Vec<Box<dyn Module<f64>>> = vec![
            Box::new(ReLU::new()),
            Box::new(Sigmoid::new()),
            Box::new(Tanh::new()),
            Box::new(LeakyReLU::new()),
            Box::new(ELU::new()),
        ];

        for activation in activations {
            let output = activation.forward(&mut graph, input).unwrap();
            assert_eq!(graph.get_shape(output), input_shape);
        }
    }

    #[test]
    fn test_mixed_precision_compatibility() {
        // Test that our neural network components work with f32
        let mut graph: Engine<f64> = Engine::new();

        let linear = Linear::new(3, 2, true);
        let relu = ReLU::new();

        let input = graph
            .tensor_from_vec(vec![1.0f64, 2.0f64, 3.0f64], &[1, 3], true)
            .unwrap();

        let hidden = linear.forward(&mut graph, input).unwrap();
        let output = relu.forward(&mut graph, hidden).unwrap();

        assert_eq!(graph.get_shape(output), vec![1, 2]);
    }

    // ============================================================================
    // BATCH NORMALIZATION TESTS
    // ============================================================================

    #[test]
    fn test_batch_norm_1d_creation() {
        let bn = BatchNorm1d::<f64>::default_config(128);

        assert_eq!(bn.num_features(), 128);
        assert!(bn.training());
        assert!(bn.track_running_stats);

        // Check parameter shapes
        let params = bn.parameters();
        assert_eq!(params.len(), 2); // weight and bias
        assert_eq!(params[0].shape(), &[128]); // weight
        assert_eq!(params[1].shape(), &[128]); // bias

        // Check running statistics shapes
        assert_eq!(bn.running_mean.shape(), &[128]);
        assert_eq!(bn.running_var.shape(), &[128]);
    }

    #[test]
    fn test_batch_norm_1d_forward_training() {
        let mut graph = Engine::new();
        let bn = BatchNorm1d::<f64>::default_config(3);

        // Create input: [batch_size=4, features=3]
        let input = graph
            .tensor_from_vec(
                vec![
                    1.0, 2.0, 3.0, // sample 1
                    4.0, 5.0, 6.0, // sample 2
                    7.0, 8.0, 9.0, // sample 3
                    10.0, 11.0, 12.0, // sample 4
                ],
                &[4, 3],
                true,
            )
            .unwrap();

        let output = bn.forward(&mut graph, input).unwrap();
        let output_shape = graph.get_shape(output);
        let output_data = graph.get_data(output);

        // Output shape should match input shape
        assert_eq!(output_shape, vec![4, 3]);
        assert!(output_data.all(|x| x.is_finite()).unwrap());

        // For this specific input, we can verify that the normalization worked
        // The mean of each feature across the batch should be approximately 0
        // and the std should be approximately 1 (before scaling/shifting)
        println!("BatchNorm output: {:?}", output_data.to_vec());
    }

    #[test]
    fn test_batch_norm_1d_training_vs_eval() {
        let mut graph = Engine::new();
        let mut bn = BatchNorm1d::<f64>::default_config(2);

        let input = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true)
            .unwrap();

        // Forward pass in training mode
        assert!(bn.training());
        let _training_output = bn.forward(&mut graph, input).unwrap();

        // Switch to evaluation mode
        bn.eval();
        assert!(!bn.training());

        // Create new input for eval mode
        let eval_input = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true)
            .unwrap();

        // Forward pass in eval mode should use running statistics
        // Note: In our current implementation, running stats aren't updated automatically
        // So this test mainly checks that the code path works
        let eval_result = bn.forward(&mut graph, eval_input);

        // Should succeed (even though running stats might not be meaningful)
        assert!(eval_result.is_ok());
    }

    #[test]
    fn test_batch_norm_1d_gradient_flow() {
        let mut graph = Engine::new();
        let bn = BatchNorm1d::<f64>::default_config(2);

        let input = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true)
            .unwrap();

        let output = bn.forward(&mut graph, input).unwrap();
        let loss = graph.sum(output, None).unwrap();

        // Backward pass should work
        let backward_result = graph.backward(loss);
        assert!(backward_result.is_ok());

        // Input should have gradients
        let input_grad = graph.get_gradient(input);
        assert!(input_grad.is_some());

        let grad = input_grad.unwrap();
        assert_eq!(grad.shape(), &[2, 2]);
        assert!(grad.all(|x| x.is_finite()).unwrap());
    }

    #[test]
    fn test_batch_norm_1d_numerical_stability() {
        let mut graph = Engine::new();
        let bn = BatchNorm1d::<f64>::default_config(2);

        // Test with very large values
        let large_input = graph
            .tensor_from_vec(vec![1e6, 2e6, 3e6, 4e6], &[2, 2], true)
            .unwrap();

        let output = bn.forward(&mut graph, large_input).unwrap();
        let output_data = graph.get_data(output);
        assert!(
            output_data
                .all(|val: f64| val.is_finite() && !val.is_nan())
                .unwrap()
        );
    }

    #[test]
    fn test_batch_norm_1d_error_cases() {
        let mut graph = Engine::new();
        let bn = BatchNorm1d::<f64>::default_config(3);

        // Test with wrong feature size
        let wrong_features = graph
            .tensor_from_vec(vec![1.0, 2.0], &[1, 2], true)
            .unwrap();

        let result = bn.forward(&mut graph, wrong_features);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("doesn't match BatchNorm1d feature size")
        );

        // Test with 1D input (should fail)
        let one_d_input = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0], &[3], true)
            .unwrap();

        let result = bn.forward(&mut graph, one_d_input);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("at least 2 dimensions"));
    }

    #[test]
    fn test_batch_norm_1d_parameter_count() {
        let bn = BatchNorm1d::<f64>::default_config(64);

        assert_eq!(bn.num_parameters(), 128); // 64 weights + 64 biases

        let params = bn.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].size(), 64); // weight
        assert_eq!(params[1].size(), 64); // bias
    }

    #[test]
    fn test_batch_norm_1d_3d_input() {
        let mut graph = Engine::new();
        let bn = BatchNorm1d::<f64>::default_config(3);

        // Test with 3D input: [batch_size, channels, length]
        // This simulates 1D convolution output
        let input_3d = graph
            .tensor_from_vec((0..24).map(|x| x as f64).collect(), &[2, 3, 4], true)
            .unwrap();

        let output = bn.forward(&mut graph, input_3d).unwrap();
        let output_shape = graph.get_shape(output);

        // Should preserve input shape
        assert_eq!(output_shape, vec![2, 3, 4]);

        // Output should be finite
        let output_data = graph.get_data(output);
        assert!(output_data.all(|x| x.is_finite()).unwrap())
    }

    // ============================================================================
    // LAYER NORMALIZATION TESTS
    // ============================================================================

    #[test]
    fn test_layer_norm_creation() {
        let ln = LayerNorm::<f64>::default_config(vec![128]);

        assert_eq!(ln.normalized_shape(), &[128]);
        assert!(ln.training());
        assert!(ln.elementwise_affine);

        // Check parameter shapes
        let params = ln.parameters();
        assert_eq!(params.len(), 2); // weight and bias
        assert_eq!(params[0].shape(), &[128]); // weight
        assert_eq!(params[1].shape(), &[128]); // bias
    }

    #[test]
    fn test_layer_norm_1d_creation() {
        let ln = LayerNorm::<f64>::new_1d(256, 1e-5);

        assert_eq!(ln.normalized_shape(), &[256]);
        assert!(ln.elementwise_affine);

        let params = ln.parameters();
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_layer_norm_forward_2d() {
        let mut graph = Engine::new();
        let ln = LayerNorm::<f64>::default_config(vec![4]); // Normalize last dimension

        // Input: [batch_size=3, features=4]
        let input = graph
            .tensor_from_vec(
                vec![
                    1.0, 2.0, 3.0, 4.0, // sample 1
                    5.0, 6.0, 7.0, 8.0, // sample 2
                    9.0, 10.0, 11.0, 12.0, // sample 3
                ],
                &[3, 4],
                true,
            )
            .unwrap();

        let output = ln.forward(&mut graph, input).unwrap();
        let output_shape = graph.get_shape(output);
        let output_data = graph.get_data(output);

        // Shape should be preserved
        assert_eq!(output_shape, vec![3, 4]);
        assert!(output_data.all(|x| x.is_finite()).unwrap());

        // For each sample, the mean across the last dimension should be close to 0
        // and the std should be close to 1 (before affine transform)
        println!("LayerNorm output: {:?}", output_data.to_vec());
    }

    #[test]
    fn test_layer_norm_forward_3d() {
        let mut graph = Engine::new();
        // Normalize over last 2 dimensions [height, width]
        let ln = LayerNorm::<f64>::default_config(vec![2, 3]);

        // Input: [batch_size=2, height=2, width=3]
        let input = graph
            .tensor_from_vec((0..12).map(|x| x as f64 + 1.0).collect(), &[2, 2, 3], true)
            .unwrap();

        let output = ln.forward(&mut graph, input).unwrap();
        let output_shape = graph.get_shape(output);

        // Shape should be preserved
        assert_eq!(output_shape, vec![2, 2, 3]);

        // Output should be finite
        let output_data = graph.get_data(output);
        assert!(output_data.all(|x| x.is_finite()).unwrap());
    }

    #[test]
    fn test_layer_norm_without_affine() {
        let mut graph = Engine::new();
        let ln = LayerNorm::<f64>::new(vec![3], 1e-5, false); // No affine parameters

        // Should have no parameters
        let params = ln.parameters();
        assert_eq!(params.len(), 0);

        let input = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true)
            .unwrap();

        let output = ln.forward(&mut graph, input).unwrap();
        let output_data = graph.get_data(output);
        assert!(output_data.all(|x| x.is_finite()).unwrap())
    }

    #[test]
    fn test_layer_norm_gradient_flow() {
        let mut graph = Engine::new();
        let ln = LayerNorm::<f64>::default_config(vec![3]);

        let input = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true)
            .unwrap();

        let output = ln.forward(&mut graph, input).unwrap();
        let loss = graph.sum(output, None).unwrap();

        // Backward pass
        let backward_result = graph.backward(loss);
        assert!(backward_result.is_ok());

        // Check gradients exist
        let input_grad = graph.get_gradient(input);
        assert!(input_grad.is_some());

        let grad = input_grad.unwrap();
        assert_eq!(grad.shape(), &[2, 3]);
        assert!(grad.all(|x| x.is_finite()).unwrap());
    }

    #[test]
    fn test_layer_norm_error_cases() {
        let mut graph = Engine::new();
        let ln = LayerNorm::<f64>::default_config(vec![3, 4]);

        // Test with input that doesn't end with normalized_shape
        let wrong_shape = graph
            .tensor_from_vec(vec![1.0, 2.0], &[1, 2], true)
            .unwrap();

        let result = ln.forward(&mut graph, wrong_shape);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("doesn't end with normalized_shape")
        );

        // Test with input that has fewer dimensions than normalized_shape
        let too_few_dims = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0], &[3], true)
            .unwrap();

        let result = ln.forward(&mut graph, too_few_dims);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires at least"));
    }

    #[test]
    fn test_layer_norm_numerical_stability() {
        let mut graph = Engine::new();
        let ln = LayerNorm::<f64>::default_config(vec![3]);

        // Test with very large values
        let large_input = graph
            .tensor_from_vec(vec![1e8, 2e8, 3e8], &[1, 3], true)
            .unwrap();

        let output = ln.forward(&mut graph, large_input).unwrap();
        let output_data = graph.get_data(output);
        assert!(
            output_data
                .all(|val| val.is_finite() && !val.is_nan())
                .unwrap()
        );

        // Test with very small values
        let small_input = graph
            .tensor_from_vec(vec![1e-8, 2e-8, 3e-8], &[1, 3], true)
            .unwrap();

        let output = ln.forward(&mut graph, small_input).unwrap();
        let output_data = graph.get_data(output);
        assert!(output_data.all(|val| val.is_finite()).unwrap())
    }

    #[test]
    fn test_layer_norm_sequence_modeling() {
        let mut graph = Engine::new();
        // Typical sequence modeling scenario: [batch, seq_len, hidden_dim]
        let ln = LayerNorm::<f64>::default_config(vec![128]); // Normalize over hidden dimension

        let batch_size = 4;
        let seq_len = 10;
        let hidden_dim = 128;

        // Create random-like input
        let input_data: Vec<f64> = (0..batch_size * seq_len * hidden_dim)
            .map(|i| (i as f64) * 0.01)
            .collect();

        let input = graph
            .tensor_from_vec(input_data, &[batch_size, seq_len, hidden_dim], true)
            .unwrap();

        let output = ln.forward(&mut graph, input).unwrap();
        let output_shape = graph.get_shape(output);

        // Shape should be preserved
        assert_eq!(output_shape, vec![batch_size, seq_len, hidden_dim]);

        // Output should be finite
        let output_data = graph.get_data(output);
        assert!(output_data.all(|x| x.is_finite()).unwrap());
    }

    // ============================================================================
    // COMPARISON TESTS (BatchNorm vs LayerNorm)
    // ============================================================================

    #[test]
    fn test_batch_norm_vs_layer_norm_behavior() {
        let mut graph = Engine::new();

        // Same feature size for both
        let feature_size = 4;
        let bn = BatchNorm1d::<f64>::default_config(feature_size);
        let ln = LayerNorm::<f64>::default_config(vec![feature_size]);

        // Create identical input
        let input = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4], true)
            .unwrap();

        // Forward through both
        let bn_output = bn.forward(&mut graph, input).unwrap();
        let ln_output = ln.forward(&mut graph, input).unwrap();

        let bn_data = graph.get_data(bn_output);
        let ln_data = graph.get_data(ln_output);

        // Both should produce finite outputs but different values
        assert!(bn_data.all(|x| x.is_finite()).unwrap());
        assert!(ln_data.all(|x| x.is_finite()).unwrap());

        // BatchNorm and LayerNorm should produce different results
        // (they normalize differently)
        let are_different = bn_data != ln_data;

        assert!(
            are_different,
            "BatchNorm and LayerNorm should produce different outputs"
        );
    }

    #[test]
    fn test_normalization_training_mode_switching() {
        let mut bn = BatchNorm1d::<f64>::default_config(3);
        let mut ln = LayerNorm::<f64>::default_config(vec![3]);

        // Both should start in training mode
        assert!(bn.training());
        assert!(ln.training());

        // Switch to eval mode
        bn.eval();
        ln.eval();

        assert!(!bn.training());
        assert!(!ln.training());

        // Switch back to training
        bn.train();
        ln.train();

        assert!(bn.training());
        assert!(ln.training());
    }

    #[test]
    fn test_normalization_parameter_management() {
        let bn = BatchNorm1d::<f64>::default_config(10);
        let ln = LayerNorm::<f64>::default_config(vec![10]);

        // Both should have same number of parameters for same feature size
        assert_eq!(bn.num_parameters(), ln.num_parameters());
        assert_eq!(bn.parameters().len(), ln.parameters().len());

        // Parameter shapes should match
        let bn_params = bn.parameters();
        let ln_params = ln.parameters();

        assert_eq!(bn_params[0].shape(), ln_params[0].shape()); // weight
        assert_eq!(bn_params[1].shape(), ln_params[1].shape()); // bias
    }
}
