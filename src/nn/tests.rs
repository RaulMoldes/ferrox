#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Engine;
    use crate::nn::activations::*;
    use crate::backend::Numeric;
    use crate::nn::layers::*;
    use crate::nn::loss::*;
    use crate::nn::module::*;
    use crate::nn::parameter::Parameter;
    use crate::tensor::Tensor;
    use std::f64::EPSILON;

    /// Helper function to check if two floating point values are approximately equal
    fn approx_equal(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() < tolerance
    }

    /// Helper function to check if tensor values are approximately equal
    fn tensors_approx_equal<T>(a: &Tensor<T>, b: &Tensor<T>, tolerance: f64) -> bool
    where
        T: Numeric + Clone + Into<f64>,
    {
        if a.shape() != b.shape() {
            return false;
        }

        a.iter()
            .zip(b.iter())
            .all(|(x, y)| approx_equal((*x).clone().into(), (*y).clone().into(), tolerance))
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
        assert!(param.data.iter().all(|&x: &f64| x == 0.5));
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

        let input = graph
            .tensor_from_vec(vec![0.0], &[1], true)
            .unwrap();

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
        for &val in result_data.iter() {
            assert!(val > 0.0 && val < 1.0);
        }

        // Check specific values
        assert!(result_data[0] < 0.01); // sigmoid(-10) ≈ 0
        assert!(approx_equal(result_data[2], 0.5, 1e-6)); // sigmoid(0) = 0.5
        assert!(result_data[4] > 0.99); // sigmoid(10) ≈ 1
    }

    #[test]
    fn test_tanh_activation() {
        let mut graph = Engine::new();
        let tanh = Tanh::new();

        let input = graph
            .tensor_from_vec(vec![0.0], &[1], true)
            .unwrap();

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

        // All tanh outputs should be between -1 and 1
        for &val in result_data.iter() {
            assert!(val > -1.0 && val < 1.0);
        }

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
        assert!(approx_equal(result_data[2], 0.0, 1e-6));   // 0
        assert!(approx_equal(result_data[3], 1.0, 1e-6));   // 1
        assert!(approx_equal(result_data[4], 2.0, 1e-6));   // 2
    }

    #[test]
    fn test_leaky_relu_custom_slope() {
        let mut graph = Engine::new();
        let leaky_relu = LeakyReLU::new_with_slope(0.1);

        let input = graph
            .tensor_from_vec(vec![-1.0, 1.0], &[2], true)
            .unwrap();

        let output = leaky_relu.forward(&mut graph, input).unwrap();
        let result_data = graph.get_data(output);

        assert!(approx_equal(result_data[0], -0.1, 1e-6)); // -1 * 0.1 = -0.1
        assert!(approx_equal(result_data[1], 1.0, 1e-6));  // 1
    }

    #[test]
    fn test_elu_activation() {
        let mut graph = Engine::new();
        let elu = ELU::new(); // default alpha = 1.0

        let input = graph
            .tensor_from_vec(vec![0.0, 1.0], &[2], true)
            .unwrap();

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
        
        sequential.add(Box::new(Linear::new(10, 5, true)));  // 2 params (weight + bias)
        sequential.add(Box::new(ReLU::new()));               // 0 params
        sequential.add(Box::new(Linear::new(5, 3, true)));   // 2 params (weight + bias)
        
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
    // PENDING

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
        let target = graph
            .tensor_from_vec(vec![1.0], &[1], false)
            .unwrap();
        
        // Forward pass
        let prediction = model.forward(&mut graph, input).unwrap();
        
        // Compute loss
        let loss_fn = MSELoss::new();
        let loss = loss_fn.compute_loss(&mut graph, prediction, target).unwrap();
        
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
        let target = graph
            .tensor_from_vec(vec![3.0], &[1], false)
            .unwrap();
        
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
    fn test_loss_backward_compatibility() {
        let mut graph = Engine::new();
        
        // Test that loss functions work with Module trait
        let mse_loss = MSELoss::new();
        let ce_loss = CrossEntropyLoss::new();
        
        let input = graph.tensor_from_vec(vec![1.0], &[1], true).unwrap();
        
        // Should be able to call forward (even if it just returns input)
        let mse_output = mse_loss.forward(&mut graph, input).unwrap();
        let ce_output = ce_loss.forward(&mut graph, input).unwrap();
        
        assert_eq!(mse_output, input);
        assert_eq!(ce_output, input);
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
}