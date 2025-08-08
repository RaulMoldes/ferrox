#[cfg(test)]
mod graph_tests {

    use crate::backend::manager::best_f32_device;
    use crate::backend::Tensor;
    use crate::graph::engine::{AutoFerroxEngine, EvaluationMode};
    use crate::ops::*;

    // Helper functions to create test tensors
    fn create_tensor_2x2(data: [f32; 4]) -> Tensor<f32> {
        let device = best_f32_device();
        Tensor::from_vec_with_device(data.to_vec(), &[2, 2], device).expect("Failed to create tensor")
    }

    fn create_tensor_1d(data: &[f32]) -> Tensor<f32> {
        let device = best_f32_device();
        Tensor::from_vec_with_device(data.to_vec(), &[data.len()], device).expect("Failed to create tensor")
    }

    fn create_engine_with_training() -> AutoFerroxEngine<f32> {
        let mut engine = AutoFerroxEngine::<f32>::new();
        engine.set_training(true); // Enable gradients
        engine
    }

    // Helper to test operation forward pass, gradient computation and shapes
    fn test_operation_with_gradients<Op: Operator<f32> + 'static>(
        op: Op,
        inputs: Vec<Tensor<f32>>,
        expected_output_shape: &[usize],
        test_name: &str,
    ) {
        let mut engine = create_engine_with_training();

        // Create input nodes with gradients enabled
        let input_nodes: Vec<_> = inputs.iter()
            .map(|tensor| engine.create_variable(tensor.clone(), true))
            .collect();

        // Apply operation
        let result_node = engine.apply_operation(Box::new(op), input_nodes.clone())
            .expect("{test_name} operation failed");

        // Verify forward pass result
        let result_tensor = engine.get_tensor(result_node)
            .expect("{test_name} result tensor not found");
        assert_eq!(result_tensor.shape(), expected_output_shape,
            "{}: shape mismatch", test_name);

        // Test backward pass - create dummy loss and run backward
        let mean_loss = Box::new(Mean::new());
        let loss_node = engine.apply_operation(mean_loss, vec![result_node])
            .expect("{test_name} loss creation failed");

        engine.backward(loss_node)
            .expect("{test_name} backward pass failed");

        // Verify gradients exist for all inputs
        for (i, &input_node) in input_nodes.iter().enumerate() {
            let grad = engine.get_gradient(input_node);
            assert!(grad.is_some(), "{}: gradient missing for input {}", test_name, i);
            assert_eq!(grad.unwrap().shape(), inputs[i].shape(),
                "{}: gradient shape mismatch for input {}", test_name, i);
        }

        println!("{} test completed successfully", test_name);
    }

    // ========================= BASIC ARITHMETIC OPERATIONS =========================

    #[test]
    fn test_add_operation() {
        // Test element-wise addition with broadcasting support
        let inputs = vec![
            create_tensor_2x2([1.0, 2.0, 3.0, 4.0]),
            create_tensor_2x2([2.0, 3.0, 4.0, 5.0]),
        ];
        test_operation_with_gradients(
            Add::default(),
            inputs,
            &[2, 2],
            "Add"
        );
    }

    #[test]
    fn test_sub_operation() {
        // Test element-wise subtraction
        let inputs = vec![
            create_tensor_2x2([5.0, 6.0, 7.0, 8.0]),
            create_tensor_2x2([1.0, 2.0, 3.0, 4.0]),
        ];
        test_operation_with_gradients(Sub, inputs, &[2, 2], "Sub");
    }

    #[test]
    fn test_mul_operation() {
        // Test element-wise multiplication
        let inputs = vec![
            create_tensor_1d(&[2.0, 3.0, 4.0]),
            create_tensor_1d(&[5.0, 6.0, 7.0]),
        ];
        test_operation_with_gradients(Mul, inputs, &[3], "Mul");
    }

    #[test]
    fn test_div_operation() {
        // Test element-wise division
        let inputs = vec![
            create_tensor_1d(&[10.0, 15.0, 20.0]),
            create_tensor_1d(&[2.0, 3.0, 4.0]),
        ];
        test_operation_with_gradients(Div, inputs, &[3], "Div");
    }

    // ========================= SCALAR OPERATIONS =========================

    #[test]
    fn test_add_scalar_operation() {
        // Test scalar addition: input + scalar
        let inputs = vec![create_tensor_1d(&[1.0, 2.0, 3.0])];
        test_operation_with_gradients(
            AddScalar::new(5.0f32),
            inputs,
            &[3],
            "AddScalar"
        );
    }

    #[test]
    fn test_sub_scalar_operation() {
        // Test scalar subtraction: input - scalar
        let inputs = vec![create_tensor_1d(&[10.0, 20.0, 30.0])];
        test_operation_with_gradients(
            SubScalar::new(5.0f32),
            inputs,
            &[3],
            "SubScalar"
        );
    }

    #[test]
    fn test_mul_scalar_operation() {
        // Test scalar multiplication: input * scalar
        let inputs = vec![create_tensor_1d(&[1.0, 2.0, 3.0])];
        test_operation_with_gradients(
            MulScalar::new(3.0f32),
            inputs,
            &[3],
            "MulScalar"
        );
    }

    #[test]
    fn test_div_scalar_operation() {
        // Test scalar division: input / scalar
        let inputs = vec![create_tensor_1d(&[6.0, 9.0, 12.0])];
        test_operation_with_gradients(
            DivScalar::new(3.0f32),
            inputs,
            &[3],
            "DivScalar"
        );
    }

    #[test]
    fn test_power_scalar_operation() {
        // Test scalar power: input ^ scalar
        let inputs = vec![create_tensor_1d(&[2.0, 3.0, 4.0])];
        test_operation_with_gradients(
            PowerScalar::new(2.0f32),
            inputs,
            &[3],
            "PowerScalar"
        );
    }

    // ========================= MATRIX OPERATIONS =========================

    #[test]
    fn test_matmul_operation() {
        let device = best_f32_device();
        // Test matrix multiplication: A @ B
        let inputs = vec![

            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], device).expect("Create matrix A"),
            Tensor::from_vec_with_device(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], device).expect("Create matrix B"),
        ];
        test_operation_with_gradients(MatMul, inputs, &[2, 2], "MatMul");
    }

    #[test]
    fn test_transpose_operation() {
        // Test matrix transpose: A^T
        let device = best_f32_device();
        let inputs = vec![
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device)
                .expect("Create matrix")
        ];
        test_operation_with_gradients(
            Transpose { axes: None },
            inputs,
            &[3, 2],
            "Transpose"
        );
    }

    // ========================= UNARY OPERATIONS =========================

    #[test]
    fn test_exp_operation() {
        // Test element-wise exponential: exp(input)
        let inputs = vec![create_tensor_1d(&[0.0, 1.0, 2.0])];
        test_operation_with_gradients(Exp, inputs, &[3], "Exp");
    }

    #[test]
    fn test_log_operation() {
        // Test element-wise natural logarithm: log(input)
        let inputs = vec![create_tensor_1d(&[1.0, 2.0, 3.0])]; // Positive values for log
        test_operation_with_gradients(Log, inputs, &[3], "Log");
    }

    #[test]
    fn test_sqrt_operation() {
        // Test element-wise square root: sqrt(input)
        let inputs = vec![create_tensor_1d(&[1.0, 4.0, 9.0])]; // Perfect squares
        test_operation_with_gradients(Sqrt, inputs, &[3], "Sqrt");
    }

    #[test]
    fn test_abs_operation() {
        // Test element-wise absolute value: abs(input)
        let inputs = vec![create_tensor_1d(&[-2.0, 0.0, 3.0])];
        test_operation_with_gradients(Abs, inputs, &[3], "Abs");
    }

    #[test]
    fn test_neg_operation() {
        // Test element-wise negation: -input
        let inputs = vec![create_tensor_1d(&[1.0, -2.0, 3.0])];
        test_operation_with_gradients(Neg, inputs, &[3], "Neg");
    }

    #[test]
    fn test_power_operation() {
        // Test element-wise power: input1 ^ input2
        let inputs = vec![
            create_tensor_1d(&[2.0, 3.0, 4.0]),
            create_tensor_1d(&[2.0, 2.0, 2.0]),
        ];
        test_operation_with_gradients(Power, inputs, &[3], "Power");
    }

    // ========================= ACTIVATION FUNCTIONS =========================

    #[test]
    fn test_relu_operation() {
        // Test ReLU activation: max(0, input)
        let inputs = vec![create_tensor_1d(&[-2.0, 0.0, 2.0])];
        test_operation_with_gradients(ReLU, inputs, &[3], "ReLU");
    }

    #[test]
    fn test_sigmoid_operation() {
        // Test sigmoid activation: 1 / (1 + exp(-input))
        let inputs = vec![create_tensor_1d(&[-1.0, 0.0, 1.0])];
        test_operation_with_gradients(Sigmoid, inputs, &[3], "Sigmoid");
    }

    #[test]
    fn test_tanh_operation() {
        // Test hyperbolic tangent activation: tanh(input)
        let inputs = vec![create_tensor_1d(&[-1.0, 0.0, 1.0])];
        test_operation_with_gradients(Tanh, inputs, &[3], "Tanh");
    }

    // ========================= REDUCTION OPERATIONS =========================

    #[test]
    fn test_sum_operation() {
        // Test sum reduction: sum(input, axes)
        let inputs = vec![create_tensor_2x2([1.0, 2.0, 3.0, 4.0])];
        test_operation_with_gradients(Sum::new(), inputs, &[], "Sum"); // Scalar result
    }

    #[test]
    fn test_mean_operation() {
        // Test mean reduction: mean(input, axes)
        let inputs = vec![create_tensor_2x2([2.0, 4.0, 6.0, 8.0])];
        test_operation_with_gradients(Mean::new(), inputs, &[], "Mean"); // Scalar result
    }

    #[test]
    fn test_max_operation() {
        // Test max reduction: max(input, axes)
        let inputs = vec![create_tensor_2x2([1.0, 4.0, 2.0, 3.0])];
        test_operation_with_gradients(Max::new(), inputs, &[], "Max"); // Scalar result
    }

    #[test]
    fn test_min_operation() {
        // Test min reduction: min(input, axes)
        let inputs = vec![create_tensor_2x2([3.0, 1.0, 4.0, 2.0])];
        test_operation_with_gradients(Min::new(), inputs, &[], "Min"); // Scalar result
    }

    // ========================= COMPARISON OPERATIONS =========================

    #[test]
    fn test_greater_operation() {
        // Test element-wise greater than: input1 > input2
        let inputs = vec![
            create_tensor_1d(&[1.0, 3.0, 5.0]),
            create_tensor_1d(&[2.0, 3.0, 4.0]),
        ];
        test_operation_with_gradients(Greater, inputs, &[3], "Greater");
    }

    #[test]
    fn test_greater_equal_operation() {
        // Test element-wise greater than or equal: input1 >= input2
        let inputs = vec![
            create_tensor_1d(&[1.0, 3.0, 5.0]),
            create_tensor_1d(&[2.0, 3.0, 4.0]),
        ];
        test_operation_with_gradients(GreaterEqual, inputs, &[3], "GreaterEqual");
    }

    #[test]
    fn test_less_operation() {
        // Test element-wise less than: input1 < input2
        let inputs = vec![
            create_tensor_1d(&[1.0, 3.0, 5.0]),
            create_tensor_1d(&[2.0, 3.0, 4.0]),
        ];
        test_operation_with_gradients(Less, inputs, &[3], "Less");
    }

    #[test]
    fn test_less_equal_operation() {
        // Test element-wise less than or equal: input1 <= input2
        let inputs = vec![
            create_tensor_1d(&[1.0, 3.0, 5.0]),
            create_tensor_1d(&[2.0, 3.0, 4.0]),
        ];
        test_operation_with_gradients(LessEqual, inputs, &[3], "LessEqual");
    }

    #[test]
    fn test_equal_operation() {
        // Test element-wise equality: input1 == input2
        let inputs = vec![
            create_tensor_1d(&[1.0, 3.0, 5.0]),
            create_tensor_1d(&[2.0, 3.0, 4.0]),
        ];
        test_operation_with_gradients(Equal, inputs, &[3], "Equal");
    }

    #[test]
    fn test_clamp_operation() {
        // Test clamp operation: clamp(input, min_val, max_val)
        let inputs = vec![create_tensor_1d(&[-5.0, 0.0, 10.0])];
        test_operation_with_gradients(
            Clamp::new(-2.0f32, 5.0f32),
            inputs,
            &[3],
            "Clamp"
        );
    }

    // ========================= ELEMENTWISE OPERATIONS =========================

    #[test]
    fn test_max_elementwise_operation() {
        // Test element-wise maximum: max(input1, input2)
        let inputs = vec![
            create_tensor_1d(&[1.0, 5.0, 3.0]),
            create_tensor_1d(&[4.0, 2.0, 6.0]),
        ];
        test_operation_with_gradients(MaxElementwise, inputs, &[3], "MaxElementwise");
    }

    #[test]
    fn test_min_elementwise_operation() {
        // Test element-wise minimum: min(input1, input2)
        let inputs = vec![
            create_tensor_1d(&[1.0, 5.0, 3.0]),
            create_tensor_1d(&[4.0, 2.0, 6.0]),
        ];
        test_operation_with_gradients(MinElementwise, inputs, &[3], "MinElementwise");
    }

    #[test]
    fn test_reciprocal_operation() {
        // Test element-wise reciprocal: 1 / input
        let inputs = vec![create_tensor_1d(&[1.0, 2.0, 4.0])]; // Non-zero values
        test_operation_with_gradients(Reciprocal, inputs, &[3], "Reciprocal");
    }

    #[test]
    fn test_sign_operation() {
        // Test element-wise sign: sign(input)
        let inputs = vec![create_tensor_1d(&[-2.0, 0.0, 3.0])];
        test_operation_with_gradients(Sign, inputs, &[3], "Sign");
    }

    // ========================= RESHAPE OPERATIONS =========================

    #[test]
    fn test_reshape_operation() {
        let device = best_f32_device();
        // Test reshape operation: reshape(input, new_shape)
        let inputs = vec![
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device)
                .expect("Create tensor")
        ];
        test_operation_with_gradients(
            Reshape::new(vec![3, 2]),
            inputs,
            &[3, 2],
            "Reshape"
        );
    }

    #[test]
    fn test_unsqueeze_operation() {
        // Test unsqueeze operation: unsqueeze(input, axis)
        let inputs = vec![create_tensor_1d(&[1.0, 2.0, 3.0])];
        test_operation_with_gradients(
            Unsqueeze::new(0),
            inputs,
            &[1, 3],
            "Unsqueeze"
        );
    }

    #[test]
    fn test_squeeze_operation() {
        let device = best_f32_device();
        // Test squeeze operation: squeeze(input, axis)
        let inputs = vec![
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0], &[1, 3], device)
                .expect("Create tensor with size-1 dimension")
        ];
        test_operation_with_gradients(
            Squeeze::at_axis(0),
            inputs,
            &[3],
            "Squeeze"
        );
    }

    #[test]
    fn test_broadcast_to_operation() {
        let device = best_f32_device();
        // Test broadcast_to operation: broadcast_to(input, target_shape)
        let inputs = vec![
            Tensor::from_vec_with_device(vec![1.0, 2.0], &[2], device)
                .expect("Create tensor")
        ];
        test_operation_with_gradients(
            BroadcastTo::new(vec![3, 2]),
            inputs,
            &[3, 2],
            "BroadcastTo"
        );
    }

    // ========================= INTEGRATION TEST =========================
    #[test]
    fn test_full_computational_graph() {
        // Complete end-to-end test demonstrating all major operation categories
        // This simulates a realistic neural network computation with multiple operation types
        let device = best_f32_device();
        let mut engine = create_engine_with_training();

        // Create input data representing a small batch of features
        let input_data = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], device)
            .expect("Failed to create input");
        let weights = Tensor::from_vec_with_device(vec![0.5, 0.3, 0.7, 0.1], &[2, 2], device)
            .expect("Failed to create weights");
        let bias = 0.2;

        let input_node = engine.create_variable(input_data, true);
        let weights_node = engine.create_variable(weights, true);


        // Step 1: Linear transformation (MatMul + Add)
        let linear_op = Box::new(MatMul);
        let linear_result = engine.apply_operation(linear_op, vec![input_node, weights_node])
            .expect("Linear transformation failed");

        println!("Matmul operation computed");
        let add_bias_op = Box::new(AddScalar::new(bias));
        let biased_result = engine.apply_operation(add_bias_op, vec![linear_result])
            .expect("Bias addition failed");

        println!("Add scalar op computed");

        // Step 2: Apply activation function (ReLU)
        let relu_op = Box::new(ReLU);
        let activated_result = engine.apply_operation(relu_op, vec![biased_result])
            .expect("ReLU activation failed");

        // Step 3: Apply mathematical transformation (Sqrt of Abs)
        let abs_op = Box::new(Abs);
        let abs_result = engine.apply_operation(abs_op, vec![activated_result])
            .expect("Abs operation failed");

        let sqrt_op = Box::new(Sqrt);
        let sqrt_result = engine.apply_operation(sqrt_op, vec![abs_result])
            .expect("Sqrt operation failed");

        // Step 4: Scalar operations (multiply by learning rate, add regularization)
        let lr_mul_op = Box::new(MulScalar::new(0.01f32));
        let scaled_result = engine.apply_operation(lr_mul_op, vec![sqrt_result])
            .expect("Learning rate scaling failed");

        let reg_add_op = Box::new(AddScalar::new(1e-6f32));
        let regularized_result = engine.apply_operation(reg_add_op, vec![scaled_result])
            .expect("Regularization failed");

        // Step 5: Apply comparison and clamp (numerical stability)
        let clamp_op = Box::new(Clamp::new(1e-10f32, 1e10f32));
        let clamped_result = engine.apply_operation(clamp_op, vec![regularized_result])
            .expect("Clamping failed");

        // Step 6: Reduction to loss (Mean)
        let mean_op = Box::new(Mean::new());
        let loss_node = engine.apply_operation(mean_op, vec![clamped_result])
            .expect("Mean reduction failed");

        // Verify forward pass completed successfully
        let loss_tensor = engine.get_tensor(loss_node).expect("Loss tensor not found");
        assert_eq!(loss_tensor.shape(), &[]); // Scalar loss

        // Step 7: Test backward pass - compute all gradients
        engine.backward(loss_node).expect("Backward pass failed");

        // Verify gradients exist for all trainable parameters
        let input_grad = engine.get_gradient(input_node);
        let weights_grad = engine.get_gradient(weights_node);


        assert!(input_grad.is_some(), "Input gradient should exist");
        assert!(weights_grad.is_some(), "Weights gradient should exist");

        // Verify gradient shapes match parameter shapes
        assert_eq!(input_grad.unwrap().shape(), &[2, 2]);
        assert_eq!(weights_grad.unwrap().shape(), &[2, 2]);

        // Step 8: Test lazy vs eager evaluation modes
        engine.set_evaluation_mode(EvaluationMode::Lazy);

        // Create new computation graph in lazy mode
        let lazy_input = create_tensor_2x2([5.0, 6.0, 7.0, 8.0]);
        let lazy_input_node = engine.create_variable(lazy_input, false);

        // Chain multiple operations in lazy mode
        let exp_op = Box::new(Exp);
        let exp_result = engine.apply_operation(exp_op, vec![lazy_input_node])
            .expect("Lazy Exp failed");

        let log_op = Box::new(Log);
        let log_result = engine.apply_operation(log_op, vec![exp_result])
            .expect("Lazy Log failed");

        let sigmoid_op = Box::new(Sigmoid);
        let sigmoid_result = engine.apply_operation(sigmoid_op, vec![log_result])
            .expect("Lazy Sigmoid failed");

        // Operations should not be evaluated yet in lazy mode
        assert!(!engine.is_evaluated(exp_result), "Exp should be lazy");
        assert!(!engine.is_evaluated(log_result), "Log should be lazy");
        assert!(!engine.is_evaluated(sigmoid_result), "Sigmoid should be lazy");

        // Force evaluation of entire lazy chain
        let final_result = engine.evaluate(sigmoid_result)
            .expect("Lazy evaluation failed");
        assert_eq!(final_result.shape(), &[2, 2]);

        // Now all nodes should be evaluated
        assert!(engine.is_evaluated(exp_result), "Exp should now be evaluated");
        assert!(engine.is_evaluated(log_result), "Log should now be evaluated");
        assert!(engine.is_evaluated(sigmoid_result), "Sigmoid should now be evaluated");

        println!("Computational graph test with ALL operations completed successfully!");
    }
}
