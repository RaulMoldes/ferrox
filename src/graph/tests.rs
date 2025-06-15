#[cfg(test)]
mod tests {

    use crate::graph::{Engine, EngineVisualization, GraphVisualizer, next_node_id};

    use crate::tensor::Tensor;

    use std::sync::Arc;
    use std::sync::Barrier;
    use std::thread;

    /// Helper function to check if two floating point values are approximately equal
    fn approx_equal(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() < tolerance
    }

    #[test]
    fn test_transpose_gradient() {
        let mut graph = Engine::new();

        // Test 2D transpose gradient
        let a = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true)
            .unwrap();

        // Default transpose
        let b = graph.transpose(a, None).unwrap();
        let loss = graph.summation(b, None).unwrap();

        graph.backward(loss).unwrap();

        let grad = graph.get_gradient(a).unwrap();
        // Gradient should have same shape as original tensor
        assert_eq!(grad.shape(), &[2, 3]);
        // Since we're summing, gradient should be all ones
        assert!(grad.data().iter().all(|&x| (x - 1.0f64).abs() < 1e-6));

        // Test custom transpose gradient
        let mut graph2 = Engine::new();
        let a2 = graph2
            .tensor_from_vec((0..24).map(|x| x as f64).collect(), &[2, 3, 4], true)
            .unwrap();

        let b2 = graph2.transpose(a2, Some(vec![2, 0, 1])).unwrap();
        let loss2 = graph2.summation(b2, None).unwrap();

        graph2.backward(loss2).unwrap();

        let grad2 = graph2.get_gradient(a2).unwrap();
        assert_eq!(grad2.shape(), &[2, 3, 4]);
        assert!(grad2.data().iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_graph_visualization() {
        let mut engine = Engine::new();

        // Create some nodes
        let a = engine.tensor_from_vec(vec![1.0, 2.0], &[2], true).unwrap();
        let b = engine.tensor_from_vec(vec![3.0, 4.0], &[2], true).unwrap();
        let c = engine.add(a, b).unwrap();
        let d = engine.mul(c, a).unwrap();

        // Test DOT generation
        let visualizer = GraphVisualizer::new();
        let dot = visualizer.to_dot(&engine, &[d]);

        assert!(dot.contains("digraph"));
        assert!(dot.contains(&format!("{}", a)));
        assert!(dot.contains(&format!("{}", b)));
        assert!(dot.contains(&format!("{}", c)));
        assert!(dot.contains(&format!("{}", d)));
    }

    #[test]
    fn test_graph_printing() {
        let mut engine = Engine::new();

        let a = engine.tensor_from_vec(vec![1.0], &[1], true).unwrap();
        let b = engine.relu(a).unwrap();

        // This shouldn't panic
        engine.plot_graph(&[b]);
    }

    #[test]
    fn test_node_id_generation() {
        let id1 = next_node_id();
        let id2 = next_node_id();
        assert_ne!(id1, id2);
        assert!(id2 > id1);
    }

    #[test]
    fn test_node_id_atomicity() {
        let num_threads = 10;
        let ids_per_thread = 100;
        let barrier = Arc::new(Barrier::new(num_threads));

        let mut handles = vec![];

        for _ in 0..num_threads {
            let barrier_clone = Arc::clone(&barrier);
            let handle = thread::spawn(move || {
                barrier_clone.wait();
                let mut ids = vec![];
                for _ in 0..ids_per_thread {
                    ids.push(next_node_id());
                }
                ids
            });
            handles.push(handle);
        }

        let mut all_ids = std::collections::HashSet::new();
        for handle in handles {
            let ids = handle.join().unwrap();
            for id in ids {
                assert!(all_ids.insert(id), "Duplicate ID found: {}", id);
            }
        }

        assert_eq!(all_ids.len(), num_threads * ids_per_thread);
    }

    #[test]
    fn test_computational_graph_basic() {
        let mut graph = Engine::new();

        let a_data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b_data = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();

        let a = graph.create_tensor(a_data, true);
        let b = graph.create_tensor(b_data, true);
        let c = graph.add(a, b).unwrap();

        let result = graph.get_data(c);
        let expected = Tensor::from_vec(vec![3.0, 5.0, 7.0, 9.0], &[2, 2]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_graph_scalar_operations() {
        let mut graph = Engine::new();

        let a = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true)
            .unwrap();

        let add_scalar = graph.add_scalar(a, 5.0).unwrap();
        let mul_scalar = graph.mul_scalar(a, 2.0).unwrap();

        let add_result = graph.get_data(add_scalar);
        let expected_add = Tensor::from_vec(vec![6.0, 7.0, 8.0, 9.0], &[2, 2]).unwrap();
        assert_eq!(add_result, expected_add);

        let mul_result = graph.get_data(mul_scalar);
        let expected_mul = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[2, 2]).unwrap();
        assert_eq!(mul_result, expected_mul);
    }

    #[test]
    fn test_graph_activations() {
        let mut graph = Engine::new();

        let a = graph
            .tensor_from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[2, 2], true)
            .unwrap();

        let relu_result = graph.relu(a).unwrap();
        let exp_result = graph.exp(a).unwrap();
        let neg_result = graph.negate(a).unwrap();

        let relu_data = graph.get_data(relu_result);
        let expected_relu = Tensor::from_vec(vec![0.0, 0.0, 1.0, 2.0], &[2, 2]).unwrap();
        assert_eq!(relu_data, expected_relu);

        let neg_data = graph.get_data(neg_result);
        let expected_neg = Tensor::from_vec(vec![1.0, 0.0, -1.0, -2.0], &[2, 2]).unwrap();
        assert_eq!(neg_data, expected_neg);

        let exp_data = graph.get_data(exp_result);
        assert!(exp_data.data().iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_backward_pass_simple() {
        let mut graph = Engine::new();

        // Simple case: z = x * y
        let x = graph.tensor_from_vec(vec![3.0], &[1], true).unwrap();
        let y = graph.tensor_from_vec(vec![4.0], &[1], true).unwrap();
        let z = graph.mul(x, y).unwrap();

        graph.backward(z).unwrap();

        // dz/dx should be y = 4.0
        // dz/dy should be x = 3.0
        let x_grad = graph.get_gradient(x).unwrap();
        let y_grad = graph.get_gradient(y).unwrap();

        assert_eq!(x_grad, Tensor::from_vec(vec![4.0], &[1]).unwrap());
        assert_eq!(y_grad, Tensor::from_vec(vec![3.0], &[1]).unwrap());
    }

    #[test]
    fn test_backward_pass_scalar_ops() {
        let mut graph = Engine::new();

        let x = graph.tensor_from_vec(vec![2.0], &[1], true).unwrap();
        let y = graph.add_scalar(x, 3.0).unwrap(); // y = x + 3
        let z = graph.mul_scalar(y, 2.0).unwrap(); // z = 2 * (x + 3)

        graph.backward(z).unwrap();

        // dz/dx = 2
        let x_grad = graph.get_gradient(x).unwrap();
        assert_eq!(x_grad, Tensor::from_vec(vec![2.0], &[1]).unwrap());
    }

    #[test]
    fn test_neural_network_forward_backward() {
        let mut graph = Engine::new();

        // Simple neural network: y = ReLU(x @ w)
        let x = graph
            .tensor_from_vec(vec![1.0, 2.0], &[1, 2], true)
            .unwrap();
        let w = graph
            .tensor_from_vec(vec![0.5, -0.3], &[2, 1], true)
            .unwrap();

        let linear = graph.matmul(x, w).unwrap();
        let output = graph.relu(linear).unwrap();
        let loss = graph.sum(output, None).unwrap();

        // Forward pass should work
        let loss_data = graph.get_data(loss);
        assert!(loss_data.data().iter().next().unwrap() >= &0.0);

        // Backward pass should work
        graph.backward(loss).unwrap();

        let x_grad = graph.get_gradient(x);
        let w_grad = graph.get_gradient(w);

        assert!(x_grad.is_some());
        assert!(w_grad.is_some());
    }

    #[test]
    fn test_relu_gradient() {
        let mut graph = Engine::new();

        let input_data = Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], &[2, 2]).unwrap();
        let a = graph.create_tensor(input_data, true);
        let b = graph.relu(a).unwrap();
        let loss = graph.sum(b, None).unwrap();

        // Check forward pass
        let result = graph.get_data(b);
        let expected = Tensor::from_vec(vec![0.0, 2.0, 0.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(result, expected);

        // Check backward pass
        graph.backward(loss).unwrap();
        let grad = graph.get_gradient(a).unwrap();
        let expected_grad = Tensor::from_vec(vec![0.0, 1.0, 0.0, 1.0], &[2, 2]).unwrap();
        assert_eq!(grad, expected_grad);
    }

    #[test]
    fn test_sum_operation() {
        let mut graph = Engine::new();

        let input_data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let a = graph.create_tensor(input_data, true);

        // Sum all elements
        let sum_all = graph.sum(a, None).unwrap();
        let result = graph.get_data(sum_all);

        assert_eq!(result.data().iter().next().unwrap(), &21.0);

        // Test summation with axes
        let sum_axis0 = graph.summation(a, Some(vec![0])).unwrap();
        let sum_axis0_result = graph.get_data(sum_axis0);
        assert_eq!(sum_axis0_result.shape(), &[3]);
    }

    #[test]
    fn test_gradient_accumulation() {
        let mut graph = Engine::new();

        // Test that gradients accumulate correctly when a node is used multiple times
        let x = graph.tensor_from_vec(vec![2.0], &[1], true).unwrap();
        let y1 = graph.mul(x, x).unwrap(); // y1 = x^2
        let y2 = graph.mul(x, x).unwrap(); // y2 = x^2
        let z = graph.add(y1, y2).unwrap(); // z = 2x^2

        graph.backward(z).unwrap();

        // dz/dx = 4x = 8.0
        let grad = graph.get_gradient(x).unwrap();
        assert_eq!(grad, Tensor::from_vec(vec![8.0], &[1]).unwrap());
    }

    #[test]
    fn test_complex_computation_graph() {
        let mut graph = Engine::new();

        // Test more complex computation: loss = sum(ReLU((x @ w1 + b1) @ w2))
        let x = graph
            .tensor_from_vec(vec![1.0, 2.0], &[1, 2], true)
            .unwrap();
        let w1 = graph
            .tensor_from_vec(vec![0.5, 0.3, -0.2, 0.4], &[2, 2], true)
            .unwrap();
        let w2 = graph
            .tensor_from_vec(vec![0.1, -0.3], &[2, 1], true)
            .unwrap();

        // Forward pass
        let h1 = graph.matmul(x, w1).unwrap();
        let h1_relu = graph.relu(h1).unwrap();
        let output = graph.matmul(h1_relu, w2).unwrap();
        let loss = graph.sum(output, None).unwrap();

        // Backward pass
        graph.backward(loss).unwrap();

        // Check that all gradients exist
        assert!(graph.get_gradient(x).is_some());
        assert!(graph.get_gradient(w1).is_some());
        assert!(graph.get_gradient(w2).is_some());

        // Check gradient shapes
        assert_eq!(graph.get_gradient(x).unwrap().shape(), &[1, 2]);
        assert_eq!(graph.get_gradient(w1).unwrap().shape(), &[2, 2]);
        assert_eq!(graph.get_gradient(w2).unwrap().shape(), &[2, 1]);
    }

    #[test]
    fn test_graph_convenience_methods() {
        let mut graph = Engine::new();

        // Test zeros and ones creation
        let zeros = graph.zeros(&[2, 3], false);
        let ones = graph.ones(&[2, 3], false);

        assert_eq!(graph.get_shape(zeros), vec![2, 3]);
        assert_eq!(graph.get_shape(ones), vec![2, 3]);

        // Test requires_grad
        assert!(!graph.requires_grad(zeros));
        assert!(!graph.requires_grad(ones));

        // Test detach
        let a = graph.tensor_from_vec(vec![1.0, 2.0], &[2], true).unwrap();
        let detached = graph.detach(a);

        assert!(graph.requires_grad(a));
        assert!(!graph.requires_grad(detached));
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = Engine::new();

        let a = graph.tensor_from_vec(vec![1.0], &[1], true).unwrap();
        let b = graph.tensor_from_vec(vec![2.0], &[1], true).unwrap();
        let c = graph.add(a, b).unwrap();
        let d = graph.mul(c, a).unwrap();

        let topo_order = graph.find_topo_sort(&[d]);

        // Check that nodes appear in correct order
        let a_pos = topo_order.iter().position(|&x| x == a).unwrap();
        let b_pos = topo_order.iter().position(|&x| x == b).unwrap();
        let c_pos = topo_order.iter().position(|&x| x == c).unwrap();
        let d_pos = topo_order.iter().position(|&x| x == d).unwrap();

        assert!(a_pos < c_pos);
        assert!(b_pos < c_pos);
        assert!(c_pos < d_pos);
    }

    /// TESTS FOR COMPLEX GRAPH OPERATIONS
    /// These tests cover more advanced features like
    /// clamping, l1/l2 norms etc.
    #[test]
    fn test_min_operation() {
        let mut graph = Engine::new();

        let a = graph
            .tensor_from_vec(vec![1.0, 5.0, 3.0, -2.0], &[4], true)
            .unwrap();
        let b = graph
            .tensor_from_vec(vec![2.0, 1.0, 4.0, -1.0], &[4], true)
            .unwrap();

        let min_result = graph.min(a, b).unwrap();
        let result_data = graph.get_data(min_result);

        let expected = Tensor::from_vec(vec![1.0, 1.0, 3.0, -2.0], &[4]).unwrap();
        assert_eq!(result_data, expected);
    }

    #[test]
    fn test_max_operation() {
        let mut graph = Engine::new();

        let a = graph
            .tensor_from_vec(vec![1.0, 5.0, 3.0, -2.0], &[4], true)
            .unwrap();
        let b = graph
            .tensor_from_vec(vec![2.0, 1.0, 4.0, -1.0], &[4], true)
            .unwrap();

        let max_result = graph.max(a, b).unwrap();
        let result_data = graph.get_data(max_result);

        let expected = Tensor::from_vec(vec![2.0, 5.0, 4.0, -1.0], &[4]).unwrap();
        assert_eq!(result_data, expected);
    }

    #[test]
    fn test_clamp_operation() {
        let mut graph = Engine::new();

        let input = graph
            .tensor_from_vec(vec![-2.0, -0.5, 0.0, 0.5, 1.5, 2.0], &[6], true)
            .unwrap();

        let clamped = graph.clamp(input, 0.0, 1.0).unwrap();
        let result_data = graph.get_data(clamped);

        let expected = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0], &[6]).unwrap();
        assert_eq!(result_data, expected);
    }

    #[test]
    fn test_sqrt_operation() {
        let mut graph = Engine::new();

        let input = graph
            .tensor_from_vec(vec![0.0, 1.0, 4.0, 9.0, 16.0], &[5], true)
            .unwrap();

        let sqrt_result = graph.sqrt(input).unwrap();
        let result_data = graph.get_data(sqrt_result);

        let expected = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0], &[5]).unwrap();
        assert_eq!(result_data, expected);
    }

    #[test]
    fn test_sqrt_negative_error() {
        let mut graph = Engine::new();

        let input = graph
            .tensor_from_vec(vec![1.0, -1.0, 4.0], &[3], true)
            .unwrap();

        let sqrt_result = graph.sqrt(input);
        assert!(sqrt_result.is_err());
        assert!(sqrt_result.unwrap_err().contains("negative values"));
    }

    #[test]
    fn test_abs_operation() {
        let mut graph = Engine::new();

        let input = graph
            .tensor_from_vec(vec![-3.0, -1.0, 0.0, 2.0, -5.0], &[5], true)
            .unwrap();

        let abs_result = graph.abs(input).unwrap();
        let result_data = graph.get_data(abs_result);

        let expected = Tensor::from_vec(vec![3.0, 1.0, 0.0, 2.0, 5.0], &[5]).unwrap();
        assert_eq!(result_data, expected);
    }

    #[test]
    fn test_min_max_gradients() {
        let mut graph = Engine::new();

        let a = graph.tensor_from_vec(vec![1.0, 5.0], &[2], true).unwrap();
        let b = graph.tensor_from_vec(vec![3.0, 2.0], &[2], true).unwrap();

        // Test min gradients
        let min_result = graph.min(a, b).unwrap();
        let loss = graph.sum(min_result, None).unwrap();

        graph.backward(loss).unwrap();

        let a_grad = graph.get_gradient(a).unwrap();
        let b_grad = graph.get_gradient(b).unwrap();

        // For min: a gets gradient where a <= b, b gets gradient where a > b
        // a = [1.0, 5.0], b = [3.0, 2.0]
        // min = [1.0, 2.0]
        // a_grad should be [1.0, 0.0] (a[0] <= b[0], a[1] > b[1])
        // b_grad should be [0.0, 1.0]
        assert_eq!(a_grad, Tensor::from_vec(vec![1.0, 0.0], &[2]).unwrap());
        assert_eq!(b_grad, Tensor::from_vec(vec![0.0, 1.0], &[2]).unwrap());
    }

    #[test]
    fn test_clamp_gradients() {
        let mut graph = Engine::new();

        let input = graph
            .tensor_from_vec(vec![-1.0, 0.5, 2.0], &[3], true)
            .unwrap();

        let clamped = graph.clamp(input, 0.0, 1.0).unwrap();
        let loss = graph.sum(clamped, None).unwrap();

        graph.backward(loss).unwrap();

        let input_grad = graph.get_gradient(input).unwrap();

        // Gradient should be 0 where clamped, 1 where not clamped
        // input = [-1.0, 0.5, 2.0], clamped to [0.0, 1.0]
        // clamped = [0.0, 0.5, 1.0]
        // gradient should be [0.0, 1.0, 0.0] (only middle value is within bounds)
        assert_eq!(
            input_grad,
            Tensor::from_vec(vec![0.0, 1.0, 0.0], &[3]).unwrap()
        );
    }

    #[test]
    fn test_sqrt_gradients() {
        let mut graph = Engine::new();

        let input = graph
            .tensor_from_vec(vec![1.0, 4.0, 9.0], &[3], true)
            .unwrap();

        let sqrt_result = graph.sqrt(input).unwrap();
        let loss = graph.sum(sqrt_result, None).unwrap();

        graph.backward(loss).unwrap();

        let input_grad = graph.get_gradient(input).unwrap();

        // Gradient of sqrt(x) is 1/(2*sqrt(x))
        // For x = [1.0, 4.0, 9.0], sqrt(x) = [1.0, 2.0, 3.0]
        // Gradient = [1/(2*1), 1/(2*2), 1/(2*3)] = [0.5, 0.25, 1/6]
        assert!(approx_equal(input_grad[0], 0.5, 1e-6));
        assert!(approx_equal(input_grad[1], 0.25, 1e-6));
        assert!(approx_equal(input_grad[2], 1.0 / 6.0, 1e-6));
    }

    #[test]
    fn test_abs_gradients() {
        let mut graph = Engine::new();

        let input = graph
            .tensor_from_vec(vec![-2.0, 3.0, 0.0], &[3], true)
            .unwrap();

        let abs_result = graph.abs(input).unwrap();
        let loss = graph.sum(abs_result, None).unwrap();

        graph.backward(loss).unwrap();

        let input_grad = graph.get_gradient(input).unwrap();

        // Gradient of abs(x): -1 for x < 0, 1 for x > 0, 0 for x = 0
        // For x = [-2.0, 3.0, 0.0]
        // Gradient = [-1.0, 1.0, 0.0]
        assert_eq!(
            input_grad,
            Tensor::from_vec(vec![-1.0, 1.0, 0.0], &[3]).unwrap()
        );
    }

    #[test]
    fn test_clamp_probabilities_convenience() {
        let mut graph = Engine::new();

        let probs = graph
            .tensor_from_vec(vec![0.0, 0.5, 1.0], &[3], true)
            .unwrap();

        let safe_probs = graph.clamp_probabilities(probs, 1e-8).unwrap();
        let result_data = graph.get_data(safe_probs);

        // Should clamp 0.0 to 1e-8 and 1.0 to 1.0-1e-8
        assert!(approx_equal(result_data[0], 1e-8, 1e-10));
        assert!(approx_equal(result_data[1], 0.5, 1e-6));
        assert!(approx_equal(result_data[2], 1.0 - 1e-8, 1e-10));
    }

    #[test]
    fn test_l2_norm_convenience() {
        let mut graph = Engine::new();

        let input = graph.tensor_from_vec(vec![3.0, 4.0], &[2], true).unwrap();

        let norm = graph.l2_norm(input).unwrap();
        let result_data = graph.get_data(norm);

        // L2 norm of [3, 4] should be sqrt(3^2 + 4^2) = sqrt(25) = 5
        assert!(approx_equal(result_data[0], 5.0, 1e-6));
    }

    #[test]
    fn test_l1_norm_convenience() {
        let mut graph = Engine::new();

        let input = graph
            .tensor_from_vec(vec![-3.0, 4.0, -1.0], &[3], true)
            .unwrap();

        let norm = graph.l1_norm(input).unwrap();
        let result_data = graph.get_data(norm);

        // L1 norm of [-3, 4, -1] should be |−3| + |4| + |−1| = 3 + 4 + 1 = 8
        assert!(approx_equal(result_data[0], 8.0, 1e-6));
    }

    #[test]
    fn test_shape_mismatch_errors() {
        let mut graph = Engine::new();

        let a = graph.tensor_from_vec(vec![1.0, 2.0], &[2], true).unwrap();
        let b = graph
            .tensor_from_vec(vec![1.0, 2.0, 3.0], &[3], true)
            .unwrap();

        // Min and Max should fail with shape mismatch
        assert!(graph.min(a, b).is_err());
        assert!(graph.max(a, b).is_err());
    }

    #[test]
    fn test_clamp_invalid_bounds() {
        // This should panic due to min_val > max_val in ClampOp::new
        let result = std::panic::catch_unwind(|| {
            let mut graph = Engine::new();
            let input = graph.tensor_from_vec(vec![1.0, 2.0], &[2], true).unwrap();
            graph.clamp(input, 1.0, 0.5)
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_numerical_stability_operations() {
        let mut graph = Engine::new();

        // Test operations with very small numbers
        let small_input = graph
            .tensor_from_vec(vec![1e-10, 1e-8, 1e-6], &[3], true)
            .unwrap();

        let sqrt_small = graph.sqrt(small_input).unwrap();
        let result_data = graph.get_data(sqrt_small);

        // Should handle very small numbers correctly
        assert!(result_data[0] > 0.0);
        assert!(result_data[1] > 0.0);
        assert!(result_data[2] > 0.0);

        // Test that sqrt of small numbers is larger than the original
        assert!(result_data[0] > 1e-10);
        assert!(result_data[1] > 1e-8);
        assert!(result_data[2] > 1e-6);
    }

    #[test]
    fn test_complex_operation_chain() {
        let mut graph = Engine::new();

        // Create a complex chain: clamp -> abs -> sqrt -> max with constant
        let input = graph
            .tensor_from_vec(vec![-2.0, 0.5, 3.0], &[3], true)
            .unwrap();

        let clamped = graph.clamp(input, -1.0, 2.0).unwrap(); // [-1.0, 0.5, 2.0]
        let abs_vals = graph.abs(clamped).unwrap(); // [1.0, 0.5, 2.0]
        let sqrt_vals = graph.sqrt(abs_vals).unwrap(); // [1.0, ~0.707, ~1.414]

        let constant = graph
            .tensor_from_vec(vec![1.0, 1.0, 1.0], &[3], false)
            .unwrap();
        let max_vals = graph.max(sqrt_vals, constant).unwrap(); // [1.0, 1.0, ~1.414]

        let result_data = graph.get_data(max_vals);

        assert!(approx_equal(result_data[0], 1.0, 1e-6));
        assert!(approx_equal(result_data[1], 1.0, 1e-6)); // max(sqrt(0.5), 1.0) = 1.0
        assert!(result_data[2] > 1.0); // sqrt(2) > 1
    }

    #[test]
    fn test_edge_cases_sqrt_zero() {
        let mut graph = Engine::new();

        let input = graph.tensor_from_vec(vec![0.0], &[1], true).unwrap();

        let sqrt_result = graph.sqrt(input).unwrap();
        let loss = graph.sum(sqrt_result, None).unwrap();

        // This should not panic and should handle gradient at zero correctly
        graph.backward(loss).unwrap();

        let input_grad = graph.get_gradient(input).unwrap();

        // Gradient at zero should be 0 (by convention to avoid infinity)
        assert_eq!(input_grad[0], 0.0);
    }

    #[test]
    fn test_edge_cases_abs_zero() {
        let mut graph = Engine::new();

        let input = graph.tensor_from_vec(vec![0.0], &[1], true).unwrap();

        let abs_result = graph.abs(input).unwrap();
        let loss = graph.sum(abs_result, None).unwrap();

        graph.backward(loss).unwrap();

        let input_grad = graph.get_gradient(input).unwrap();

        // Gradient of abs at zero should be 0 (by convention)
        assert_eq!(input_grad[0], 0.0);
    }
}
