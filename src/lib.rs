//! # Ferrox
//!
//! Ferrox is a lightweight, CPU-based automatic differentiation engine written in Rust,
//! inspired by PyTorch's autograd and the CMU Deep Learning Systems course.
//!
//! ## Features
//!
//! - Reverse-mode automatic differentiation (backpropagation)
//! - Dynamic computation graph construction
//! - Scalar and tensor support via `ndarray`
//! - Operator overloading for intuitive expressions
//! - Gradient accumulation
//! - Graph visualization (requires GraphViz installed)
//! - High-level neural network modules
//! - Written 100% in safe Rust
//!
//! ## Quick Start
//!
//! ```rust
//! use ferrox::{Engine, Tensor, nn::*};
//!
//! // Create computation graph
//! let mut graph = Engine::new();
//!
//! // Create a simple neural network
//! let mut model = Sequential::new();
//! model.add(Box::new(Linear::new(784, 256, true)));
//! model.add(Box::new(ReLU::new()));
//! model.add(Box::new(Linear::new(256, 10, true)));
//!
//! // Create input data
//! let input_data = Tensor::randn(&[32, 784]); // batch_size=32, features=784
//! let input_node = graph.create_tensor(input_data, true);
//!
//! // Forward pass
//! let output = model.forward(&mut graph, input_node).unwrap();
//!
//! // Compute loss and backward pass
//! let loss = graph.sum(output, None).unwrap();
//! graph.backward(loss).unwrap();
//! ```

pub mod backend;
pub mod graph;
pub mod initializers;
pub mod nn;
pub mod optim;
pub mod tensor; // New neural network module

// Re-export commonly used types for convenience
pub use backend::{Device, Float, Numeric, cpu, default_device};
pub use graph::{
    Engine, EngineVisualization, GraphVisualizer, NodeId, VisualizationConfig, next_node_id,
};
pub use initializers::{
    init_tensor_kaiming_uniform, init_tensor_xavier_uniform, kaiming_uniform, xavier_uniform,
};
pub use nn::{Module, Parameter};
pub use tensor::Tensor;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::thread;

    #[test]
    fn test_device_operations() {
        let device = cpu();

        let zeros = device.zeros::<f64>(&[2, 3]);
        assert_eq!(zeros.shape(), &[2, 3]);
        assert!(zeros.iter().all(|&x| x == 0.0));

        let ones = device.ones::<f64>(&[2, 3]);
        assert_eq!(ones.shape(), &[2, 3]);
        assert!(ones.iter().all(|&x| x == 1.0));

        let full = device.full::<f64>(&[2, 2], 5.0);
        assert_eq!(full.shape(), &[2, 2]);
        assert!(full.iter().all(|&x| x == 5.0));
    }

    #[test]
    fn test_tensor_creation_with_device() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.len(), 4);
        assert_eq!(tensor.device(), &cpu());

        let zeros = Tensor::<f64>::zeros_with_device(&[3, 3], cpu());
        assert_eq!(zeros.shape(), &[3, 3]);
    }

    #[test]
    fn test_tensor_operations() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();

        let sum = a.add(&b).unwrap();
        let expected = Tensor::from_vec(vec![3.0, 5.0, 7.0, 9.0], &[2, 2]).unwrap();
        assert_eq!(sum, expected);

        let mul = a.mul(&b).unwrap();
        let expected_mul = Tensor::from_vec(vec![2.0, 6.0, 12.0, 20.0], &[2, 2]).unwrap();
        assert_eq!(mul, expected_mul);
    }

    #[test]
    fn test_tensor_matmul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();

        let result = a.matmul(&b).unwrap();
        let expected = Tensor::from_vec(vec![22.0, 28.0, 49.0, 64.0], &[2, 2]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_tensor_activations() {
        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

        // Test ReLU
        let relu_result = input.relu();
        let expected_relu = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        assert_eq!(relu_result, expected_relu);

        // Test Sigmoid (should be between 0 and 1)
        let sigmoid_result = input.sigmoid();
        for &val in sigmoid_result.data().iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }

        // Test Exp
        let exp_result = input.exp();
        assert!(exp_result.data().iter().all(|&x| x > 0.0));

        // Test Negate
        let neg_result = input.negate();
        let expected_neg = Tensor::from_vec(vec![2.0, 1.0, 0.0, -1.0, -2.0], &[5]).unwrap();
        assert_eq!(neg_result, expected_neg);
    }

    #[test]
    fn test_tensor_scalar_operations() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let add_scalar = tensor.add_scalar(5.0);
        let expected_add = Tensor::from_vec(vec![6.0, 7.0, 8.0, 9.0], &[2, 2]).unwrap();
        assert_eq!(add_scalar, expected_add);

        let mul_scalar = tensor.mul_scalar(2.0);
        let expected_mul = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[2, 2]).unwrap();
        assert_eq!(mul_scalar, expected_mul);

        let div_scalar = tensor.div_scalar(2.0);
        let expected_div = Tensor::from_vec(vec![0.5, 1.0, 1.5, 2.0], &[2, 2]).unwrap();
        assert_eq!(div_scalar, expected_div);
    }

    #[test]
    fn test_tensor_transpose_comprehensive() {
        // Test 2D transpose
        let tensor_2d = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Default transpose (should swap axes)
        let transposed_default = tensor_2d.transpose(None).unwrap();
        assert_eq!(transposed_default.shape(), &[3, 2]);

        // Explicit axes transpose
        let transposed_explicit = tensor_2d.transpose(Some(&[1, 0])).unwrap();
        assert_eq!(transposed_explicit.shape(), &[3, 2]);
        assert_eq!(transposed_default.data(), transposed_explicit.data());

        // Test 3D transpose
        let tensor_3d = Tensor::from_vec((0..24).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();

        // Default transpose (reverse all axes: [2,3,4] -> [4,3,2])
        let transposed_3d_default = tensor_3d.transpose(None).unwrap();
        assert_eq!(transposed_3d_default.shape(), &[4, 3, 2]);

        // Custom permutation: [2,3,4] -> [4,2,3] (axes [2,0,1])
        let transposed_3d_custom = tensor_3d.transpose(Some(&[2, 0, 1])).unwrap();
        assert_eq!(transposed_3d_custom.shape(), &[4, 2, 3]);

        // Test 1D and 0D tensors (should be unchanged)
        let tensor_1d = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let transposed_1d = tensor_1d.transpose(None).unwrap();
        assert_eq!(transposed_1d.shape(), &[3]);
        assert_eq!(transposed_1d.data(), tensor_1d.data());

        let tensor_0d = Tensor::from_vec(vec![42.0], &[]).unwrap();
        let transposed_0d = tensor_0d.transpose(None).unwrap();
        assert_eq!(transposed_0d.shape(), &[]);
        assert_eq!(transposed_0d.data(), tensor_0d.data());
    }

    #[test]
    fn test_transpose_error_cases() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Invalid axes length
        assert!(tensor.transpose(Some(&[0])).is_err());
        assert!(tensor.transpose(Some(&[0, 1, 2])).is_err());

        // Invalid axes values
        assert!(tensor.transpose(Some(&[0, 2])).is_err()); // 2 is out of bounds
        assert!(tensor.transpose(Some(&[0, 0])).is_err()); // duplicate axis
        assert!(tensor.transpose(Some(&[1, 1])).is_err()); // duplicate axis
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
    fn test_tensor_broadcasting() {
        let a = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let target_shape = &[2, 3];

        let broadcasted = a.broadcast_to(target_shape).unwrap();
        let expected = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[2, 3]).unwrap();
        assert_eq!(broadcasted, expected);
    }

    #[test]
    fn test_tensor_squeeze_unsqueeze() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();

        // Test squeeze
        let squeezed = tensor.squeeze(Some(0)).unwrap();
        assert_eq!(squeezed.shape(), &[3]);

        // Test unsqueeze
        let unsqueezed = squeezed.unsqueeze(1);
        assert_eq!(unsqueezed.shape(), &[3, 1]);
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
    fn test_tensor_error_handling() {
        // Test shape mismatch in addition
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        assert!(a.add(&b).is_err());

        // Test invalid matrix multiplication
        let c = Tensor::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
        let d = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();

        assert!(c.matmul(&d).is_err());
    }

    #[test]
    fn test_tensor_reductions() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Test sum
        let sum_all = tensor.sum(None);
        assert_eq!(sum_all.data().iter().next().unwrap(), &21.0);

        // Test mean
        let mean_all = tensor.mean(None);
        assert_eq!(mean_all.data().iter().next().unwrap(), &3.5);

        // Test sum along axis
        let sum_axis0 = tensor.sum(Some(0));
        assert_eq!(sum_axis0.shape(), &[3]);

        let sum_axis1 = tensor.sum(Some(1));
        assert_eq!(sum_axis1.shape(), &[2]);
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

    #[test]
    fn test_xavier_uniform_bounds() {
        let fan_in = 100;
        let fan_out = 50;
        let gain = 1.0;
        let expected_bound = gain * (6.0 / (fan_in + fan_out) as f64).sqrt();

        let init = xavier_uniform(fan_in, fan_out, gain);
        for _ in 0..1000 {
            let val = init();
            assert!(val >= -expected_bound && val <= expected_bound);
        }
    }

    #[test]
    fn test_kaiming_uniform_bounds() {
        let fan_in = 784;
        let fan_out = 256;
        let expected_bound = (6.0 / fan_in as f64).sqrt();

        let init = kaiming_uniform(fan_in, fan_out, "relu");
        for _ in 0..1000 {
            let val = init();
            assert!(val >= -expected_bound && val <= expected_bound);
        }
    }

    #[test]
    fn test_tensor_initialization() {
        let shape = [784, 256];
        let weights = init_tensor_xavier_uniform(&shape, 1.0);

        assert_eq!(weights.shape(), &[784, 256]);
    }

    // TESTS FOR THE NEURAL NETWORK MODULE (ADDED ON 2025-06-10)
    
}
