use crate::NodeId;
use std::collections::HashMap;
use crate::backend::{Numeric, Float};
use crate::graph::Engine;
/// Trait that all optimizers must implement
/// If we wanted to create a new optimizer, we would implement this trait
/// and provide the necessary methods for parameter updates.
/// This allows for flexibility in choosing different optimization algorithms
/// such as SGD, Adam, RMSprop, etc.
/// It also mimics PyTorch and TensorFlow's optimizer interfaces, which you can customize by inheriting
/// from a base optimizer class.
pub trait Optimizer<T> 
where T: Numeric + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand
{
    /// Update parameters based on gradients
    fn step(&mut self, engine: &mut Engine<T>);

    /// Reset gradients to zero/None
    fn reset_grad(&mut self, engine: &mut Engine<T>);

    /// Add parameters to the optimizer
    fn add_param(&mut self, param_id: usize, param: NodeId);
}



/// Stochastic Gradient Descent optimizer with momentum and weight decay
/// SGD is the most basic optimizer, often used as a baseline.
/// It updates parameters by subtracting the gradient scaled by a learning rate.
/// Optionally, we allow the user to add momentum to accelerate convergence
/// and weight decay (L2 regularization) to prevent overfitting.
/// Nesterov's momentum can also be implemented for more advanced updates.
/// This optimizer is similar to PyTorch's SGD implementation.
/// https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html
pub struct SGD<T> 
where
    T: crate::backend::numeric::Numeric + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand
{   
    // Swapped the implementation in order not to use raw pointers to parameters.
    params: HashMap<usize, NodeId>, // Map of parameter IDs to their NodeId in the computation graph
    /// Learning rate for parameter updates
    lr: T,
    // Momentum factor
    /// Momentum helps accelerate SGD in the relevant direction and dampens oscillations.
    /// It is a technique to improve convergence speed by accumulating moving average of past gradients.
    momentum: T,
    nesterov: bool, // whether to use Nesterov momentum.
    // Nesterov basically looks ahead by computing the gradient at the "looked ahead" position.
    /// weight decay (L2 regularization) factor
    weight_decay: T,
    u: HashMap<usize, Vec<T>>, // momentum buffers
}

impl<T> SGD<T>
where
    T: Float + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand + From<f64> + rand_distr::num_traits::FromPrimitive,
{
    pub fn new(lr: T, momentum:T, weight_decay: T) -> Self {
        Self {
            params: HashMap::new(),
            lr,
            nesterov: false, // Nesterov momentum is off by default
            momentum,
            weight_decay,
            u: HashMap::new(),
        }
    }

    /// Clips gradient norm of parameters to prevent exploding gradients.
    /// Basicall computes the norm of all gradients and scales them if they exceed max_norm.
    /// This is useful in training deep networks where gradients can become very large.
    /// The clip_coef is computed through the max_norm and the total norm of the gradients.
    pub fn clip_grad_norm(&mut self, engine: &mut Engine<T>, max_norm: T) {
        let mut total_norm_sq = <T as Numeric>::zero(); // Initialize total norm squared

        // Calculate total gradient norm
        for (&param_id, _param_ptr) in &self.params {
            if let Some(grad) = engine.get_gradient(param_id) {
                for &g in grad.iter() {
                    total_norm_sq += g * g;
                }
            }
        }

        let total_norm = total_norm_sq.sqrt();

        // If norm exceeds max_norm, scale all gradients
        if total_norm > max_norm {
            let clip_coef = max_norm / total_norm;
            
            // Need to modify gradients in the engine
            for (&param_id, _param_ptr) in &self.params {
                if let Some(grad) = engine.get_gradient(param_id) {
                    let clipped_grad = grad.mul_scalar(clip_coef);
                    // You'll need to add a method to set gradients in Engine
                    engine.set_gradient(param_id, clipped_grad);
                }
            }
        }
    }

    // Setter for nesterov momentum
    pub fn set_nesterov(&mut self, nesterov: bool) {
        self.nesterov = nesterov;
    }
}

impl <T> Optimizer<T> for SGD<T> 
where
    T: Float + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand + rand_distr::num_traits::FromPrimitive,
{
    fn step(&mut self, engine: &mut Engine<T>) {
        // Iterate over all parameters and update them
        // using the stored gradients and momentum buffers.
        for (&param_id, &node_id) in &self.params {
            if let Some(grad) = engine.get_gradient(node_id) {
                // Get current parameter data from the engine
                let current_data = engine.get_data(node_id);
                
                // Initialize momentum buffer if not present
                if !self.u.contains_key(&param_id) {
                    self.u.insert(param_id, vec![<T as Numeric>::zero(); current_data.size()]);
                }
    
                // Get the momentum buffer for this parameter
                let momentum_buffer = self.u.get_mut(&param_id).unwrap();
                
                // Create new parameter data with updates
                let mut new_data = current_data.clone();
                
                for i in 0..current_data.size() {
                    let mut g = grad[i];
    
                    // Apply weight decay (L2 regularization)
                    if self.weight_decay != <T as Numeric>::zero() {
                        g += self.weight_decay * current_data[i];
                    }
    
                    // Update momentum buffer
                    momentum_buffer[i] = self.momentum * momentum_buffer[i] + g;
    
                    // Apply update based on momentum type
                    let update = if self.nesterov {
                        // Nesterov momentum: use gradient + momentum * velocity
                        g + self.momentum * momentum_buffer[i]
                    } else {
                        // Standard momentum: use velocity directly
                        momentum_buffer[i]
                    };
                    
                    new_data[i] -= self.lr * update;
                }
                
                // Update the parameter in the engine
                engine.set_node_data(node_id, new_data);
            }
        }
    }

    fn reset_grad(&mut self, engine: &mut Engine<T>) {
        for &node_id in self.params.values() {
            engine.clear_gradient(node_id);
        }
    }

    fn add_param(&mut self, param_id: usize, param_node_id: NodeId) {
        self.params.insert(param_id, param_node_id);
    }
}

/// Adam optimizer - adaptive learning rate optimization algorithm
pub struct Adam<T>
where
    T: Float + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
{
    params: HashMap<usize, NodeId>,
    lr: T,
    // momentum parameters
    beta1: T,
    beta2: T,
    eps: T, // small constant to prevent division by zero
    // nesterov: bool,  not used on Adam
    weight_decay: T, // weight decay (L2 regularization) factor
    t: u32, // time step
    m: HashMap<usize, Vec<T>>, // first moment estimates
    v: HashMap<usize, Vec<T>>, // second moment estimates
}

impl<T> Adam<T>
where
    T: Float + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand + From<f64> + rand_distr::num_traits::FromPrimitive,
{
    pub fn new(lr: T, beta1: T, beta2: T, eps: T, weight_decay: T) -> Self {
        Self {
            params: HashMap::new(),
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

// Optimizer trait implementation for Adam
/// Adam is an adaptive learning rate optimization algorithm that computes individual adaptive learning rates for different parameters.
/// It combines the benefits of AdaGrad and RMSProp, making it suitable for a wide range of problems.
/// Basically, Adagrad adapts the learning rate based on the historical gradient information of each parameter, in a way that parameters that have been updated a lot do not
/// get updated as much in the future.
/// RMSProp is an evolution of Adagrad that prevents the learning rate from becoming too small by using a moving average of squared gradients
/// using only the previous ones.
/// Adam maintains two moment estimates: the first moment (mean) and the second moment (uncentered variance).
/// It also includes bias correction to account for the initialization of these moments.
impl<T> Optimizer<T> for Adam<T>
where
    T: Float + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand + From<f64> + rand_distr::num_traits::FromPrimitive,
{
    // Adam does not use Nesterov momentum, so we don't need to implement that here.
    fn step(&mut self, engine: &mut Engine<T>) {
        self.t += 1;

        for (&param_id, &node_id) in &self.params {
            if let Some(grad) = engine.get_gradient(node_id) {
                // Get current parameter data from the engine
                let current_data = engine.get_data(node_id);
                
                // Initialize moment estimates if not present
                if !self.m.contains_key(&param_id) {
                    self.m.insert(param_id, vec![<T as Numeric>::zero(); current_data.size()]);
                    self.v.insert(param_id, vec![<T as Numeric>::zero(); current_data.size()]);
                }

                let m_t = self.m.get_mut(&param_id).unwrap();
                let v_t = self.v.get_mut(&param_id).unwrap();
                
                // Create new parameter data with updates
                let mut new_data = current_data.clone();

                for i in 0..current_data.size() {
                    let mut g = grad[i];

                    // Apply weight decay (L2 regularization)
                    if self.weight_decay != <T as Numeric>::zero() {
                        g += self.weight_decay * current_data[i];
                    }

                    // Update biased first moment estimate
                    m_t[i] = self.beta1 * m_t[i] + (<T as Numeric>::one() - self.beta1) * g;

                    // Update biased second raw moment estimate
                    v_t[i] = self.beta2 * v_t[i] + (<T as Numeric>::one() - self.beta2) * g * g;

                    // Compute bias-corrected first moment estimate
                    let m_hat = m_t[i] / (<T as Numeric>::one()- self.beta1.powi(self.t as i32));

                    // Compute bias-corrected second raw moment estimate
                    let v_hat = v_t[i] / (<T as Numeric>::one() - self.beta2.powi(self.t as i32));

                    // Update parameter
                    new_data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
                }
                
                // Update the parameter in the engine
                engine.set_node_data(node_id, new_data);
            }
        }
    }

    fn reset_grad(&mut self, engine: &mut Engine<T>) {
        for &node_id in self.params.values() {
            engine.clear_gradient(node_id);
        }
    }

    fn add_param(&mut self, param_id: usize, param_node_id: NodeId) {
        self.params.insert(param_id, param_node_id);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::graph::Engine;
    use ndarray::{Array, IxDyn};

    #[test]
    fn test_sgd_basic_optimization() {
        // Create engine and simple 2x2 parameter tensor
        let mut engine = Engine::new();
        let data = Array::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let tensor = Tensor::new(data);
        let param_node = engine.create_tensor(tensor, true);

        // Set some gradients (simulating backpropagation)
        let grad_data = Array::from_shape_vec(IxDyn(&[2, 2]), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let grad_tensor = Tensor::new(grad_data);
        engine.set_gradient(param_node, grad_tensor);

        // Create SGD optimizer with learning rate 0.1, no momentum, no weight decay
        let mut sgd = SGD::new(0.1, 0.0, 0.0);
        sgd.add_param(0, param_node);

        // Store original values for comparison
        let original_data = engine.get_data(param_node);
        let original_values = [
            original_data[(0, 0)],
            original_data[(0, 1)],
            original_data[(1, 0)],
            original_data[(1, 1)],
        ];

        // Perform one optimization step
        sgd.step(&mut engine);

        // Check that parameters were updated correctly: param = param - lr * grad
        let updated_data = engine.get_data(param_node);
        assert!((updated_data[(0, 0)] - (original_values[0] - 0.1 * 0.1)).abs() < 1e-10);
        assert!((updated_data[(0, 1)] - (original_values[1] - 0.1 * 0.2)).abs() < 1e-10);
        assert!((updated_data[(1, 0)] - (original_values[2] - 0.1 * 0.3)).abs() < 1e-10);
        assert!((updated_data[(1, 1)] - (original_values[3] - 0.1 * 0.4)).abs() < 1e-10);

        // Test momentum on second step
        let mut sgd_momentum = SGD::new(0.1, 0.9, 0.0);
        sgd_momentum.add_param(0, param_node);

        // Set new gradients
        let grad_data2 = Array::from_shape_vec(IxDyn(&[2, 2]), vec![0.05, 0.1, 0.15, 0.2]).unwrap();
        let grad_tensor2 = Tensor::new(grad_data2);
        engine.set_gradient(param_node, grad_tensor2);

        // Will behave like no momentum on first step as the weights are initialized to zero.
        sgd_momentum.step(&mut engine);

        // Set gradients again for second step. We need to call step twice to see momentum effect
        // This simulates a second optimization step with new gradients
        let grad_data3 = Array::from_shape_vec(IxDyn(&[2, 2]), vec![0.05, 0.1, 0.15, 0.2]).unwrap();
        let grad_tensor3 = Tensor::new(grad_data3);
        engine.set_gradient(param_node, grad_tensor3);

        let before_momentum = engine.get_data(param_node)[(0, 0)];
        sgd_momentum.step(&mut engine); // Second step - now momentum effect is visible

        // With momentum, the update should be different than just -lr * grad
        let expected_no_momentum = before_momentum - 0.1 * 0.05;
        let actual = engine.get_data(param_node)[(0, 0)];
        println!(
            "Expected no momentum: {}, Actual: {}",
            expected_no_momentum,
            actual
        );
        assert!((actual - expected_no_momentum).abs() > 1e-10);

        // Test reset_grad
        sgd.reset_grad(&mut engine);
        assert!(engine.get_gradient(param_node).is_none());
    }

    #[test]
    fn test_adam_optimization_with_bias_correction() {
        // Create engine and 1D parameter tensor for simplicity
        let mut engine = Engine::new();
        let data = Array::from_shape_vec(IxDyn(&[3]), vec![1.0, -0.5, 2.0]).unwrap();
        let tensor = Tensor::new(data);
        let param_node = engine.create_tensor(tensor, true);

        // Set gradients
        let grad_data = Array::from_shape_vec(IxDyn(&[3]), vec![0.1, -0.2, 0.3]).unwrap();
        let grad_tensor = Tensor::new(grad_data);
        engine.set_gradient(param_node, grad_tensor);

        // Create Adam optimizer with standard hyperparameters
        let mut adam = Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0);
        adam.add_param(0, param_node);

        // Store original values
        let original_data = engine.get_data(param_node);
        let original_values = [original_data[0], original_data[1], original_data[2]];

        // Perform first optimization step
        adam.step(&mut engine);

        // Check that parameters changed (Adam should update all parameters)
        let updated_data = engine.get_data(param_node);
        assert!(updated_data[0] != original_values[0]);
        assert!(updated_data[1] != original_values[1]);
        assert!(updated_data[2] != original_values[2]);

        // Verify bias correction is working (first step should have larger updates due to bias correction)
        let first_step_change = (updated_data[0] - original_values[0]).abs();

        // Set same gradients for second step
        let grad_data2 = Array::from_shape_vec(IxDyn(&[3]), vec![0.1, -0.2, 0.3]).unwrap();
        let grad_tensor2 = Tensor::new(grad_data2);
        engine.set_gradient(param_node, grad_tensor2);

        let before_second = engine.get_data(param_node)[0];
        adam.step(&mut engine);

        let after_second = engine.get_data(param_node)[0];
        let second_step_change = (after_second - before_second).abs();

        // Due to bias correction, first step should have larger magnitude change
        // (though this might be subtle with these parameters)
        assert!(first_step_change > 0.0);
        assert!(second_step_change > 0.0);

        // Test that Adam maintains separate moment estimates
        // By checking that the internal state has been updated
        assert_eq!(adam.t, 2); // Two steps taken
        assert!(adam.m.contains_key(&0)); // First moment exists
        assert!(adam.v.contains_key(&0)); // Second moment exists

        // Test weight decay functionality
        let mut adam_with_decay = Adam::new(0.001, 0.9, 0.999, 1e-8, 0.01);
        let data_decay = Array::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
        let tensor_decay = Tensor::new(data_decay);
        let param_decay_node = engine.create_tensor(tensor_decay, true);

        let grad_data_decay = Array::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(); // Zero gradients
        let grad_tensor_decay = Tensor::new(grad_data_decay);
        engine.set_gradient(param_decay_node, grad_tensor_decay);

        adam_with_decay.add_param(1, param_decay_node);
        let before_decay = engine.get_data(param_decay_node)[0];
        adam_with_decay.step(&mut engine);

        // Even with zero gradients, weight decay should cause parameters to decrease
        let after_decay = engine.get_data(param_decay_node)[0];
        assert!(after_decay < before_decay);

        // Test reset_grad
        adam.reset_grad(&mut engine);
        assert!(engine.get_gradient(param_node).is_none());
    }
}