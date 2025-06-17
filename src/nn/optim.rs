use super::module::Module;
use crate::NodeId;
use crate::backend::{Float, Numeric, NumericCuda};
use crate::graph::Engine;
use crate::tensor::Tensor;
use core::panic;
use std::collections::HashMap;
use std::collections::HashSet;

/// Trait that all optimizers must implement
/// This trait provides a common interface for all optimization algorithms,
/// allowing easy swapping between SGD, Adam, RMSprop, etc.
///
/// If we wanted to create a new optimizer, we would implement this trait
/// and provide the necessary methods for parameter updates.
/// This allows for flexibility in choosing different optimization algorithms
/// such as SGD, Adam, RMSprop, etc.
/// It also mimics PyTorch and TensorFlow's optimizer interfaces, which you can customize by inheriting
/// from a base optimizer class.
pub trait Optimizer<T>
where
    T: NumericCuda + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
{
    /// Perform one optimization step using computed gradients
    ///
    /// This method should:
    /// 1. Read gradients from the engine for all registered parameters
    /// 2. Update internal optimizer state (momentum buffers, etc.)
    /// 3. Compute parameter updates according to the optimization algorithm
    /// 4. Update parameter values in the engine
    fn step(&mut self, engine: &mut Engine<T>);

    /// Clear all gradients for registered parameters
    ///
    /// This should be called after each optimization step to prepare
    /// for the next forward/backward pass.
    fn reset_grad(&mut self, engine: &mut Engine<T>);

    /// Add a parameter to be optimized
    ///
    /// # Arguments
    /// * `param_id` - Unique identifier for the parameter (can be ignored if not needed)
    /// * `param_node_id` - NodeId of the parameter in the computation graph
    fn add_param(&mut self, param_id: usize, param_node_id: NodeId);

    /// Add all parameters from a module to the optimizer
    ///
    /// This is a convenience method that automatically registers all
    /// parameters from a neural network module.
    fn add_param_group<M>(&mut self, module: &M, engine: &mut Engine<T>)
    where
        M: Module<T>,
        T: rand_distr::num_traits::FromPrimitive,
    {
        let param_map = module.create_parameters_in_graph(engine);
        for (i, (_, node_id)) in param_map.into_iter().enumerate() {
            self.add_param(i, node_id);
        }
    }
}

/// Stochastic Gradient Descent optimizer with momentum and weight decay
/// SGD is the most basic optimizer, often used as a baseline.
/// It updates parameters by subtracting the gradient scaled by a learning rate.
/// Optionally, we allow the user to add momentum to accelerate convergence
/// and weight decay (L2 regularization) to prevent overfitting.
/// Nesterov's momentum can also be implemented for more advanced updates.
/// This optimizer is similar to PyTorch's SGD implementation.
/// https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html
///
/// This implementation follows the standard SGD update rule:
/// ```bash
/// v_t = momentum * v_{t-1} + (1 - dampening) * (g_t + weight_decay * p_t)
/// p_t = p_t - lr * v_t  (standard momentum)
/// p_t = p_t - lr * (g_t + momentum * v_t)  (Nesterov momentum)
/// ```
///
/// Where:
/// - v_t: momentum buffer at time t
/// - g_t: gradient at time t  
/// - p_t: parameter at time t
/// - lr: learning rate
pub struct SGD<T>
where
    T: Float
        + NumericCuda
        + Clone
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand,
{
    /// Map of NodeId to their momentum buffers
    /// Each NodeId corresponds to a parameter tensor in the computation graph
    momentum_buffers: HashMap<NodeId, Tensor<T>>,

    /// Set of parameter NodeIds being optimized
    param_nodes: HashSet<NodeId>,

    /// Learning rate for parameter updates
    lr: T,

    /// Momentum factor (typically 0.9)
    /// Controls how much of the previous velocity to retain
    momentum: T,

    /// Dampening factor for momentum (typically 0.0)
    /// Reduces the contribution of current gradient: (1 - dampening) * gradient
    dampening: T, // Prevent bias in momentum updates

    /// Whether to use Nesterov momentum
    /// Nesterov looks ahead by computing gradient at the "looked ahead" position
    nesterov: bool,

    /// Weight decay (L2 regularization) factor
    /// Adds weight_decay * parameter to the gradient
    weight_decay: T,
}

impl<T> SGD<T>
where
    T: Float
        + NumericCuda
        + Clone
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand
        + From<f64>
        + rand_distr::num_traits::FromPrimitive,
{
    /// Creates a new SGD optimizer
    ///
    /// # Arguments
    /// * `lr` - Learning rate
    /// * `momentum` - Momentum factor (0.0 for no momentum)
    /// * `dampening` - Dampening factor (typically 0.0)
    /// * `weight_decay` - Weight decay coefficient (0.0 for no weight decay)
    /// * `nesterov` - Whether to use Nesterov momentum
    pub fn new(lr: T, momentum: T, dampening: T, weight_decay: T, nesterov: bool) -> Self {
        Self {
            momentum_buffers: HashMap::new(),
            param_nodes: HashSet::new(),
            lr,
            momentum,
            dampening,
            nesterov,
            weight_decay,
        }
    }

    /// Convenience constructor with common defaults
    pub fn with_defaults(lr: T) -> Self {
        Self::new(
            lr,
            T::from(0.0), // no momentum
            T::from(0.0), // no dampening
            T::from(0.0), // no weight decay
            false,        // no nesterov
        )
    }

    /// Constructor with momentum but other defaults
    pub fn with_momentum(lr: T, momentum: T) -> Self {
        Self::new(
            lr,
            momentum,
            T::from(0.0), // no dampening
            T::from(0.0), // no weight decay
            false,        // no nesterov
        )
    }

    /// Clips gradient norm of all parameters to prevent exploding gradients
    ///
    /// This computes the global norm across all parameters and scales all gradients
    /// proportionally if the norm exceeds max_norm.
    ///
    /// # Arguments
    /// * `engine` - The computation graph engine
    /// * `max_norm` - Maximum allowed gradient norm
    pub fn clip_grad_norm(&mut self, engine: &mut Engine<T>, max_norm: T) {
        let mut total_norm_sq = <T as Numeric>::zero();

        // Calculate total gradient norm across all parameters
        for &param_node in &self.param_nodes {
            if let Some(grad) = engine.get_gradient(param_node) {
                for &g in grad.iter() {
                    total_norm_sq += g * g;
                }
            }
        }

        let total_norm = total_norm_sq.sqrt();

        // If norm exceeds max_norm, scale all gradients
        if total_norm > max_norm {
            let clip_coef = max_norm / total_norm;

            for &param_node in &self.param_nodes {
                if let Some(grad) = engine.get_gradient(param_node) {
                    let clipped_grad = grad.mul_scalar(clip_coef);
                    engine.set_gradient(param_node, clipped_grad);
                }
            }
        }
    }

    /// Sets whether to use Nesterov momentum
    pub fn set_nesterov(&mut self, nesterov: bool) {
        self.nesterov = nesterov;
    }

    /// Gets current learning rate
    pub fn get_lr(&self) -> T {
        self.lr
    }

    /// Sets learning rate (useful for learning rate scheduling)
    pub fn set_lr(&mut self, lr: T) {
        self.lr = lr;
    }
}

impl<T> Optimizer<T> for SGD<T>
where
    T: Float
        + NumericCuda
        + Clone
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand
        + rand_distr::num_traits::FromPrimitive,
{
    fn step(&mut self, engine: &mut Engine<T>) {
        let zero = <T as Numeric>::zero();
        let one = <T as Numeric>::one();

        for &param_node in &self.param_nodes {
            if let Some(grad) = engine.get_gradient(param_node) {
                let current_params = engine.get_data(param_node);

                // Apply weight decay: g = g + weight_decay * p
                let mut effective_grad = if self.weight_decay != zero {
                    if let Ok(updated_grad) =
                        grad.add(&current_params.mul_scalar(self.weight_decay))
                    {
                        updated_grad
                    } else {
                        panic!("Failed to apply weight decay to gradient");
                    }
                } else {
                    grad
                };

                // Initialize momentum buffer if it doesn't exist
                if !self.momentum_buffers.contains_key(&param_node) {
                    self.momentum_buffers
                        .insert(param_node, Tensor::zeros(current_params.shape()));
                }

                let momentum_buffer = self.momentum_buffers.get_mut(&param_node).unwrap();

                // Update momentum buffer: v = momentum * v + (1 - dampening) * g
                if self.momentum != zero {
                    *momentum_buffer = momentum_buffer
                        .mul_scalar(self.momentum)
                        .add(&effective_grad.mul_scalar(one - self.dampening))
                        .unwrap_or_else(|err| {
                            panic!("Failed to update momentum buffer: {}", err);
                        });

                    // For Nesterov: update = g + momentum * v
                    // For standard: update = v
                    effective_grad = if self.nesterov {
                        effective_grad
                            .add(&momentum_buffer.mul_scalar(self.momentum))
                            .unwrap_or_else(|err| {
                                panic!("Failed to update gradients: {}", err);
                            })
                    } else {
                        momentum_buffer.clone()
                    };
                }

                // Update parameters: p = p - lr * update
                let new_params = current_params
                    .add(&effective_grad.mul_scalar(-self.lr))
                    .unwrap_or_else(|err| {
                        panic!("Failed to update parameters: {}", err);
                    });
                engine.set_node_data(param_node, new_params);
            }
        }
    }

    fn reset_grad(&mut self, engine: &mut Engine<T>) {
        for &param_node in &self.param_nodes {
            engine.clear_gradient(param_node);
        }
    }

    fn add_param(&mut self, _param_id: usize, param_node_id: NodeId) {
        self.param_nodes.insert(param_node_id);
    }
}
/// Adam optimizer - Adaptive Moment Estimation
///
/// Adam computes individual adaptive learning rates for different parameters from
/// estimates of first and second moments of the gradients.
///
/// The algorithm:
/// ```text
/// m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
/// v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
/// m_hat_t = m_t / (1 - beta1^t)  // bias correction
/// v_hat_t = v_t / (1 - beta2^t)  // bias correction  
/// p_t = p_t - lr * m_hat_t / (sqrt(v_hat_t) + eps)
/// ```
///
/// Where:
/// - m_t: first moment estimate (momentum)
/// - v_t: second moment estimate (uncentered variance)
/// - m_hat_t, v_hat_t: bias-corrected estimates
/// - beta1: exponential decay rate for first moment (typically 0.9)
/// - beta2: exponential decay rate for second moment (typically 0.999)
/// - eps: small constant for numerical stability (typically 1e-8)
pub struct Adam<T>
where
    T: Float
        + NumericCuda
        + Clone
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand,
{
    /// Map of NodeId to their first moment estimates (momentum)
    first_moments: HashMap<NodeId, Tensor<T>>,

    /// Map of NodeId to their second moment estimates (uncentered variance)
    second_moments: HashMap<NodeId, Tensor<T>>,

    /// Set of parameter NodeIds being optimized
    param_nodes: HashSet<NodeId>,

    /// Learning rate
    lr: T,

    /// Exponential decay rate for first moment estimates
    /// Controls the momentum-like behavior (typically 0.9)
    beta1: T,

    /// Exponential decay rate for second moment estimates  
    /// Controls the adaptive learning rate behavior (typically 0.999)
    beta2: T,

    /// Small constant for numerical stability (typically 1e-8)
    /// Added to denominator to prevent division by zero
    eps: T,

    /// Weight decay (L2 regularization) factor
    weight_decay: T,

    /// Whether to use AMSGrad variant
    /// AMSGrad maintains the maximum of past second moments instead of exponential average
    amsgrad: bool,

    /// Maximum second moments for AMSGrad (only used if amsgrad=true)
    max_second_moments: HashMap<NodeId, Tensor<T>>,

    /// Current time step (starts at 0, incremented each step)
    /// Used for bias correction: beta^t
    step_count: u64,
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
impl<T> Adam<T>
where
    T: Float
        + NumericCuda
        + Clone
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand
        + From<f64>
        + rand_distr::num_traits::FromPrimitive,
{
    /// Creates a new Adam optimizer with custom parameters
    ///
    /// # Arguments
    /// * `lr` - Learning rate (typically 1e-3)
    /// * `beta1` - Exponential decay rate for first moment (typically 0.9)
    /// * `beta2` - Exponential decay rate for second moment (typically 0.999)
    /// * `eps` - Small constant for numerical stability (typically 1e-8)
    /// * `weight_decay` - Weight decay coefficient (typically 0.0)
    /// * `amsgrad` - Whether to use AMSGrad variant (typically false)
    pub fn new(lr: T, beta1: T, beta2: T, eps: T, weight_decay: T, amsgrad: bool) -> Self {
        Self {
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            param_nodes: HashSet::new(),
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            amsgrad,
            max_second_moments: HashMap::new(),
            step_count: 0,
        }
    }

    /// Creates Adam with commonly used default parameters
    /// lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, amsgrad=false
    pub fn with_defaults(lr: T) -> Self {
        Self::new(
            lr,
            T::from(0.9),
            T::from(0.999),
            T::from(1e-8),
            T::from(0.0),
            false,
        )
    }

    /// Creates Adam with weight decay
    pub fn with_weight_decay(lr: T, weight_decay: T) -> Self {
        Self::new(
            lr,
            T::from(0.9),
            T::from(0.999),
            T::from(1e-8),
            weight_decay,
            false,
        )
    }

    /// Creates AMSGrad variant
    pub fn amsgrad(lr: T) -> Self {
        Self::new(
            lr,
            T::from(0.9),
            T::from(0.999),
            T::from(1e-8),
            T::from(0.0),
            true,
        )
    }

    /// Gets current learning rate
    pub fn get_lr(&self) -> T {
        self.lr
    }

    /// Sets learning rate (useful for learning rate scheduling)
    pub fn set_lr(&mut self, lr: T) {
        self.lr = lr;
    }

    /// Gets current step count
    pub fn get_step_count(&self) -> u64 {
        self.step_count
    }

    /// Resets the optimizer state (clears all moment estimates and step count)
    pub fn reset_state(&mut self) {
        self.first_moments.clear();
        self.second_moments.clear();
        self.max_second_moments.clear();
        self.step_count = 0;
    }
}

impl<T> Optimizer<T> for Adam<T>
where
    T: Float
        + NumericCuda
        + Clone
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand
        + From<f64>
        + rand_distr::num_traits::FromPrimitive,
{
    fn step(&mut self, engine: &mut Engine<T>) {
        let zero = <T as Numeric>::zero();
        let one = <T as Numeric>::one();

        // Increment step count for bias correction
        self.step_count += 1;

        // Compute bias correction factors
        // As t increases, these approach 1.0, removing the bias
        let bias_correction1 = one - self.beta1.powi(self.step_count as i32);
        let bias_correction2 = one - self.beta2.powi(self.step_count as i32);

        for &param_node in &self.param_nodes {
            if let Some(grad) = engine.get_gradient(param_node) {
                let current_params = engine.get_data(param_node);

                // Apply weight decay: g = g + weight_decay * p
                let effective_grad = if self.weight_decay != zero {
                    grad.add(&current_params.mul_scalar(self.weight_decay))
                        .unwrap_or_else(|err| {
                            panic!("Failed to apply weight decay to gradient: {}", err);
                        })
                } else {
                    grad
                };

                // Initialize moment estimates if they don't exist
                if !self.first_moments.contains_key(&param_node) {
                    self.first_moments
                        .insert(param_node, Tensor::zeros(current_params.shape()));
                    self.second_moments
                        .insert(param_node, Tensor::zeros(current_params.shape()));

                    if self.amsgrad {
                        self.max_second_moments
                            .insert(param_node, Tensor::zeros(current_params.shape()));
                    }
                }

                let first_moment = self.first_moments.get_mut(&param_node).unwrap();
                let second_moment = self.second_moments.get_mut(&param_node).unwrap();

                // Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                *first_moment = first_moment
                    .mul_scalar(self.beta1)
                    .add(&effective_grad.mul_scalar(one - self.beta1))
                    .unwrap_or_else(|err| {
                        panic!("Failed to update first moment estimate: {}", err);
                    });

                // Update biased second moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                let grad_squared = effective_grad.mul(&effective_grad).unwrap_or_else(|err| {
                    panic!("Failed to square gradient: {}", err);
                });

                *second_moment = second_moment
                    .mul_scalar(self.beta2)
                    .add(&grad_squared.mul_scalar(one - self.beta2))
                    .unwrap_or_else(|err| {
                        panic!("Failed to update second moment estimate: {}", err);
                    });

                // Compute bias-corrected first moment estimate: m_hat_t = m_t / (1 - beta1^t)
                let first_moment_corrected = first_moment.div_scalar(bias_correction1);

                // Compute bias-corrected second moment estimate: v_hat_t = v_t / (1 - beta2^t)
                let mut second_moment_corrected = second_moment.div_scalar(bias_correction2);

                // AMSGrad: use max of current and past second moments
                if self.amsgrad {
                    let max_second_moment = self.max_second_moments.get_mut(&param_node).unwrap();

                    // Element-wise maximum: max_v_t = max(max_v_{t-1}, v_hat_t)
                    let updated_max = Tensor::new_with_device(
                        ndarray::Zip::from(max_second_moment.data())
                            .and(second_moment_corrected.data())
                            .map_collect(|&a, &b| if a > b { a } else { b }),
                        max_second_moment.device().clone(),
                    );

                    *max_second_moment = updated_max.clone();
                    second_moment_corrected = updated_max;
                }

                // Compute parameter update: p_t = p_t - lr * m_hat_t / (sqrt(v_hat_t) + eps)
                let denominator = Tensor::new_with_device(
                    second_moment_corrected.data().mapv(|x| x.sqrt() + self.eps),
                    second_moment_corrected.device().clone(),
                );

                let update = first_moment_corrected
                    .div(&denominator)
                    .unwrap_or_else(|err| {
                        panic!("Failed to compute the moment correction form {}", err)
                    })
                    .mul_scalar(self.lr);
                let new_params = current_params.add(&update.negate()).unwrap_or_else(|err| {
                    panic!("Failed to compute the moment correction form: {}", err)
                });

                engine.set_node_data(param_node, new_params);
            }
        }
    }

    fn reset_grad(&mut self, engine: &mut Engine<T>) {
        for &param_node in &self.param_nodes {
            engine.clear_gradient(param_node);
        }
    }

    fn add_param(&mut self, _param_id: usize, param_node_id: NodeId) {
        self.param_nodes.insert(param_node_id);
    }
}

#[cfg(test)]
mod optimizer_tests {
    use super::*;
    use crate::graph::Engine;
    use crate::nn::{Linear, Module};
    use crate::tensor::Tensor;

    /// Helper function to check if two floating point values are approximately equal
    fn approx_equal(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() < tolerance
    }

    #[test]
    fn test_sgd_momentum_bias_correction() {
        let mut engine = Engine::new();

        // Create a simple parameter tensor
        let param_tensor = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let param_node = engine.create_tensor(param_tensor, true);

        // Set a gradient
        let grad_tensor = Tensor::from_vec(vec![0.1, 0.2], &[2]).unwrap();
        engine.set_gradient(param_node, grad_tensor);

        // Create SGD with momentum
        let mut sgd = SGD::with_momentum(0.1, 0.9);
        sgd.add_param(0, param_node);

        // First step - should behave like no momentum since buffer is zero
        sgd.step(&mut engine);
        let after_first_step = engine.get_data(param_node);

        // Expected: param - lr * grad = [1.0, 2.0] - 0.1 * [0.1, 0.2] = [0.99, 1.98]
        assert!(approx_equal(after_first_step[0], 0.99, 1e-6));
        assert!(approx_equal(after_first_step[1], 1.98, 1e-6));

        // Set gradient again for second step
        let grad_tensor2 = Tensor::from_vec(vec![0.1, 0.2], &[2]).unwrap();
        engine.set_gradient(param_node, grad_tensor2);

        // Second step - now momentum should take effect
        sgd.step(&mut engine);
        let after_second_step = engine.get_data(param_node);

        // Momentum buffer after first step: [0.1, 0.2]
        // Momentum buffer after second step: 0.9 * [0.1, 0.2] + [0.1, 0.2] = [0.19, 0.38]
        // Update: [0.99, 1.98] - 0.1 * [0.19, 0.38] = [0.971, 1.942]
        assert!(approx_equal(after_second_step[0], 0.971, 1e-6));
        assert!(approx_equal(after_second_step[1], 1.942, 1e-6));
    }

    #[test]
    fn test_sgd_nesterov_momentum() {
        let mut engine = Engine::new();

        let param_tensor = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let param_node = engine.create_tensor(param_tensor, true);

        let grad_tensor = Tensor::from_vec(vec![0.1], &[1]).unwrap();
        engine.set_gradient(param_node, grad_tensor);

        // SGD with Nesterov momentum
        let mut sgd = SGD::new(
            0.1, 0.9,
            1.0, // One dampening factor prevents the momentum from causing any update at the first iteration
            0.0, true,
        );
        sgd.add_param(0, param_node);

        // First step
        sgd.step(&mut engine);
        let after_first = engine.get_data(param_node);

        // First step with Nesterov should be same as regular SGD (momentum buffer is zero)
        // Expected: 1.0 - 0.1 * 0.1 = 0.99
        println!("After first step: {:?}", after_first[0]);
        assert!(approx_equal(after_first[0], 0.99, 1e-6));

        // Set gradient for second step
        let grad_tensor2 = Tensor::from_vec(vec![0.1], &[1]).unwrap();
        engine.set_gradient(param_node, grad_tensor2);

        // Second step with Nesterov
        sgd.step(&mut engine);
        let after_second = engine.get_data(param_node);

        // With Nesterov: update = grad + momentum * velocity
        assert!(approx_equal(after_second[0], 0.98, 1e-6));
    }

    #[test]
    fn test_sgd_weight_decay() {
        let mut engine = Engine::new();

        let param_tensor = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let param_node = engine.create_tensor(param_tensor, true);

        // Zero gradient to isolate weight decay effect
        let grad_tensor = Tensor::from_vec(vec![0.0, 0.0], &[2]).unwrap();
        engine.set_gradient(param_node, grad_tensor);

        // SGD with weight decay but no momentum
        let mut sgd = SGD::new(0.1, 0.0, 0.0, 0.01, false);
        sgd.add_param(0, param_node);

        sgd.step(&mut engine);
        let after_step = engine.get_data(param_node);

        // With weight decay: effective_grad = grad + weight_decay * param = [0,0] + 0.01 * [1,2] = [0.01, 0.02]
        // Update: param - lr * effective_grad = [1,2] - 0.1 * [0.01, 0.02] = [0.999, 1.998]
        assert!(approx_equal(after_step[0], 0.999, 1e-6));
        assert!(approx_equal(after_step[1], 1.998, 1e-6));
    }

    #[test]
    fn test_adam_bias_correction() {
        let mut engine = Engine::new();

        let param_tensor = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let param_node = engine.create_tensor(param_tensor, true);

        let grad_tensor = Tensor::from_vec(vec![0.1], &[1]).unwrap();
        engine.set_gradient(param_node, grad_tensor);

        // Adam with default parameters
        let mut adam = Adam::with_defaults(0.001);
        adam.add_param(0, param_node);

        let original_param = engine.get_data(param_node)[0];

        // First step
        adam.step(&mut engine);
        let after_first = engine.get_data(param_node)[0];

        // Manual calculation for first step:
        // m1 = 0.9 * 0 + 0.1 * 0.1 = 0.01
        // v1 = 0.999 * 0 + 0.001 * 0.01 = 0.00001
        // m_hat = 0.01 / (1 - 0.9^1) = 0.01 / 0.1 = 0.1
        // v_hat = 0.00001 / (1 - 0.999^1) = 0.00001 / 0.001 = 0.01
        // update = 0.001 * 0.1 / (sqrt(0.01) + 1e-8) ≈ 0.001 * 0.1 / 0.1 = 0.001
        // new_param = 1.0 - 0.001 = 0.999

        let expected_change = 0.001;
        assert!(approx_equal(
            after_first,
            original_param - expected_change,
            1e-4
        ));
    }

    #[test]
    fn test_adam_convergence_behavior() {
        let mut engine = Engine::new();

        // Test Adam's ability to adapt learning rates
        let param_tensor = Tensor::from_vec(vec![10.0, 0.1], &[2]).unwrap();
        let param_node = engine.create_tensor(param_tensor, true);

        let mut adam = Adam::with_defaults(0.01);
        adam.add_param(0, param_node);

        // Simulate gradients with different magnitudes
        for step in 0..10 {
            let grad_tensor = if step < 5 {
                // Large gradient for first parameter, small for second
                Tensor::from_vec(vec![1.0, 0.01], &[2]).unwrap()
            } else {
                // Switch: small gradient for first parameter, large for second
                Tensor::from_vec(vec![0.01, 1.0], &[2]).unwrap()
            };

            engine.set_gradient(param_node, grad_tensor);
            adam.step(&mut engine);
            engine.clear_gradient(param_node);
        }

        let final_params = engine.get_data(param_node);

        // Both parameters should have moved towards zero, demonstrating adaptive learning
        assert!(final_params[0] < 10.0);
        assert!(final_params[1] < 0.1);
    }

    #[test]
    fn test_adam_amsgrad_variant() {
        let mut engine = Engine::new();

        let param_tensor = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let param_node = engine.create_tensor(param_tensor, true);

        // Create AMSGrad variant
        let mut amsgrad = Adam::amsgrad(0.001);
        amsgrad.add_param(0, param_node);

        // Test with varying gradient magnitudes
        let gradients = vec![1.0, 0.1, 0.5, 0.05, 0.2];
        let mut param_history = vec![];

        for &grad_val in &gradients {
            let grad_tensor = Tensor::from_vec(vec![grad_val], &[1]).unwrap();
            engine.set_gradient(param_node, grad_tensor);

            amsgrad.step(&mut engine);
            param_history.push(engine.get_data(param_node)[0]);

            engine.clear_gradient(param_node);
        }

        // AMSGrad should show stable progress (parameters should decrease monotonically for positive gradients)
        for i in 1..param_history.len() {
            assert!(param_history[i] < param_history[i - 1]);
        }
    }

    #[test]
    fn test_optimizer_parameter_groups() {
        let mut engine = Engine::new();

        // Create a simple neural network
        let linear1 = Linear::new(3, 2, true);
        let linear2 = Linear::new(2, 1, false); // No bias

        let mut sgd = SGD::with_momentum(0.01, 0.9);

        // Add parameters from both layers
        sgd.add_param_group(&linear1, &mut engine);
        sgd.add_param_group(&linear2, &mut engine);

        // Check that parameters were added (we should have 3 parameters: 2 weights + 1 bias)
        assert_eq!(sgd.param_nodes.len(), 3);
    }

    #[test]
    fn test_gradient_clipping() {
        let mut engine = Engine::new();

        // Create parameters with large gradients
        let param_tensor = Tensor::from_vec(vec![1.0, 1.0], &[2]).unwrap();
        let param_node = engine.create_tensor(param_tensor, true);

        // Set large gradients that exceed max norm
        let large_grad = Tensor::from_vec(vec![10.0, 10.0], &[2]).unwrap();
        engine.set_gradient(param_node, large_grad);

        let mut sgd = SGD::with_defaults(0.1);
        sgd.add_param(0, param_node);

        // Apply gradient clipping with max_norm = 1.0
        sgd.clip_grad_norm(&mut engine, 1.0);

        // Check that gradients were clipped
        let clipped_grad = engine.get_gradient(param_node).unwrap();
        let grad_norm: f64 = clipped_grad.iter().map(|&x| x * x).sum::<f64>().sqrt();

        assert!(approx_equal(grad_norm, 1.0, 1e-6));
    }

    #[test]
    fn test_optimizer_reset_grad() {
        let mut engine = Engine::new();

        let param_tensor = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let param_node = engine.create_tensor(param_tensor, true);

        let grad_tensor = Tensor::from_vec(vec![0.1], &[1]).unwrap();
        engine.set_gradient(param_node, grad_tensor);

        let mut sgd = SGD::with_defaults(0.1);
        sgd.add_param(0, param_node);

        // Verify gradient exists
        assert!(engine.get_gradient(param_node).is_some());

        // Reset gradients
        sgd.reset_grad(&mut engine);

        // Verify gradient was cleared
        assert!(engine.get_gradient(param_node).is_none());
    }

    #[test]
    fn test_neural_network_training_integration() {
        let mut engine = Engine::new();

        // Create a simple 2-layer network
        let linear1 = Linear::new(2, 3, true);
        let linear2 = Linear::new(3, 1, true);

        // Create input and target
        let input = engine
            .tensor_from_vec(vec![1.0, 2.0], &[1, 2], true)
            .unwrap();
        let target = engine.tensor_from_vec(vec![1.0], &[1, 1], false).unwrap();

        // Forward pass
        let hidden = linear1.forward(&mut engine, input).unwrap();
        let hidden_relu = engine.relu(hidden).unwrap();
        let output = linear2.forward(&mut engine, hidden_relu).unwrap();

        // Compute loss (simple MSE)
        let negated_target = engine.negate(target).unwrap();
        let diff = engine.add(output, negated_target).unwrap();
        let loss = engine.mul(diff, diff).unwrap();

        // Backward pass
        engine.backward(loss).unwrap();

        // Create optimizer and add parameters
        let mut adam = Adam::with_defaults(0.001);
        adam.add_param_group(&linear1, &mut engine);
        adam.add_param_group(&linear2, &mut engine);

        adam.step(&mut engine);

        // Verify that parameters were updated
        assert_eq!(adam.step_count, 1);

        // The test passes if no panics occur and optimizer step completes
    }

    #[test]
    fn test_learning_rate_scheduling() {
        let mut engine = Engine::new();

        let param_tensor = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let param_node = engine.create_tensor(param_tensor, true);

        let mut sgd = SGD::with_defaults(0.1);
        sgd.add_param(0, param_node);

        // Test learning rate getter/setter
        assert!(approx_equal(sgd.get_lr(), 0.1, 1e-6));

        sgd.set_lr(0.05);
        assert!(approx_equal(sgd.get_lr(), 0.05, 1e-6));

        // Verify the new learning rate is used
        let grad_tensor = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        engine.set_gradient(param_node, grad_tensor);

        let before_step = engine.get_data(param_node)[0];
        sgd.step(&mut engine);
        let after_step = engine.get_data(param_node)[0];

        // Expected change: -0.05 * 1.0 = -0.05
        let actual_change = after_step - before_step;
        assert!(approx_equal(actual_change, -0.05, 1e-6));
    }

    #[test]
    fn test_optimizer_state_management() {
        let mut engine = Engine::new();

        let param_tensor = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let param_node = engine.create_tensor(param_tensor, true);

        let mut adam = Adam::with_defaults(0.001);
        adam.add_param(0, param_node);

        // Perform a few steps to build up state
        for _ in 0..3 {
            let grad_tensor = Tensor::from_vec(vec![0.1], &[1]).unwrap();
            engine.set_gradient(param_node, grad_tensor);
            adam.step(&mut engine);
            engine.clear_gradient(param_node);
        }

        assert_eq!(adam.get_step_count(), 3);
        assert!(adam.first_moments.contains_key(&param_node));
        assert!(adam.second_moments.contains_key(&param_node));

        // Reset state
        adam.reset_state();

        assert_eq!(adam.get_step_count(), 0);
        assert!(adam.first_moments.is_empty());
        assert!(adam.second_moments.is_empty());
    }
}
