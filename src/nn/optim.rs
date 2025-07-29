use super::module::Module;
use crate::NodeId;
use crate::backend::{CPUNumber, GPUFloat};
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
    T: GPUFloat,
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
    T: GPUFloat,
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
    T: GPUFloat + From<f64>,
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
        let mut total_norm_sq = <T as CPUNumber>::zero();

        // Calculate total gradient norm across all parameters using CPUTensor methods
        for &param_node in &self.param_nodes {
            if let Some(grad) = engine.get_gradient(param_node) {
                // Use CPUTensor methods to compute squared norm
                let grad_squared = grad.mul(&grad)
                    .unwrap_or_else(|err| panic!("Failed to square gradient: {}", err));

                // Sum all elements - we need to access CPU data for this scalar operation
                if let Ok(cpu_data) = grad_squared.cpu_data() {
                    for &g_sq in cpu_data.iter() {
                        total_norm_sq += g_sq;
                    }
                } else {
                    panic!("Failed to access CPU data for gradient norm calculation");
                }
            }
        }

        let total_norm = total_norm_sq.sqrt();

        // If norm exceeds max_norm, scale all gradients
        if total_norm > max_norm {
            let clip_coef = max_norm / total_norm;

            for &param_node in &self.param_nodes {
                if let Some(grad) = engine.get_gradient(param_node) {
                    let clipped_grad = grad.mul_scalar(clip_coef)
                        .unwrap_or_else(|err| panic!("Failed to clip gradient: {}", err));
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
    T: GPUFloat,
{
    fn step(&mut self, engine: &mut Engine<T>) {
        let zero = <T as CPUNumber>::zero();
        let one = <T as CPUNumber>::one();

        for &param_node in &self.param_nodes {
            if let Some(grad) = engine.get_gradient(param_node) {
                let current_params = engine.get_data(param_node);

                // Apply weight decay using CPUTensor methods: g = g + weight_decay * p
                let mut effective_grad = if self.weight_decay != zero {
                    // Use CPUTensor::add and mul_scalar instead of direct operations
                    let weight_decay_term = current_params.mul_scalar(self.weight_decay)
                        .unwrap_or_else(|err| panic!("Failed to compute weight decay term: {}", err));

                    grad.add(&weight_decay_term)
                        .unwrap_or_else(|err| panic!("Failed to apply weight decay to gradient: {}", err))
                } else {
                    grad.clone()  // Clone since we need owned value
                };

                // Initialize momentum buffer if it doesn't exist
                self.momentum_buffers
                    .entry(param_node)
                    .or_insert_with(|| Tensor::zeros(current_params.shape()));

                let momentum_buffer = self.momentum_buffers.get_mut(&param_node).unwrap();

                // Update momentum buffer: v = momentum * v + (1 - dampening) * g
                if self.momentum != zero {
                    let momentum_term = momentum_buffer.mul_scalar(self.momentum)
                        .unwrap_or_else(|err| panic!("Failed to compute momentum term: {}", err));

                    let grad_term = effective_grad.mul_scalar(one - self.dampening)
                        .unwrap_or_else(|err| panic!("Failed to compute gradient term: {}", err));

                    *momentum_buffer = momentum_term.add(&grad_term)
                        .unwrap_or_else(|err| panic!("Failed to update momentum buffer: {}", err));

                    // For Nesterov: update = g + momentum * v
                    // For standard: update = v
                    effective_grad = if self.nesterov {
                        let nesterov_term = momentum_buffer.mul_scalar(self.momentum)
                            .unwrap_or_else(|err| panic!("Failed to compute Nesterov term: {}", err));

                        effective_grad.add(&nesterov_term)
                            .unwrap_or_else(|err| panic!("Failed to compute Nesterov update: {}", err))
                    } else {
                        momentum_buffer.clone()
                    };
                }

                // Update parameters: p = p - lr * update
                // Use neg() method instead of negate(), and add() instead of direct operations
                let lr_scaled_update = effective_grad.mul_scalar(self.lr)
                    .unwrap_or_else(|err| panic!("Failed to scale update by learning rate: {}", err));

                let negative_update = lr_scaled_update.neg()
                    .unwrap_or_else(|err| panic!("Failed to negate update: {}", err));

                let new_params = current_params.add(&negative_update)
                    .unwrap_or_else(|err| panic!("Failed to update parameters: {}", err));

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
/// - eps: small constant for CPUNumbereral stability (typically 1e-8)
pub struct Adam<T>
where
    T: GPUFloat,
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

    /// Small constant for CPUNumbereral stability (typically 1e-8)
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
impl<T> Optimizer<T> for Adam<T>
where
    T: GPUFloat + From<f64>,
{
    fn step(&mut self, engine: &mut Engine<T>) {
        let zero = <T as CPUNumber>::zero();
        let one = <T as CPUNumber>::one();

        // Increment step count for bias correction
        self.step_count += 1;

        // Compute bias correction factors
        let bias_correction1 = one - self.beta1.powi(self.step_count as i32);
        let bias_correction2 = one - self.beta2.powi(self.step_count as i32);

        for &param_node in &self.param_nodes {
            if let Some(grad) = engine.get_gradient(param_node) {
                let current_params = engine.get_data(param_node);

                // Apply weight decay using CPUTensor methods: g = g + weight_decay * p
                let effective_grad = if self.weight_decay != zero {
                    let weight_decay_term = current_params.mul_scalar(self.weight_decay)
                        .unwrap_or_else(|err| panic!("Failed to compute weight decay term: {}", err));

                    grad.add(&weight_decay_term)
                        .unwrap_or_else(|err| panic!("Failed to apply weight decay to gradient: {}", err))
                } else {
                    grad.clone()
                };

                // Initialize moment estimates if they don't exist
                if !self.first_moments.contains_key(&param_node) {
                    self.first_moments.insert(param_node, Tensor::zeros(current_params.shape()));
                    self.second_moments.insert(param_node, Tensor::zeros(current_params.shape()));

                    if self.amsgrad {
                        self.max_second_moments.insert(param_node, Tensor::zeros(current_params.shape()));
                    }
                }

                let first_moment = self.first_moments.get_mut(&param_node).unwrap();
                let second_moment = self.second_moments.get_mut(&param_node).unwrap();

                // Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                let first_momentum_term = first_moment.mul_scalar(self.beta1)
                    .unwrap_or_else(|err| panic!("Failed to compute first momentum term: {}", err));

                let first_grad_term = effective_grad.mul_scalar(one - self.beta1)
                    .unwrap_or_else(|err| panic!("Failed to compute first gradient term: {}", err));

                *first_moment = first_momentum_term.add(&first_grad_term)
                    .unwrap_or_else(|err| panic!("Failed to update first moment estimate: {}", err));

                // Update biased second moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                let grad_squared = effective_grad.mul(&effective_grad)
                    .unwrap_or_else(|err| panic!("Failed to square gradient: {}", err));

                let second_momentum_term = second_moment.mul_scalar(self.beta2)
                    .unwrap_or_else(|err| panic!("Failed to compute second momentum term: {}", err));

                let second_grad_term = grad_squared.mul_scalar(one - self.beta2)
                    .unwrap_or_else(|err| panic!("Failed to compute second gradient term: {}", err));

                *second_moment = second_momentum_term.add(&second_grad_term)
                    .unwrap_or_else(|err| panic!("Failed to update second moment estimate: {}", err));

                // Compute bias-corrected first moment estimate: m_hat_t = m_t / (1 - beta1^t)
                let first_moment_corrected = first_moment.div_scalar(bias_correction1)
                    .unwrap_or_else(|err| panic!("Failed to compute bias correction for first moment: {}", err));

                // Compute bias-corrected second moment estimate: v_hat_t = v_t / (1 - beta2^t)
                let mut second_moment_corrected = second_moment.div_scalar(bias_correction2)
                    .unwrap_or_else(|err| panic!("Failed to compute bias correction for second moment: {}", err));

                // AMSGrad: use max of current and past second moments
                if self.amsgrad {
                    let max_second_moment = self.max_second_moments.get_mut(&param_node).unwrap();

                    // Element-wise maximum using CPUTensor::max method
                    let updated_max = max_second_moment.max(&second_moment_corrected)
                        .unwrap_or_else(|err| panic!("Failed to compute element-wise maximum: {}", err));

                    *max_second_moment = updated_max.clone();
                    second_moment_corrected = updated_max;
                }

                // Compute parameter update: p_t = p_t - lr * m_hat_t / (sqrt(v_hat_t) + eps)
                // Create denominator tensor: sqrt(v_hat_t) + eps
                let denominator = self.create_sqrt_plus_eps_tensor(&second_moment_corrected).unwrap_or_else(|err| panic!("Failed to create sqrt plus eps tensor: {}", err));

                let update_direction = first_moment_corrected.div(&denominator)
                    .unwrap_or_else(|err| panic!("Failed to compute update direction: {}", err));

                let scaled_update = update_direction.mul_scalar(self.lr)
                    .unwrap_or_else(|err| panic!("Failed to scale update by learning rate: {}", err));

                let negative_update = scaled_update.neg()
                    .unwrap_or_else(|err| panic!("Failed to negate update: {}", err));

                let new_params = current_params.add(&negative_update)
                    .unwrap_or_else(|err| panic!("Failed to update parameters: {}", err));

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

impl<T> Adam<T>
where
    T: GPUFloat + From<f64>,
{
    /// Creates a new Adam optimizer with custom parameters
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

    /// Creates a tensor with sqrt(input) + eps using CPUTensor methods only
    /// This replaces the direct ndarray mapv operation that was breaking
    fn create_sqrt_plus_eps_tensor(&self, input: &Tensor<T>) -> Result<Tensor<T>, String> {
        // Use CPUTensor methods instead of direct element access
        // First apply sqrt to all elements, then add eps scalar
        let sqrt_tensor = input.sqrt()?;
        let result = sqrt_tensor.add_scalar(self.eps)?;
        Ok(result)
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

    /// Helper to get scalar value from tensor using CPUTensor methods
    fn get_scalar_value<T>(tensor: &Tensor<T>) -> T
    where
        T: GPUFloat + Clone
    {
        // Use CPUTensor method to access data instead of direct indexing
        let cpu_data = tensor.cpu_data().expect("Failed to get CPU data");
        cpu_data[[0]].clone()  // ndarray indexing for scalar
    }

    /// Helper to get vector values from tensor using CPUTensor methods
    fn get_vector_values<T>(tensor: &Tensor<T>) -> Vec<T>
    where
        T: GPUFloat + Clone
    {
        let cpu_data = tensor.cpu_data().expect("Failed to get CPU data");
        cpu_data.iter().cloned().collect()
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

        // First step
        sgd.step(&mut engine);
        let after_first_step = engine.get_data(param_node);
        let values = get_vector_values(&after_first_step);

        // Expected: param - lr * grad = [1.0, 2.0] - 0.1 * [0.1, 0.2] = [0.99, 1.98]
        assert!(approx_equal(values[0], 0.99, 1e-6));
        assert!(approx_equal(values[1], 1.98, 1e-6));

        // Set gradient again for second step
        let grad_tensor2 = Tensor::from_vec(vec![0.1, 0.2], &[2]).unwrap();
        engine.set_gradient(param_node, grad_tensor2);

        // Second step - now momentum should take effect
        sgd.step(&mut engine);
        let after_second_step = engine.get_data(param_node);
        let values2 = get_vector_values(&after_second_step);

        // Momentum buffer after first step: [0.1, 0.2]
        // Momentum buffer after second step: 0.9 * [0.1, 0.2] + [0.1, 0.2] = [0.19, 0.38]
        // Update: [0.99, 1.98] - 0.1 * [0.19, 0.38] = [0.971, 1.942]
        assert!(approx_equal(values2[0], 0.971, 1e-6));
        assert!(approx_equal(values2[1], 1.942, 1e-6));
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
        let values = get_vector_values(&after_step);

        // With weight decay: effective_grad = grad + weight_decay * param = [0,0] + 0.01 * [1,2] = [0.01, 0.02]
        // Update: param - lr * effective_grad = [1,2] - 0.1 * [0.01, 0.02] = [0.999, 1.998]
        assert!(approx_equal(values[0], 0.999, 1e-6));
        assert!(approx_equal(values[1], 1.998, 1e-6));
    }

    #[test]
    fn test_adam_bias_correction() {
        let mut engine = Engine::new();

        let param_tensor = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let param_node = engine.create_tensor(param_tensor, true);

        let grad_tensor = Tensor::from_vec(vec![0.1], &[1]).unwrap();
        engine.set_gradient(param_node, grad_tensor);

        // Adam with default parameters - specify type for From<f64> bound
        let mut adam = Adam::<f64>::with_defaults(0.001);
        adam.add_param(0, param_node);

        let original_param = get_scalar_value(&engine.get_data(param_node));

        // First step
        adam.step(&mut engine);
        let after_first = get_scalar_value(&engine.get_data(param_node));

        // Verify parameter changed in expected direction (decreased for positive gradient)
        assert!(after_first < original_param);

        // Verify change magnitude is reasonable
        let change = original_param - after_first;
        assert!(change > 0.0 && change < 0.01); // Should be small but positive
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
        let grad_values = get_vector_values(&clipped_grad);

        // Calculate norm using the clipped values
        let grad_norm: f64 = grad_values.iter().map(|&x| x * x).sum::<f64>().sqrt();

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

        let before_step = get_scalar_value(&engine.get_data(param_node));
        sgd.step(&mut engine);
        let after_step = get_scalar_value(&engine.get_data(param_node));

        // Expected change: -0.05 * 1.0 = -0.05
        let actual_change = after_step - before_step;
        assert!(approx_equal(actual_change, -0.05, 1e-6));
    }
}
