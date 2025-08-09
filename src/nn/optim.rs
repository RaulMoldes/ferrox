use super::module::Module;
use crate::NodeId;
use crate::backend::{CPUNumber, FerroxCudaF};
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
    T: FerroxCudaF,
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
    T: FerroxCudaF,
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
    T: FerroxCudaF + From<f64>,
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
                let grad_squared = grad
                    .mul(&grad)
                    .unwrap_or_else(|err| panic!("Failed to square gradient: {}", err));

                // Sum all elements - we need to access CPU data for this scalar operation
                if let Ok(cpu_data) = grad_squared.as_slice() {
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
                    let clipped_grad = grad
                        .mul_scalar(clip_coef)
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
    T: FerroxCudaF,
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
                    let weight_decay_term = current_params
                        .mul_scalar(self.weight_decay)
                        .unwrap_or_else(|err| {
                            panic!("Failed to compute weight decay term: {}", err)
                        });

                    grad.add(&weight_decay_term).unwrap_or_else(|err| {
                        panic!("Failed to apply weight decay to gradient: {}", err)
                    })
                } else {
                    grad.clone() // Clone since we need owned value
                };

                // Initialize momentum buffer if it doesn't exist
                self.momentum_buffers.entry(param_node).or_insert_with(|| {
                    Tensor::zeros(current_params.shape())
                        .expect("Zeroed tensor failed to be created")
                });

                let momentum_buffer = self.momentum_buffers.get_mut(&param_node).unwrap();

                // Update momentum buffer: v = momentum * v + (1 - dampening) * g
                if self.momentum != zero {
                    let momentum_term = momentum_buffer
                        .mul_scalar(self.momentum)
                        .unwrap_or_else(|err| panic!("Failed to compute momentum term: {}", err));

                    let grad_term = effective_grad
                        .mul_scalar(one - self.dampening)
                        .unwrap_or_else(|err| panic!("Failed to compute gradient term: {}", err));

                    *momentum_buffer = momentum_term
                        .add(&grad_term)
                        .unwrap_or_else(|err| panic!("Failed to update momentum buffer: {}", err));

                    // For Nesterov: update = g + momentum * v
                    // For standard: update = v
                    effective_grad = if self.nesterov {
                        let nesterov_term = momentum_buffer
                            .mul_scalar(self.momentum)
                            .unwrap_or_else(|err| {
                                panic!("Failed to compute Nesterov term: {}", err)
                            });

                        effective_grad.add(&nesterov_term).unwrap_or_else(|err| {
                            panic!("Failed to compute Nesterov update: {}", err)
                        })
                    } else {
                        momentum_buffer.clone()
                    };
                }

                // Update parameters: p = p - lr * update
                // Use neg() method instead of negate(), and add() instead of direct operations
                let lr_scaled_update = effective_grad.mul_scalar(self.lr).unwrap_or_else(|err| {
                    panic!("Failed to scale update by learning rate: {}", err)
                });

                let negative_update = lr_scaled_update
                    .neg()
                    .unwrap_or_else(|err| panic!("Failed to negate update: {}", err));

                let new_params = current_params
                    .add(&negative_update)
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
    T: FerroxCudaF,
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
    T: FerroxCudaF + From<f64>,
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
                    let weight_decay_term = current_params
                        .mul_scalar(self.weight_decay)
                        .unwrap_or_else(|err| {
                            panic!("Failed to compute weight decay term: {}", err)
                        });

                    grad.add(&weight_decay_term).unwrap_or_else(|err| {
                        panic!("Failed to apply weight decay to gradient: {}", err)
                    })
                } else {
                    grad.clone()
                };

                // Initialize moment estimates if they don't exist
                if !self.first_moments.contains_key(&param_node) {
                    self.first_moments.insert(
                        param_node,
                        Tensor::zeros(current_params.shape())
                            .expect("Zeroed tensor failed to be created"),
                    );
                    self.second_moments.insert(
                        param_node,
                        Tensor::zeros(current_params.shape())
                            .expect("Zeroed tensor failed to be created"),
                    );

                    if self.amsgrad {
                        self.max_second_moments.insert(
                            param_node,
                            Tensor::zeros(current_params.shape())
                                .expect("Zeroed tensor failed to be created"),
                        );
                    }
                }

                let first_moment = self.first_moments.get_mut(&param_node).unwrap();
                let second_moment = self.second_moments.get_mut(&param_node).unwrap();

                // Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                let first_momentum_term = first_moment
                    .mul_scalar(self.beta1)
                    .unwrap_or_else(|err| panic!("Failed to compute first momentum term: {}", err));

                let first_grad_term = effective_grad
                    .mul_scalar(one - self.beta1)
                    .unwrap_or_else(|err| panic!("Failed to compute first gradient term: {}", err));

                *first_moment = first_momentum_term
                    .add(&first_grad_term)
                    .unwrap_or_else(|err| {
                        panic!("Failed to update first moment estimate: {}", err)
                    });

                // Update biased second moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                let grad_squared = effective_grad
                    .mul(&effective_grad)
                    .unwrap_or_else(|err| panic!("Failed to square gradient: {}", err));

                let second_momentum_term =
                    second_moment.mul_scalar(self.beta2).unwrap_or_else(|err| {
                        panic!("Failed to compute second momentum term: {}", err)
                    });

                let second_grad_term =
                    grad_squared
                        .mul_scalar(one - self.beta2)
                        .unwrap_or_else(|err| {
                            panic!("Failed to compute second gradient term: {}", err)
                        });

                *second_moment =
                    second_momentum_term
                        .add(&second_grad_term)
                        .unwrap_or_else(|err| {
                            panic!("Failed to update second moment estimate: {}", err)
                        });

                // Compute bias-corrected first moment estimate: m_hat_t = m_t / (1 - beta1^t)
                let first_moment_corrected = first_moment
                    .div_scalar(bias_correction1)
                    .unwrap_or_else(|err| {
                        panic!(
                            "Failed to compute bias correction for first moment: {}",
                            err
                        )
                    });

                // Compute bias-corrected second moment estimate: v_hat_t = v_t / (1 - beta2^t)
                let mut second_moment_corrected = second_moment
                    .div_scalar(bias_correction2)
                    .unwrap_or_else(|err| {
                        panic!(
                            "Failed to compute bias correction for second moment: {}",
                            err
                        )
                    });

                // AMSGrad: use max of current and past second moments
                if self.amsgrad {
                    let max_second_moment = self.max_second_moments.get_mut(&param_node).unwrap();

                    // Element-wise maximum using CPUTensor::max method
                    let updated_max = max_second_moment
                        .max()
                        .unwrap_or_else(|err| {
                            panic!("Failed to compute element-wise maximum: {}", err)
                        });

                    *max_second_moment = updated_max.clone();
                    second_moment_corrected = updated_max;
                }

                // Compute parameter update: p_t = p_t - lr * m_hat_t / (sqrt(v_hat_t) + eps)
                // Create denominator tensor: sqrt(v_hat_t) + eps
                let denominator = self
                    .create_sqrt_plus_eps_tensor(&second_moment_corrected)
                    .unwrap_or_else(|err| panic!("Failed to create sqrt plus eps tensor: {}", err));

                let update_direction = first_moment_corrected
                    .div(&denominator)
                    .unwrap_or_else(|err| panic!("Failed to compute update direction: {}", err));

                let scaled_update = update_direction.mul_scalar(self.lr).unwrap_or_else(|err| {
                    panic!("Failed to scale update by learning rate: {}", err)
                });

                let negative_update = scaled_update
                    .neg()
                    .unwrap_or_else(|err| panic!("Failed to negate update: {}", err));

                let new_params = current_params
                    .add(&negative_update)
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
    T: FerroxCudaF + From<f64>,
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
