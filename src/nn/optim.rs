use crate::nn::Module;
use crate::graph::NodeId;
use crate::backend::{FerroxF, FerroxCudaF};
use crate::graph::AutoFerroxEngine;
use crate::backend::Tensor;
use std::collections::{HashMap, HashSet};

/// Trait that all optimizers must implement
/// Provides a common interface for all optimization algorithms, enabling easy
/// swapping between SGD, Adam, RMSprop, etc. This design follows PyTorch's optimizer interface.
pub trait Optimizer<T>
where
    T: FerroxCudaF,
{
    /// Perform one optimization step using computed gradients
    /// This method should: read gradients, update optimizer state, compute updates, apply updates
    fn step(&mut self, engine: &mut AutoFerroxEngine<T>);

    /// Clear all gradients for registered parameters
    /// Should be called after each optimization step to prepare for next forward/backward pass
    fn reset_grad(&mut self, engine: &mut AutoFerroxEngine<T>);

    /// Add a parameter to be optimized
    fn add_param(&mut self, param_id: usize, param_node_id: NodeId);

    /// Add all parameters from a module to the optimizer
    /// Convenience method that automatically registers all parameters from a neural network module
    fn add_param_group<M>(&mut self, module: &M, engine: &mut AutoFerroxEngine<T>)
    where
        M: Module<T>,
        T: FerroxCudaF,
    {
        let param_map = module.create_parameters_in_graph(engine);
        for (i, (_, node_id)) in param_map.into_iter().enumerate() {
            self.add_param(i, node_id);
        }
    }
}

/// Stochastic Gradient Descent optimizer with momentum and weight decay
/// Implements: param = param - lr * (grad + weight_decay * param)
/// With momentum: velocity = momentum * velocity + grad, param = param - lr * velocity
pub struct SGD<T>
where
    T: FerroxCudaF,
{
    lr: T,
    momentum: T,
    weight_decay: T,
    nesterov: bool,
    param_nodes: HashSet<NodeId>,
    momentum_buffers: HashMap<NodeId, Tensor<T>>,
}

impl<T> SGD<T>
where
    T: FerroxCudaF,
{
    /// Create new SGD optimizer
    pub fn new(lr: T, momentum: T, weight_decay: T, nesterov: bool) -> Self {
        Self {
            lr,
            momentum,
            weight_decay,
            nesterov,
            param_nodes: HashSet::new(),
            momentum_buffers: HashMap::new(),
        }
    }

    /// Create SGD with commonly used defaults (no momentum, no weight decay)
    pub fn with_defaults(lr: T) -> Self {
        Self::new(lr, <T as FerroxF>::zero(), <T as FerroxF>::zero(), false)
    }

    /// Create SGD with momentum
    pub fn with_momentum(lr: T, momentum: T) -> Self {
        Self::new(lr, momentum, <T as FerroxF>::zero(), false)
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> T {
        self.lr
    }

    /// Set learning rate for scheduling
    pub fn set_lr(&mut self, lr: T) {
        self.lr = lr;
    }

    /// Enable/disable Nesterov momentum
    pub fn set_nesterov(&mut self, nesterov: bool) {
        self.nesterov = nesterov;
    }

    /// Gradient clipping by norm - prevents exploding gradients
    pub fn clip_grad_norm(&mut self, engine: &mut AutoFerroxEngine<T>, max_norm: T) {
        let zero = <T as FerroxF>::zero();
        let mut total_norm_sq = zero;

        // Calculate total gradient norm using tensor operations
        for &param_node in &self.param_nodes {
            if let Some(grad) = engine.get_gradient(param_node) {
                let grad_squared = grad.mul(&grad)
                    .expect("Failed to compute gradient squared");

                // Sum all elements to get scalar norm contribution
                let norm_contrib_tensor = grad_squared.sum(None)
                    .expect("Failed to compute gradient norm");

                // Extract scalar value from the resulting tensor
                if let Ok(cpu_data) = norm_contrib_tensor.as_slice() {
                    if !cpu_data.is_empty() {
                        total_norm_sq += cpu_data[0];
                    }
                }
            }
        }

        let total_norm = total_norm_sq.sqrt();

        // Apply clipping if needed
        if total_norm > max_norm {
            let clip_coef = max_norm / total_norm;

            for &param_node in &self.param_nodes {
                if let Some(grad) = engine.get_gradient(param_node) {
                    let clipped_grad = grad.mul_scalar(clip_coef)
                        .expect("Failed to clip gradient");
                    engine.set_gradient(param_node, clipped_grad);
                }
            }
        }
    }
}

impl<T> Optimizer<T> for SGD<T>
where
    T: FerroxCudaF,
{
    fn step(&mut self, engine: &mut AutoFerroxEngine<T>) {
        let zero = <T as FerroxF>::zero();

        for &param_node in &self.param_nodes {
            if let Some(grad) = engine.get_gradient(param_node) {
                // Get current parameter values
                let current_params = engine.get_tensor_data(param_node)
                    .expect("Parameter tensor not found").clone();

                // Apply weight decay: effective_grad = grad + weight_decay * params
                let effective_grad = if self.weight_decay != zero {
                    let weight_decay_term = current_params.mul_scalar(self.weight_decay)
                        .expect("Failed to compute weight decay term");
                    grad.add(&weight_decay_term)
                        .expect("Failed to add weight decay")
                } else {
                    grad.clone()
                };

                let update = if self.momentum != zero {
                    // Get or initialize momentum buffer
                    let momentum_buffer = self.momentum_buffers.entry(param_node)
                        .or_insert_with(|| {
                            Tensor::zeros(current_params.shape())
                                .expect("Failed to create momentum buffer")
                        });

                    // Update momentum: buffer = momentum * buffer + effective_grad
                    let momentum_term = momentum_buffer.mul_scalar(self.momentum)
                        .expect("Failed to scale momentum buffer");
                    let new_buffer = momentum_term.add(&effective_grad)
                        .expect("Failed to update momentum buffer");

                    *momentum_buffer = new_buffer.clone();

                    if self.nesterov {
                        // Nesterov: update = momentum * new_buffer + effective_grad
                        let nesterov_momentum = new_buffer.mul_scalar(self.momentum)
                            .expect("Failed to compute Nesterov momentum term");
                        nesterov_momentum.add(&effective_grad)
                            .expect("Failed to compute Nesterov update")
                    } else {
                        new_buffer
                    }
                } else {
                    effective_grad
                };

                // Apply learning rate and update parameters
                let scaled_update = update.mul_scalar(self.lr)
                    .expect("Failed to scale update by learning rate");
                let new_params = current_params.sub(&scaled_update)
                    .expect("Failed to update parameters");

                // Update parameter in the graph using your new method
                engine.update_parameter(param_node, new_params)
                    .expect("Failed to update parameter in graph");
            }
        }
    }

    fn reset_grad(&mut self, engine: &mut AutoFerroxEngine<T>) {
        for &param_node in &self.param_nodes {
            engine.clear_gradient(param_node);
        }
    }

    fn add_param(&mut self, _param_id: usize, param_node_id: NodeId) {
        self.param_nodes.insert(param_node_id);
    }
}

/// Adam optimizer - Adaptive Moment Estimation
/// Maintains separate adaptive learning rates for each parameter based on first and second moments
pub struct Adam<T>
where
    T: FerroxCudaF,
{
    lr: T,
    beta1: T,
    beta2: T,
    eps: T,
    weight_decay: T,
    amsgrad: bool,
    param_nodes: HashSet<NodeId>,
    first_moments: HashMap<NodeId, Tensor<T>>,
    second_moments: HashMap<NodeId, Tensor<T>>,
    max_second_moments: HashMap<NodeId, Tensor<T>>,
    step_count: u64,
}

impl<T> Adam<T>
where
    T: FerroxCudaF + From<f64>,
{
    /// Create new Adam optimizer with custom parameters
    pub fn new(lr: T, beta1: T, beta2: T, eps: T, weight_decay: T, amsgrad: bool) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            amsgrad,
            param_nodes: HashSet::new(),
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            max_second_moments: HashMap::new(),
            step_count: 0,
        }
    }

    /// Create Adam with standard defaults
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

    /// Create Adam with weight decay (AdamW-style)
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

    /// Create AMSGrad variant
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

    pub fn get_lr(&self) -> T { self.lr }
    pub fn set_lr(&mut self, lr: T) { self.lr = lr; }
    pub fn get_step_count(&self) -> u64 { self.step_count }

    /// Reset optimizer state
    pub fn reset_state(&mut self) {
        self.first_moments.clear();
        self.second_moments.clear();
        self.max_second_moments.clear();
        self.step_count = 0;
    }

    /// Compute bias correction factors
    fn compute_bias_corrections(&self) -> (T, T) {
        let step_f64 = self.step_count as f64;
        let beta1_power = FerroxF::to_f64(self.beta1).powf(step_f64);
        let beta2_power = FerroxF::to_f64(self.beta2).powf(step_f64);

        (T::from(1.0 - beta1_power), T::from(1.0 - beta2_power))
    }
}

impl<T> Optimizer<T> for Adam<T>
where
    T: FerroxCudaF + From<f64>,
{
    fn step(&mut self, engine: &mut AutoFerroxEngine<T>) {
        let zero = <T as FerroxF>::zero();
        let one = <T as FerroxF>::one();

        self.step_count += 1;
        let (bias_correction1, bias_correction2) = self.compute_bias_corrections();

        for &param_node in &self.param_nodes {
            if let Some(grad) = engine.get_gradient(param_node) {
                let current_params = engine.get_tensor_data(param_node)
                    .expect("Parameter tensor not found").clone();

                // Apply weight decay if specified
                let effective_grad = if self.weight_decay != zero {
                    let weight_decay_term = current_params.mul_scalar(self.weight_decay)
                        .expect("Failed to compute weight decay term");
                    grad.add(&weight_decay_term)
                        .expect("Failed to add weight decay")
                } else {
                    grad.clone()
                };

                // Get or initialize moment buffers
                let first_moment = self.first_moments.entry(param_node)
                    .or_insert_with(|| {
                        let device = current_params.device();
                        Tensor::zeros_with_device(current_params.shape(), device)
                            .expect("Failed to create first moment buffer")
                    });

                let second_moment = self.second_moments.entry(param_node)
                    .or_insert_with(|| {
                        let device = current_params.device();
                        Tensor::zeros_with_device(current_params.shape(), device)
                            .expect("Failed to create second moment buffer")
                    });

                // Update first moment: m = beta1 * m + (1 - beta1) * grad
                let one_minus_beta1 = one - self.beta1;
                let first_term = first_moment.mul_scalar(self.beta1)
                    .expect("Failed to scale first moment");
                let second_term = effective_grad.mul_scalar(one_minus_beta1)
                    .expect("Failed to scale gradient for first moment");
                let new_first_moment = first_term.add(&second_term)
                    .expect("Failed to update first moment");
                *first_moment = new_first_moment.clone();

                // Update second moment: v = beta2 * v + (1 - beta2) * grad^2
                let one_minus_beta2 = one - self.beta2;
                let grad_squared = effective_grad.mul(&effective_grad)
                    .expect("Failed to compute gradient squared");
                let first_term = second_moment.mul_scalar(self.beta2)
                    .expect("Failed to scale second moment");
                let second_term = grad_squared.mul_scalar(one_minus_beta2)
                    .expect("Failed to scale gradient squared for second moment");
                let new_second_moment = first_term.add(&second_term)
                    .expect("Failed to update second moment");
                *second_moment = new_second_moment.clone();

                // For AMSGrad, maintain maximum of second moments
                let moment_for_update = if self.amsgrad {
                    let max_second_moment = self.max_second_moments.entry(param_node)
                        .or_insert_with(|| {
                            let device = current_params.device();
                            Tensor::zeros_with_device(current_params.shape(), device)
                                .expect("Failed to create max second moment buffer")
                        });

                    // max_v = max(max_v, v) - use element-wise maximum
                    let updated_max = max_second_moment.max(&new_second_moment)
                        .expect("Failed to compute element-wise maximum");
                    *max_second_moment = updated_max.clone();
                    updated_max
                } else {
                    new_second_moment
                };

                // Bias-corrected first moment: m_hat = m / (1 - beta1^t)
                let corrected_first_moment = new_first_moment.div_scalar(bias_correction1)
                    .expect("Failed to compute bias-corrected first moment");

                // Bias-corrected second moment: v_hat = v / (1 - beta2^t)
                let corrected_second_moment = moment_for_update.div_scalar(bias_correction2)
                    .expect("Failed to compute bias-corrected second moment");

                // Compute update: update = lr * m_hat / (sqrt(v_hat) + eps)
                let sqrt_second_moment = corrected_second_moment.sqrt()
                    .expect("Failed to compute square root of second moment");
                let denominator = sqrt_second_moment.add_scalar(self.eps)
                    .expect("Failed to add epsilon for numerical stability");
                let update_direction = corrected_first_moment.div(&denominator)
                    .expect("Failed to compute update direction");
                let scaled_update = update_direction.mul_scalar(self.lr)
                    .expect("Failed to scale update by learning rate");

                // Apply update: params = params - scaled_update
                let new_params = current_params.sub(&scaled_update)
                    .expect("Failed to update parameters");

                // Update parameter in the computation graph
                engine.update_parameter(param_node, new_params)
                    .expect("Failed to update parameter in graph");
            }
        }
    }

    fn reset_grad(&mut self, engine: &mut AutoFerroxEngine<T>) {
        for &param_node in &self.param_nodes {
            engine.clear_gradient(param_node);
        }
    }

    fn add_param(&mut self, _param_id: usize, param_node_id: NodeId) {
        self.param_nodes.insert(param_node_id);
    }
}

/// RMSprop optimizer - Root Mean Square Propagation
/// Maintains a moving average of squared gradients to normalize the gradient
pub struct RMSprop<T>
where
    T: FerroxCudaF,
{
    lr: T,
    alpha: T,
    eps: T,
    weight_decay: T,
    momentum: T,
    centered: bool,
    param_nodes: HashSet<NodeId>,
    square_averages: HashMap<NodeId, Tensor<T>>,
    momentum_buffers: HashMap<NodeId, Tensor<T>>,
    grad_averages: HashMap<NodeId, Tensor<T>>, // For centered variant
}

impl<T> RMSprop<T>
where
    T: FerroxCudaF + From<f64>,
{
    /// Create new RMSprop optimizer
    pub fn new(lr: T, alpha: T, eps: T, weight_decay: T, momentum: T, centered: bool) -> Self {
        Self {
            lr,
            alpha,
            eps,
            weight_decay,
            momentum,
            centered,
            param_nodes: HashSet::new(),
            square_averages: HashMap::new(),
            momentum_buffers: HashMap::new(),
            grad_averages: HashMap::new(),
        }
    }

    /// Create RMSprop with standard defaults
    pub fn with_defaults(lr: T) -> Self {
        Self::new(
            lr,
            T::from(0.99),  // alpha
            T::from(1e-8),  // eps
            T::from(0.0),   // weight_decay
            T::from(0.0),   // momentum
            false,          // centered
        )
    }

    /// Create RMSprop with momentum
    pub fn with_momentum(lr: T, momentum: T) -> Self {
        Self::new(
            lr,
            T::from(0.99),
            T::from(1e-8),
            T::from(0.0),
            momentum,
            false,
        )
    }

    pub fn get_lr(&self) -> T { self.lr }
    pub fn set_lr(&mut self, lr: T) { self.lr = lr; }
}

impl<T> Optimizer<T> for RMSprop<T>
where
    T: FerroxCudaF + From<f64>,
{
    fn step(&mut self, engine: &mut AutoFerroxEngine<T>) {
        let zero = <T as FerroxF>::zero();
        let one = <T as FerroxF>::one();

        for &param_node in &self.param_nodes {
            if let Some(grad) = engine.get_gradient(param_node) {
                let current_params = engine.get_tensor_data(param_node)
                    .expect("Parameter tensor not found").clone();

                // Apply weight decay if specified
                let effective_grad = if self.weight_decay != zero {
                    let weight_decay_term = current_params.mul_scalar(self.weight_decay)
                        .expect("Failed to compute weight decay term");
                    grad.add(&weight_decay_term)
                        .expect("Failed to add weight decay")
                } else {
                    grad.clone()
                };

                // Get or initialize square average buffer
                let square_avg = self.square_averages.entry(param_node)
                    .or_insert_with(|| {
                        let device = current_params.device();
                        Tensor::zeros_with_device(current_params.shape(), device)
                            .expect("Failed to create square average buffer")
                    });

                // Update square average: sq_avg = alpha * sq_avg + (1 - alpha) * grad^2
                let one_minus_alpha = one - self.alpha;
                let grad_squared = effective_grad.mul(&effective_grad)
                    .expect("Failed to compute gradient squared");
                let first_term = square_avg.mul_scalar(self.alpha)
                    .expect("Failed to scale square average");
                let second_term = grad_squared.mul_scalar(one_minus_alpha)
                    .expect("Failed to scale gradient squared");
                let new_square_avg = first_term.add(&second_term)
                    .expect("Failed to update square average");
                *square_avg = new_square_avg.clone();

                // Compute denominator
                let denominator = if self.centered {
                    // Centered variant: use variance instead of second moment
                    let grad_avg = self.grad_averages.entry(param_node)
                        .or_insert_with(|| {
                            let device = current_params.device();
                            Tensor::zeros_with_device(current_params.shape(), device)
                                .expect("Failed to create gradient average buffer")
                        });

                    // Update gradient average: g_avg = alpha * g_avg + (1 - alpha) * grad
                    let first_term = grad_avg.mul_scalar(self.alpha)
                        .expect("Failed to scale gradient average");
                    let second_term = effective_grad.mul_scalar(one_minus_alpha)
                        .expect("Failed to scale gradient for average");
                    let new_grad_avg = first_term.add(&second_term)
                        .expect("Failed to update gradient average");
                    *grad_avg = new_grad_avg.clone();

                    // Compute variance: sq_avg - (g_avg)^2
                    let grad_avg_squared = new_grad_avg.mul(&new_grad_avg)
                        .expect("Failed to compute gradient average squared");
                    let variance = new_square_avg.sub(&grad_avg_squared)
                        .expect("Failed to compute variance");
                    let sqrt_variance = variance.sqrt()
                        .expect("Failed to compute square root of variance");
                    sqrt_variance.add_scalar(self.eps)
                        .expect("Failed to add epsilon to variance")
                } else {
                    // Standard variant: sqrt(sq_avg) + eps
                    let sqrt_square_avg = new_square_avg.sqrt()
                        .expect("Failed to compute square root of square average");
                    sqrt_square_avg.add_scalar(self.eps)
                        .expect("Failed to add epsilon to square average")
                };

                // Compute update direction
                let update_direction = effective_grad.div(&denominator)
                    .expect("Failed to compute update direction");

                let final_update = if self.momentum != zero {
                    // Apply momentum
                    let momentum_buffer = self.momentum_buffers.entry(param_node)
                        .or_insert_with(|| {
                            let device = current_params.device();
                            Tensor::zeros_with_device(current_params.shape(), device)
                                .expect("Failed to create momentum buffer")
                        });

                    // Update momentum: buf = momentum * buf + update_direction
                    let momentum_term = momentum_buffer.mul_scalar(self.momentum)
                        .expect("Failed to scale momentum buffer");
                    let new_buffer = momentum_term.add(&update_direction)
                        .expect("Failed to update momentum buffer");
                    *momentum_buffer = new_buffer.clone();
                    new_buffer
                } else {
                    update_direction
                };

                // Apply learning rate and update parameters
                let scaled_update = final_update.mul_scalar(self.lr)
                    .expect("Failed to scale update by learning rate");
                let new_params = current_params.sub(&scaled_update)
                    .expect("Failed to update parameters");

                // Update parameter in the computation graph
                engine.update_parameter(param_node, new_params)
                    .expect("Failed to update parameter in graph");
            }
        }
    }

    fn reset_grad(&mut self, engine: &mut AutoFerroxEngine<T>) {
        for &param_node in &self.param_nodes {
            engine.clear_gradient(param_node);
        }
    }

    fn add_param(&mut self, _param_id: usize, param_node_id: NodeId) {
        self.param_nodes.insert(param_node_id);
    }
}
