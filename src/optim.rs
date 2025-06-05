use std::collections::HashMap;
use crate::tensor::Tensor;
/// Trait that all optimizers must implement
/// If we wanted to create a new optimizer, we would implement this trait
/// and provide the necessary methods for parameter updates.
/// This allows for flexibility in choosing different optimization algorithms
/// such as SGD, Adam, RMSprop, etc.
/// It also mimics PyTorch and TensorFlow's optimizer interfaces, which you can customize by inheriting 
/// from a base optimizer class.
pub trait Optimizer {
    /// Update parameters based on gradients
    fn step(&mut self);
    
    /// Reset gradients to zero/None
    fn reset_grad(&mut self);
    
    /// Add parameters to the optimizer
    fn add_param(&mut self, param_id: usize, param: &mut Parameter);
}

/// Represents a parameter in the neural network
#[derive(Debug, Clone)]
pub struct Parameter {
    pub data: Tensor,  // Using Tensor type for data representation
    pub grad: Option<Tensor>, // Gradient of the parameter, None if not computed
    //I use tensors to mimic the graph API in thsis crate which also uses Tensors to represent data or gradients.
    pub shape: Vec<usize>,
}

impl Parameter {
    pub fn new(data: Tensor, shape: Vec<usize>) -> Self {
        Self {
            data,
            grad: None,
            shape,
        }
    }
    
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }
    
    pub fn size(&self) -> usize {
        self.data.len()
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
pub struct SGD {
    params: HashMap<usize, *mut Parameter>,
    lr: f64,
    // Momentum factor
    /// Momentum helps accelerate SGD in the relevant direction and dampens oscillations.
    /// It is a technique to improve convergence speed by accumulating moving average of past gradients.
    momentum: f64,
    nesterov: bool, // whether to use Nesterov momentum.
    // Nesterov basically looks ahead by computing the gradient at the "looked ahead" position.
    /// weight decay (L2 regularization) factor
    weight_decay: f64,
    u: HashMap<usize, Vec<f64>>, // momentum buffers
}

impl SGD {
    pub fn new(lr: f64, momentum: f64, weight_decay: f64) -> Self {
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
    pub fn clip_grad_norm(&mut self, max_norm: f64) {
        let mut total_norm_sq = 0.0;
        
        // Calculate total gradient norm
        for (_, param_ptr) in &self.params {
            let param = unsafe { &**param_ptr };
            if let Some(ref grad) = param.grad {
                for &g in grad {
                    total_norm_sq += g * g;
                }
            }
        }
        
        let total_norm = total_norm_sq.sqrt();
        
        // If norm exceeds max_norm, scale all gradients
        if total_norm > max_norm {
            let clip_coef = max_norm / total_norm;
            
            for (_, param_ptr) in &self.params {
                let param = unsafe { &mut **param_ptr };
                if let Some(ref mut grad) = param.grad {
                    for g in grad.iter_mut() {
                        *g *= clip_coef;
                    }
                }
            }
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
      // Iterate over all parameters and update them
        // using the stored gradients and momentum buffers.
        for (&param_id, param_ptr) in &self.params {
            let param = unsafe { &mut **param_ptr };
            
            if let Some(ref grad) = param.grad {
                // Initialize momentum buffer if not present
                if !self.u.contains_key(&param_id) {
                    self.u.insert(param_id, vec![0.0; param.size()]);
                }
                
                /// TODO: implement Nesterov momentum (not done)
                let momentum_buffer = self.u.get_mut(&param_id).unwrap();
                
                for i in 0..param.size() {
                    let mut g = grad[i];
                    
                    // Apply weight decay (L2 regularization)
                    if self.weight_decay != 0.0 {
                        g += self.weight_decay * param.data[i];
                    }
                    
                    // Update momentum buffer
                    momentum_buffer[i] = self.momentum * momentum_buffer[i] + g;
                    
                    // Update parameter
                    param.data[i] -= self.lr * momentum_buffer[i];
                }
            }
        }
    }
    
    fn reset_grad(&mut self) {
        for (_, param_ptr) in &self.params {
            let param = unsafe { &mut **param_ptr };
            param.zero_grad();
        }
    }
    
    fn add_param(&mut self, param_id: usize, param: &mut Parameter) {
        self.params.insert(param_id, param as *mut Parameter);
    }
}

/// Adam optimizer - adaptive learning rate optimization algorithm
pub struct Adam {
    params: HashMap<usize, *mut Parameter>,
    lr: f64,
    // momentum parameters
    beta1: f64, 
    beta2: f64,
    eps: f64, // small constant to prevent division by zero
    // nesterov: bool,  not used on Adam
    weight_decay: f64, // weight decay (L2 regularization) factor
    t: u32, // time step
    m: HashMap<usize, Vec<f64>>, // first moment estimates
    v: HashMap<usize, Vec<f64>>, // second moment estimates
}

impl Adam {
    pub fn new(lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
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
impl Optimizer for Adam {
  // Adam does not use Nesterov momentum, so we don't need to implement that here.
    fn step(&mut self) {
        self.t += 1;
        
        for (&param_id, param_ptr) in &self.params {
            let param = unsafe { &mut **param_ptr };
            
            if let Some(ref grad) = param.grad {
                // Initialize moment estimates if not present
                if !self.m.contains_key(&param_id) {
                    self.m.insert(param_id, vec![0.0; param.size()]);
                    self.v.insert(param_id, vec![0.0; param.size()]);
                }
                
                let m_t = self.m.get_mut(&param_id).unwrap();
                let v_t = self.v.get_mut(&param_id).unwrap();
                
                for i in 0..param.size() {
                    let mut g = grad[i];
                    
                    // Apply weight decay (L2 regularization)
                    if self.weight_decay != 0.0 {
                        g += self.weight_decay * param.data[i];
                    }
                    
                    // Update biased first moment estimate
                    m_t[i] = self.beta1 * m_t[i] + (1.0 - self.beta1) * g;
                    
                    // Update biased second raw moment estimate
                    v_t[i] = self.beta2 * v_t[i] + (1.0 - self.beta2) * g * g;
                    
                    // Compute bias-corrected first moment estimate
                    let m_hat = m_t[i] / (1.0 - self.beta1.powi(self.t as i32));
                    
                    // Compute bias-corrected second raw moment estimate
                    let v_hat = v_t[i] / (1.0 - self.beta2.powi(self.t as i32));
                    
                    // Update parameter
                    param.data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
                }
            }
        }
    }
    
    fn reset_grad(&mut self) {
        for (_, param_ptr) in &self.params {
            let param = unsafe { &mut **param_ptr };
            param.zero_grad();
        }
    }
    
    fn add_param(&mut self, param_id: usize, param: &mut Parameter) {
        self.params.insert(param_id, param as *mut Parameter);
    }
}
