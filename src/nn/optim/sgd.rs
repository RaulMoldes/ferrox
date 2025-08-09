use crate::backend::Device;
use crate::backend::Tensor;
use crate::backend::{FerroxCudaF, FerroxF};
use crate::graph::AutoFerroxEngine;
use crate::graph::Node;
use crate::graph::NodeId;
use crate::nn::optim::{Optimizer, OptimizerError, ParameterGroup, OptimizerStateDict};
use crate::nn::Module;
use std::collections::{HashMap, HashSet};
use std::cell::RefCell;

/// Stochastic Gradient Descent optimizer with momentum and weight decay
/// Uses proper L2 weight decay (not AdamW-style decoupled weight decay)
pub struct SGD<T>
where
    T: FerroxCudaF,
{
    lr: T,
    momentum: T,
    weight_decay: T,
    nesterov: bool,
    param_groups: Vec<ParameterGroup<T>>,
    momentum_buffers: RefCell<HashMap<NodeId, Tensor<T>>>,
}

impl<T> SGD<T>
where
    T: FerroxCudaF,
{
    pub fn new(lr: T, momentum: T, weight_decay: T, nesterov: bool) -> Self {
        let mut default_group = ParameterGroup::new("default".to_string());
        default_group.lr = Some(lr);
        default_group.weight_decay = Some(weight_decay);
        default_group.momentum = Some(momentum);

        Self {
            lr,
            momentum,
            weight_decay,
            nesterov,
            param_groups: vec![default_group],
            momentum_buffers: RefCell::new(HashMap::new()),
        }
    }

    pub fn with_defaults(lr: T) -> Self {
        Self::new(lr, <T as FerroxF>::zero(), <T as FerroxF>::zero(), false)
    }

    pub fn with_momentum(lr: T, momentum: T) -> Self {
        Self::new(lr, momentum, <T as FerroxF>::zero(), false)
    }

    pub fn set_nesterov(&mut self, nesterov: bool) {
        self.nesterov = nesterov;
    }

    fn gradient_norm(&self, engine: &mut AutoFerroxEngine<T>) -> Result<T, OptimizerError> {
        let mut total_norm_sq = <T as FerroxF>::zero();
        for group in &self.param_groups {
            for &param_node in &group.params {
                if let Some(grad) = engine.get_gradient(param_node) {
                    let grad_squared = grad
                        .mul(grad)
                        .map_err(|e| OptimizerError::TensorOperation(e))?;

                    let norm_contrib = grad_squared
                        .sum(None)
                        .map_err(|e| OptimizerError::TensorOperation(e))?;

                    // Extract scalar from tensor
                    let scalar = norm_contrib
                        .first()
                        .map_err(|e| OptimizerError::TensorOperation(e))?;
                    total_norm_sq += scalar;
                }
            }
        }
        let total_norm = total_norm_sq.sqrt();
        Ok(total_norm)
    }

    /// Gradient clipping by global norm to prevent exploding gradients
    pub fn clip_grad_norm(
        &mut self,
        engine: &mut AutoFerroxEngine<T>,
        max_norm: T,
    ) -> Result<T, OptimizerError> {
        // Calculate total gradient norm across all parameter groups
        let total_norm = self.gradient_norm(engine)?;
        // Apply clipping if norm exceeds threshold
        if total_norm > max_norm {
            let clip_coef = max_norm / total_norm;

            for group in &self.param_groups {
                for &param_node in &group.params {
                    engine
                        .clip_gradient(param_node, clip_coef)
                        .map_err(|e| OptimizerError::TensorOperation(e))?;
                }
            }
        }

        Ok(total_norm)
    }

    /// Helper to get effective parameter values for a group
    fn get_group_lr(&self, group: &ParameterGroup<T>) -> T {
        group.lr.unwrap_or(self.lr)
    }

    fn get_group_weight_decay(&self, group: &ParameterGroup<T>) -> T {
        group.weight_decay.unwrap_or(self.weight_decay)
    }

    fn get_group_momentum(&self, group: &ParameterGroup<T>) -> T {
        group.momentum.unwrap_or(self.momentum)
    }

    // PRIVATE METHODS FOR READABILITY
    fn apply_weight_decay(
    &self,
    decay_rate: T,
    grad: &Tensor<T>,  // Cambiar: ahora recibe el gradiente
    params: &Tensor<T>,
) -> Result<Tensor<T>, OptimizerError> {
    if decay_rate == <T as FerroxF>::zero() {
        return Ok(grad.clone());
    }

    // L2 weight decay: effective_grad = grad + weight_decay * params
    let weight_decay_term = params
        .mul_scalar(decay_rate)
        .map_err(|e| OptimizerError::TensorOperation(e))?;

    grad.add(&weight_decay_term)
        .map_err(|e| OptimizerError::TensorOperation(e))
}

    fn apply_lr(
        &self,
        lr: T,
        params: &Tensor<T>,
        update: &Tensor<T>,
    ) -> Result<Tensor<T>, OptimizerError> {
        // Apply learning rate and update parameters
        let scaled_update = update
            .mul_scalar(lr)
            .map_err(|e| OptimizerError::TensorOperation(e))?;

        params
            .sub(&scaled_update)
            .map_err(|e| OptimizerError::TensorOperation(e))
    }

    fn update_momentum_buffer(
    &self,
    param_id: NodeId,
    device: Device,
    shape: &[usize],
    momentum: T,
    effective_grad: &Tensor<T>  // Cambiar: recibe gradiente efectivo
) -> Result<(), OptimizerError> {

    let mut buffers = self.momentum_buffers.borrow_mut();
    let momentum_buffer = buffers.entry(param_id).or_insert_with(|| {
        Tensor::zeros_with_device(shape, device)
            .expect("Failed to create device-aware momentum buffer")
    });
    *momentum_buffer = momentum_buffer
        .mul_scalar(momentum)
        .map_err(|e| OptimizerError::TensorOperation(e))?
        .add(effective_grad)  // Cambiar: suma el gradiente efectivo
        .map_err(|e| OptimizerError::TensorOperation(e))?;
    Ok(())
}
fn compute_update(
    &self,
    momentum: T,
    lr: T,
    param_id: NodeId,
    params: &Tensor<T>,
    effective_grad: &Tensor<T>,  // Nuevo parámetro
) -> Result<Tensor<T>, OptimizerError> {
    if momentum == <T as FerroxF>::zero() {
        return self.apply_lr(lr, params, effective_grad);  // Usar gradiente efectivo
    }

    let device = params.device();
    let shape = params.shape();

    self.update_momentum_buffer(param_id, device, shape, momentum, effective_grad)?;

    if self.nesterov {
        let buffers = self.momentum_buffers.borrow();
        let buffer = buffers.get(&param_id).expect("Parameter not found!");
        let update = buffer
            .mul_scalar(momentum)
            .map_err(|e| OptimizerError::TensorOperation(e))?
            .add(effective_grad)
            .map_err(|e| OptimizerError::TensorOperation(e))?;
        return self.apply_lr(lr, params, &update);
    }

    let buffers = self.momentum_buffers.borrow();
    let momentum_buffer = buffers.get(&param_id).expect("Buffer should exist!");
    self.apply_lr(lr, params, momentum_buffer)
}
}

impl<T> Optimizer<T> for SGD<T>
where
    T: FerroxCudaF,
{
   fn step(&mut self, engine: &mut AutoFerroxEngine<T>) -> Result<(), OptimizerError> {

    for group in &self.param_groups {
        let group_lr = self.get_group_lr(group);
        let group_weight_decay = self.get_group_weight_decay(group);
        let group_momentum = self.get_group_momentum(group);

        for &param_node in &group.params {
            if let Some(grad) = engine.get_gradient(param_node) {
                let params = engine
                    .get_tensor_data(param_node)
                    .ok_or(OptimizerError::ParameterNotFound(param_node))?;

                // Calcular gradiente efectivo con weight decay
                let effective_grad = self.apply_weight_decay(group_weight_decay, grad, params)?;

                let updated_params = if group_momentum != FerroxF::zero() {
                    self.compute_update(group_momentum, group_lr, param_node, params, &effective_grad)?
                } else {
                    self.apply_lr(group_lr, params, &effective_grad)?
                };

                engine
                    .update_parameter(param_node, updated_params)
                    .map_err(|_| OptimizerError::ParameterNotFound(param_node))?;
            }
        }
    }
    Ok(())
}

    fn reset_grad(&mut self, engine: &mut AutoFerroxEngine<T>) {
        for group in &self.param_groups {
            for &param_node in &group.params {
                engine.clear_gradient(param_node);
            }
        }
    }

    fn add_param(&mut self, _param_id: usize, param_node_id: NodeId) {
        self.param_groups[0].params.insert(param_node_id);
    }

    fn add_param_group(&mut self, group: ParameterGroup<T>) -> Result<(), OptimizerError> {
        if group.params.is_empty() {
            return Err(OptimizerError::InvalidParameterGroup);
        }
        self.param_groups.push(group);
        Ok(())
    }

    fn get_lr(&self) -> T {
        self.lr
    }

    fn set_lr(&mut self, lr: T) {
        self.lr = lr;
        for group in &mut self.param_groups {
            group.lr = Some(lr);
        }
    }

    /// TODO : IMPLEMENT PROPER STATE DICT LOADING SUPPORTING PARAMETER GROUPS.
    fn state_dict(&self) -> OptimizerStateDict
    where
        T: FerroxCudaF,
    {
        unimplemented!("Not implemented");
    }

    fn load_state_dict(&mut self, state_dict: &OptimizerStateDict) -> Result<(), OptimizerError>
    where
        T: FerroxCudaF,
    {
        unimplemented!("Not implemented");
    }
}
