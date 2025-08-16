use crate::backend::Device;
use crate::backend::Tensor;
use crate::backend::{FerroxCudaF, FerroxF};
use crate::graph::AutoFerroxEngine;
use crate::graph::NodeId;
use crate::nn::optim::{Optimizer, OptimizerError, OptimizerStateDict, ParameterGroup};
use crate::FerroxN;
use std::cell::RefCell;
use std::collections::HashMap;
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
                    let grad_squared = grad.mul(grad).map_err(OptimizerError::TensorOperation)?;

                    let norm_contrib = grad_squared
                        .sum(None, false)
                        .map_err(OptimizerError::TensorOperation)?;

                    // Extract scalar from tensor
                    let scalar = norm_contrib
                        .first()
                        .map_err(OptimizerError::TensorOperation)?;
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
                        .map_err(OptimizerError::TensorOperation)?;
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
        grad: &Tensor<T>, // Cambiar: ahora recibe el gradiente
        params: &Tensor<T>,
    ) -> Result<Tensor<T>, OptimizerError> {
        if decay_rate == <T as FerroxF>::zero() {
            return Ok(grad.clone());
        }

        // L2 weight decay: effective_grad = grad + weight_decay * params
        let weight_decay_term = params
            .mul_scalar(decay_rate)
            .map_err(OptimizerError::TensorOperation)?;

        grad.add(&weight_decay_term)
            .map_err(OptimizerError::TensorOperation)
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
            .map_err(OptimizerError::TensorOperation)?;

        params
            .sub(&scaled_update)
            .map_err(OptimizerError::TensorOperation)
    }

    fn update_momentum_buffer(
        &self,
        param_id: NodeId,
        device: Device,
        shape: &[usize],
        momentum: T,
        effective_grad: &Tensor<T>, // Cambiar: recibe gradiente efectivo
    ) -> Result<(), OptimizerError> {
        let mut buffers = self.momentum_buffers.borrow_mut();
        let momentum_buffer = buffers.entry(param_id).or_insert_with(|| {
            Tensor::zeros_with_device(shape, device)
                .expect("Failed to create device-aware momentum buffer")
        });
        *momentum_buffer = momentum_buffer
            .mul_scalar(momentum)
            .map_err(OptimizerError::TensorOperation)?
            .add(effective_grad) // Cambiar: suma el gradiente efectivo
            .map_err(OptimizerError::TensorOperation)?;
        Ok(())
    }
    fn compute_update(
        &self,
        momentum: T,
        lr: T,
        param_id: NodeId,
        params: &Tensor<T>,
        effective_grad: &Tensor<T>, // Nuevo parámetro
    ) -> Result<Tensor<T>, OptimizerError> {
        if momentum == <T as FerroxF>::zero() {
            return self.apply_lr(lr, params, effective_grad); // Usar gradiente efectivo
        }

        let device = params.device();
        let shape = params.shape();

        self.update_momentum_buffer(param_id, device, shape, momentum, effective_grad)?;

        if self.nesterov {
            let buffers = self.momentum_buffers.borrow();
            let buffer = buffers.get(&param_id).expect("Parameter not found!");
            let update = buffer
                .mul_scalar(momentum)
                .map_err(OptimizerError::TensorOperation)?
                .add(effective_grad)
                .map_err(OptimizerError::TensorOperation)?;
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
                    let effective_grad =
                        self.apply_weight_decay(group_weight_decay, grad, params)?;

                    let updated_params = if group_momentum != <T as FerroxF>::zero() {
                        self.compute_update(
                            group_momentum,
                            group_lr,
                            param_node,
                            params,
                            &effective_grad,
                        )?
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

    fn add_param(&mut self, param_group: usize, param_node_id: NodeId) {
        self.param_groups[param_group].params.insert(param_node_id);
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

    fn state_dict(&self) -> OptimizerStateDict
    where
        T: FerroxCudaF,
    {
        let mut state_dict = OptimizerStateDict::new("SGD".to_string());

        // SGD doesn't track step count like Adam, but we set it to 0 for consistency
        state_dict.step_count = 0;

        // Save hyperparameters as f64 for serialization
        state_dict
            .hyperparameters
            .insert("lr".to_string(), <T as FerroxN>::to_f64(self.lr));
        state_dict.hyperparameters.insert(
            "momentum".to_string(),
            <T as FerroxN>::to_f64(self.momentum),
        );
        state_dict.hyperparameters.insert(
            "weight_decay".to_string(),
            <T as FerroxN>::to_f64(self.weight_decay),
        );
        state_dict.hyperparameters.insert(
            "nesterov".to_string(),
            if self.nesterov { 1.0 } else { 0.0 },
        );

        // Save all parameter groups with their settings
        for group in &self.param_groups {
            state_dict.add_parameter_group(group);
        }

        // Save momentum buffers for all parameters that have them
        let momentum_buffers = self.momentum_buffers.borrow();
        for (&node_id, tensor) in momentum_buffers.iter() {
            if state_dict
                .add_buffer(node_id, tensor, "momentum".to_string())
                .is_err()
            {
                // Skip parameters that cannot be serialized
                continue;
            }
        }

        state_dict
    }

    fn load_state_dict(&mut self, state_dict: &OptimizerStateDict) -> Result<(), OptimizerError>
    where
        T: FerroxCudaF,
    {
        // Verify this is the correct optimizer type
        if state_dict.optimizer_type != "SGD" {
            return Err(OptimizerError::TypeMismatch(format!(
                "Expected SGD state dict, got {}",
                state_dict.optimizer_type
            )));
        }

        // Clear existing momentum buffers
        self.momentum_buffers.borrow_mut().clear();

        // Restore hyperparameters if they exist
        if let Some(&lr) = state_dict.hyperparameters.get("lr") {
            if let Some(lr_val) = <T as FerroxN>::from_f64(lr) {
                self.lr = lr_val;
            }
        }
        if let Some(&momentum) = state_dict.hyperparameters.get("momentum") {
            if let Some(momentum_val) = <T as FerroxN>::from_f64(momentum) {
                self.momentum = momentum_val;
            }
        }
        if let Some(&weight_decay) = state_dict.hyperparameters.get("weight_decay") {
            if let Some(wd_val) = <T as FerroxN>::from_f64(weight_decay) {
                self.weight_decay = wd_val;
            }
        }
        if let Some(&nesterov) = state_dict.hyperparameters.get("nesterov") {
            self.nesterov = nesterov > 0.5;
        }

        // Restore parameter groups with their custom settings
        self.param_groups.clear();
        for serializable_group in &state_dict.parameter_groups {
            let group = ParameterGroup::from_serializable(serializable_group)?;
            self.param_groups.push(group);
        }

        // Restore momentum buffers for parameters that had them
        for buffer in state_dict.parameter_buffers.values() {
            if buffer.buffer_type == "momentum" {
                let node_id = NodeId(buffer.node_id);
                let restored_tensor = buffer.to_tensor::<T>()?;
                self.momentum_buffers
                    .borrow_mut()
                    .insert(node_id, restored_tensor);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod sgd_tests {

    use crate::backend::manager::best_f32_device;
    use crate::backend::Tensor;
    use crate::graph::AutoFerroxEngine;
    use crate::nn::optim::{sgd::SGD, Optimizer};

    #[test]
    fn test_sgd_basic_step() {
        let mut engine = AutoFerroxEngine::<f32>::new(true);
        let device = best_f32_device();
        let param = Tensor::from_vec_with_device(vec![2.0f32], &[1], device).unwrap();
        let node = engine.create_variable(param, true);

        let mut optimizer = SGD::with_defaults(0.1f32);
        optimizer.add_param(0, node);

        let grad = Tensor::from_vec_with_device(vec![1.0f32], &[1], device).unwrap();
        engine.set_gradient(node, grad);

        let _before = engine
            .get_tensor_data(node)
            .unwrap()
            .clone()
            .first()
            .unwrap();
        optimizer.step(&mut engine).unwrap();
        let after = engine
            .get_tensor_data(node)
            .unwrap()
            .clone()
            .first()
            .unwrap();

        // Expected: param - lr * grad = 2.0 - 0.1 * 1.0 = 1.9
        assert!((after - 1.9).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_momentum() {
        let mut engine = AutoFerroxEngine::<f32>::new(true);
        let device = best_f32_device();
        let param = Tensor::from_vec_with_device(vec![1.0f32], &[1], device).unwrap();
        let node = engine.create_variable(param, true);

        let mut optimizer = SGD::with_momentum(0.1f32, 0.9f32);
        optimizer.add_param(0, node);

        // First step - no momentum yet
        let grad = Tensor::from_vec_with_device(vec![1.0f32], &[1], device).unwrap();
        engine.set_gradient(node, grad.clone());

        optimizer.step(&mut engine).unwrap();
        optimizer.reset_grad(&mut engine);

        // Second step - momentum kicks in
        engine.set_gradient(node, grad);

        let before = engine
            .get_tensor_data(node)
            .unwrap()
            .clone()
            .first()
            .unwrap();
        optimizer.step(&mut engine).unwrap();
        let after = engine
            .get_tensor_data(node)
            .unwrap()
            .clone()
            .first()
            .unwrap();

        let step_size = (before - after).abs();

        // Second step should be larger than lr due to momentum
        assert!(step_size > 0.1);
    }

    #[test]
    fn test_sgd_gradient_clipping() {
        let mut engine = AutoFerroxEngine::<f32>::new(true);
        let device = best_f32_device();

        let param = Tensor::from_vec_with_device(vec![1.0f32], &[1], device).unwrap();
        let node = engine.create_variable(param, true);

        let mut optimizer = SGD::with_defaults(0.1f32);
        optimizer.add_param(0, node);

        // Large gradient
        let grad = Tensor::from_vec_with_device(vec![10.0f32], &[1], device).unwrap();
        engine.set_gradient(node, grad);

        let norm = optimizer.clip_grad_norm(&mut engine, 1.0f32).unwrap();

        // Original norm should be 10.0
        assert!((norm - 10.0).abs() < 1e-6);

        // Clipped gradient should have norm 1.0
        let clipped_grad = engine.get_gradient(node).unwrap().clone();
        let clipped_val = clipped_grad.first().unwrap();
        assert!((clipped_val - 1.0).abs() < 1e-6);
    }
}
