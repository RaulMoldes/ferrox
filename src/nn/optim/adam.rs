use crate::backend::Device;
use crate::backend::Tensor;
use crate::backend::{FerroxCudaF, FerroxF};
use crate::graph::AutoFerroxEngine;
use crate::graph::NodeId;
use crate::nn::optim::{Optimizer, OptimizerError, OptimizerStateDict, ParameterGroup};
use std::cell::RefCell;
use std::collections::HashMap;

// Adam optimizer with proper AdamW-style decoupled weight decay
/// Implements bias correction and AMSGrad variant
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
    param_groups: Vec<ParameterGroup<T>>,
    first_moments: RefCell<HashMap<NodeId, Tensor<T>>>,
    second_moments: RefCell<HashMap<NodeId, Tensor<T>>>,
    max_second_moments: RefCell<HashMap<NodeId, Tensor<T>>>,
    step_count: u64,
}

enum MomentBuffer {
    First,
    Second,
    Max,
}

impl<T> Adam<T>
where
    T: FerroxCudaF,
{
    pub fn new(lr: T, beta1: T, beta2: T, eps: T, weight_decay: T, amsgrad: bool) -> Self {
        let mut default_group = ParameterGroup::new("default".to_string());
        default_group.lr = Some(lr);
        default_group.weight_decay = Some(weight_decay);

        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            amsgrad,
            param_groups: vec![default_group],
            first_moments: RefCell::new(HashMap::new()),
            second_moments: RefCell::new(HashMap::new()),
            max_second_moments: RefCell::new(HashMap::new()),
            step_count: 0,
        }
    }

    pub fn with_defaults(lr: T) -> Self {
        Self::new(
            lr,
            <T as FerroxF>::from_f64(0.9).unwrap(),
            <T as FerroxF>::from_f64(0.999).unwrap(),
            <T as FerroxF>::from_f64(1e-8).unwrap(),
            <T as FerroxF>::zero(),
            false,
        )
    }

    /// AdamW variant with decoupled weight decay
    pub fn adamw(lr: T, weight_decay: T) -> Self {
        Self::new(
            lr,
            <T as FerroxF>::from_f64(0.9).unwrap(),
            <T as FerroxF>::from_f64(0.999).unwrap(),
            <T as FerroxF>::from_f64(1e-8).unwrap(),
            weight_decay,
            false,
        )
    }

    pub fn amsgrad(lr: T) -> Self {
        Self::new(
            lr,
            <T as FerroxF>::from_f64(0.9).unwrap(),
            <T as FerroxF>::from_f64(0.999).unwrap(),
            <T as FerroxF>::from_f64(1e-8).unwrap(),
            <T as FerroxF>::zero(),
            true,
        )
    }

    pub fn get_step_count(&self) -> u64 {
        self.step_count
    }

    /// Reset optimizer state for fresh training
    pub fn reset_state(&mut self) {
        self.first_moments.borrow_mut().clear();
        self.second_moments.borrow_mut().clear();
        self.max_second_moments.borrow_mut().clear();
        self.step_count = 0;
    }

    /// Compute bias correction factors for current step
    fn compute_bias_corrections(&self) -> (T, T) {
        let step_f64 = <T as FerroxF>::from_f64(self.step_count as f64).expect("Failed to cast step count to float");
        let beta1_power = self.beta1.powf(step_f64);
        let beta2_power = self.beta2.powf(step_f64);

        (<T as FerroxF>::one()  - beta1_power, <T as FerroxF>::one() - beta2_power)
    }

    fn get_group_lr(&self, group: &ParameterGroup<T>) -> T {
        group.lr.unwrap_or(self.lr)
    }

    fn get_group_weight_decay(&self, group: &ParameterGroup<T>) -> T {
        group.weight_decay.unwrap_or(self.weight_decay)
    }

    fn update_moment_buffer(
        &self,
        grad: &Tensor<T>,
        param_id: NodeId,
        moment_type: MomentBuffer,
        device: Device,
        shape: &[usize],
    ) -> Result<(), OptimizerError> {
        match moment_type {
            MomentBuffer::First => {
                let mut first = self.first_moments.borrow_mut();
                let buffer = first.entry(param_id).or_insert_with(|| {
                    Tensor::zeros_with_device(shape, device)
                        .expect("Failed to create device-aware first moment buffer")
                });

                // Update first moment: m = beta1 * m + (1 - beta1) * grad
                let one_minus_beta1 = <T as FerroxF>::one() - self.beta1;
                let first_term = buffer
                    .mul_scalar(self.beta1)
                    .map_err(OptimizerError::TensorOperation)?;
                let second_term = grad
                    .mul_scalar(one_minus_beta1)
                    .map_err(OptimizerError::TensorOperation)?;
                *buffer = first_term
                    .add(&second_term)
                    .map_err(OptimizerError::TensorOperation)?;
            }
            MomentBuffer::Second => {
                let mut second = self.second_moments.borrow_mut();
                let buffer = second.entry(param_id).or_insert_with(|| {
                    Tensor::zeros_with_device(shape, device)
                        .expect("Failed to create device-aware first moment buffer")
                });
                // Update second moment: v = beta2 * v + (1 - beta2) * grad^2
                let one_minus_beta2 = <T as FerroxF>::one() - self.beta2;
                let grad_squared = grad
                    .mul(grad)
                    .map_err(OptimizerError::TensorOperation)?;
                let first_term = buffer
                    .mul_scalar(self.beta2)
                    .map_err(OptimizerError::TensorOperation)?;
                let second_term = grad_squared
                    .mul_scalar(one_minus_beta2)
                    .map_err(OptimizerError::TensorOperation)?;
                *buffer = first_term
                    .add(&second_term)
                    .map_err(OptimizerError::TensorOperation)?;
            }
            MomentBuffer::Max => {
                let mut max = self.max_second_moments.borrow_mut();
                let buffer = max.entry(param_id).or_insert_with(|| {
                    Tensor::zeros_with_device(shape, device)
                        .expect("Failed to create device-aware first moment buffer")
                });
                let seconds = self.second_moments.borrow();
                let second_moment = seconds
                    .get(&param_id)
                    .expect("Second moment should exist !");
                *buffer = buffer
                    .max(second_moment)
                    .map_err(OptimizerError::TensorOperation)?;
            }
        };

        Ok(())
    }

    fn take_current_buffer(&self, param_id: NodeId, moment_type: MomentBuffer) -> Tensor<T> {
        let buffer = match moment_type {
            MomentBuffer::First => {
                let first = self.first_moments.borrow_mut();
                first.get(&param_id).expect("Buffer not found").clone()
            }
            MomentBuffer::Second => {
                let second = self.second_moments.borrow();
                second.get(&param_id).expect("Buffer not found").clone()
            }
            MomentBuffer::Max => {
                let max = self.max_second_moments.borrow();
                max.get(&param_id).expect("Buffer not found").clone()
            }
        };
        buffer
    }

    fn compute_update(
        &self,
        param_id: NodeId,
        second_moment: &Tensor<T>,
        bias_correction1: T,
        bias_correction2: T,
        lr: T,
    ) -> Result<Tensor<T>, OptimizerError> {
        // Compute update: lr * m_hat / (sqrt(v_hat) + eps)
        let denominator = second_moment
            .div_scalar(bias_correction2)
            .map_err(OptimizerError::TensorOperation)?
            .sqrt()
            .map_err(OptimizerError::TensorOperation)?
            .add_scalar(self.eps)
            .map_err(OptimizerError::TensorOperation)?;

        let scaled_update = self
            .take_current_buffer(param_id, MomentBuffer::First)
            .div_scalar(bias_correction1)
            .map_err(OptimizerError::TensorOperation)?
            .div(&denominator)
            .map_err(OptimizerError::TensorOperation)?
            .mul_scalar(lr)
            .map_err(OptimizerError::TensorOperation)?;
        Ok(scaled_update)
    }

    fn apply_weight_decay(
        &self,
        weight_decay: T,
        lr: T,
        params: &Tensor<T>,
        update: &Tensor<T>,
    ) -> Result<Tensor<T>, OptimizerError> {
        let new_params = if weight_decay != <T as FerroxF>::zero() {
            let weight_decay_term = params
                .mul_scalar(weight_decay * lr)
                .map_err(OptimizerError::TensorOperation)?;
            params
                .sub(update)
                .map_err(OptimizerError::TensorOperation)?
                .sub(&weight_decay_term)
                .map_err(OptimizerError::TensorOperation)?
        } else {
            params
                .sub(update)
                .map_err(OptimizerError::TensorOperation)?
        };
        Ok(new_params)
    }
}
impl<T> Optimizer<T> for Adam<T>
where
    T: FerroxCudaF,
{
    fn step(&mut self, engine: &mut AutoFerroxEngine<T>) -> Result<(), OptimizerError> {
        self.step_count += 1;
        let (bias_correction1, bias_correction2) = self.compute_bias_corrections();

        for group in &self.param_groups {
            let group_lr = self.get_group_lr(group);
            let group_weight_decay = self.get_group_weight_decay(group);

            for &param_node in &group.params {
                if let Some(grad) = engine.get_gradient(param_node) {
                    let params = engine
                        .get_tensor_data(param_node)
                        .ok_or(OptimizerError::ParameterNotFound(param_node))?
                        .clone();
                    let device = params.device();
                    let shape = params.shape();
                    // Initialize moment buffers with proper device placement
                    self.update_moment_buffer(
                        grad,
                        param_node,
                        MomentBuffer::First,
                        device,
                        shape,
                    )?;

                    self.update_moment_buffer(
                        grad,
                        param_node,
                        MomentBuffer::Second,
                        device,
                        shape,
                    )?;

                    // AMSGrad: maintain maximum of second moments
                    let moment_for_update = if self.amsgrad {
                        self.update_moment_buffer(
                            grad,
                            param_node,
                            MomentBuffer::Max,
                            device,
                            shape,
                        )?;

                        self.take_current_buffer(param_node, MomentBuffer::Max)
                    } else {
                        self.take_current_buffer(param_node, MomentBuffer::Second)
                    };

                    // Compute update: lr * m_hat / (sqrt(v_hat) + eps)
                    let update = self.compute_update(
                        param_node,
                        &moment_for_update,
                        bias_correction1,
                        bias_correction2,
                        group_lr,
                    )?;

                    // Apply AdamW-style decoupled weight decay directly to parameters
                    let new_params =
                        self.apply_weight_decay(group_weight_decay, group_lr, &params, &update)?;

                    engine
                        .update_parameter(param_node, new_params)
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
        let mut state_dict = OptimizerStateDict::new("Adam".to_string());

        // Save step count for bias correction
        state_dict.step_count = self.step_count;

        // Save hyperparameters as f64 for serialization
        state_dict
            .hyperparameters
            .insert("lr".to_string(), FerroxF::to_f64(self.lr));
        state_dict
            .hyperparameters
            .insert("beta1".to_string(), FerroxF::to_f64(self.beta1));
        state_dict
            .hyperparameters
            .insert("beta2".to_string(), FerroxF::to_f64(self.beta2));
        state_dict
            .hyperparameters
            .insert("eps".to_string(), FerroxF::to_f64(self.eps));
        state_dict.hyperparameters.insert(
            "weight_decay".to_string(),
            FerroxF::to_f64(self.weight_decay),
        );
        state_dict
            .hyperparameters
            .insert("amsgrad".to_string(), if self.amsgrad { 1.0 } else { 0.0 });

        // Save all parameter groups with their settings
        for group in &self.param_groups {
            state_dict.add_parameter_group(group);
        }

        // Save first moment buffers for all parameters
        let first_moments = self.first_moments.borrow();
        for (&node_id, tensor) in first_moments.iter() {
            if state_dict.add_buffer(node_id, tensor, "first_moment".to_string()).is_err() {
                // Skip parameters that cannot be serialized
                continue;
            }
        }

        // Save second moment buffers for all parameters
        let second_moments = self.second_moments.borrow();
        for (&node_id, tensor) in second_moments.iter() {
            if state_dict.add_buffer(node_id, tensor, "second_moment".to_string()).is_err() {
                continue;
            }
        }

        // Save max second moment buffers if using AMSGrad
        if self.amsgrad {
            let max_second_moments = self.max_second_moments.borrow();
            for (&node_id, tensor) in max_second_moments.iter() {
                if state_dict.add_buffer(node_id, tensor, "max_second_moment".to_string()).is_err()
                {
                    continue;
                }
            }
        }

        state_dict
    }

    fn load_state_dict(&mut self, state_dict: &OptimizerStateDict) -> Result<(), OptimizerError>
    where
        T: FerroxCudaF,
    {
        // Verify this is the correct optimizer type
        if state_dict.optimizer_type != "Adam" {
            return Err(OptimizerError::TypeMismatch(format!(
                "Expected Adam state dict, got {}",
                state_dict.optimizer_type
            )));
        }

        // Clear existing state
        self.first_moments.borrow_mut().clear();
        self.second_moments.borrow_mut().clear();
        self.max_second_moments.borrow_mut().clear();

        // Restore step count for proper bias correction
        self.step_count = state_dict.step_count;

        // Restore hyperparameters if they exist
        if let Some(&lr) = state_dict.hyperparameters.get("lr") {
            if let Some(lr_val) = FerroxF::from_f64(lr) {
                self.lr = lr_val;
            }
        }
        if let Some(&beta1) = state_dict.hyperparameters.get("beta1") {
            if let Some(beta1_val) = FerroxF::from_f64(beta1) {
                self.beta1 = beta1_val;
            }
        }
        if let Some(&beta2) = state_dict.hyperparameters.get("beta2") {
            if let Some(beta2_val) = FerroxF::from_f64(beta2) {
                self.beta2 = beta2_val;
            }
        }
        if let Some(&eps) = state_dict.hyperparameters.get("eps") {
            if let Some(eps_val) = FerroxF::from_f64(eps) {
                self.eps = eps_val;
            }
        }
        if let Some(&weight_decay) = state_dict.hyperparameters.get("weight_decay") {
            if let Some(wd_val) = FerroxF::from_f64(weight_decay) {
                self.weight_decay = wd_val;
            }
        }
        if let Some(&amsgrad) = state_dict.hyperparameters.get("amsgrad") {
            self.amsgrad = amsgrad > 0.5;
        }

        // Restore parameter groups with their custom settings
        self.param_groups.clear();
        for serializable_group in &state_dict.parameter_groups {
            let group = ParameterGroup::from_serializable(serializable_group)?;
            self.param_groups.push(group);
        }



        // Restore first moment buffers
        for buffer in state_dict.parameter_buffers.values() {
            if buffer.buffer_type == "first_moment" {
                let node_id = NodeId(buffer.node_id);
                let restored_tensor = buffer.to_tensor::<T>()?;
                self.first_moments
                    .borrow_mut()
                    .insert(node_id, restored_tensor);
            }
        }

        // Restore second moment buffers
        for buffer in state_dict.parameter_buffers.values(){
            if buffer.buffer_type == "second_moment" {
                let node_id = NodeId(buffer.node_id);
                let restored_tensor = buffer.to_tensor::<T>()?;
                self.second_moments
                    .borrow_mut()
                    .insert(node_id, restored_tensor);
            }
        }

        // Restore max second moment buffers if using AMSGrad
        if self.amsgrad {
            for buffer in state_dict.parameter_buffers.values() {
                if buffer.buffer_type == "max_second_moment" {
                    let node_id = NodeId(buffer.node_id);
                    let restored_tensor = buffer.to_tensor::<T>()?;
                    self.max_second_moments
                        .borrow_mut()
                        .insert(node_id, restored_tensor);
                }
            }
        }

        Ok(())
    }
}


#[cfg(test)]
mod adam_tests {
    use crate::backend::Tensor;
    use crate::backend::manager::best_f32_device;
    use crate::graph::AutoFerroxEngine;
    use crate::nn::optim::{Optimizer, adam::Adam};

    #[test]
    fn test_adam_basic_step() {
        let mut engine = AutoFerroxEngine::<f32>::new();
        let device = best_f32_device();
        // Single parameter optimization
        let param = Tensor::from_vec_with_device(vec![1.0f32], &[1], device).unwrap();
        let node = engine.create_variable(param, true);

        let mut optimizer = Adam::with_defaults(0.1f32);
        optimizer.add_param(0, node);

        // Set gradient and step
        let grad = Tensor::from_vec_with_device(vec![0.5f32], &[1], device).unwrap();
        engine.set_gradient(node, grad);

        let before = engine.get_tensor_data(node).unwrap().clone().first().unwrap();
        optimizer.step(&mut engine).unwrap();
        let after = engine.get_tensor_data(node).unwrap().clone().first().unwrap();

        // Parameter should decrease with positive gradient
        assert!(after < before);
        assert_eq!(optimizer.get_step_count(), 1);
    }

    #[test]
    fn test_adam_weight_decay() {
        let mut engine = AutoFerroxEngine::<f32>::new();
        let device = best_f32_device();
        let param = Tensor::from_vec_with_device(vec![1.0f32], &[1],device ).unwrap();
        let node = engine.create_variable(param, true);

        // AdamW with weight decay
        let mut optimizer = Adam::adamw(0.1f32, 0.01f32);
        optimizer.add_param(0, node);

        // Zero gradient - only weight decay acts
        let grad = Tensor::zeros(&[1]).unwrap();
        engine.set_gradient(node, grad);

        let before = engine.get_tensor_data(node).unwrap().clone().first().unwrap();
        optimizer.step(&mut engine).unwrap();
        let after = engine.get_tensor_data(node).unwrap().clone().first().unwrap();

        // Weight decay should reduce parameter
        assert!(after < before);
    }

    #[test]
    fn test_adam_state_reset() {
        let mut optimizer = Adam::with_defaults(0.1f32);

        // Simulate some steps
        optimizer.step_count = 10;

        optimizer.reset_state();

        assert_eq!(optimizer.get_step_count(), 0);
    }
}
