use crate::backend::Tensor;
use crate::backend::{FerroxCudaF, FerroxF};
use crate::graph::AutoFerroxEngine;
use crate::graph::NodeId;
use std::collections::HashMap;
use std::cell::RefCell;
use crate::nn::optim::{Optimizer, OptimizerError, ParameterGroup, OptimizerStateDict};
use crate::backend::Device;

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
    Max
}

impl<T> Adam<T>
where
    T: FerroxCudaF + From<f64>,
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
            T::from(0.9),
            T::from(0.999),
            T::from(1e-8),
            T::from(0.0),
            false,
        )
    }

    /// AdamW variant with decoupled weight decay
    pub fn adamw(lr: T, weight_decay: T) -> Self {
        Self::new(
            lr,
            T::from(0.9),
            T::from(0.999),
            T::from(1e-8),
            weight_decay,
            false,
        )
    }

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
        let step_f64 = self.step_count as f64;
        let beta1_power = FerroxF::to_f64(self.beta1).powf(step_f64);
        let beta2_power = FerroxF::to_f64(self.beta2).powf(step_f64);

        (T::from(1.0 - beta1_power), T::from(1.0 - beta2_power))
    }

    fn get_group_lr(&self, group: &ParameterGroup<T>) -> T {
        group.lr.unwrap_or(self.lr)
    }

    fn get_group_weight_decay(&self, group: &ParameterGroup<T>) -> T {
        group.weight_decay.unwrap_or(self.weight_decay)
    }


    fn update_moment_buffer(&self,  grad: &Tensor<T>, param_id: NodeId, moment_type: MomentBuffer, device: Device, shape: &[usize]) -> Result<(), OptimizerError> {
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
            .map_err(|e| OptimizerError::TensorOperation(e))?;
        let second_term = grad
            .mul_scalar(one_minus_beta1)
            .map_err(|e| OptimizerError::TensorOperation(e))?;
        *buffer = first_term
            .add(&second_term)
            .map_err(|e| OptimizerError::TensorOperation(e))?;

    },MomentBuffer::Second => {

            let mut second = self.second_moments.borrow_mut();
            let buffer = second.entry(param_id).or_insert_with(|| {
                    Tensor::zeros_with_device(shape, device)
                    .expect("Failed to create device-aware first moment buffer")
                    });
        // Update second moment: v = beta2 * v + (1 - beta2) * grad^2
                    let one_minus_beta2 = <T as FerroxF>::one() - self.beta2;
                    let grad_squared = grad
                        .mul(grad)
                        .map_err(|e| OptimizerError::TensorOperation(e))?;
                    let first_term = buffer
                        .mul_scalar(self.beta2)
                        .map_err(|e| OptimizerError::TensorOperation(e))?;
                    let second_term = grad_squared
                        .mul_scalar(one_minus_beta2)
                        .map_err(|e| OptimizerError::TensorOperation(e))?;
                    *buffer = first_term
                        .add(&second_term)
                        .map_err(|e| OptimizerError::TensorOperation(e))?;

    }, MomentBuffer::Max => {
        let  mut max = self.max_second_moments.borrow_mut();
        let buffer = max.entry(param_id).or_insert_with(|| {
                    Tensor::zeros_with_device(shape, device)
                    .expect("Failed to create device-aware first moment buffer")
                    });
        let seconds = self.second_moments.borrow();
        let second_moment = seconds.get(&param_id).expect("Second moment should exist !");
        *buffer = buffer.max(&second_moment)
                            .map_err(|e| OptimizerError::TensorOperation(e))?;

    },
};

    Ok(())
}


    fn take_current_buffer(&self,  param_id: NodeId, moment_type: MomentBuffer) -> Tensor<T> {
        let buffer = match moment_type {

            MomentBuffer::First => {

            let first = self.first_moments.borrow_mut();
            first.get(&param_id).expect("Buffer not found").clone()
    },MomentBuffer::Second => {

            let second = self.second_moments.borrow();
            second.get(&param_id).expect("Buffer not found").clone()

    }, MomentBuffer::Max => {
        let  max = self.max_second_moments.borrow();
        max.get(&param_id).expect("Buffer not found").clone()

    },


};
    buffer
    }


    fn compute_update(&self, param_id: NodeId, second_moment: &Tensor<T>, bias_correction1: T, bias_correction2: T, lr: T) -> Result<Tensor<T>, OptimizerError>{
                // Compute update: lr * m_hat / (sqrt(v_hat) + eps)
                    let denominator = second_moment
                        .div_scalar(bias_correction2)
                        .map_err(|e| OptimizerError::TensorOperation(e))?
                        .sqrt()
                        .map_err(|e| OptimizerError::TensorOperation(e))?
                        .add_scalar(self.eps)
                        .map_err(|e| OptimizerError::TensorOperation(e))?;

                    let scaled_update = self.take_current_buffer(param_id, MomentBuffer::First)
                        .div_scalar(bias_correction1)
                        .map_err(|e| OptimizerError::TensorOperation(e))?
                        .div(&denominator)
                        .map_err(|e| OptimizerError::TensorOperation(e))?
                        .mul_scalar(lr)
                        .map_err(|e| OptimizerError::TensorOperation(e))?;
                    Ok(scaled_update)
    }

    fn apply_weight_decay(&self, weight_decay: T, lr: T, params: &Tensor<T>, update: &Tensor<T>) -> Result<Tensor<T>, OptimizerError> {
        let new_params = if weight_decay != <T as FerroxF>::zero() {
                        let weight_decay_term = params
                            .mul_scalar(weight_decay * lr)
                            .map_err(|e| OptimizerError::TensorOperation(e))?;
                        params
                            .sub(update)
                            .map_err(|e| OptimizerError::TensorOperation(e))?
                            .sub(&weight_decay_term)
                            .map_err(|e| OptimizerError::TensorOperation(e))?
                    } else {
                        params
                            .sub(&update)
                            .map_err(|e| OptimizerError::TensorOperation(e))?
                    };
                    Ok(new_params)
    }
}
impl<T> Optimizer<T> for Adam<T>
where
    T: FerroxCudaF + From<f64>,
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
                    self.update_moment_buffer(grad, param_node, MomentBuffer::First, device, shape)?;

                    self.update_moment_buffer(grad, param_node, MomentBuffer::Second, device, shape)?;

                    // AMSGrad: maintain maximum of second moments
                    let moment_for_update = if self.amsgrad {
                        self.update_moment_buffer(grad, param_node, MomentBuffer::Max, device, shape)?;

                        self.take_current_buffer(param_node, MomentBuffer::Max)
                    } else {
                        self.take_current_buffer(param_node, MomentBuffer::Second)
                    };

                    // Compute update: lr * m_hat / (sqrt(v_hat) + eps)
                    let update = self.compute_update(param_node, &moment_for_update, bias_correction1, bias_correction2, group_lr)?;

                    // Apply AdamW-style decoupled weight decay directly to parameters
                    let new_params = self.apply_weight_decay(group_weight_decay, group_lr, &params, &update)?;

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

    fn state_dict(&self) -> OptimizerStateDict
    where
        T: FerroxCudaF,
    {
        unimplemented!("Not implemneted");
    }

    fn load_state_dict(&mut self, state_dict: &OptimizerStateDict) -> Result<(), OptimizerError>
    where
        T: FerroxCudaF,
    {

        unimplemented!("Not implemneted");
    }
}
