use crate::backend::Device;
use crate::backend::Tensor;
use crate::backend::{FerroxCudaF, FerroxF};
use crate::graph::AutoFerroxEngine;
use crate::graph::NodeId;
use crate::nn::Module;
use bincode::{config, decode_from_slice, encode_to_vec, Decode, Encode};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{read, write};

pub mod adam;
pub mod sgd;

/// Error types for optimizer operations
#[derive(Debug)]
pub enum OptimizerError {
    GradientNotFound(NodeId),
    ParameterNotFound(NodeId),
    TensorOperation(String),
    DeviceMismatch,
    InvalidParameterGroup,
    TypeMismatch(String),
}

impl std::fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            OptimizerError::GradientNotFound(id) => {
                write!(f, "Gradient not found for node {}", id.0)
            }
            OptimizerError::ParameterNotFound(id) => {
                write!(f, "Parameter not found for node {}", id.0)
            }
            OptimizerError::TensorOperation(msg) => write!(f, "Tensor operation failed: {}", msg),
            OptimizerError::DeviceMismatch => write!(f, "Device mismatch between tensors"),
            OptimizerError::InvalidParameterGroup => {
                write!(f, "Invalid parameter group configuration")
            }
            OptimizerError::TypeMismatch(msg) => write!(f, "Type mismatch: {}", msg),
        }
    }
}

impl std::error::Error for OptimizerError {}

/// Type-erased tensor data for serialization
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub enum TensorData {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl TensorData {
    /// Create from f32 tensor
    pub fn from_f32_tensor(tensor: &Tensor<f32>) -> Result<Self, OptimizerError> {
        let data = tensor
            .as_slice()
            .map_err(OptimizerError::TensorOperation)?
            .to_vec();
        Ok(TensorData::F32(data))
    }

    /// Create from f64 tensor
    pub fn from_f64_tensor(tensor: &Tensor<f64>) -> Result<Self, OptimizerError> {
        let data = tensor
            .as_slice()
            .map_err(OptimizerError::TensorOperation)?
            .to_vec();
        Ok(TensorData::F64(data))
    }

    /// Convert to f32 tensor
    pub fn to_f32_tensor(
        &self,
        shape: &[usize],
        device: Device,
    ) -> Result<Tensor<f32>, OptimizerError> {
        match self {
            TensorData::F32(data) => Tensor::from_vec_with_device(data.clone(), shape, device)
                .map_err(OptimizerError::TensorOperation),
            _ => Err(OptimizerError::TypeMismatch(
                "Expected f32 data".to_string(),
            )),
        }
    }

    /// Convert to f64 tensor
    pub fn to_f64_tensor(
        &self,
        shape: &[usize],
        device: Device,
    ) -> Result<Tensor<f64>, OptimizerError> {
        match self {
            TensorData::F64(data) => Tensor::from_vec_with_device(data.clone(), shape, device)
                .map_err(OptimizerError::TensorOperation),
            _ => Err(OptimizerError::TypeMismatch(
                "Expected f64 data".to_string(),
            )),
        }
    }
}

/// Serializable tensor buffer for optimizer state
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct TensorBuffer {
    pub data: TensorData,
    pub shape: Vec<usize>,
    pub device: Device,
    pub buffer_type: String, // "momentum", "first_moment", "second_moment", etc.
    pub node_id: usize,
}

impl TensorBuffer {
    /// Create from tensor with type detection
    pub fn from_tensor<T>(
        tensor: &Tensor<T>,
        buffer_type: String,
        node_id: NodeId,
    ) -> Result<Self, OptimizerError>
    where
        T: FerroxCudaF + 'static,
    {
        let data = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // This cast is safe as the type is already verified.
            let f32_tensor = unsafe { &*(tensor as *const Tensor<T> as *const Tensor<f32>) };
            TensorData::from_f32_tensor(f32_tensor)?
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            // Safe cast since type is already verified
            let f64_tensor = unsafe { &*(tensor as *const Tensor<T> as *const Tensor<f64>) };
            TensorData::from_f64_tensor(f64_tensor)?
        } else {
            return Err(OptimizerError::TypeMismatch(
                "Unsupported tensor type".to_string(),
            ));
        };

        Ok(Self {
            data,
            device: tensor.device(),
            shape: tensor.shape().to_vec(),
            buffer_type,
            node_id: node_id.0,
        })
    }

    /// Convert back to tensor with type detection
    pub fn to_tensor<T>(&self) -> Result<Tensor<T>, OptimizerError>
    where
        T: FerroxCudaF + 'static,
    {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let f32_tensor = self.data.to_f32_tensor(&self.shape, self.device)?;
            // Safe cast since we verified the type
            let tensor =
                unsafe { std::ptr::read(&f32_tensor as *const Tensor<f32> as *const Tensor<T>) };
            std::mem::forget(f32_tensor);
            Ok(tensor)
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            let f64_tensor = self.data.to_f64_tensor(&self.shape, self.device)?;
            // Safe cast since we verified the type
            let tensor =
                unsafe { std::ptr::read(&f64_tensor as *const Tensor<f64> as *const Tensor<T>) };
            std::mem::forget(f64_tensor);
            Ok(tensor)
        } else {
            Err(OptimizerError::TypeMismatch(
                "Unsupported tensor type".to_string(),
            ))
        }
    }
}

/// Serializable parameter group
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct SerializableParameterGroup {
    pub params: Vec<usize>, // NodeId values
    pub lr: Option<f64>,
    pub weight_decay: Option<f64>,
    pub momentum: Option<f64>,
    pub name: String,
}

impl<T> From<&ParameterGroup<T>> for SerializableParameterGroup
where
    T: FerroxCudaF,
{
    fn from(group: &ParameterGroup<T>) -> Self {
        Self {
            params: group.params.iter().map(|id| id.0).collect(),
            lr: group.lr.map(|v| FerroxF::to_f64(v)),
            weight_decay: group.weight_decay.map(|v| FerroxF::to_f64(v)),
            momentum: group.momentum.map(|v| FerroxF::to_f64(v)),
            name: group.name.clone(),
        }
    }
}

/// Parameter group for flexible per-layer optimization settings
#[derive(Debug, Clone)]
pub struct ParameterGroup<T>
where
    T: FerroxCudaF,
{
    pub params: HashSet<NodeId>,
    pub lr: Option<T>,
    pub weight_decay: Option<T>,
    pub momentum: Option<T>,
    pub name: String,
}

impl<T> ParameterGroup<T>
where
    T: FerroxCudaF,
{
    pub fn new(name: String) -> Self {
        Self {
            params: HashSet::new(),
            lr: None,
            weight_decay: None,
            momentum: None,
            name,
        }
    }

    pub fn with_lr(mut self, lr: T) -> Self {
        self.lr = Some(lr);
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: T) -> Self {
        self.weight_decay = Some(weight_decay);
        self
    }

    pub fn add_param(&mut self, param_id: NodeId) {
        self.params.insert(param_id);
    }

    /// Convert from serializable format
    pub fn from_serializable(
        serializable: &SerializableParameterGroup,
    ) -> Result<Self, OptimizerError>
    where
        T: crate::backend::number::CPUNumber,
    {
        Ok(Self {
            params: serializable.params.iter().map(|&id| NodeId(id)).collect(),
            lr: serializable.lr.and_then(FerroxF::from_f64),
            weight_decay: serializable.weight_decay.and_then(FerroxF::from_f64),
            momentum: serializable.momentum.and_then(FerroxF::from_f64),
            name: serializable.name.clone(),
        })
    }
}

/// State dictionary for saving and loading optimizer state
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct OptimizerStateDict {
    pub optimizer_type: String,
    pub step_count: u64,
    pub hyperparameters: HashMap<String, f64>,
    pub parameter_buffers: HashMap<String, TensorBuffer>,
    pub parameter_groups: Vec<SerializableParameterGroup>,
}

impl OptimizerStateDict {
    pub fn new(optimizer_type: String) -> Self {
        Self {
            optimizer_type,
            step_count: 0,
            hyperparameters: HashMap::new(),
            parameter_buffers: HashMap::new(),
            parameter_groups: Vec::new(),
        }
    }

    /// Add tensor buffer to state dict
    pub fn add_buffer<T>(
        &mut self,
        node_id: NodeId,
        tensor: &Tensor<T>,
        buffer_type: String,
    ) -> Result<(), OptimizerError>
    where
        T: FerroxCudaF + 'static,
    {
        let buffer = TensorBuffer::from_tensor(tensor, buffer_type, node_id)?;
        let key = format!("{}_{}", node_id.0, buffer.buffer_type);
        self.parameter_buffers.insert(key, buffer);
        Ok(())
    }

    /// Get tensor buffer by node_id and buffer_type
    pub fn get_buffer(&self, node_id: NodeId, buffer_type: &str) -> Option<&TensorBuffer> {
        let key = format!("{}_{}", node_id.0, buffer_type);
        self.parameter_buffers.get(&key)
    }

    /// Add parameter group to state dict
    pub fn add_parameter_group<T>(&mut self, group: &ParameterGroup<T>)
    where
        T: FerroxCudaF,
    {
        self.parameter_groups
            .push(SerializableParameterGroup::from(group));
    }

    /// Save to bytes using bincode
    pub fn save_to_bytes(&self) -> Result<Vec<u8>, OptimizerError> {
        encode_to_vec(self, config::standard())
            .map_err(|e| OptimizerError::TensorOperation(e.to_string()))
    }

    /// Load from bytes using bincode
    pub fn load_from_bytes(data: &[u8]) -> Result<Self, OptimizerError> {
        decode_from_slice(data, config::standard())
            .map(|(val, _)| val)
            .map_err(|e| OptimizerError::TensorOperation(e.to_string()))
    }
}

/// Core trait for all optimizers with proper error handling
pub trait Optimizer<T>
where
    T: FerroxCudaF,
{
    /// Perform one optimization step using computed gradients
    fn step(&mut self, engine: &mut AutoFerroxEngine<T>) -> Result<(), OptimizerError>;

    /// Clear all gradients for registered parameters
    fn reset_grad(&mut self, engine: &mut AutoFerroxEngine<T>);

    /// Add a single parameter to be optimized
    fn add_param(&mut self, param_id: usize, param_node_id: NodeId);

    /// Add parameter group with custom settings
    fn add_param_group(&mut self, group: ParameterGroup<T>) -> Result<(), OptimizerError>;

    /// Add all parameters from a module (convenience method)
    fn add_module_params<M>(
        &mut self,
        module: &M,
        engine: &mut AutoFerroxEngine<T>,
    ) -> Result<(), OptimizerError>
    where
        M: Module<T>,
        T: FerroxCudaF,
    {
        let param_map = module.create_parameters_in_graph(engine);
        for (i, (_, node_id)) in param_map.into_iter().enumerate() {
            self.add_param(i, node_id);
        }
        Ok(())
    }

    /// Get current learning rate for main parameter group
    fn get_lr(&self) -> T;

    /// Set learning rate for all parameter groups
    fn set_lr(&mut self, lr: T);

    /// Save optimizer state to state dict
    fn state_dict(&self) -> OptimizerStateDict
    where
        T: Clone + 'static;

    /// Load optimizer state from state dict
    fn load_state_dict(&mut self, state_dict: &OptimizerStateDict) -> Result<(), OptimizerError>
    where
        T: Clone + crate::backend::number::CPUNumber + 'static;

    /// Save state dict to file
    fn save_checkpoint(&self, filepath: &str) -> Result<(), OptimizerError>
    where
        T: Clone + 'static,
    {
        let state_dict = self.state_dict();
        let bytes = state_dict.save_to_bytes()?;
        write(filepath, bytes).map_err(|e| {
            OptimizerError::TensorOperation(format!("Failed to write checkpoint: {}", e))
        })
    }

    /// Load state dict from file
    fn load_checkpoint(&mut self, filepath: &str) -> Result<(), OptimizerError>
    where
        T: Clone + crate::backend::number::CPUNumber + 'static,
    {
        let bytes = read(filepath).map_err(|e| {
            OptimizerError::TensorOperation(format!("Failed to read checkpoint: {}", e))
        })?;
        let state_dict = OptimizerStateDict::load_from_bytes(&bytes)?;
        self.load_state_dict(&state_dict)
    }
}
