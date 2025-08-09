pub mod optim;
pub mod parameter;

use crate::backend::number::FerroxCudaF;
use crate::backend::Device;
use crate::graph::AutoFerroxEngine;
use crate::graph::NodeId;
use crate::nn::parameter::Parameter;
use std::collections::HashMap;
use std::slice::{Iter, IterMut};

/// Core trait for all neural network modules
/// This mirrors PyTorch's nn.Module interface, providing parameter management
/// and hierarchical composition of neural network components
pub trait Module<T>
where
    T: FerroxCudaF,
{
    /// Forward pass computation - the main interface for module execution
    /// All tensor operations must go through the computational graph engine
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String>;

    /// Collect all parameters from this module and submodules
    /// Used by optimizers to register parameters for gradient computation
    fn parameters(&self) -> Vec<&Parameter<T>> {
        Vec::new()
    }

    /// Mutable access to parameters for initialization and modification
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        Vec::new()
    }

    /// Create parameter nodes in the computational graph and return mapping
    /// This is the key method that connects parameters to the gradient system
    /// Returns (parameter_name, node_id) pairs for optimizer registration
    fn create_parameters_in_graph(
        &self,
        engine: &mut AutoFerroxEngine<T>,
    ) -> HashMap<String, NodeId>
    where
        T: rand_distr::num_traits::FromPrimitive,
    {
        let mut param_map = HashMap::new();
        let mut param_counter = 0;

        for param in self.parameters() {
            // Create a unique name if parameter doesn't have one
            let param_name = match &param.name {
                Some(name) => name.clone(),
                None => format!("param_{}", param_counter),
            };

            // Create the parameter node in the graph with gradient tracking enabled
            let node_id = engine.create_variable(param.data.clone(), param.requires_grad);
            param_map.insert(param_name, node_id);
            param_counter += 1;
        }

        param_map
    }

    /// Get training mode - affects layers like Dropout and BatchNorm
    fn training(&self) -> bool {
        true // Default to training mode
    }

    /// Set training mode for this module and all submodules
    fn set_training(&mut self, training: bool);

    /// Switch to evaluation mode - calls set_training(false)
    fn eval(&mut self) {
        self.set_training(false);
    }

    /// Switch to training mode - calls set_training(true)
    fn train(&mut self) {
        self.set_training(true);
    }

    /// Count total number of parameters in this module
    fn num_parameters(&self) -> usize {
        self.parameters()
            .iter()
            .map(|param| param.num_elements())
            .sum()
    }

    /// Helper method to apply a function to all parameters
    /// Useful for initialization, regularization, etc.
    fn apply_to_parameters<F>(&mut self, mut func: F)
    where
        F: FnMut(&mut Parameter<T>),
    {
        for param in self.parameters_mut() {
            func(param);
        }
    }

    /// Move all parameters to a specific device
    /// This is useful for moving models from CPU to GPU
    fn to_device(&mut self, device: Device) -> Result<(), String> {
        for param in self.parameters_mut() {
            param.data = param.data.clone().to_device(device)?;
        }
        Ok(())
    }

    /// Get the device of the first parameter (assumes all parameters are on same device)
    /// Returns None if module has no parameters
    fn device(&self) -> Option<Device> {
        if let Some(param) = self.parameters().first() {
            Some(param.data.device())
        } else {
            None
        }
    }

    /// Check if all parameters are on the same device
    fn check_device_consistency(&self) -> Result<(), String> {
        let params = self.parameters();
        if params.is_empty() {
            return Ok(());
        }

        let first_device = params[0].data.device();
        for (i, param) in params.iter().enumerate().skip(1) {
            if param.data.device() != first_device {
                return Err(format!(
                    "Parameter {} is on device {:?}, but parameter 0 is on device {:?}",
                    i,
                    param.data.device(),
                    first_device
                ));
            }
        }
        Ok(())
    }
}

/// Container for multiple modules - similar to PyTorch's ModuleList
/// Allows treating a collection of modules as a single module
pub struct ModuleList<T, M>
where
    T: FerroxCudaF,
    M: Module<T>,
{
    modules: Vec<M>,
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, M> ModuleList<T, M>
where
    T: FerroxCudaF,
    M: Module<T>,
{
    /// Create a new module list
    pub fn new(modules: Vec<M>) -> Self {
        Self {
            modules,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Add a module to the list
    pub fn add_module(&mut self, module: M) {
        self.modules.push(module);
    }

    /// Get reference to module at index
    pub fn get(&self, index: usize) -> Option<&M> {
        self.modules.get(index)
    }

    /// Get mutable reference to module at index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut M> {
        self.modules.get_mut(index)
    }

    /// Number of modules in the list
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Check if module list is empty
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Iterator over modules
    pub fn iter(&self) -> Iter<'_, M> {
        self.modules.iter()
    }

    /// Mutable iterator over modules
    pub fn iter_mut(&mut self) -> IterMut<'_, M> {
        self.modules.iter_mut()
    }
}

impl<T, M> Module<T> for ModuleList<T, M>
where
    T: FerroxCudaF,
    M: Module<T>,
{
    /// ModuleList doesn't define forward behavior - subclasses should override
    fn forward(&self, _graph: &mut AutoFerroxEngine<T>, _input: NodeId) -> Result<NodeId, String> {
        Err("ModuleList doesn't define forward pass - override in subclass".to_string())
    }

    /// Collect parameters from all modules in the list
    fn parameters(&self) -> Vec<&Parameter<T>> {
        self.modules
            .iter()
            .flat_map(|module| module.parameters())
            .collect()
    }

    /// Collect mutable parameter references from all modules
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.modules
            .iter_mut()
            .flat_map(|module| module.parameters_mut())
            .collect()
    }

    /// Create parameter nodes for all modules in the list
    fn create_parameters_in_graph(
        &self,
        engine: &mut AutoFerroxEngine<T>,
    ) -> HashMap<String, NodeId>
    where
        T: FerroxCudaF,
    {
        let mut all_params = HashMap::new();

        for (module_idx, module) in self.modules.iter().enumerate() {
            let module_params = module.create_parameters_in_graph(engine);

            // Prefix parameter names with module index to avoid conflicts
            for (param_name, node_id) in module_params {
                let prefixed_name = format!("module_{}_{}", module_idx, param_name);
                all_params.insert(prefixed_name, node_id);
            }
        }

        all_params
    }

    fn training(&self) -> bool {
        self.training
    }

    /// Set training mode for all modules in the list
    fn set_training(&mut self, training: bool) {
        self.training = training;
        for module in &mut self.modules {
            module.set_training(training);
        }
    }
}

/// Sequential module that applies modules in order
/// Similar to PyTorch's nn.Sequential
pub struct Sequential<T, M>
where
    T: FerroxCudaF,
    M: Module<T>,
{
    modules: ModuleList<T, M>,
}

impl<T, M> Sequential<T, M>
where
    T: FerroxCudaF,
    M: Module<T>,
{
    /// Create a new sequential module
    pub fn new(modules: Vec<M>) -> Self {
        Self {
            modules: ModuleList::new(modules),
        }
    }

    /// Add a module to the end of the sequence
    pub fn add_module(&mut self, module: M) {
        self.modules.add_module(module);
    }

    /// Get number of modules in sequence
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Check if sequence is empty
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

impl<T, M> Module<T> for Sequential<T, M>
where
    T: FerroxCudaF,
    M: Module<T>,
{
    /// Forward pass applies modules sequentially
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        let mut current_input = input;

        // Apply each module in sequence
        for (i, module) in self.modules.iter().enumerate() {
            current_input = module
                .forward(graph, current_input)
                .map_err(|e| format!("Sequential module {} failed: {}", i, e))?;
        }

        Ok(current_input)
    }

    /// Delegate parameter collection to underlying ModuleList
    fn parameters(&self) -> Vec<&Parameter<T>> {
        self.modules.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.modules.parameters_mut()
    }

    fn create_parameters_in_graph(
        &self,
        engine: &mut AutoFerroxEngine<T>,
    ) -> HashMap<String, NodeId>
    where
        T: rand_distr::num_traits::FromPrimitive,
    {
        self.modules.create_parameters_in_graph(engine)
    }

    fn training(&self) -> bool {
        self.modules.training()
    }

    fn set_training(&mut self, training: bool) {
        self.modules.set_training(training);
    }
}
