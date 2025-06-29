use crate::backend::number::GPUNumber;
use crate::graph::Engine;
use crate::graph::node::NodeId;
use crate::nn::parameter::Parameter;
use std::collections::HashMap;

// While implementing this trait, I have taken inspiration from PyTorch's Module class
// https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
// https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html

/// The base trait for all neural network modules.
///
/// This trait defines the interface that all neural network components must implement.
/// It provides methods for parameter management, training/evaluation mode switching,
/// and the forward pass computation.
///
/// I have followed a similar approach to PyTorch's Module class,
/// this trait allows for:
///
/// - Hierarchical composition of neural network layers
/// - Automatic parameter collection and management
/// - Training/evaluation mode switching
/// - Clean forward pass interface
///
/// Other modules can implement this trait to define their own behavior,
/// while still being compatible with the overall framework.
pub trait Module<T>
where
    T: GPUNumber,
{
    /// Performs the forward pass of the module.
    ///
    /// # Arguments
    ///
    /// * `graph` - The computation graph engine
    /// * `input` - The input node ID
    ///
    /// # Returns
    ///
    /// The output node ID after applying this module
    fn forward(&self, graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String>;

    /// Returns all parameters of this module.
    ///
    /// This method should recursively collect parameters from all submodules.
    /// The default implementation returns an empty vector, but most modules
    /// will override this to return their learnable parameters.
    fn parameters(&self) -> Vec<&Parameter<T>> {
        Vec::new()
    }

    /// Returns mutable references to all parameters of this module.
    ///
    /// This is useful for optimizers that need to modify parameters directly.
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        Vec::new()
    }

    /// Returns whether the module is in training mode.
    ///
    /// Training mode affects certain layers like Dropout and BatchNorm.
    fn training(&self) -> bool {
        true // Default to training mode
    }

    /// Sets the training mode for this module and all submodules.
    ///
    /// # Arguments
    ///
    /// * `training` - Whether to set training mode (true) or evaluation mode (false)
    fn set_training(&mut self, training: bool);

    /// Sets the module to evaluation mode.
    ///
    /// This is equivalent to calling `set_training(false)`.
    fn eval(&mut self) {
        self.set_training(false);
    }

    /// Sets the module to training mode.
    ///
    /// This is equivalent to calling `set_training(true)`.
    fn train(&mut self) {
        self.set_training(true);
    }

    /// Returns the number of parameters in this module.
    ///
    /// This includes parameters from all submodules.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.size()).sum()
    }

    /// Creates parameter NodeIds in the computation graph for all parameters.
    ///
    /// This method should be called before doing any forward passes to ensure
    /// all parameters are registered in the computation graph.
    fn create_parameters_in_graph(&self, graph: &mut Engine<T>) -> HashMap<String, NodeId> {
        let mut param_map = HashMap::new();

        for (i, param) in self.parameters().iter().enumerate() {
            let name = param
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|| format!("param_{i}"));

            let node_id = Parameter::create_in_graph(graph, param.data.clone());
            param_map.insert(name, node_id);
        }

        param_map
    }
}

/// A container that holds multiple modules in a list.
///
/// This is similar to PyTorch's ModuleList and allows for dynamic
/// construction of neural networks.
/// It looks like duplicated logic if you compare it to `Sequential`, but it is not.
/// The main difference is that `ModuleList` does not enforce a sequential
/// forward pass, allowing for more flexible architectures.
/// In Pytorch the module list is just a generalization of a sequential module,
/// allowing for any type of module to be stored in the list.
/// In Rust we do not have inheritance so I decided to keep both.
pub struct ModuleList<T>
where
    T: GPUNumber,
{
    modules: Vec<Box<dyn Module<T>>>, // Dynamic dispatching modules here allows for any type of module to be stored in the mpdule list.
    training: bool,
}

impl<T> ModuleList<T>
where
    T: GPUNumber,
{
    /// Creates a new empty ModuleList.
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            training: true,
        }
    }

    /// Adds a module to the list.
    ///
    /// # Arguments
    ///
    /// * `module` - The module to add (must be boxed for trait object storage)
    pub fn push(&mut self, module: Box<dyn Module<T>>) {
        self.modules.push(module);
    }

    /// Returns the number of modules in the list.
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Returns true if the list is empty.
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Returns an iterator over the modules.
    pub fn iter(&self) -> std::slice::Iter<Box<dyn Module<T>>> {
        self.modules.iter()
    }

    /// Returns a mutable iterator over the modules.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<Box<dyn Module<T>>> {
        self.modules.iter_mut()
    }

    /// Applies a function to each module in sequence.
    ///
    /// This is useful for implementing sequential forward passes.
    ///
    /// # Arguments
    ///
    /// * `graph` - The computation graph engine
    /// * `input` - The initial input node ID
    ///
    /// # Returns
    ///
    /// The final output node ID after applying all modules
    pub fn forward_sequential(
        &self,
        graph: &mut Engine<T>,
        input: NodeId,
    ) -> Result<NodeId, String> {
        let mut current = input;
        for module in &self.modules {
            current = module.forward(graph, current)?;
        }
        Ok(current)
    }
}

impl<T> Default for ModuleList<T>
where
    T: GPUNumber,
{
    fn default() -> Self {
        Self::new()
    }
}

// This is quite tricky, because a module list is itself a module that contains other modules.
// Therefore, it implements the Module trait, but also contains other modules that implement Module.
// You know you are deep into computer science when everything gets so recursive that you have to implement a trait for a trait that contains itself.
impl<T> Module<T> for ModuleList<T>
where
    T: GPUNumber,
{
    fn forward(&self, graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
        self.forward_sequential(graph, input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        for module in &self.modules {
            params.extend(module.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        for module in &mut self.modules {
            params.extend(module.parameters_mut());
        }
        params
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        for module in &mut self.modules {
            module.set_training(training);
        }
    }
}

// Add indexing and iteration support for ModuleList
// I decided to ad this because it would be handy to access modules by index
// also you can check in Pytorch docs that ModuleList supports indexing like a Python list does.
impl<T> std::ops::Index<usize> for ModuleList<T>
where
    T: GPUNumber,
{
    type Output = Box<dyn Module<T>>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.modules[index]
    }
}

impl<T> std::ops::IndexMut<usize> for ModuleList<T>
where
    T: GPUNumber,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.modules[index]
    }
}

impl<T> IntoIterator for ModuleList<T>
where
    T: GPUNumber,
{
    type Item = Box<dyn Module<T>>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.modules.into_iter()
    }
}
impl<T> From<Vec<Box<dyn Module<T>>>> for ModuleList<T>
where
    T: GPUNumber,
{
    fn from(modules: Vec<Box<dyn Module<T>>>) -> Self {
        Self {
            modules,
            training: true,
        }
    }
}
