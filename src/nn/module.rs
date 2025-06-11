use crate::backend::numeric::Numeric;
use crate::graph::Engine;
use crate::graph::node::NodeId;
use crate::nn::parameter::Parameter;
use std::collections::HashMap;

/// The base trait for all neural network modules.
///
/// This trait defines the interface that all neural network components must implement.
/// It provides methods for parameter management, training/evaluation mode switching,
/// and the forward pass computation.
///
/// # Design Philosophy
///
/// Similar to PyTorch's Module class, this trait allows for:
/// - Hierarchical composition of neural network layers
/// - Automatic parameter collection and management
/// - Training/evaluation mode switching
/// - Clean forward pass interface
///
/// # Examples
///
/// ```rust
/// use ferrox::nn::{Module, Parameter};
/// use ferrox::graph::Engine;
/// use ferrox::NodeId; 
///
/// struct MyLayer {
///     weight: Parameter<f64>,
///     bias: Parameter<f64>,
///     training: bool,
/// }
///
/// impl Module<f64> for MyLayer {
///     fn forward(&self, graph: &mut Engine<f64>, input: NodeId) -> Result<NodeId, String> {
///         // Implementation here
///         Ok(input)
///     }
///     
///     fn parameters(&self) -> Vec<&Parameter<f64>> {
///         vec![&self.weight, &self.bias]
///     }
///     
///     fn training(&self) -> bool {
///         self.training
///     }
///     
///     fn set_training(&mut self, training: bool) {
///         self.training = training;
///     }
/// }
/// ```
pub trait Module<T>
where
    T: Numeric + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
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
                .unwrap_or_else(|| format!("param_{}", i));

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
///
/// # Examples
///
/// ```rust
/// use ferrox::nn::{ModuleList, Linear};
///
/// let mut layers = ModuleList::new();
/// layers.push(Box::new(Linear::<f64>::new(784, 256, true)));
/// layers.push(Box::new(Linear::<f64>::new(256, 128, true)));
/// layers.push(Box::new(Linear::<f64>::new(128, 10, true)));
/// ```
pub struct ModuleList<T>
where
    T: Numeric + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
{
    modules: Vec<Box<dyn Module<T>>>,
    training: bool,
}

impl<T> ModuleList<T>
where
    T: Numeric + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
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
    T: Numeric + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for ModuleList<T>
where
    T: Numeric + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
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
