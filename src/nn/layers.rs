use super::initializers::xavier_uniform;
use crate::backend::number::GPUNumber;
use crate::graph::Engine;
use crate::graph::node::NodeId;
use crate::nn::{Module, Parameter};
use crate::tensor::Tensor;

/// Identity layer that returns the input unchanged.
///
/// This is useful as a placeholder or for skip connections.
/// It does not perform any computation and simply passes the input through.
#[derive(Debug, Clone)]
pub struct Identity {
    training: bool,
}

impl Identity {
    /// Creates a new Identity layer.
    pub fn new() -> Self {
        Self { training: true }
    }
}

impl Default for Identity {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for Identity
where
    T: GPUNumber,
{
    fn forward(&self, _graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
        Ok(input)
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// Linear (fully connected) layer.
///
/// Applies a linear transformation: y = xW^T + b
///
/// # Parameters
///
/// * `weight` - Weight matrix of shape (out_features, in_features)
/// * `bias` - Bias vector of shape (out_features,) (optional)`
#[derive(Debug)]
pub struct Linear<T>
where
    T: GPUNumber,
{
    pub weight: Parameter<T>,
    pub bias: Option<Parameter<T>>, // Bias is optional as it is common to have layers without bias
    pub in_features: usize,
    pub out_features: usize,
    training: bool,
    weight_node_cache: std::cell::RefCell<Option<NodeId>>,
    bias_node_cache: std::cell::RefCell<Option<NodeId>>,
}

impl<T> Linear<T>
where
    T: GPUNumber
        + Clone
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand
        + From<f64>,
{
    /// Creates a new Linear layer.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features  
    /// * `bias` - Whether to include bias term
    ///
    /// # Weight Initialization
    ///
    /// Weights are initialized using Xavier/Glorot uniform initialization.
    /// Bias is initialized to zeros if present.
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Initialize weights using Xavier uniform initialization.
        // This should be changed to use a more flexible initializer in the future.
        let weight_init = xavier_uniform(in_features, out_features, 1.0);
        let weight = Parameter::from_init(&[out_features, in_features], weight_init);

        // Initialize bias to zeros if requested
        let bias_param = if bias {
            let bias_data = Tensor::zeros(&[out_features]);
            Some(Parameter::new(bias_data))
        } else {
            None
        };

        Self {
            weight,
            bias: bias_param,
            in_features,
            out_features,
            training: true,
            weight_node_cache: std::cell::RefCell::new(None),
            bias_node_cache: std::cell::RefCell::new(None),
        }
    }

    /// Creates a new Linear layer with custom weight initialization.
    /// Added this for flexibility, whilst keeping backward compatibility.
    /// The xavier-glorot initialization, will be kept as the default one.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    /// * `bias` - Whether to include bias term
    /// * `weight_init` - Function to initialize each weight element
    pub fn with_init<F>(in_features: usize, out_features: usize, bias: bool, weight_init: F) -> Self
    where
        F: FnMut() -> f64,
    {
        let weight = Parameter::from_init(&[out_features, in_features], weight_init);

        let bias_param = if bias {
            let bias_data = Tensor::zeros(&[out_features]);
            Some(Parameter::new(bias_data))
        } else {
            None
        };

        Self {
            weight,
            bias: bias_param,
            in_features,
            out_features,
            training: true,
            weight_node_cache: std::cell::RefCell::new(None),
            bias_node_cache: std::cell::RefCell::new(None),
        }
    }

    // Get the cached parameter NodeIds (if they exist)
    /// Critical for this layer to learn and update weights/biases.
    /// Returns a vector of NodeIds for the weight and bias parameters.
    pub fn get_cached_parameter_nodes(&self) -> Vec<Option<NodeId>> {
        let mut nodes = Vec::new();

        // Get weight node
        nodes.push(*self.weight_node_cache.borrow());

        // Get bias node if it exists
        if self.bias.is_some() {
            nodes.push(*self.bias_node_cache.borrow());
        }

        nodes
    }

    /// Check if parameters have been initialized (cached)
    pub fn parameters_initialized(&self) -> bool {
        let weight_initialized = self.weight_node_cache.borrow().is_some();
        let bias_initialized = self.bias.is_none() || self.bias_node_cache.borrow().is_some();
        weight_initialized && bias_initialized
    }

    /// Returns whether this layer has bias.
    pub fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    /// Returns the input feature size.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Returns the output feature size.
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl<T> Module<T> for Linear<T>
where
    T: GPUNumber
        + Clone
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand
        + rand_distr::num_traits::FromPrimitive,
{
    fn forward(&self, graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
        // Validate input shape
        let input_shape = graph.get_shape(input);
        if input_shape.len() < 2 {
            return Err(
                "Linear layer requires input with at least 2 dimensions (batch_size, features)"
                    .to_string(),
            );
        }

        let input_features = input_shape[input_shape.len() - 1];
        if input_features != self.in_features {
            return Err(format!(
                "Input feature size mismatch: expected {}, got {}",
                self.in_features, input_features
            ));
        }

        // Get or create weight node (create once, reuse forever)
        let weight_node = {
            let mut cache = self.weight_node_cache.borrow_mut();
            if let Some(node_id) = *cache {
                node_id
            } else {
                let new_node = Parameter::create_in_graph(graph, self.weight.data.clone());
                *cache = Some(new_node);
                new_node
            }
        };

        // Weight matrix has shape [out_features, in_features]
        // We need to transpose it to [in_features, out_features] for matrix multiplication
        // So that: input [batch_size, in_features] @ weight.T [in_features, out_features] = output [batch_size, out_features]
        let weight_t = graph.transpose(weight_node, None)?;

        // Verify the transposed weight shape is correct
        let weight_t_shape = graph.get_shape(weight_t);
        if weight_t_shape != vec![self.in_features, self.out_features] {
            return Err(format!(
                "Weight transpose shape mismatch: expected [{}, {}], got {:?}",
                self.in_features, self.out_features, weight_t_shape
            ));
        }

        // Compute: output = input @ weight^T
        // This gives us: [batch_size, in_features] @ [in_features, out_features] = [batch_size, out_features]
        let output = graph.matmul(input, weight_t)?;

        // Verify output shape is correct before adding bias
        let output_shape = graph.get_shape(output);
        let expected_output_shape = {
            let mut expected = input_shape.clone();
            let last_idx = expected.len() - 1;
            expected[last_idx] = self.out_features;
            expected
        };

        if output_shape != expected_output_shape {
            return Err(format!(
                "Output shape mismatch after matrix multiplication: expected {:?}, got {:?}",
                expected_output_shape, output_shape
            ));
        }

        // Add bias if present
        if let Some(ref bias_param) = self.bias {
            // Get or create bias node (create once, reuse forever)
            let bias_node = {
                let mut cache = self.bias_node_cache.borrow_mut();
                if let Some(node_id) = *cache {
                    node_id
                } else {
                    let new_node = Parameter::create_in_graph(graph, bias_param.data.clone());
                    *cache = Some(new_node);
                    new_node
                }
            };

            // Bias has shape [out_features]
            // We need to broadcast it to match the output shape for addition
            let bias_shape = graph.get_shape(bias_node);
            if bias_shape != vec![self.out_features] {
                return Err(format!(
                    "Bias shape mismatch: expected [{}], got {:?}",
                    self.out_features, bias_shape
                ));
            }

            // Broadcast bias to match output shape
            // From [out_features] to [batch_size, ..., out_features]
            let broadcasted_bias_node = graph.broadcast_to(bias_node, output_shape)?;

            // Add bias: output = output + bias
            let final_output = graph.add(output, broadcasted_bias_node)?;

            // Verify final output shape
            let final_output_shape = graph.get_shape(final_output);
            if final_output_shape != expected_output_shape {
                return Err(format!(
                    "Final output shape mismatch: expected {:?}, got {:?}",
                    expected_output_shape, final_output_shape
                ));
            }

            Ok(final_output)
        } else {
            Ok(output)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// Flatten layer that reshapes input to 2D.
///
/// Flattens all dimensions except the batch dimension (first dimension).
/// For example, an input of shape (N, C, H, W) becomes (N, C*H*W).
#[derive(Debug, Clone)]
pub struct Flatten {
    training: bool,
}

impl Flatten {
    /// Creates a new Flatten layer.
    pub fn new() -> Self {
        Self { training: true }
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for Flatten
where
    T: GPUNumber
        + Clone
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand
        + rand_distr::num_traits::FromPrimitive,
{
    fn forward(&self, graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
        let input_shape = graph.get_shape(input);

        if input_shape.len() < 2 {
            return Err("Flatten requires input with at least 2 dimensions".to_string());
        }

        // Calculate flattened size: keep batch dimension, flatten the rest
        let batch_size = input_shape[0];
        let flattened_size: usize = input_shape[1..].iter().product();

        let new_shape = vec![batch_size, flattened_size];
        graph.reshape(input, new_shape)
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// Sequential container for chaining modules.
///
/// Applies modules in the order they were added. Each module's output
/// becomes the input to the next module.
///
/// NOTE THAT this should be a module and NOT A MODULE LIST as the module list is a struct.
/// Rust does not allow tinheritance of structs.
pub struct Sequential<T>
where
    T: GPUNumber,
{
    modules: Vec<Box<dyn Module<T>>>,
    training: bool,
}

impl<T> Sequential<T>
where
    T: GPUNumber,
{
    /// Creates a new empty Sequential container.
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            training: true,
        }
    }

    /// Creates a Sequential container from a vector of modules.
    ///
    /// # Arguments
    ///
    /// * `modules` - Vector of boxed modules to chain together
    pub fn from_modules(modules: Vec<Box<dyn Module<T>>>) -> Self {
        Self {
            modules,
            training: true,
        }
    }

    /// Adds a module to the end of the sequence.
    ///
    /// # Arguments
    ///
    /// * `module` - The module to add (must be boxed for trait object storage)
    pub fn add(&mut self, module: Box<dyn Module<T>>) {
        self.modules.push(module);
    }

    /// Returns the number of modules in the sequence.
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Returns true if the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Inserts a module at the specified position.
    ///
    /// # Arguments
    ///
    /// * `index` - Position to insert at
    /// * `module` - The module to insert
    ///
    /// # Panics
    ///
    /// Panics if index > len().
    pub fn insert(&mut self, index: usize, module: Box<dyn Module<T>>) {
        self.modules.insert(index, module);
    }

    /// Removes and returns the module at the specified position.
    ///
    /// # Arguments
    ///
    /// * `index` - Position to remove from
    ///
    /// # Panics
    ///
    /// Panics if index >= len().
    pub fn remove(&mut self, index: usize) -> Box<dyn Module<T>> {
        self.modules.remove(index)
    }
}

impl<T> Default for Sequential<T>
where
    T: GPUNumber,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for Sequential<T>
where
    T: GPUNumber,
{
    fn forward(&self, graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
        let mut current = input;
        for module in &self.modules {
            current = module.forward(graph, current)?;
        }
        Ok(current)
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

// Added index-based access to modules. Utility following the ModuleList impl.
impl<T> std::ops::Index<usize> for Sequential<T>
where
    T: GPUNumber,
{
    type Output = Box<dyn Module<T>>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.modules[index]
    }
}
impl<T> std::ops::IndexMut<usize> for Sequential<T>
where
    T: GPUNumber,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.modules[index]
    }
}

impl<T> IntoIterator for Sequential<T>
where
    T: GPUNumber,
{
    type Item = Box<dyn Module<T>>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.modules.into_iter()
    }
}
impl<T> std::iter::FromIterator<Box<dyn Module<T>>> for Sequential<T>
where
    T: GPUNumber,
{
    fn from_iter<I: IntoIterator<Item = Box<dyn Module<T>>>>(iter: I) -> Self {
        Self {
            modules: iter.into_iter().collect(),
            training: true,
        }
    }
}

impl<T> From<Vec<Box<dyn Module<T>>>> for Sequential<T>
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
