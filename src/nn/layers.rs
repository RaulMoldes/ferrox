use crate::backend::numeric::Numeric;
use crate::graph::Engine;
use crate::graph::node::NodeId;
use crate::initializers::{xavier_normal, xavier_uniform};
use crate::nn::{Module, Parameter};
use crate::tensor::Tensor;

/// Identity layer that returns the input unchanged.
///
/// This is useful as a placeholder or for skip connections.
///
/// # Examples
///
/// ```rust
/// use ferrox::nn::{Identity, Module};
///
/// let identity = Identity::new();
/// // forward pass returns input unchanged
/// ```
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
    T: Numeric + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
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
/// * `bias` - Bias vector of shape (out_features,) (optional)
///
/// # Examples
///
/// ```rust
/// use ferrox::nn::Linear;
///
/// // Create a linear layer: 784 inputs -> 256 outputs with bias
/// let linear = Linear::new(784, 256, true);
///
/// // Create without bias
/// let linear_no_bias = Linear::new(784, 256, false);
/// ```
#[derive(Debug)]
pub struct Linear<T>
where
    T: Numeric + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
{
    pub weight: Parameter<T>,
    pub bias: Option<Parameter<T>>,
    pub in_features: usize,
    pub out_features: usize,
    training: bool,
}

impl<T> Linear<T>
where
    T: Numeric
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
        // Initialize weights using Xavier uniform initialization
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
        }
    }

    /// Creates a new Linear layer with custom weight initialization.
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
        }
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
    T: Numeric + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
{
    fn forward(&self, graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
        // Create weight and bias nodes in the graph
        let weight_node = Parameter::create_in_graph(graph, self.weight.data.clone());

        // Compute: output = input @ weight^T
        let output = graph.matmul(input, weight_node)?;

        // Add bias if present
        if let Some(ref bias_param) = self.bias {
            let bias_node = Parameter::create_in_graph(graph, bias_param.data.clone());
            graph.add(output, bias_node)
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
///
/// # Examples
///
/// ```rust
/// use ferrox::nn::Flatten;
///
/// let flatten = Flatten::new();
/// // Input shape: (32, 3, 28, 28) -> Output shape: (32, 2352)
/// ```
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
    T: Numeric
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
/// # Examples
///
/// ```rust
/// use ferrox::nn::{Sequential, Linear, ReLU};
///
/// let mut model = Sequential::new();
/// model.add(Box::new(Linear::new(784, 256, true)));
/// model.add(Box::new(ReLU::new()));
/// model.add(Box::new(Linear::new(256, 10, true)));
/// ```

pub struct Sequential<T>
where
    T: Numeric + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
{
    modules: Vec<Box<dyn Module<T>>>,
    training: bool,
}

impl<T> Sequential<T>
where
    T: Numeric + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
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
    T: Numeric + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for Sequential<T>
where
    T: Numeric + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
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

/// Macro for easily creating Sequential models.
///
/// # Examples
///
/// ```rust
/// use ferrox::sequential;
/// use ferrox::nn::{Linear, ReLU};
///
/// let model = sequential![
///     Linear::new(784, 256, true),
///     ReLU::new(),
///     Linear::new(256, 128, true),
///     ReLU::new(),
///     Linear::new(128, 10, true)
/// ];
/// ```
#[macro_export]
macro_rules! sequential {
    ($($module:expr),* $(,)?) => {
        {
            let mut seq = $crate::nn::Sequential::new();
            $(
                seq.add(Box::new($module));
            )*
            seq
        }
    };
}
