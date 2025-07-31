use crate::backend::number::FerroxCudaF;
use crate::graph::Engine;
use crate::graph::node::NodeId;
use crate::tensor::Tensor;

/// A Parameter is a special kind of tensor that represents learnable parameters in a neural network.
///
/// Parameters are automatically included when collecting parameters from a module,
/// and they always require gradients for training. This is similar to PyTorch's Parameter class.
/// Parameters can be initialized with specific values, or created from random initialization functions.
/// They can also be named for easier debugging and visualization.``
#[derive(Debug, Clone)]
pub struct Parameter<T>
where
    T: FerroxCudaF,
{
    /// The actual tensor data
    pub data: Tensor<T>,
    /// Whether this parameter requires gradients (always true for parameters)
    pub requires_grad: bool,
    /// Optional name for debugging and visualization
    pub name: Option<String>,
}

impl<T> Parameter<T>
where
    T: FerroxCudaF,
{
    /// Creates a new parameter from tensor data.
    ///
    /// # Arguments
    ///
    /// * `data` - The tensor data for this parameter
    pub fn new(data: Tensor<T>) -> Self {
        Self {
            data,
            requires_grad: true, // Parameters always require gradients
            name: None,
        }
    }

    /// Creates a new parameter with a name for debugging.
    ///
    /// # Arguments
    ///
    /// * `data` - The tensor data
    /// * `name` - A name for this parameter
    pub fn new_named(data: Tensor<T>, name: String) -> Self {
        Self {
            data,
            requires_grad: true,
            name: Some(name),
        }
    }

    /// Creates a parameter from existing tensor data and adds it to the computation graph.
    ///
    /// # Arguments
    ///
    /// * `graph` - The computation graph engine
    /// * `data` - The tensor data
    ///
    /// # Returns
    ///
    /// The NodeId of the parameter in the computation graph
    pub fn create_in_graph(graph: &mut Engine<T>, data: Tensor<T>) -> NodeId {
        graph.create_tensor(data, true)
    }

    /// Creates a parameter with random initialization using specified bounds.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the parameter tensor
    /// * `init_fn` - Initialization function that returns a value for each element
    pub fn from_init<F>(shape: &[usize], mut init_fn: F) -> Self
    where
        F: FnMut() -> f64,
        T: From<f64>,
    {
        let total_elements: usize = shape.iter().product();
        let data: Vec<T> = (0..total_elements).map(|_| T::from(init_fn())).collect();

        let tensor = Tensor::from_vec(data, shape).expect("Failed to create parameter tensor");

        Self::new(tensor)
    }

    /// Returns the shape of the parameter.
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Returns the number of elements in the parameter.
    pub fn size(&self) -> usize {
        self.data.size()
    }

    /// Gets the parameter name if available.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Sets the parameter name.
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }

    /// Creates a detached copy of this parameter (no gradient tracking).
    /// Useful for inference or when you want to stop gradient flow.
    pub fn detach(&self) -> Tensor<T> {
        self.data
            .detach()
            .expect("Could not detach Tensor from graph")
    }
}

impl<T> From<Tensor<T>> for Parameter<T>
where
    T: FerroxCudaF,
{
    fn from(tensor: Tensor<T>) -> Self {
        Self::new(tensor)
    }
}

/// Helper trait for converting various types to parameters
pub trait ToParameter<T>
where
    T: FerroxCudaF,
{
    fn to_parameter(self) -> Parameter<T>;
}

impl<T> ToParameter<T> for Tensor<T>
where
    T: FerroxCudaF,
{
    fn to_parameter(self) -> Parameter<T> {
        Parameter::new(self)
    }
}

impl<T> ToParameter<T> for Vec<T>
where
    T: FerroxCudaF,
{
    fn to_parameter(self) -> Parameter<T> {
        // Default to 1D tensor
        let tensor = Tensor::from_vec(self, &[1]).expect("Failed to create tensor from vector");
        Parameter::new(tensor)
    }
}
