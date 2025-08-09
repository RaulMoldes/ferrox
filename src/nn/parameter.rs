use crate::backend::number::FerroxCudaF;
use crate::backend::Device;
use crate::backend::Tensor;
use rand::Rng;

/// A Parameter wraps a tensor that requires gradients for optimization.
/// Parameters are automatically registered for gradient computation and collected by optimizers.
/// This design follows PyTorch's Parameter class - parameters are special tensors that are
/// automatically added to the module's parameter list and included in optimization.
#[derive(Debug, Clone)]
pub struct Parameter<T>
where
    T: FerroxCudaF,
{
    /// The tensor data - this should be accessed through the computational graph
    pub data: Tensor<T>,
    /// Parameters always require gradients by design
    pub requires_grad: bool,
    /// Optional name for debugging and parameter identification
    pub name: Option<String>,
}

impl<T> Parameter<T>
where
    T: FerroxCudaF,
{
    /// Creates a new parameter from tensor data
    /// Parameters always require gradients by default
    pub fn new(data: Tensor<T>) -> Self {
        Self {
            data,
            requires_grad: true,
            name: None,
        }
    }

    /// Creates a named parameter for easier debugging and identification
    pub fn new_named(data: Tensor<T>, name: String) -> Self {
        Self {
            data,
            requires_grad: true,
            name: Some(name),
        }
    }

    /// Returns the shape of this parameter
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Returns the number of elements in this parameter
    pub fn num_elements(&self) -> usize {
        self.data.len()
    }

    /// Returns the parameter name if set
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Creates a parameter from random initialization using specified function
    /// Uses the best available device for tensor creation
    pub fn from_init<F>(shape: &[usize], mut init_fn: F) -> Self
    where
        F: FnMut() -> f64,
        T: From<f64>,
    {
        let total_elements: usize = shape.iter().product();
        let data: Vec<T> = (0..total_elements).map(|_| T::from(init_fn())).collect();

        // Use best available device instead of defaulting to CPU
        let device = crate::backend::default_device();
        let tensor = Tensor::from_vec_with_device(data, shape, device)
            .expect("Failed to create parameter tensor from initialization");

        Self::new(tensor)
    }

    /// Creates a parameter from initialization with specific device
    pub fn from_init_with_device<F>(shape: &[usize], mut init_fn: F, device: Device) -> Self
    where
        F: FnMut() -> f64,
        T: From<f64>,
    {
        let total_elements: usize = shape.iter().product();
        let data: Vec<T> = (0..total_elements).map(|_| T::from(init_fn())).collect();

        let tensor = Tensor::from_vec_with_device(data, shape, device)
            .expect("Failed to create parameter tensor from initialization");

        Self::new(tensor)
    }

    /// Creates a parameter with Xavier/Glorot uniform initialization
    /// Suitable for layers with tanh or sigmoid activations
    pub fn xavier_uniform(shape: &[usize]) -> Self
    where
        T: From<f64>,
    {
        let mut rng = rand::rng();

        // Xavier initialization: U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
        let (fan_in, fan_out) = Self::calculate_fan_in_out(shape);
        let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();

        Self::from_init(shape, move || rng.random_range(-bound..bound))
    }

    /// Creates a parameter with Xavier initialization on specific device
    pub fn xavier_uniform_with_device(shape: &[usize], device: Device) -> Self
    where
        T: From<f64>,
    {
        use rand::Rng;
        let mut rng = rand::rng();

        let (fan_in, fan_out) = Self::calculate_fan_in_out(shape);
        let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();

        Self::from_init_with_device(shape, move || rng.random_range(-bound..bound), device)
    }

    /// Creates a parameter with Kaiming/He uniform initialization
    /// Suitable for layers with ReLU activations
    pub fn kaiming_uniform(shape: &[usize]) -> Self
    where
        T: From<f64>,
    {
        let mut rng = rand::rng();

        // Kaiming initialization: U(-sqrt(6/fan_in), sqrt(6/fan_in))
        let (fan_in, _) = Self::calculate_fan_in_out(shape);
        let bound = (6.0 / fan_in as f64).sqrt();

        Self::from_init(shape, move || rng.random_range(-bound..bound))
    }

    /// Creates a parameter with Kaiming initialization on specific device
    pub fn kaiming_uniform_with_device(shape: &[usize], device: Device) -> Self
    where
        T: From<f64>,
    {
        let mut rng = rand::rng();

        let (fan_in, _) = Self::calculate_fan_in_out(shape);
        let bound = (6.0 / fan_in as f64).sqrt();

        Self::from_init_with_device(shape, move || rng.random_range(-bound..bound), device)
    }

    /// Creates a parameter initialized with zeros
    pub fn zeros(shape: &[usize]) -> Self
    where
        T: From<f64>,
    {
        Self::from_init(shape, || 0.0)
    }

    /// Creates a parameter initialized with zeros on specific device
    pub fn zeros_with_device(shape: &[usize], device: Device) -> Self
    where
        T: From<f64>,
    {
        Self::from_init_with_device(shape, || 0.0, device)
    }

    /// Creates a parameter initialized with ones
    pub fn ones(shape: &[usize]) -> Self
    where
        T: From<f64>,
    {
        Self::from_init(shape, || 1.0)
    }

    /// Creates a parameter initialized with ones on specific device
    pub fn ones_with_device(shape: &[usize], device: Device) -> Self
    where
        T: From<f64>,
    {
        Self::from_init_with_device(shape, || 1.0, device)
    }

    /// Move parameter to a specific device
    /// This allows moving parameters from CPU to GPU after creation
    pub fn to_device(mut self, device: Device) -> Result<Self, String> {
        self.data = self.data.to_device(device)?;
        Ok(self)
    }

    /// Helper function to calculate fan_in and fan_out for initialization
    fn calculate_fan_in_out(shape: &[usize]) -> (usize, usize) {
        match shape.len() {
            0 => (1, 1),
            1 => (shape[0], shape[0]),
            2 => (shape[0], shape[1]), // [in_features, out_features]
            _ => {
                // For higher dimensional tensors (conv layers, etc.)
                let num_input_fmaps = shape[1];
                let num_output_fmaps = shape[0];
                let receptive_field_size: usize = shape[2..].iter().product();

                let fan_in = num_input_fmaps * receptive_field_size;
                let fan_out = num_output_fmaps * receptive_field_size;
                (fan_in, fan_out)
            }
        }
    }
}

/// Trait for converting tensors into parameters
/// Provides convenient methods for creating parameters from existing tensors
pub trait ToParameter<T>
where
    T: FerroxCudaF,
{
    /// Converts this tensor into a parameter
    fn to_parameter(self) -> Parameter<T>;

    /// Converts this tensor into a named parameter
    fn to_parameter_named(self, name: String) -> Parameter<T>;
}

impl<T> ToParameter<T> for Tensor<T>
where
    T: FerroxCudaF,
{
    fn to_parameter(self) -> Parameter<T> {
        Parameter::new(self)
    }

    fn to_parameter_named(self, name: String) -> Parameter<T> {
        Parameter::new_named(self, name)
    }
}
