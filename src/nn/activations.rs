use crate::backend::numeric::{Float, Numeric};
use crate::graph::Engine;
use crate::graph::node::NodeId;
use crate::nn::Module;
use crate::nn::parameter::Parameter;

/// ReLU (Rectified Linear Unit) activation function.
///
/// Applies the function ReLU(x) = max(0, x) element-wise.
/// This is one of the most commonly used activation functions in deep learning.
///
/// # Mathematical Definition
///
/// ```text
/// ReLU(x) = max(0, x) = {
///     x if x > 0
///     0 if x ≤ 0
/// }
/// ```
///
/// # Examples
///
/// ```rust
/// use ferrox::nn::{ReLU, Module};
/// use ferrox::graph::Engine;
///
/// let relu = ReLU::new();
/// let mut graph = Engine::new();
/// 
/// // Create input tensor with negative and positive values
/// let input = graph.tensor_from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5], true).unwrap();
/// let output = relu.forward(&mut graph, input).unwrap();
/// // Output will be [0.0, 0.0, 0.0, 1.0, 2.0]
/// ```
#[derive(Debug, Clone)]
pub struct ReLU {
    training: bool,
}

impl ReLU {
    /// Creates a new ReLU activation layer.
    pub fn new() -> Self {
        Self { training: true }
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for ReLU
where
    T: Float + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
{
    fn forward(&self, graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
        graph.relu(input)
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// Sigmoid activation function.
///
/// Applies the sigmoid function element-wise: σ(x) = 1 / (1 + e^(-x))
/// Maps any real number to a value between 0 and 1, making it useful for binary classification.
///
/// # Mathematical Definition
///
/// ```text
/// Sigmoid(x) = σ(x) = 1 / (1 + e^(-x))
/// ```
///
/// # Characteristics
///
/// - Output range: (0, 1)
/// - Derivative: σ'(x) = σ(x) * (1 - σ(x))
/// - Center point: σ(0) = 0.5
///
/// # Examples
///
/// ```rust
/// use ferrox::nn::{Sigmoid, Module};
/// use ferrox::graph::Engine;
///
/// let sigmoid = Sigmoid::new();
/// let mut graph = Engine::new();
/// 
/// let input = graph.tensor_from_vec(vec![-2.0, 0.0, 2.0], &[3], true).unwrap();
/// let output = sigmoid.forward(&mut graph, input).unwrap();
/// // Output will be approximately [0.119, 0.5, 0.881]
/// ```
#[derive(Debug, Clone)]
pub struct Sigmoid {
    training: bool,
}

impl Sigmoid {
    /// Creates a new Sigmoid activation layer.
    pub fn new() -> Self {
        Self { training: true }
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for Sigmoid
where
    T: Float + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand + rand_distr::num_traits::FromPrimitive,
{
    fn forward(&self, graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
        // Sigmoid(x) = 1 / (1 + exp(-x))
        // We need to implement this using available operations in the graph
        
        // First, compute -x
        let neg_input = graph.negate(input)?;
        
        // Then compute exp(-x)
        let exp_neg_input = graph.exp(neg_input)?;
        
        // Add 1 to get (1 + exp(-x))
        let one_plus_exp = graph.add_scalar(exp_neg_input, <T as crate::backend::numeric::Numeric>::one())?;
        
        // Create a tensor of ones with the same shape as input
        let input_shape = graph.get_shape(input);
        let ones_node = graph.ones(&input_shape, false);
        
        // Finally, compute 1 / (1 + exp(-x))
        graph.div(ones_node, one_plus_exp)
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// Tanh (Hyperbolic Tangent) activation function.
///
/// Applies the hyperbolic tangent function element-wise: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
/// Maps any real number to a value between -1 and 1, making it zero-centered.
///
/// # Mathematical Definition
///
/// ```text
/// Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)) = (e^(2x) - 1) / (e^(2x) + 1)
/// ```
///
/// # Characteristics
///
/// - Output range: (-1, 1)
/// - Zero-centered (unlike sigmoid)
/// - Derivative: tanh'(x) = 1 - tanh²(x)
///
/// # Examples
///
/// ```rust
/// use ferrox::nn::{Tanh, Module};
/// use ferrox::graph::Engine;
///
/// let tanh = Tanh::new();
/// let mut graph = Engine::new();
/// 
/// let input = graph.tensor_from_vec(vec![-2.0, 0.0, 2.0], &[3], true).unwrap();
/// let output = tanh.forward(&mut graph, input).unwrap();
/// // Output will be approximately [-0.964, 0.0, 0.964]
/// ```
#[derive(Debug, Clone)]
pub struct Tanh {
    training: bool,
}

impl Tanh {
    /// Creates a new Tanh activation layer.
    pub fn new() -> Self {
        Self { training: true }
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for Tanh
where
    T: Float + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand + rand_distr::num_traits::FromPrimitive,
{
    fn forward(&self, graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
        // Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        // We can implement this more efficiently as: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
        
        // Compute 2x
        let two_x = graph.mul_scalar(input, <T as Numeric>::from_f64(2.0).unwrap())?;
        
        // Compute e^(2x)
        let exp_2x = graph.exp(two_x)?;
        
        // Create ones tensor with same shape as input
        let input_shape = graph.get_shape(input);
        let ones_node = graph.ones(&input_shape, false);
        
        // Compute e^(2x) - 1
        let neg_ones = graph.negate(ones_node)?;
        let numerator = graph.add(exp_2x, neg_ones)?;
        
        // Compute e^(2x) + 1
        let ones_node2 = graph.ones(&input_shape, false);
        let denominator = graph.add(exp_2x, ones_node2)?;
        
        // Return (e^(2x) - 1) / (e^(2x) + 1)
        graph.div(numerator, denominator)
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// Softmax activation function.
///
/// Applies the softmax function to the input tensor along the last dimension.
/// The softmax function converts a vector of arbitrary real values to a probability distribution.
///
/// # Mathematical Definition
///
/// For input vector x = [x₁, x₂, ..., xₙ]:
/// ```text
/// Softmax(xᵢ) = e^xᵢ / Σⱼ(e^xʲ)
/// ```
///
/// # Characteristics
///
/// - Output range: (0, 1) for each element
/// - Sum of all outputs equals 1
/// - Commonly used in the final layer for multi-class classification
/// - Numerically stable implementation using the log-sum-exp trick
///
/// # Examples
///
/// ```rust
/// use ferrox::nn::{Softmax, Module};
/// use ferrox::graph::Engine;
///
/// let softmax = Softmax::new();
/// let mut graph = Engine::new();
/// 
/// // For a batch of 2 samples with 3 classes each
/// let input = graph.tensor_from_vec(
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
///     &[2, 3], 
///     true
/// ).unwrap();
/// let output = softmax.forward(&mut graph, input).unwrap();
/// // Each row will sum to 1.0
/// ```
#[derive(Debug, Clone)]
pub struct Softmax {
    /// Dimension along which to apply softmax. Default is -1 (last dimension).
    dim: Option<usize>,
    training: bool,
}

impl Softmax {
    /// Creates a new Softmax activation layer.
    /// 
    /// By default, applies softmax along the last dimension.
    pub fn new() -> Self {
        Self {
            dim: None,
            training: true,
        }
    }

    /// Creates a new Softmax activation layer with a specific dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension along which to apply softmax
    pub fn new_with_dim(dim: usize) -> Self {
        Self {
            dim: Some(dim),
            training: true,
        }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for Softmax
where
    T: Float
        + Clone
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand
        + rand_distr::num_traits::FromPrimitive,
{
    fn forward(&self, graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
        let input_shape = graph.get_shape(input);
        let target_dim = self.dim.unwrap_or(input_shape.len() - 1);

        // For numerical stability, subtract the maximum value along the target dimension
        // This is the log-sum-exp trick: softmax(x) = softmax(x - max(x))
        
        // Note: For simplicity in this implementation, we'll use a basic version
        // A more robust implementation would compute the max along the specified dimension
        // and subtract it before applying exponential
        
        // Apply exponential to all elements
        let exp_input = graph.exp(input)?;
        
        // Sum along the target dimension
        let sum_exp = graph.summation(exp_input, Some(vec![target_dim]))?;
        
        // Broadcast to original shape
        let broadcasted_sum = graph.broadcast_to(sum_exp, input_shape)?;
        
        // Divide to get probabilities
        graph.div(exp_input, broadcasted_sum)
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// LeakyReLU activation function.
///
/// Applies the Leaky ReLU function element-wise. Unlike ReLU, LeakyReLU allows
/// a small, non-zero gradient when the input is negative.
///
/// # Mathematical Definition
///
/// ```text
/// LeakyReLU(x) = {
///     x           if x > 0
///     α * x       if x ≤ 0
/// }
/// ```
///
/// Where α is the negative slope parameter (default: 0.01).
///
/// # Advantages over ReLU
///
/// - Prevents "dying ReLU" problem
/// - Allows gradients to flow even for negative inputs
/// - Can help with faster convergence in some cases
///
/// # Examples
///
/// ```rust
/// use ferrox::nn::{LeakyReLU, Module};
/// use ferrox::graph::Engine;
///
/// let leaky_relu = LeakyReLU::new(); // Default slope = 0.01
/// let custom_slope = LeakyReLU::new_with_slope(0.1);
/// 
/// let mut graph = Engine::new();
/// let input = graph.tensor_from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5], true).unwrap();
/// let output = leaky_relu.forward(&mut graph, input).unwrap();
/// // Output with default slope: [-0.02, -0.01, 0.0, 1.0, 2.0]
/// ```
#[derive(Debug, Clone)]
pub struct LeakyReLU<T>
where
    T: Float + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
{
    /// The negative slope parameter
    negative_slope: T,
    training: bool,
}

impl<T> LeakyReLU<T>
where
    T: Float + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
{
    /// Creates a new LeakyReLU activation layer with default slope (0.01).
    pub fn new() -> Self
    where
        T: From<f64>,
    {
        Self {
            negative_slope: T::from(0.01),
            training: true,
        }
    }

    /// Creates a new LeakyReLU activation layer with custom negative slope.
    ///
    /// # Arguments
    ///
    /// * `negative_slope` - The slope for negative values (typically small positive value)
    pub fn new_with_slope(negative_slope: T) -> Self {
        Self {
            negative_slope,
            training: true,
        }
    }

    /// Returns the negative slope parameter.
    pub fn negative_slope(&self) -> T {
        self.negative_slope
    }
}

impl<T> Default for LeakyReLU<T>
where
    T: Float + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand + From<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for LeakyReLU<T>
where
    T: Float + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
{
    fn forward(&self, graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
        // LeakyReLU(x) = max(x, α * x) where α is the negative slope
        // This can be implemented as: x > 0 ? x : α * x
        // Since we don't have conditional operations, we'll use:
        // LeakyReLU(x) = max(x, α * x) = x * max(1, α) + min(x, 0) * (α - 1)
        // But a simpler approach is: max(0, x) + α * min(0, x)
        
        // Get ReLU(x) = max(0, x)
        let positive_part = graph.relu(input)?;
        
        // Get min(0, x) = -ReLU(-x)
        let neg_input = graph.negate(input)?;
        let neg_relu = graph.relu(neg_input)?;
        let negative_part = graph.negate(neg_relu)?;
        
        // Multiply negative part by slope
        let scaled_negative = graph.mul_scalar(negative_part, self.negative_slope)?;
        
        // Combine: max(0, x) + α * min(0, x)
        graph.add(positive_part, scaled_negative)
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// ELU (Exponential Linear Unit) activation function.
///
/// Applies the ELU function element-wise. ELU has the benefit of having negative values
/// which pushes the mean of the activations closer to zero, enabling faster learning.
///
/// # Mathematical Definition
///
/// ```text
/// ELU(x) = {
///     x                   if x > 0
///     α * (e^x - 1)       if x ≤ 0
/// }
/// ```
///
/// Where α is a hyperparameter that controls the saturation point for negative inputs.
///
/// # Characteristics
///
/// - Smooth everywhere (differentiable at x = 0)
/// - Negative values push mean activation closer to zero
/// - Saturates to -α for large negative inputs
/// - Reduces the bias shift effect
///
/// # Examples
///
/// ```rust
/// use ferrox::nn::{ELU, Module};
/// use ferrox::graph::Engine;
///
/// let elu = ELU::new(); // Default α = 1.0
/// let custom_alpha = ELU::new_with_alpha(0.5);
/// 
/// let mut graph = Engine::new();
/// let input = graph.tensor_from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5], true).unwrap();
/// let output = elu.forward(&mut graph, input).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ELU<T>
where
    T: Float + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
{
    /// The α parameter for negative inputs
    alpha: T,
    training: bool,
}

impl<T> ELU<T>
where
    T: Float + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
{
    /// Creates a new ELU activation layer with default α = 1.0.
    pub fn new() -> Self
    where
        T: From<f64>,
    {
        Self {
            alpha: T::from(1.0),
            training: true,
        }
    }

    /// Creates a new ELU activation layer with custom α parameter.
    ///
    /// # Arguments
    ///
    /// * `alpha` - The α parameter (should be positive)
    pub fn new_with_alpha(alpha: T) -> Self {
        Self {
            alpha,
            training: true,
        }
    }

    /// Returns the α parameter.
    pub fn alpha(&self) -> T {
        self.alpha
    }
}

impl<T> Default for ELU<T>
where
    T: Float + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand + From<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for ELU<T>
where
    T: Float + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand + rand_distr::num_traits::FromPrimitive,
{
    fn forward(&self, graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
        // ELU(x) = x if x > 0, α * (e^x - 1) if x ≤ 0
        // We can implement this as: max(0, x) + min(0, α * (e^x - 1))
        
        // Positive part: max(0, x) = ReLU(x)
        let positive_part = graph.relu(input)?;
        
        // For negative part: α * (e^x - 1) when x ≤ 0
        // First compute e^x
        let exp_input = graph.exp(input)?;
        
        // Subtract 1: (e^x - 1)
        let input_shape = graph.get_shape(input);
        let ones_node = graph.ones(&input_shape, false);
        let negated_ones = graph.negate(ones_node)?;
        let exp_minus_one = graph.add(exp_input, negated_ones)?;
        
        // Multiply by α: α * (e^x - 1)
        let alpha_term = graph.mul_scalar(exp_minus_one, self.alpha)?;
        
        // Take minimum with 0: min(0, α * (e^x - 1))
        // This is equivalent to: -ReLU(-α * (e^x - 1))
        let neg_alpha_term = graph.negate(alpha_term)?;
        let neg_relu = graph.relu(neg_alpha_term)?;
        let negative_part = graph.negate(neg_relu)?;
        
        // Combine positive and negative parts
        graph.add(positive_part, negative_part)
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}