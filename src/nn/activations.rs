use crate::backend::number::{CPUNumber, FerroxCudaF, FerroxF};
use crate::graph::Engine;
use crate::graph::node::NodeId;
use crate::nn::Module;

// ReLU (Rectified Linear Unit) activation function.
//
// Applies the function ReLU(x) = max(0, x) element-wise.
// This is one of the most commonly used activation functions in deep learning.
// The expected output is to be zero for all negative inputs and equal to the input for all positive inputs.
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
    T: FerroxCudaF,
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

// Sigmoid activation function.
//
// Applies the sigmoid function element-wise: σ(x) = 1 / (1 + e^(-x))
// Maps any real number to a value between 0 and 1, making it useful for binary classification.
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
    T: FerroxCudaF,
{
    fn forward(&self, graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
        // Sigmoid(x) = 1 / (1 + exp(-x))
        // We need to implement this using available operations in the graph

        // First, compute -x
        let neg_input = graph.negate(input)?;

        // Then compute exp(-x)
        let exp_neg_input = graph.exp(neg_input)?;

        // Add 1 to get (1 + exp(-x))
        let one_plus_exp = graph.add_scalar(
            exp_neg_input,
            <T as crate::backend::number::CPUNumber>::one(),
        )?;

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

// Tanh (Hyperbolic Tangent) activation function.
//
// Applies the hyperbolic tangent function element-wise: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
// Maps any real number to a value between -1 and 1, making it zero-centered.
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
    T: FerroxCudaF,
{
    fn forward(&self, graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
        // Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        // We can implement this more efficiently as: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)

        // Compute 2x
        let two_x = graph.mul_scalar(input, <T as CPUNumber>::from_f64(2.0).unwrap())?;

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

/// I decided to implement LeakyReLU and ELU activation functions as well.
/// Their standard implementation requires the conditional operation,
/// As my `Engine` does not support this, I will implement them on top the ReLU function.
///
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
/// - Allows gradients to flow even for negative inputs, helping with faster convergence in some cases.
#[derive(Debug, Clone)]
pub struct LeakyReLU<T>
where
    T: FerroxCudaF,
{
    /// The negative slope parameter
    negative_slope: T,
    training: bool,
}

impl<T> LeakyReLU<T>
where
    T: FerroxCudaF,
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
    T: FerroxCudaF + From<f64>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for LeakyReLU<T>
where
    T: FerroxCudaF,
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
/// Applies the ELU function element-wise. ELU has the benefit over ReLU and LeakyReLU
/// of taking negative values
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
/// Where α is a hyperparameter that controls the saturation point for negative inputs.
/// The idea is to allow the function to have a non-zero output for negative inputs,
/// which helps to reduce the bias shift effect during training.
/// Check this conversation on why this happens: https://www.quora.com/Why-do-non-zero-mean-activations-induce-a-bias-shift-for-units-in-the-next-layer-and-why-is-that-bad
/// Also take a look at my `initialiers.rs` module.
///
/// Another advantage of ELU is that it is differentiable everywhere, including at zero,
/// which does not occur with standard ReLU.
///
/// Negative values of the ELU function push mean activation closer to zero
/// ELU saturates to -α for large negative inputs a.k.a. lim[f(x)] where x -> -∞ is -α
///
///
#[derive(Debug, Clone)]
pub struct ELU<T>
where
    T: FerroxCudaF,
{
    /// The α parameter for negative inputs
    alpha: T,
    training: bool,
}

impl<T> ELU<T>
where
    T: FerroxCudaF,
{
    /// Creates a new ELU activation layer with default α = 1.0.
    pub fn new() -> Self
    where
        T: FerroxCudaF,
    {
        Self {
            alpha: FerroxF::from_f64(1.0).unwrap(),
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
    T: FerroxCudaF,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for ELU<T>
where
    T: FerroxCudaF,
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

/// Softmax activation function.
///
/// Applies the softmax function along a specified dimension. Softmax is commonly used
/// as the final activation function in multi-class classification problems as it
/// converts logits to a probability distribution where all values sum to 1.
///
/// # Mathematical Definition
///
/// For a vector x and dimension d:
/// ```text
/// softmax(x_i) = exp(x_i - max(x)) / Σⱼ exp(x_j - max(x))
/// ```
///
/// The subtraction of max(x) is for numerical stability to prevent overflow.
/// This is known as the "stable softmax" implementation.
///
/// # Properties
///
/// - All outputs are in the range (0, 1)
/// - The sum of all outputs along the specified dimension equals 1
/// - The function is differentiable everywhere
/// - Preserves the relative ordering of inputs (monotonic)
///
/// # Usage
///
/// Typically used as the final layer in classification networks:
/// ```text
/// logits -> Softmax -> probabilities
/// ```
///
/// # numerical stability
///
/// This implementation uses the CPUNumberally stable version that subtracts
/// the maximum value before exponentiation to prevent overflow with large inputs.
#[derive(Debug, Clone)]
pub struct Softmax {
    /// Dimension along which to apply softmax
    /// For typical use cases:
    /// - 1 for shape [batch_size, num_classes] (most common)
    /// - -1 for last dimension (equivalent to 1 in the above case)
    dim: i32,
    training: bool,
}

impl Softmax {
    /// Creates a new Softmax activation layer.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension along which to apply softmax (default: 1 for [batch, classes])
    pub fn new(dim: i32) -> Self {
        Self {
            dim,
            training: true,
        }
    }

    /// Creates a new Softmax with default dimension 1.
    /// This is the most common case for classification where input is [batch_size, num_classes].
    pub fn default_dim() -> Self {
        Self::new(1)
    }

    /// Returns the dimension along which softmax is applied.
    pub fn dim(&self) -> i32 {
        self.dim
    }

    /// Sets the dimension along which softmax is applied.
    pub fn set_dim(&mut self, dim: i32) {
        self.dim = dim;
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::default_dim()
    }
}

impl<T> Module<T> for Softmax
where
    T: FerroxCudaF,
{
    fn forward(&self, graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
        // Get input shape to validate dimension
        let input_shape = graph.get_shape(input);
        let ndim = input_shape.len() as i32;

        // Convert negative dimension to positive
        let dim = if self.dim < 0 {
            (ndim + self.dim) as usize
        } else {
            self.dim as usize
        };

        // Validate dimension is within bounds
        if dim >= input_shape.len() {
            return Err(format!(
                "Softmax dimension {} is out of bounds for tensor with {} dimensions",
                self.dim,
                input_shape.len()
            ));
        }

        // Step 1: Subtract max for numerical stability
        // Find maximum along the specified dimension
        let max_vals = graph.max_along_dim(input, dim)?;

        // Expand max_vals to match input shape for broadcasting
        let expanded_max = graph.expand_dims_at(max_vals, dim)?;
        let broadcasted_max = graph.broadcast_to(expanded_max, input_shape.clone())?;

        // Subtract max from input: x - max(x)
        let neg_max = graph.negate(broadcasted_max)?;
        let shifted_input = graph.add(input, neg_max)?;

        // Step 2: Compute exponentials
        let exp_vals = graph.exp(shifted_input)?;

        // Step 3: Sum exponentials along the specified dimension
        let sum_exp = graph.sum_along_dim(exp_vals, dim)?;

        // Step 4: Expand sum to match input shape for broadcasting
        let expanded_sum = graph.expand_dims_at(sum_exp, dim)?;
        let broadcasted_sum = graph.broadcast_to(expanded_sum, input_shape)?;

        // Step 5: Divide exponentials by their sum
        let softmax_output = graph.div(exp_vals, broadcasted_sum)?;

        Ok(softmax_output)
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}
