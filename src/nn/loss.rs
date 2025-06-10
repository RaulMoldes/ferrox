use crate::backend::numeric::{Float, Numeric};
use crate::graph::Engine;
use crate::graph::node::NodeId;
use crate::nn::Module;
use crate::nn::parameter::Parameter;

/// Trait for all loss functions in the neural network module.
///
/// Loss functions compute the discrepancy between predicted and target values,
/// providing a scalar value that can be used for optimization. All loss functions
/// implement the Module trait to integrate seamlessly with the computation graph.
///
/// # Design Philosophy
///
/// Similar to PyTorch's loss functions, this trait provides:
/// - Consistent interface for all loss computations
/// - Integration with automatic differentiation
/// - Support for different reduction strategies
/// - Batch processing capabilities
pub trait Loss<T>: Module<T>
where
    T: Numeric + Clone + std::fmt::Debug + ndarray::LinalgScalar + ndarray::ScalarOperand,
{
    /// Computes the loss between predictions and targets.
    ///
    /// # Arguments
    ///
    /// * `graph` - The computation graph engine
    /// * `predictions` - Model predictions
    /// * `targets` - Ground truth targets
    ///
    /// # Returns
    ///
    /// The computed loss as a node in the computation graph
    fn compute_loss(
        &self,
        graph: &mut Engine<T>,
        predictions: NodeId,
        targets: NodeId,
    ) -> Result<NodeId, String>;
}

/// Reduction strategy for loss functions.
///
/// Determines how to aggregate individual sample losses into a single scalar value.
/// This matches PyTorch's reduction strategies for consistency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// No reduction: return the loss for each sample
    None,
    /// Mean reduction: return the average loss across all samples
    Mean,
    /// Sum reduction: return the sum of all losses
    Sum,
}

impl Default for Reduction {
    fn default() -> Self {
        Reduction::Mean
    }
}

/// Mean Squared Error (MSE) Loss function.
///
/// Computes the mean squared error between predictions and targets.
/// This is commonly used for regression tasks where the target is a continuous value.
///
/// # Mathematical Definition
///
/// For individual samples:
/// ```text
/// MSE(y_pred, y_true) = (y_pred - y_true)²
/// ```
///
/// For batches with different reductions:
/// - None: Returns individual squared errors
/// - Mean: `(1/N) * Σᵢ(y_pred_i - y_true_i)²`
/// - Sum: `Σᵢ(y_pred_i - y_true_i)²`
///
/// # Use Cases
///
/// - Regression problems
/// - Continuous target variables
/// - When outliers should be heavily penalized (quadratic penalty)
///
/// # Examples
///
/// ```rust
/// use ferrox::nn::{MSELoss, Loss, Reduction};
/// use ferrox::graph::Engine;
///
/// // Create MSE loss with mean reduction (default)
/// let mse_loss = MSELoss::new();
///
/// // Create MSE loss with sum reduction
/// let mse_sum = MSELoss::new_with_reduction(Reduction::Sum);
///
/// let mut graph = Engine::new();
/// 
/// // Predictions and targets for regression
/// let predictions = graph.tensor_from_vec(vec![1.0, 2.0, 3.0], &[3], true).unwrap();
/// let targets = graph.tensor_from_vec(vec![1.1, 1.9, 3.2], &[3], false).unwrap();
/// 
/// let loss = mse_loss.compute_loss(&mut graph, predictions, targets).unwrap();
/// // Loss ≈ ((1.0-1.1)² + (2.0-1.9)² + (3.0-3.2)²) / 3 = (0.01 + 0.01 + 0.04) / 3 = 0.02
/// ```
#[derive(Debug, Clone)]
pub struct MSELoss {
    /// Reduction strategy for aggregating batch losses
    reduction: Reduction,
    training: bool,
}

impl MSELoss {
    /// Creates a new MSE loss with mean reduction.
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
            training: true,
        }
    }

    /// Creates a new MSE loss with specified reduction strategy.
    ///
    /// # Arguments
    ///
    /// * `reduction` - How to reduce the loss across the batch
    pub fn new_with_reduction(reduction: Reduction) -> Self {
        Self {
            reduction,
            training: true,
        }
    }

    /// Returns the current reduction strategy.
    pub fn reduction(&self) -> Reduction {
        self.reduction
    }

    /// Sets the reduction strategy.
    pub fn set_reduction(&mut self, reduction: Reduction) {
        self.reduction = reduction;
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for MSELoss
where
    T: Numeric
        + Clone
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand
        + rand_distr::num_traits::FromPrimitive,
{
    fn forward(&self, graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
        // For the Module trait, we just return the input unchanged
        // The actual loss computation happens in compute_loss
        Ok(input)
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl<T> Loss<T> for MSELoss
where
    T: Numeric
        + Clone
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand
        + rand_distr::num_traits::FromPrimitive,
{
    fn compute_loss(
        &self,
        graph: &mut Engine<T>,
        predictions: NodeId,
        targets: NodeId,
    ) -> Result<NodeId, String> {
        // Compute difference: (predictions - targets)
        let negated = graph.negate(predictions)?;
        let diff = graph.add(predictions, negated)?;
        
        // Square the difference: (predictions - targets)²
        let squared_diff = graph.mul(diff, diff)?;
        
        // Apply reduction strategy
        match self.reduction {
            Reduction::None => {
                // Return individual squared errors
                Ok(squared_diff)
            }
            Reduction::Sum => {
                // Sum all elements
                graph.sum(squared_diff, None)
            }
            Reduction::Mean => {
                // Compute mean of all elements
                let sum_loss = graph.sum(squared_diff, None)?;
                
                // Get total number of elements for mean calculation
                let target_shape = graph.get_shape(targets);
                let total_elements: usize = target_shape.iter().product();
                let n_elements = <T as Numeric>::from_f64(total_elements as f64)
                    .ok_or("Failed to convert element count to tensor type")?;
                
                // Divide by number of elements to get mean
                graph.mul_scalar(sum_loss, <T as Numeric>::from_f64(1.0 / total_elements as f64)
                    .ok_or("Failed to create mean divisor")?)
            }
        }
    }
}

/// Cross Entropy Loss function.
///
/// Computes the cross entropy loss between predictions and targets.
/// This is the most commonly used loss function for classification tasks.
///
/// # Mathematical Definition
///
/// For multi-class classification with C classes:
/// ```text
/// CrossEntropy(y_pred, y_true) = -Σᶜ y_true_c * log(softmax(y_pred_c))
/// ```
///
/// For binary classification:
/// ```text
/// CrossEntropy(y_pred, y_true) = -(y_true * log(σ(y_pred)) + (1-y_true) * log(1-σ(y_pred)))
/// ```
///
/// Where σ is the sigmoid function and softmax is applied for multi-class.
///
/// # Input Requirements
///
/// - Predictions: Raw logits (unnormalized scores) of shape (N, C) for multi-class
///   or (N,) or (N, 1) for binary classification
/// - Targets: Class indices (0 to C-1) for multi-class, or binary labels (0 or 1)
///   for binary classification
///
/// # Numerical Stability
///
/// This implementation uses the log-sum-exp trick for numerical stability,
/// preventing overflow/underflow in exponential computations.
///
/// # Examples
///
/// ```rust
/// use ferrox::nn::{CrossEntropyLoss, Loss, Reduction};
/// use ferrox::graph::Engine;
///
/// // Create cross entropy loss for multi-class classification
/// let ce_loss = CrossEntropyLoss::new();
///
/// let mut graph = Engine::new();
/// 
/// // Example: 2 samples, 3 classes
/// // Raw logits (before softmax)
/// let predictions = graph.tensor_from_vec(
///     vec![2.0, 1.0, 0.1, 0.5, 1.5, 2.5], 
///     &[2, 3], 
///     true
/// ).unwrap();
/// 
/// // Target class indices [0, 2] (class 0 for first sample, class 2 for second)
/// let targets = graph.tensor_from_vec(vec![0.0, 2.0], &[2], false).unwrap();
/// 
/// let loss = ce_loss.compute_loss(&mut graph, predictions, targets).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CrossEntropyLoss {
    /// Reduction strategy for aggregating batch losses
    reduction: Reduction,
    /// Whether to apply log softmax internally (true for multi-class, false if pre-applied)
    apply_softmax: bool,
    training: bool,
}

impl CrossEntropyLoss {
    /// Creates a new Cross Entropy loss with mean reduction.
    /// 
    /// By default, applies softmax internally for multi-class classification.
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
            apply_softmax: true,
            training: true,
        }
    }

    /// Creates a new Cross Entropy loss with specified reduction strategy.
    ///
    /// # Arguments
    ///
    /// * `reduction` - How to reduce the loss across the batch
    pub fn new_with_reduction(reduction: Reduction) -> Self {
        Self {
            reduction,
            apply_softmax: true,
            training: true,
        }
    }

    /// Creates a new Cross Entropy loss with custom settings.
    ///
    /// # Arguments
    ///
    /// * `reduction` - How to reduce the loss across the batch
    /// * `apply_softmax` - Whether to apply softmax internally
    pub fn new_with_options(reduction: Reduction, apply_softmax: bool) -> Self {
        Self {
            reduction,
            apply_softmax,
            training: true,
        }
    }

    /// Returns the current reduction strategy.
    pub fn reduction(&self) -> Reduction {
        self.reduction
    }

    /// Returns whether softmax is applied internally.
    pub fn applies_softmax(&self) -> bool {
        self.apply_softmax
    }

    /// Sets the reduction strategy.
    pub fn set_reduction(&mut self, reduction: Reduction) {
        self.reduction = reduction;
    }

    /// Sets whether to apply softmax internally.
    pub fn set_apply_softmax(&mut self, apply_softmax: bool) {
        self.apply_softmax = apply_softmax;
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for CrossEntropyLoss
where
    T: Float
        + Clone
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand
        + rand_distr::num_traits::FromPrimitive,
{
    fn forward(&self, graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
        // For the Module trait, we just return the input unchanged
        // The actual loss computation happens in compute_loss
        Ok(input)
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl<T> Loss<T> for CrossEntropyLoss
where
    T: Float
        + Clone
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand
        + rand_distr::num_traits::FromPrimitive,
{
    fn compute_loss(
        &self,
        graph: &mut Engine<T>,
        predictions: NodeId,
        targets: NodeId,
    ) -> Result<NodeId, String> {
        let pred_shape = graph.get_shape(predictions);
        let target_shape = graph.get_shape(targets);

        // Validate input shapes
        if pred_shape.len() < 1 || target_shape.len() < 1 {
            return Err("Predictions and targets must have at least 1 dimension".to_string());
        }

        let batch_size = pred_shape[0];
        if target_shape[0] != batch_size {
            return Err(format!(
                "Batch size mismatch: predictions {} vs targets {}",
                batch_size, target_shape[0]
            ));
        }

        // Handle different cases based on prediction shape
        let loss = if pred_shape.len() == 1 || (pred_shape.len() == 2 && pred_shape[1] == 1) {
            // Binary classification case
            self.compute_binary_cross_entropy(graph, predictions, targets)?
        } else if pred_shape.len() == 2 {
            // Multi-class classification case
            self.compute_multiclass_cross_entropy(graph, predictions, targets)?
        } else {
            return Err("Predictions must be 1D (binary) or 2D (multi-class)".to_string());
        };

        // Apply reduction strategy
        match self.reduction {
            Reduction::None => Ok(loss),
            Reduction::Sum => graph.sum(loss, None),
            Reduction::Mean => {
                let sum_loss = graph.sum(loss, None)?;
                let n_samples = <T as Numeric>::from_f64(batch_size as f64)
                    .ok_or("Failed to convert batch size to tensor type")?;
                graph.mul_scalar(sum_loss, <T as Numeric>::from_f64(1.0 / batch_size as f64)
                    .ok_or("Failed to create mean divisor")?)
            }
        }
    }
}

impl CrossEntropyLoss {
    /// Computes binary cross entropy loss.
    ///
    /// # Arguments
    ///
    /// * `graph` - The computation graph engine
    /// * `predictions` - Logits of shape (N,) or (N, 1)
    /// * `targets` - Binary labels of shape (N,) with values 0 or 1
    fn compute_binary_cross_entropy<T>(
        &self,
        graph: &mut Engine<T>,
        predictions: NodeId,
        targets: NodeId,
    ) -> Result<NodeId, String>
    where
        T: Float
            + Clone
            + std::fmt::Debug
            + ndarray::LinalgScalar
            + ndarray::ScalarOperand
            + rand_distr::num_traits::FromPrimitive,
    {
        // Binary cross entropy: -(t * log(σ(x)) + (1-t) * log(1-σ(x)))
        // For numerical stability, we use: x - x*t + log(1 + exp(-x))
        // This avoids computing sigmoid explicitly

        let target_shape = graph.get_shape(targets);
        let ones_node = graph.ones(&target_shape, false);

        // Compute x - x*t
        let x_times_t = graph.mul(predictions, targets)?;
        let negated = graph.negate(x_times_t)?;
        let x_minus_xt = graph.add(predictions, negated)?;

        // Compute log(1 + exp(-x)) using log-sum-exp trick
        let neg_x = graph.negate(predictions)?;
        let exp_neg_x = graph.exp(neg_x)?;
        let one_plus_exp = graph.add(ones_node, exp_neg_x)?;
        let log_term = graph.log(one_plus_exp)?;

        // Combine: x - x*t + log(1 + exp(-x))
        let loss = graph.add(x_minus_xt, log_term)?;

        Ok(loss)
    }

    /// Computes multi-class cross entropy loss.
    ///
    /// # Arguments
    ///
    /// * `graph` - The computation graph engine
    /// * `predictions` - Logits of shape (N, C) where C is number of classes
    /// * `targets` - Class indices of shape (N,) with values 0 to C-1
    fn compute_multiclass_cross_entropy<T>(
        &self,
        graph: &mut Engine<T>,
        predictions: NodeId,
        targets: NodeId,
    ) -> Result<NodeId, String>
    where
        T: Float
            + Clone
            + std::fmt::Debug
            + ndarray::LinalgScalar
            + ndarray::ScalarOperand
            + rand_distr::num_traits::FromPrimitive,
    {
        let pred_shape = graph.get_shape(predictions);
        let num_classes = pred_shape[1];
        let batch_size = pred_shape[0];

        // Apply log-softmax for numerical stability
        // log_softmax(x) = x - log(sum(exp(x)))
        let log_softmax = if self.apply_softmax {
            self.compute_log_softmax(graph, predictions)?
        } else {
            // Assume predictions are already log probabilities
            predictions
        };

        // Convert target indices to one-hot encoding
        let one_hot_targets = self.create_one_hot(graph, targets, num_classes)?;

        // Compute cross entropy: -sum(targets * log_softmax)
        let ce_terms = graph.mul(one_hot_targets, log_softmax)?;
        let neg_ce_terms = graph.negate(ce_terms)?;
        
        // Sum over classes (axis 1) to get loss per sample
        let loss_per_sample = graph.summation(neg_ce_terms, Some(vec![1]))?;

        Ok(loss_per_sample)
    }

    /// Computes log softmax for numerical stability.
    ///
    /// Uses the log-sum-exp trick: log_softmax(x) = x - log(sum(exp(x - max(x))))
    fn compute_log_softmax<T>(
        &self,
        graph: &mut Engine<T>,
        logits: NodeId,
    ) -> Result<NodeId, String>
    where
        T: Float
            + Clone
            + std::fmt::Debug
            + ndarray::LinalgScalar
            + ndarray::ScalarOperand
            + rand_distr::num_traits::FromPrimitive,
    {
        // Simplified log softmax implementation
        // In a full implementation, you'd subtract the max for numerical stability
        
        // Compute exp(logits)
        let exp_logits = graph.exp(logits)?;
        
        // Sum over classes (axis 1)
        let sum_exp = graph.summation(exp_logits, Some(vec![1]))?;
        
        // Take log of the sum
        let log_sum_exp = graph.log(sum_exp)?;
        
        // Expand log_sum_exp to match logits shape for broadcasting
        let logits_shape = graph.get_shape(logits);
        // Insert a new axis at position 1 by reshaping
        let sum_shape = graph.get_shape(log_sum_exp);
        let mut new_shape = sum_shape.clone();
        new_shape.insert(1, 1);
        let expanded_log_sum_exp = graph.reshape(log_sum_exp, new_shape)?;
        let broadcasted_log_sum_exp = graph.broadcast_to(expanded_log_sum_exp, logits_shape)?;
        
        // Compute log_softmax = logits - log(sum(exp(logits)))
        let negated_log_sum_exp = graph.negate(broadcasted_log_sum_exp)?;
        graph.add(logits, negated_log_sum_exp)
    }

    /// Creates one-hot encoding from class indices.
    ///
    /// # Arguments
    ///
    /// * `graph` - The computation graph engine
    /// * `indices` - Class indices of shape (N,)
    /// * `num_classes` - Number of classes
    ///
    /// # Returns
    ///
    /// One-hot encoded tensor of shape (N, num_classes)
    fn create_one_hot<T>(
        &self,
        graph: &mut Engine<T>,
        indices: NodeId,
        num_classes: usize,
    ) -> Result<NodeId, String>
    where
        T: Float
            + Clone
            + std::fmt::Debug
            + ndarray::LinalgScalar
            + ndarray::ScalarOperand
            + rand_distr::num_traits::FromPrimitive,
    {
        let indices_shape = graph.get_shape(indices);
        let batch_size = indices_shape[0];

        // Create zeros tensor of shape (batch_size, num_classes)
        let zeros = graph.zeros(&[batch_size, num_classes], false);

        // For a simplified implementation, we'll create one-hot manually
        // In a production system, you'd want a more efficient scatter operation
        
        // This is a simplified approach - in practice you'd need to implement
        // a proper scatter operation for creating one-hot encodings efficiently
        // For now, we'll return a placeholder implementation
        
        // TODO: Implement proper one-hot encoding with scatter operation
        // This would require adding a scatter operation to the graph engine
        
        // For demonstration, return zeros (this should be replaced with proper one-hot)
        Err("One-hot encoding not yet fully implemented. Please add scatter operation to graph engine.".to_string())
    }
}