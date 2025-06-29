use crate::backend::number::{CPUNumber, GPUFloat, GPUNumber};
use crate::graph::Engine;
use crate::graph::node::NodeId;
use crate::nn::Module;

/// Trait for all loss functions in the neural network module.
///
/// Loss functions compute the discrepancy between predicted and target values,
/// providing a scalar value that can be used for optimization. All loss functions
/// implement the Module trait to integrate seamlessly with the computation graph.
///
/// Similar to PyTorch's loss functions, this trait provides:
/// - Consistent interface for all loss computations
/// - Integration with automatic differentiation
/// - Support for different reduction strategies
/// - Batch processing capabilities
pub trait Loss<T>: Module<T>
where
    T: GPUNumber,
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
/// TODO: Probably we should add Median or Log Reduction in the future. For now, we keep it simple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reduction {
    /// No reduction: return the loss for each sample
    None,
    /// Mean reduction: return the average loss across all samples
    #[default]
    Mean,
    /// Sum reduction: return the sum of all losses
    Sum,
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
    T: GPUNumber
        + Clone
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand
        + rand_distr::num_traits::FromPrimitive,
{
    fn forward(&self, _graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
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
    T: GPUNumber
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
        // Shpw inputs

        // Compute difference: (predictions - targets)
        let negated = graph.negate(targets)?;
        if graph.get_shape(predictions) != graph.get_shape(negated) {
            return Err(format!(
                "Shape mismatch: predictions {:?} vs targets {:?}",
                graph.get_shape(predictions),
                graph.get_shape(negated)
            ));
        }
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

                // Divide by number of elements to get mean
                graph.mul_scalar(
                    sum_loss,
                    <T as CPUNumber>::from_f64(1.0 / total_elements as f64)
                        .ok_or("Failed to create mean divisor")?,
                )
            }
        }
    }
}

/// Binary Cross Entropy (BCE) Loss function.
///
/// Computes the binary cross entropy loss between predictions and binary targets.
/// This is the standard loss function for binary classification tasks where each sample
/// belongs to one of two classes (0 or 1).
///
/// # Mathematical Definition
///
/// For individual samples:
/// ```text
/// BCE(y_pred, y_true) = -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
/// ```
///
/// Where:
/// - y_pred: predicted probabilities (should be in range [0, 1])
/// - y_true: binary targets (0 or 1)
#[derive(Debug, Clone)]
pub struct BCELoss {
    /// Reduction strategy for aggregating batch losses
    reduction: Reduction,
    /// Small epsilon value for numerical stability (prevents log(0))
    eps: f64,
    training: bool,
}

impl BCELoss {
    /// Creates a new BCE loss with mean reduction and default epsilon.
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
            eps: 1e-8,
            training: true,
        }
    }

    /// Creates a new BCE loss with specified reduction strategy.
    ///
    /// # Arguments
    ///
    /// * `reduction` - How to reduce the loss across the batch
    pub fn new_with_reduction(reduction: Reduction) -> Self {
        Self {
            reduction,
            eps: 1e-8,
            training: true,
        }
    }

    /// Creates a new BCE loss with custom epsilon for numerical stability.
    ///
    /// # Arguments
    ///
    /// * `reduction` - How to reduce the loss across the batch
    /// * `eps` - Small value to prevent log(0), should be very small (e.g., 1e-8)
    pub fn new_with_eps(reduction: Reduction, eps: f64) -> Self {
        Self {
            reduction,
            eps,
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

    /// Returns the epsilon value used for numerical stability.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Sets the epsilon value for numerical stability.
    pub fn set_eps(&mut self, eps: f64) {
        self.eps = eps;
    }
}

impl Default for BCELoss {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for BCELoss
where
    T: GPUFloat,
{
    fn forward(&self, _graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
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

impl<T> Loss<T> for BCELoss
where
    T: GPUFloat,
{
    fn compute_loss(
        &self,
        graph: &mut Engine<T>,
        predictions: NodeId,
        targets: NodeId,
    ) -> Result<NodeId, String> {
        // Validate that predictions and targets have the same shape
        let pred_shape = graph.get_shape(predictions);
        let target_shape = graph.get_shape(targets);
        if pred_shape != target_shape {
            return Err(format!(
                "Shape mismatch: predictions {pred_shape:?} vs targets {target_shape:?}"
            ));
        }

        // Clamp predictions to [eps, 1-eps] for numerical stability using our new clamp operation
        let eps_val = <T as CPUNumber>::from_f64(self.eps)
            .ok_or("Failed to convert epsilon to tensor type")?;
        let one_minus_eps = <T as CPUNumber>::from_f64(1.0 - self.eps)
            .ok_or("Failed to convert 1-epsilon to tensor type")?;

        let clamped_preds = graph.clamp(predictions, eps_val, one_minus_eps)?;

        // Compute log(predictions)
        let log_preds = graph.log(clamped_preds)?;

        // Compute 1 - predictions
        let ones = graph.ones(&pred_shape, false);
        let neg_preds = graph.negate(clamped_preds)?;
        let one_minus_preds = graph.add(ones, neg_preds)?;

        // Clamp (1 - predictions) as well for stability
        let clamped_one_minus_preds = graph.clamp(one_minus_preds, eps_val, one_minus_eps)?;

        // Compute log(1 - predictions)
        let log_one_minus_preds = graph.log(clamped_one_minus_preds)?;

        // Compute 1 - targets
        let ones_target = graph.ones(&target_shape, false);
        let neg_targets = graph.negate(targets)?;
        let one_minus_targets = graph.add(ones_target, neg_targets)?;

        // Compute targets * log(predictions)
        let first_term = graph.mul(targets, log_preds)?;

        // Compute (1 - targets) * log(1 - predictions)
        let second_term = graph.mul(one_minus_targets, log_one_minus_preds)?;

        // Compute BCE = -(first_term + second_term)
        let bce_terms = graph.add(first_term, second_term)?;
        let bce_loss = graph.negate(bce_terms)?;

        // Apply reduction strategy
        match self.reduction {
            Reduction::None => {
                // Return individual losses
                Ok(bce_loss)
            }
            Reduction::Sum => {
                // Sum all elements
                graph.sum(bce_loss, None)
            }
            Reduction::Mean => {
                // Compute mean of all elements
                let sum_loss = graph.sum(bce_loss, None)?;

                // Get total number of elements for mean calculation
                let total_elements: usize = pred_shape.iter().product();

                // Divide by number of elements to get mean
                graph.mul_scalar(
                    sum_loss,
                    <T as CPUNumber>::from_f64(1.0 / total_elements as f64)
                        .ok_or("Failed to create mean divisor")?,
                )
            }
        }
    }
}
/// Categorical Cross Entropy (CCE) Loss function.
///
/// Computes the categorical cross entropy loss between predictions and categorical targets.
/// This is the standard loss function for multi-class classification tasks where each sample
/// belongs to exactly one of C classes.
///
/// # Mathematical Definition
///
/// For individual samples:
/// ```text
/// CCE(y_pred, y_true) = -Σᵢ y_true_i * log(y_pred_i)
/// ```
///
/// Where:
/// - y_pred: predicted probabilities (should sum to 1, typically from softmax)
/// - y_true: one-hot encoded targets or class indices
///
/// # Important Notes
///
/// - Predictions should be passed through a softmax activation before computing CCE
/// - For numerical stability, predictions are clamped to [eps, 1-eps] range
/// - This implementation supports both one-hot encoded targets and class indices
/// - For class indices, they are converted to one-hot encoding internally
#[derive(Debug, Clone)]
pub struct CCELoss {
    /// Reduction strategy for aggregating batch losses
    reduction: Reduction,
    /// Small epsilon value for numerical stability (prevents log(0))
    eps: f64,
    /// Whether targets are provided as class indices (true) or one-hot vectors (false)
    from_indices: bool,
    training: bool,
}

impl CCELoss {
    /// Creates a new CCE loss with mean reduction and default epsilon.
    /// Assumes targets are one-hot encoded.
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
            eps: 1e-8,
            from_indices: false,
            training: true,
        }
    }

    /// Creates a new CCE loss with specified reduction strategy.
    /// Assumes targets are one-hot encoded.
    ///
    /// # Arguments
    ///
    /// * `reduction` - How to reduce the loss across the batch
    pub fn new_with_reduction(reduction: Reduction) -> Self {
        Self {
            reduction,
            eps: 1e-8,
            from_indices: false,
            training: true,
        }
    }

    /// Creates a new CCE loss for class indices.
    /// Use this when targets are provided as class indices (0, 1, 2, ...) instead of one-hot vectors.
    ///
    /// # Arguments
    ///
    /// * `reduction` - How to reduce the loss across the batch
    pub fn new_from_indices(reduction: Reduction) -> Self {
        Self {
            reduction,
            eps: 1e-8,
            from_indices: true,
            training: true,
        }
    }

    /// Creates a new CCE loss with custom epsilon for numerical stability.
    ///
    /// # Arguments
    ///
    /// * `reduction` - How to reduce the loss across the batch
    /// * `eps` - Small value to prevent log(0), should be very small (e.g., 1e-8)
    /// * `from_indices` - Whether targets are class indices (true) or one-hot encoded (false)
    pub fn new_with_eps(reduction: Reduction, eps: f64, from_indices: bool) -> Self {
        Self {
            reduction,
            eps,
            from_indices,
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

    /// Returns the epsilon value used for numerical stability.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Sets the epsilon value for numerical stability.
    pub fn set_eps(&mut self, eps: f64) {
        self.eps = eps;
    }

    /// Returns whether the loss expects targets as class indices.
    pub fn from_indices(&self) -> bool {
        self.from_indices
    }

    /// Sets whether targets are provided as class indices.
    pub fn set_from_indices(&mut self, from_indices: bool) {
        self.from_indices = from_indices;
    }
}

impl Default for CCELoss {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for CCELoss
where
    T: GPUFloat,
{
    fn forward(&self, _graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
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

impl<T> Loss<T> for CCELoss
where
    T: GPUFloat,
{
    fn compute_loss(
        &self,
        graph: &mut Engine<T>,
        predictions: NodeId,
        targets: NodeId,
    ) -> Result<NodeId, String> {
        let pred_shape = graph.get_shape(predictions);
        let target_shape = graph.get_shape(targets);

        // Handle class indices vs one-hot encoded targets
        let one_hot_targets = if self.from_indices {
            // Convert class indices to one-hot encoding
            // For now, we'll assume targets are already in the correct format
            // TODO: Implement proper index to one-hot conversion
            if target_shape.len() != pred_shape.len() - 1 {
                return Err("When using from_indices=true, target shape should be [batch_size] and prediction shape should be [batch_size, num_classes]".to_string());
            }
            // For simplicity, we'll require pre-converted one-hot targets for now
            return Err("Index to one-hot conversion not yet implemented. Please provide one-hot encoded targets.".to_string());
        } else {
            // Validate shapes for one-hot targets
            if pred_shape != target_shape {
                return Err(format!(
                    "Shape mismatch: predictions {pred_shape:?} vs targets {target_shape:?}"
                ));
            }
            targets
        };

        // Clamp predictions for numerical stability using our new clamp operation
        let eps_val = <T as CPUNumber>::from_f64(self.eps)
            .ok_or("Failed to convert epsilon to tensor type")?;
        let one_minus_eps = <T as CPUNumber>::from_f64(1.0 - self.eps)
            .ok_or("Failed to convert 1-epsilon to tensor type")?;

        let clamped_preds = graph.clamp(predictions, eps_val, one_minus_eps)?;

        // Compute log(predictions)
        let log_preds = graph.log(clamped_preds)?;

        // Compute element-wise product: targets * log(predictions)
        let products = graph.mul(one_hot_targets, log_preds)?;

        // Sum over the class dimension (last dimension)
        // For shape [batch_size, num_classes], we want to sum over the last dimension
        let class_dim = pred_shape.len() - 1;
        let per_sample_loss = graph.summation(products, Some(vec![class_dim]))?;

        // Negate to get the cross entropy loss
        let cce_loss = graph.negate(per_sample_loss)?;

        // Apply reduction strategy
        match self.reduction {
            Reduction::None => {
                // Return individual losses for each sample
                Ok(cce_loss)
            }
            Reduction::Sum => {
                // Sum all sample losses
                graph.sum(cce_loss, None)
            }
            Reduction::Mean => {
                // Compute mean of all sample losses
                let sum_loss = graph.sum(cce_loss, None)?;

                // Get batch size for mean calculation
                let batch_size = pred_shape[0];

                // Divide by batch size to get mean
                graph.mul_scalar(
                    sum_loss,
                    <T as CPUNumber>::from_f64(1.0 / batch_size as f64)
                        .ok_or("Failed to create mean divisor")?,
                )
            }
        }
    }
}
/// Binary Cross Entropy with Logits Loss function.
///
/// This is a CPUNumberally stable version of Binary Cross Entropy that combines
/// sigmoid activation and BCE loss in a single operation. This is more stable
/// than applying sigmoid followed by BCE separately.
///
/// # Mathematical Definition
///
/// ```text
/// BCE_with_logits(logits, targets) = max(logits, 0) - logits * targets + log(1 + exp(-abs(logits)))
/// ```
///
/// This formulation avoids CPUNumberal issues with very large or small logits.
#[derive(Debug, Clone)]
pub struct BCEWithLogitsLoss {
    /// Reduction strategy for aggregating batch losses
    reduction: Reduction,
    training: bool,
}

impl BCEWithLogitsLoss {
    /// Creates a new BCE with logits loss with mean reduction.
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
            training: true,
        }
    }

    /// Creates a new BCE with logits loss with specified reduction strategy.
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

impl Default for BCEWithLogitsLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Module<T> for BCEWithLogitsLoss
where
    T: GPUFloat,
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

impl<T> Loss<T> for BCEWithLogitsLoss
where
    T: GPUFloat,
{
    fn compute_loss(
        &self,
        graph: &mut Engine<T>,
        logits: NodeId,
        targets: NodeId,
    ) -> Result<NodeId, String> {
        let logits_shape = graph.get_shape(logits);
        let target_shape = graph.get_shape(targets);

        if logits_shape != target_shape {
            return Err(format!(
                "Shape mismatch: logits {logits_shape:?} vs targets {target_shape:?}"
            ));
        }

        // Implement the CPUNumberally stable formula:
        // loss = max(logits, 0) - logits * targets + log(1 + exp(-abs(logits)))

        // Compute max(logits, 0) using ReLU
        let relu_logits = graph.relu(logits)?;

        // Compute logits * targets
        let logits_times_targets = graph.mul(logits, targets)?;

        // Compute abs(logits) using our new abs operation
        let abs_logits = graph.abs(logits)?;

        // Compute -abs(logits)
        let neg_abs_logits = graph.negate(abs_logits)?;

        // Compute exp(-abs(logits))
        let exp_neg_abs_logits = graph.exp(neg_abs_logits)?;

        // Compute 1 + exp(-abs(logits))
        let ones = graph.ones(&logits_shape, false);
        let one_plus_exp = graph.add(ones, exp_neg_abs_logits)?;

        // Compute log(1 + exp(-abs(logits)))
        let log_one_plus_exp = graph.log(one_plus_exp)?;

        // Combine terms: max(logits, 0) - logits * targets + log(1 + exp(-abs(logits)))
        let neg_logits_times_targets = graph.negate(logits_times_targets)?;
        let first_part = graph.add(relu_logits, neg_logits_times_targets)?;
        let bce_loss = graph.add(first_part, log_one_plus_exp)?;

        // Apply reduction strategy
        match self.reduction {
            Reduction::None => Ok(bce_loss),
            Reduction::Sum => graph.sum(bce_loss, None),
            Reduction::Mean => {
                let sum_loss = graph.sum(bce_loss, None)?;
                let total_elements: usize = logits_shape.iter().product();
                graph.mul_scalar(
                    sum_loss,
                    <T as CPUNumber>::from_f64(1.0 / total_elements as f64)
                        .ok_or("Failed to create mean divisor")?,
                )
            }
        }
    }
}
