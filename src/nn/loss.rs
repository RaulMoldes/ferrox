use crate::backend::numeric::Numeric;
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
/// TODO: Probably we should add Median or Log Reduction in the future. For now, we keep it simple.
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
/// Note that it is to be used with continuous targets, not categorical ones.
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

                // Divide by number of elements to get mean
                graph.mul_scalar(
                    sum_loss,
                    <T as Numeric>::from_f64(1.0 / total_elements as f64)
                        .ok_or("Failed to create mean divisor")?,
                )
            }
        }
    }
}


// TODO: Binary Cross Entropy Loss, Categorical Cross Entropy Loss. We currently do not have any implementation of loss functions for categorical targets.