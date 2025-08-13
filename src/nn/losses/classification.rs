// src/nn/losses/classification.rs
// Classification loss functions implemented using computational graph operations
// BCE and CCE losses for binary and multiclass classification tasks

use crate::backend::FerroxCudaF;
use crate::graph::{AutoFerroxEngine, NodeId};
use crate::nn::losses::{Loss, ReductionType};
use crate::ops::{
    basic::{Add, Mul},
    comparison::Clamp,
    reduction::{Mean, Sum},
    scalar::AddScalar,
    unary::{Log, Softmax, Neg},
};
use crate::FerroxN;
use std::marker::PhantomData;

/// Binary Cross Entropy Loss: BCE = -mean(targets * log(predictions) + (1-targets) * log(1-predictions))
/// Used for binary classification where targets are in [0, 1] and predictions are probabilities
#[derive(Debug, Clone)]
pub struct BCELoss<T> {
    reduction: ReductionType,
    _phantom: PhantomData<T>,
}

impl<T> BCELoss<T>
where
    T: FerroxCudaF,
{
    pub fn new(reduction: ReductionType) -> Self {
        Self {
            reduction,
            _phantom: PhantomData,
        }
    }
}

impl<T> Default for BCELoss<T>
where
    T: FerroxCudaF,
{
    fn default() -> Self {
        Self::new(ReductionType::Mean)
    }
}

impl<T> Loss<T> for BCELoss<T>
where
    T: FerroxCudaF,
{
    fn forward(
        &self,
        graph: &mut AutoFerroxEngine<T>,
        predictions: NodeId,
        targets: NodeId,
    ) -> Result<NodeId, String> {
        // BCE = -[targets * log(predictions) + (1-targets) * log(1-predictions)]
        // Add epsilon for numerical stability to avoid log(0)

        // Step 1: Clip predictions to [epsilon, 1-epsilon] for numerical stability
        let epsilon = FerroxN::from_f64(1e-7).unwrap();
        let one_minus_eps = FerroxN::from_f64(1.0 - 1e-7).unwrap();
        let clamp_op = Box::new(Clamp::new(epsilon, one_minus_eps));
        let clipped_pred = graph
            .apply_operation(clamp_op, vec![predictions])
            .map_err(|e| format!("BCE clipping failed: {}", e))?;

        // Step 2: Compute log(clipped_predictions)
        let log_op = Box::new(Log);
        let log_pred = graph
            .apply_operation(log_op, vec![clipped_pred])
            .map_err(|e| format!("BCE log(predictions) failed: {}", e))?;

        // Step 2: Compute targets * log(predictions)
        let mul_op1 = Box::new(Mul);
        let targets_log_pred = graph
            .apply_operation(mul_op1, vec![targets, log_pred])
            .map_err(|e| format!("BCE targets * log(pred) failed: {}", e))?;

        // Step 3: Compute (1 - targets) - subtract targets from scalar 1
        let one = FerroxN::one();
        let add_scalar_op = Box::new(AddScalar::new(one));
        let neg_op = Box::new(Neg);
        let neg_targets = graph
            .apply_operation(neg_op, vec![targets])
            .map_err(|e| format!("BCE negate targets failed: {}", e))?;
        let one_minus_targets = graph
            .apply_operation(add_scalar_op, vec![neg_targets])
            .map_err(|e| format!("BCE (1-targets) computation failed: {}", e))?;

        // Step 4: Compute (1 - predictions) with clipping
        let one = FerroxN::one();
        let add_scalar_op2 = Box::new(AddScalar::new(one));
        let neg_op2 = Box::new(Neg);
        let neg_clipped_pred = graph
            .apply_operation(neg_op2, vec![clipped_pred])
            .map_err(|e| format!("BCE negate predictions failed: {}", e))?;
        let one_minus_pred = graph
            .apply_operation(add_scalar_op2, vec![neg_clipped_pred])
            .map_err(|e| format!("BCE (1-predictions) computation failed: {}", e))?;

        // Step 5: Compute log(1 - predictions)
        let log_op2 = Box::new(Log);
        let log_one_minus_pred = graph
            .apply_operation(log_op2, vec![one_minus_pred])
            .map_err(|e| format!("BCE log(1-predictions) failed: {}", e))?;

        // Step 6: Compute (1-targets) * log(1-predictions)
        let mul_op2 = Box::new(Mul);
        let one_minus_targets_log_pred = graph
            .apply_operation(mul_op2, vec![one_minus_targets, log_one_minus_pred])
            .map_err(|e| format!("BCE (1-targets) * log(1-pred) failed: {}", e))?;

        // Step 7: Add both terms: targets*log(pred) + (1-targets)*log(1-pred)
        let add_op = Box::new(Add::new());
        let cross_entropy = graph
            .apply_operation(add_op, vec![targets_log_pred, one_minus_targets_log_pred])
            .map_err(|e| format!("BCE cross entropy sum failed: {}", e))?;

        // Step 8: Apply negative sign to get final BCE
        let neg_op3 = Box::new(Neg);
        let bce_loss = graph
            .apply_operation(neg_op3, vec![cross_entropy])
            .map_err(|e| format!("BCE final negation failed: {}", e))?;

        // Step 9: Apply reduction strategy
        match self.reduction {
            ReductionType::Mean => {
                let mean_op = Box::new(Mean::new());
                graph
                    .apply_operation(mean_op, vec![bce_loss])
                    .map_err(|e| format!("BCE mean reduction failed: {}", e))
            }
            ReductionType::Sum => {
                let sum_op = Box::new(Sum::new());
                graph
                    .apply_operation(sum_op, vec![bce_loss])
                    .map_err(|e| format!("BCE sum reduction failed: {}", e))
            }
            ReductionType::None => Ok(bce_loss),
        }
    }

    fn reduction(&self) -> ReductionType {
        self.reduction
    }
}

/// Categorical Cross Entropy Loss: CCE = -mean(sum(targets * log(softmax(predictions))))
/// Used for multiclass classification where targets are one-hot encoded and predictions are logits
#[derive(Debug, Clone)]
pub struct CCELoss<T> {
    reduction: ReductionType,
    /// Whether to apply softmax to predictions (true for logits, false for probabilities)
    apply_softmax: bool,
    _phantom: PhantomData<T>,
}

impl<T> CCELoss<T>
where
    T: FerroxCudaF,
{
    /// Create new CCE loss
    /// apply_softmax: true if predictions are logits, false if already probabilities
    pub fn new(reduction: ReductionType, apply_softmax: bool) -> Self {
        Self {
            reduction,
            apply_softmax,
            _phantom: PhantomData,
        }
    }

    /// Create CCE loss for logits (most common case)
    pub fn from_logits(reduction: ReductionType) -> Self {
        Self::new(reduction, true)
    }

    /// Create CCE loss for probabilities (less common, when softmax already applied)
    pub fn from_probabilities(reduction: ReductionType) -> Self {
        Self::new(reduction, false)
    }
}

impl<T> Default for CCELoss<T>
where
    T: FerroxCudaF,
{
    /// Default CCE loss expects logits and uses mean reduction (most common setup)
    fn default() -> Self {
        Self::from_logits(ReductionType::Mean)
    }
}

impl<T> Loss<T> for CCELoss<T>
where
    T: FerroxCudaF,
{
    fn forward(
        &self,
        graph: &mut AutoFerroxEngine<T>,
        predictions: NodeId,
        targets: NodeId,
    ) -> Result<NodeId, String> {
        // CCE = -mean(sum(targets * log(softmax(predictions)), axis=1))

        // Step 2: Apply softmax and add epsilon for numerical stability
        let probabilities = if self.apply_softmax {
            let softmax_op = Box::new(Softmax);
            let raw_probs = graph
                .apply_operation(softmax_op, vec![predictions])
                .map_err(|e| format!("CCE softmax application failed: {}", e))?;

            // Add epsilon for numerical stability
            let epsilon = FerroxN::from_f64(1e-7).unwrap();
            let add_eps_op = Box::new(AddScalar::new(epsilon));
            graph
                .apply_operation(add_eps_op, vec![raw_probs])
                .map_err(|e| format!("CCE epsilon addition failed: {}", e))?
        } else {
            // Add epsilon even if probabilities are provided
            let epsilon = FerroxN::from_f64(1e-7).unwrap();
            let add_eps_op = Box::new(AddScalar::new(epsilon));
            graph
                .apply_operation(add_eps_op, vec![predictions])
                .map_err(|e| format!("CCE epsilon addition failed: {}", e))?
        };

        // Step 3: Compute log of stabilized probabilities
        let log_op = Box::new(Log);
        let log_probabilities = graph
            .apply_operation(log_op, vec![probabilities])
            .map_err(|e| format!("CCE log(probabilities) failed: {}", e))?;

        // Step 3: Element-wise multiplication with targets (one-hot encoding)
        let mul_op = Box::new(Mul);
        let targets_log_prob = graph
            .apply_operation(mul_op, vec![targets, log_probabilities])
            .map_err(|e| format!("CCE targets * log(prob) failed: {}", e))?;

        // Step 4: Sum across classes (axis=1 for batch processing)
        // This reduces [batch_size, num_classes] to [batch_size]
        let sum_op = Box::new(Sum::new());
        let cross_entropy_per_sample = graph
            .apply_operation(sum_op, vec![targets_log_prob])
            .map_err(|e| format!("CCE sum across classes failed: {}", e))?;

        // Step 5: Apply negative sign to get final CCE per sample
        let neg_op = Box::new(Neg);
        let cce_per_sample = graph
            .apply_operation(neg_op, vec![cross_entropy_per_sample])
            .map_err(|e| format!("CCE negation failed: {}", e))?;

        // Step 6: Apply batch reduction strategy
        match self.reduction {
            ReductionType::Mean => {
                let mean_op = Box::new(Mean::new());
                graph
                    .apply_operation(mean_op, vec![cce_per_sample])
                    .map_err(|e| format!("CCE mean reduction failed: {}", e))
            }
            ReductionType::Sum => {
                let sum_op = Box::new(Sum::new());
                graph
                    .apply_operation(sum_op, vec![cce_per_sample])
                    .map_err(|e| format!("CCE sum reduction failed: {}", e))
            }
            ReductionType::None => Ok(cce_per_sample),
        }
    }

    fn reduction(&self) -> ReductionType {
        self.reduction
    }
}
