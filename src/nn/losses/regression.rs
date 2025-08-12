// src/nn/losses.rs
// Loss functions for neural networks using the computational graph
// All losses are implemented as operations that work through the tensor API and graph system

use crate::backend::FerroxCudaF;
use crate::graph::{AutoFerroxEngine, NodeId};
use crate::nn::losses::{Loss, ReductionType};
use crate::ops::{
    basic::{Mul, Sub},
    reduction::{Mean, Sum},
    unary::Abs,
};
use std::marker::PhantomData;

/// Mean Squared Error Loss: MSE = mean((predictions - targets)²)
/// Used for regression tasks where targets are continuous values
#[derive(Debug, Clone)]
pub struct MSELoss<T> {
    reduction: ReductionType,
    _phantom: PhantomData<T>,
}

impl<T> MSELoss<T>
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

impl<T> Default for MSELoss<T>
where
    T: FerroxCudaF,
{
    fn default() -> Self {
        Self::new(ReductionType::Mean)
    }
}

impl<T> Loss<T> for MSELoss<T>
where
    T: FerroxCudaF,
{
    fn forward(
        &self,
        graph: &mut AutoFerroxEngine<T>,
        predictions: NodeId,
        targets: NodeId,
    ) -> Result<NodeId, String> {
        // MSE = (predictions - targets)²

        // Step 1: Compute difference (predictions - targets)
        let sub_op = Box::new(Sub);
        let diff = graph
            .apply_operation(sub_op, vec![predictions, targets])
            .map_err(|e| format!("MSE difference computation failed: {}", e))?;

        // Step 2: Square the difference
        let mul_op = Box::new(Mul);
        let squared_diff = graph
            .apply_operation(mul_op, vec![diff, diff])
            .map_err(|e| format!("MSE squaring failed: {}", e))?;

        // Step 3: Apply reduction strategy
        match self.reduction {
            ReductionType::Mean => {
                let mean_op = Box::new(Mean::new());
                graph
                    .apply_operation(mean_op, vec![squared_diff])
                    .map_err(|e| format!("MSE mean reduction failed: {}", e))
            }
            ReductionType::Sum => {
                let sum_op = Box::new(Sum::new());
                graph
                    .apply_operation(sum_op, vec![squared_diff])
                    .map_err(|e| format!("MSE sum reduction failed: {}", e))
            }
            ReductionType::None => Ok(squared_diff),
        }
    }

    fn reduction(&self) -> ReductionType {
        self.reduction
    }
}

/// L1 Loss (Mean Absolute Error): L1 = mean(|predictions - targets|)
/// More robust to outliers than MSE, used in regression tasks
#[derive(Debug, Clone)]
pub struct L1Loss<T> {
    reduction: ReductionType,
    _phantom: PhantomData<T>,
}

impl<T> L1Loss<T>
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

impl<T> Default for L1Loss<T>
where
    T: FerroxCudaF,
{
    fn default() -> Self {
        Self::new(ReductionType::Mean)
    }
}

impl<T> Loss<T> for L1Loss<T>
where
    T: FerroxCudaF,
{
    fn forward(
        &self,
        graph: &mut AutoFerroxEngine<T>,
        predictions: NodeId,
        targets: NodeId,
    ) -> Result<NodeId, String> {
        // L1 = |predictions - targets|

        // Step 1: Compute difference
        let sub_op = Box::new(Sub);
        let diff = graph
            .apply_operation(sub_op, vec![predictions, targets])
            .map_err(|e| format!("L1 difference computation failed: {}", e))?;

        // Step 2: Take absolute value using the computational graph
        let abs_op = Box::new(Abs);
        let abs_diff = graph
            .apply_operation(abs_op, vec![diff])
            .map_err(|e| format!("L1 absolute value computation failed: {}", e))?;

        // Step 3: Apply reduction strategy
        match self.reduction {
            ReductionType::Mean => {
                let mean_op = Box::new(Mean::new());
                graph
                    .apply_operation(mean_op, vec![abs_diff])
                    .map_err(|e| format!("L1 mean reduction failed: {}", e))
            }
            ReductionType::Sum => {
                let sum_op = Box::new(Sum::new());
                graph
                    .apply_operation(sum_op, vec![abs_diff])
                    .map_err(|e| format!("L1 sum reduction failed: {}", e))
            }
            ReductionType::None => Ok(abs_diff),
        }
    }

    fn reduction(&self) -> ReductionType {
        self.reduction
    }
}
