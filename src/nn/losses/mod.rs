pub mod classification;
pub mod regression;

use crate::backend::FerroxCudaF;
use crate::graph::{AutoFerroxEngine, NodeId};
pub use classification::{BCELoss, CCELoss};
pub use regression::{L1Loss, MSELoss};

/// Base trait for all loss functions
/// Mirrors PyTorch's loss interface, providing forward computation through computational graph
pub trait Loss<T>
where
    T: FerroxCudaF,
{
    /// Compute loss using computational graph for automatic differentiation
    /// predictions: model outputs [batch_size, num_classes] or [batch_size]
    /// targets: ground truth labels [batch_size, num_classes] or [batch_size]
    /// Returns: scalar loss node for backpropagation
    fn forward(
        &self,
        graph: &mut AutoFerroxEngine<T>,
        predictions: NodeId,
        targets: NodeId,
    ) -> Result<NodeId, String>;

    /// Optional reduction type for the loss (mean, sum, none)
    fn reduction(&self) -> ReductionType {
        ReductionType::Mean
    }
}

/// Loss reduction strategies - determines how batch losses are aggregated
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReductionType {
    /// Average loss across batch (most common)
    Mean,
    /// Sum all losses in batch
    Sum,
    /// Return individual losses without reduction
    None,
}
