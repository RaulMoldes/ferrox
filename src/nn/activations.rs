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
