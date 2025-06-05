use std::sync::atomic::{AtomicUsize, Ordering};
use super::op::Operator;
use crate::tensor::Tensor;

// Unique ID generator for computational graph nodes
static NODE_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub fn next_node_id() -> usize {
    // The use of `Ordering::Relaxed` allows for relaxed consistency,
    // which provides better performance and is sufficient for generating unique IDs in this context.
    // You can look at `test_node_id_atomicity` for a detailed test to validate the atomicity
    // of the incremental counter across multiple threads.
    NODE_COUNTER.fetch_add(1, Ordering::Relaxed)
}

// Id of the node in the computational graph.
pub type NodeId = usize;

// Represents a value in the computational graph
// A value can be either a leaf node (input tensor) or an intermediate result from an operation.
#[derive(Debug)]
pub struct Node {
    pub id: NodeId,
    pub op: Option<Box<dyn Operator>>, // Use Box here to allow dynamic dispatch for different operators
    // Apparently, dynamic dispatch is less efficient than static dispatch, but it allows us to use different operators in the same graph.
    // https://softwaremill.com/rust-static-vs-dynamic-dispatch/
    // The thing is that with static dispatch the compiler will perform monomorphization, which means that it will generate a new version of the code for each type used.
    // On the other hand, dynamic dispatch will use a vtable to call the methods of the trait, which is less efficient.
    // I an not sure which one is better for our use case, but I will stick with dynamic dispatch for now.
    // Static dispatch would consume more memory, but would also allow the compiler to perform more optimizations (by means of increasig the compile time).
    pub inputs: Vec<NodeId>,
    pub cached_data: Tensor,
    pub requires_grad: bool,
}

impl Node {
    // Create a new leaf node in the graph
    pub fn new(data: Tensor, requires_grad: bool) -> Self {
        Self {
            id: next_node_id(),
            op: None,
            inputs: Vec::new(),
            cached_data: data, //THe cached data is the input tensor.
            requires_grad,
        }
    }

    // Create a new node from an operation
    pub fn from_op(op: Box<dyn Operator>, inputs: Vec<NodeId>, data: Tensor) -> Self {
        Self {
            id: next_node_id(),
            op: Some(op),
            inputs,
            cached_data: data,
            requires_grad: true, // Assume intermediate values require gradients
                                 // Refering to PytOrch documentation, the requires_grad flag is always set to false on torch.autograd unless the node is teh output of an
                                 // operation. In PyTorch you can manipulate this flag to control which nodes will have their gradients computed,
                                 // allowing to optimize memory usage and computation time. For example one might not want to execute the backward pass on
                                 // all intermediate nodes, but only on the final output of the graph.
                                 // For now, I will assume that all intermediate nodes require gradients, let's keep optimization for later.
        }

        // Anyway, the ´detach()´ operation will allow us to remove the gradients from the node,
    }

    // Check if this is a leaf node (no operation)
    pub fn is_leaf(&self) -> bool {
        self.op.is_none()
    }

    // Detach from computation graph
    pub fn detach(&self) -> Self {
        Self {
            id: next_node_id(),
            op: None,
            inputs: Vec::new(),
            cached_data: self.cached_data.detach(),
            requires_grad: false,
        }
    }
}
