use super::op::{
    AbsOp, AddOp, AddScalarOp, BroadcastToOp, ClampOp, DivOp, DivScalarOp, ExpOp, LogOp, MatMulOp,
    MaxAlongDimOp, MaxOp, MinOp, MulOp, MulScalarOp, NegateOp, Operator, PowOp, ReLUOp, ReshapeOp,
    SqrtOp, SumOp, SummationOp, TransposeOp,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::node::{Node, NodeId};
use crate::backend::number::{CPUNumber, GPUFloat};
use crate::tensor::Tensor;

// Computational graph engine that manages all nodes and their relationships
#[derive(Debug)]
pub struct Engine<T>
where
    T: GPUFloat,
{
    // T must implement Clone and Debug traits{
    // Creating a graph is kind of boilerplate in Rust
    // Thing is i do not want to hhave to grab the graph mutably the graph every time I want to read an item
    // RefCell allows us to have interior mutability, which means we can mutate the node inside the Rc without having to grab a mutable reference to it.
    pub nodes: HashMap<NodeId, Rc<RefCell<Node<T>>>>,
    pub gradients: HashMap<NodeId, Tensor<T>>, // To store gradients for each node
}

impl<T> Default for Engine<T>
where
    T: GPUFloat,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Engine<T>
where
    T: GPUFloat,
{
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            gradients: HashMap::new(),
        }
    }

    // Add a Node to the graph
    pub fn add_node(&mut self, node: Node<T>) -> NodeId {
        let id = node.id;
        self.nodes.insert(id, Rc::new(RefCell::new(node)));
        id
    }

    // Create a leaf node
    pub fn create_tensor(&mut self, data: Tensor<T>, requires_grad: bool) -> NodeId {
        let node = Node::new(data, requires_grad);
        self.add_node(node)
    }

    // Create a tensor from data with default device
    pub fn tensor_from_vec(
        &mut self,
        data: Vec<T>,
        shape: &[usize],
        requires_grad: bool,
    ) -> Result<NodeId, String> {
        let tensor = Tensor::from_vec(data, shape)?;
        Ok(self.create_tensor(tensor, requires_grad))
    }
}

impl<T> Engine<T>
where
    T: GPUFloat, // T must implement Clone and Debug traits
{
    // Create zeros tensor
    pub fn zeros(&mut self, shape: &[usize], requires_grad: bool) -> NodeId {
        let tensor = Tensor::zeros(shape).expect("Failed to create zeroed tensor");
        self.create_tensor(tensor, requires_grad)
    }
}

impl<T> Engine<T>
where
    T: GPUFloat, // T must implement Clone and Debug traits
{
    // Create ones tensor

    pub fn ones(&mut self, shape: &[usize], requires_grad: bool) -> NodeId {
        let tensor = Tensor::ones(shape).expect("Failed to create ones Tensor");
        self.create_tensor(tensor, requires_grad)
    }
}
impl<T> Engine<T>
where
    T: GPUFloat
        + Clone
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand
        + rand_distr::num_traits::FromPrimitive, // T must implement Clone and Debug traits
{
    // Main gradient computation function. This can actually be called on any node.
    // Note that my graph implementation is based on reverse-mode automatic differentiation.
    // It is a Define-By-Run (dynamic) graph, meaning that the graph is built dynamically as operations are performed.
    // Similarly to other frameworks like PyTorch, TensorFlow 2.X on eager mode, Chainer, etc.
    // This function computes the gradient of the output node with respect to all input nodes that require gradients.
    // The main difference between this approach and the Define-And-Run (static) approach (Tensorflow 1.X or Tensorflow graph mode)
    // is quite well explained here: https://medium.com/@zzemb6/define-and-run-vs-define-by-run-b527d127e13a
    // Mainly, in Define-By-Run, the graph is built dynamically as operations are performed,
    // and the gradients are computed in reverse order, starting from the output node.
    pub fn compute_gradient_of_variables(
        &mut self,
        output_tensor: NodeId,
        out_grad: Option<Tensor<T>>,
    ) -> Result<(), String> {
        // Clear previous gradients
        self.gradients.clear();

        // Initialize gradient of output
        let output_grad = match out_grad {
            Some(grad) => grad,
            None => {
                let output_data = self.nodes[&output_tensor].borrow().cached_data.clone();
                Tensor::ones(output_data.shape())?
            }
        };

        // Map from node to list of gradient contributions
        let mut node_to_output_grads_list: HashMap<NodeId, Vec<Tensor<T>>> = HashMap::new();
        node_to_output_grads_list.insert(output_tensor, vec![output_grad]);

        // Get reverse topological order
        let reverse_topo_order: Vec<NodeId> = self
            .find_topo_sort(&[output_tensor])
            .into_iter()
            .rev()
            .collect();

        // Process nodes in reverse topological order
        for &node_id in &reverse_topo_order {
            if let Some(grad_list) = node_to_output_grads_list.get(&node_id) {
                // Sum all gradient contributions for this node
                let mut accumulated_grad = grad_list[0].clone();
                for grad in grad_list.iter().skip(1) {
                    accumulated_grad = accumulated_grad.add(grad)?;
                }

                // Store the accumulated gradient
                self.gradients.insert(node_id, accumulated_grad.clone());

                let node = self.nodes[&node_id].borrow();

                // If this node has an operation, compute gradients for its inputs
                if let Some(ref op) = node.op {
                    // Collect input data for gradient computation
                    let input_data: Vec<Tensor<T>> = node
                        .inputs
                        .iter()
                        .map(|&input_id| self.nodes[&input_id].borrow().cached_data.clone())
                        .collect();

                    // Compute gradients for all inputs
                    let input_grads: Vec<Tensor<T>> =
                        op.gradient(&accumulated_grad, &input_data)?;

                    // Accumulate gradients for input nodes
                    for (i, &input_id) in node.inputs.iter().enumerate() {
                        if self.nodes[&input_id].borrow().requires_grad {
                            node_to_output_grads_list
                                .entry(input_id)
                                .or_default()
                                .push(input_grads[i].clone());
                        }
                    }
                }
            }
        }

        Ok(())
    }

    // Backward pass: compute gradients using reverse-mode automatic differentiation
    // I am using reverse mode as it is more efficent for neural networks and other models where the number of outputs is much smaller than the number of inputs.
    // In forward mode, we would compute the gradient of each output with respect to each input, which is not efficient for large models that can have thousands of inputs.
    // I created this wrapper  over the private method compute_gradient_of_variables to make it easier to use.
    // Additionally my idea is to implement more gradient computation methods in the future, such as finite differences, so I want to keep this method as the main entrypoint for the backward pass while the other methods will hold the complexity of the computation.
    pub fn backward(&mut self, output_node: NodeId) -> Result<(), String> {
        // the backard function calls the main gradient computation function. By passing None as the out_grad, we will use the default gradient of ones.
        // I think this could be done better if we passed  the actual gradient of the output node, but for now we will use the default gradient of ones.
        if let Some(grad) = self.get_gradient(output_node) {
            return self.compute_gradient_of_variables(output_node, Some(grad));
        }
        // Fallback to default gradient of ones if no gradient is set
        self.compute_gradient_of_variables(output_node, None)
    }
}

impl<T> Engine<T>
where
    T: GPUFloat,
{
    /// GRAPH OPERATIONS:
    ///
    /// Currently, implemented graph operations include:
    ///
    /// - Addition (element-wise and scalar)
    /// - Multiplication (element-wise and scalar)
    /// - Division (element-wise and scalar)
    /// - Power (element-wise and scalar)
    /// - Matrix multiplication
    /// - ReLU activation
    /// - Exponential function
    /// - Logarithm function
    /// - Negation
    /// - Sum operation (with axes support)
    /// - Transpose
    /// - Reshape
    /// - Broadcast
    ///
    /// Operations in the TO-DO list include:
    ///
    /// - More activation functions (tanh, softmax, etc.)
    /// - Convolution operations
    /// - Pooling operations
    /// - Batch normalization
    /// - Dropout
    ///
    /// Any other activation functions or operations can be added as needed.
    /// At the end of each operation, the resulting OPNode is added to the graph and its ID is returned.
    /// This is critical so that tensor operations are recorded on the grph strcture.
    ///
    // Addition operation
    pub fn add(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();
        let data_b = self.nodes[&b].borrow().cached_data.clone();

        let op = Box::new(AddOp);
        let result_data = op.compute(&[data_a, data_b])?;

        let node = Node::from_op(op, vec![a, b], result_data);
        Ok(self.add_node(node))
    }

    // Scalar addition
    pub fn add_scalar(&mut self, a: NodeId, scalar: T) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();

        let op = Box::new(AddScalarOp::new(scalar));
        let result_data = op.compute(&[data_a])?;

        let node = Node::from_op(op, vec![a], result_data);
        Ok(self.add_node(node))
    }

    // Multiplication operation
    pub fn mul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();
        let data_b = self.nodes[&b].borrow().cached_data.clone();

        let op = Box::new(MulOp);
        let result_data = op.compute(&[data_a, data_b])?;

        let node = Node::from_op(op, vec![a, b], result_data);
        Ok(self.add_node(node))
    }

    // Scalar multiplication
    pub fn mul_scalar(&mut self, a: NodeId, scalar: T) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();

        let op = Box::new(MulScalarOp::new(scalar));
        let result_data = op.compute(&[data_a])?;

        let node = Node::from_op(op, vec![a], result_data);
        Ok(self.add_node(node))
    }

    // Division operation
    pub fn div(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();
        let data_b = self.nodes[&b].borrow().cached_data.clone();

        let op = Box::new(DivOp);
        let result_data = op.compute(&[data_a, data_b])?;

        let node = Node::from_op(op, vec![a, b], result_data);
        Ok(self.add_node(node))
    }

    // Matrix multiplication operation
    pub fn matmul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();
        let data_b = self.nodes[&b].borrow().cached_data.clone();

        let op = Box::new(MatMulOp);
        let result_data = op.compute(&[data_a, data_b])?;

        let node = Node::from_op(op, vec![a, b], result_data);
        Ok(self.add_node(node))
    }

    // Negation operation
    pub fn negate(&mut self, a: NodeId) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();

        let op = Box::new(NegateOp);
        let result_data = op.compute(&[data_a])?;

        let node = Node::from_op(op, vec![a], result_data);
        Ok(self.add_node(node))
    }

    // Division by scalar
    pub fn div_scalar(&mut self, a: NodeId, scalar: T) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();

        // Check for zero scalar before creating the operation
        let zero = <T as CPUNumber>::zero();
        if scalar == zero {
            return Err("Cannot divide by zero scalar".to_string());
        }

        let op = Box::new(DivScalarOp::new(scalar));
        let result_data = op.compute(&[data_a])?;

        let node = Node::from_op(op, vec![a], result_data);
        Ok(self.add_node(node))
    }

    // Division operation with two nodes
    pub fn divide(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();
        let data_b = self.nodes[&b].borrow().cached_data.clone();
        let op = Box::new(DivOp);
        let result_data = op.compute(&[data_a, data_b])?;
        let node = Node::from_op(op, vec![a, b], result_data);
        Ok(self.add_node(node))
    }
}

impl<T> Engine<T>
where
    T: GPUFloat
        + Clone
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand
        + rand_distr::num_traits::FromPrimitive, // T must implement Clone and Debug traits
{
    /// Find maximum values along a specified dimension.
    ///
    /// This is essential for CPUNumberally stable softmax computation.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor node
    /// * `dim` - Dimension along which to find maximum
    ///
    /// # Returns
    ///
    /// Tensor with maximum values, with the specified dimension reduced
    ///
    /// # Examples
    ///
    /// For input shape [2, 3] and dim=1:
    /// - Input: [[1, 5, 3], [2, 1, 4]]
    /// - Output: [5, 4] (shape [2])
    pub fn max_along_dim(&mut self, input: NodeId, dim: usize) -> Result<NodeId, String> {
        let input_data = self.nodes[&input].borrow().cached_data.clone();
        let input_shape = input_data.shape();

        // Validate dimension
        if dim >= input_shape.len() {
            return Err(format!(
                "Dimension {} out of bounds for tensor with {} dimensions",
                dim,
                input_shape.len()
            ));
        }


        let result_tensor = input_data.max_reduce(Some(&[dim])
        )?;



        // Create operation for gradient computation
        let op = Box::new(MaxAlongDimOp::new(dim));
        let node = Node::from_op(op, vec![input], result_tensor);
        Ok(self.add_node(node))
    }

    /// Sum values along a specified dimension.
    ///
    /// Similar to the existing sum operation but specifically for one dimension.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor node
    /// * `dim` - Dimension along which to sum
    ///
    /// # Returns
    ///
    /// Tensor with summed values, with the specified dimension reduced
    pub fn sum_along_dim(&mut self, input: NodeId, dim: usize) -> Result<NodeId, String> {
        // This can reuse the existing summation operation
        self.summation(input, Some(vec![dim]))
    }

    /// Expand dimensions by inserting a size-1 dimension at the specified position.
    /// This is used to restore dimensions after reduction operations for broadcasting.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor node
    /// * `dim` - Position where to insert the new dimension
    ///
    /// # Returns
    ///
    /// Tensor with an additional dimension of size 1 at the specified position
    ///
    /// # Examples
    ///
    /// For input shape [2] and dim=1:
    /// - Input shape: [2]
    /// - Output shape: [2, 1]
    ///
    /// For input shape [2, 3] and dim=1:
    /// - Input shape: [2, 3]
    /// - Output shape: [2, 1, 3]
    pub fn expand_dims_at(&mut self, input: NodeId, dim: usize) -> Result<NodeId, String> {
        let input_data = self.nodes[&input].borrow().cached_data.clone();
        let input_shape = input_data.shape();

        // Validate dimension
        if dim > input_shape.len() {
            return Err(format!(
                "Cannot expand at dimension {} for tensor with {} dimensions",
                dim,
                input_shape.len()
            ));
        }

        // Create new shape with size-1 dimension inserted
        let mut new_shape = input_shape.to_vec();
        new_shape.insert(dim, 1);

        // Use existing reshape operation
        self.reshape(input, new_shape)
    }

    // Sum operation
    pub fn sum(&mut self, a: NodeId, axis: Option<usize>) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();

        let op = Box::new(SumOp::new(axis));
        let result_data = op.compute(&[data_a])?;

        let node = Node::from_op(op, vec![a], result_data);
        Ok(self.add_node(node))
    }

    // Summation with multiple axes
    pub fn summation(&mut self, a: NodeId, axes: Option<Vec<usize>>) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();

        let op = Box::new(SummationOp::new(axes));
        let result_data = op.compute(&[data_a])?;

        let node = Node::from_op(op, vec![a], result_data);
        Ok(self.add_node(node))
    }

    // Transpose operation
    pub fn transpose(&mut self, a: NodeId, axes: Option<Vec<usize>>) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();

        let op = Box::new(TransposeOp::new(axes));
        let result_data = op.compute(&[data_a])?;

        let node = Node::from_op(op, vec![a], result_data);
        Ok(self.add_node(node))
    }

    // Reshape operation
    pub fn reshape(&mut self, a: NodeId, new_shape: Vec<usize>) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();

        let op = Box::new(ReshapeOp::new(new_shape));
        let result_data = op.compute(&[data_a])?;

        let node = Node::from_op(op, vec![a], result_data);
        Ok(self.add_node(node))
    }

    // Broadcast operation
    pub fn broadcast_to(&mut self, a: NodeId, target_shape: Vec<usize>) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();

        let op = Box::new(BroadcastToOp::new(target_shape));
        let result_data = op.compute(&[data_a])?;

        let node = Node::from_op(op, vec![a], result_data);
        Ok(self.add_node(node))
    }

    /// Element-wise minimum operation between two tensors.
    ///
    /// Computes min(a, b) element-wise. Both tensors must have the same shape.
    pub fn min(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();
        let data_b = self.nodes[&b].borrow().cached_data.clone();

        let op = Box::new(MinOp);
        let result_data = op.compute(&[data_a, data_b])?;

        let node = Node::from_op(op, vec![a, b], result_data);
        Ok(self.add_node(node))
    }

    /// Element-wise maximum operation between two tensors.
    ///
    /// Computes max(a, b) element-wise. Both tensors must have the same shape.
    pub fn max(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();
        let data_b = self.nodes[&b].borrow().cached_data.clone();

        let op = Box::new(MaxOp);
        let result_data = op.compute(&[data_a, data_b])?;

        let node = Node::from_op(op, vec![a, b], result_data);
        Ok(self.add_node(node))
    }

    /// Clamp operation that constrains tensor values to a specified range.
    ///
    /// Clamps all elements in the input tensor to the range [min_val, max_val].
    /// This is essential for numerical stability, especially in loss functions.
    pub fn clamp(&mut self, input: NodeId, min_val: T, max_val: T) -> Result<NodeId, String> {
        let data_input = self.nodes[&input].borrow().cached_data.clone();

        let op = Box::new(ClampOp::new(min_val, max_val));
        let result_data = op.compute(&[data_input])?;

        let node = Node::from_op(op, vec![input], result_data);
        Ok(self.add_node(node))
    }

    /// Element-wise absolute value operation.
    ///
    /// Computes the absolute value of each element in the input tensor.
    pub fn abs(&mut self, input: NodeId) -> Result<NodeId, String> {
        let data_input = self.nodes[&input].borrow().cached_data.clone();

        let op = Box::new(AbsOp);
        let result_data = op.compute(&[data_input])?;

        let node = Node::from_op(op, vec![input], result_data);
        Ok(self.add_node(node))
    }

    // Get data from a node
    pub fn get_data(&self, node_id: NodeId) -> Tensor<T> {
        self.nodes[&node_id].borrow().cached_data.clone()
    }

    // Get shape of a node
    pub fn get_shape(&self, node_id: NodeId) -> Vec<usize> {
        self.nodes[&node_id].borrow().cached_data.shape().to_vec()
    }

    // Check if node requires gradient
    pub fn requires_grad(&self, node_id: NodeId) -> bool {
        self.nodes[&node_id].borrow().requires_grad
    }

    // Detach a node from the computation graph
    pub fn detach(&mut self, node_id: NodeId) -> NodeId {
        let detached_node = self.nodes[&node_id].borrow().detach();
        self.add_node(detached_node)
    }

    // Find topological sort - needed for backward pass
    pub fn find_topo_sort(&self, output_nodes: &[NodeId]) -> Vec<NodeId> {
        let mut visited = std::collections::HashSet::new();
        let mut topo_order = Vec::new();

        // Iterate over the output nodes and find the route towards the root.
        for &node_id in output_nodes {
            self.topo_sort_dfs(node_id, &mut visited, &mut topo_order);
        }

        topo_order
    }

    // DFS helper for topological sort.
    // This is the inner function that performs the  recursive DFS traversal,
    // Maintains a visited and a topo_order list.
    // This way it is very easy to avoid recursive cycles.
    pub fn topo_sort_dfs(
        &self,
        node_id: NodeId,
        visited: &mut std::collections::HashSet<NodeId>,
        topo_order: &mut Vec<NodeId>,
    ) {
        if visited.contains(&node_id) {
            return;
        }

        visited.insert(node_id);

        let node = self.nodes[&node_id].borrow();
        for &input_id in &node.inputs {
            self.topo_sort_dfs(input_id, visited, topo_order);
        }

        topo_order.push(node_id);
    }

    /// Set gradient for a node (needed for gradient clipping)
    pub fn set_gradient(&mut self, node_id: NodeId, gradient: Tensor<T>) {
        self.gradients.insert(node_id, gradient);
    }

    // Get gradient for a node
    pub fn get_gradient(&self, node_id: NodeId) -> Option<Tensor<T>> {
        self.gradients.get(&node_id).cloned()
    }

    /// Update the data of a node
    pub fn set_node_data(&mut self, node_id: NodeId, new_data: Tensor<T>) {
        if let Some(node_ref) = self.nodes.get(&node_id) {
            node_ref.borrow_mut().cached_data = new_data;
        }
    }

    /// Clear gradient for a node
    pub fn clear_gradient(&mut self, node_id: NodeId) {
        self.gradients.remove(&node_id);
    }

    // Helper function to sum a list of nodes (useful for loss functions)
    pub fn sum_node_list(&mut self, node_list: Vec<NodeId>) -> Result<NodeId, String> {
        if node_list.is_empty() {
            return Err("Cannot sum empty node list".to_string());
        }

        let mut result = node_list[0];
        for &node in node_list.iter().skip(1) {
            result = self.add(result, node)?;
        }

        Ok(result)
    }
}

// Put here the operations that require Float trait
// This is a trait bound for operations that require floating point numbers.
impl<T> Engine<T>
where
    T: GPUFloat,
{
    // Power operation
    pub fn pow(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();
        let data_b = self.nodes[&b].borrow().cached_data.clone();

        let op = Box::new(PowOp);
        let result_data = op.compute(&[data_a, data_b])?;

        let node = Node::from_op(op, vec![a, b], result_data);
        Ok(self.add_node(node))
    }

    // ReLU activation
    pub fn relu(&mut self, a: NodeId) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();

        let op = Box::new(ReLUOp);
        let result_data = op.compute(&[data_a])?;

        let node = Node::from_op(op, vec![a], result_data);
        Ok(self.add_node(node))
    }

    // Exponential operation
    pub fn exp(&mut self, a: NodeId) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();

        let op = Box::new(ExpOp);
        let result_data = op.compute(&[data_a])?;

        let node = Node::from_op(op, vec![a], result_data);
        Ok(self.add_node(node))
    }

    // Logarithm operation
    pub fn log(&mut self, a: NodeId) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();

        let op = Box::new(LogOp);
        let result_data = op.compute(&[data_a])?;

        let node = Node::from_op(op, vec![a], result_data);
        Ok(self.add_node(node))
    }

    /// Element-wise square root operation.
    ///
    /// Computes the square root of each element in the input tensor.
    /// Input values must be non-negative.
    pub fn sqrt(&mut self, input: NodeId) -> Result<NodeId, String> {
        let data_input = self.nodes[&input].borrow().cached_data.clone();

        let op = Box::new(SqrtOp);
        let result_data = op.compute(&[data_input])?;

        let node = Node::from_op(op, vec![input], result_data);
        Ok(self.add_node(node))
    }
}

impl<T> Engine<T>
where
    T: GPUFloat,
{
    /// Convenience method to clamp probabilities for numerical stability.
    ///
    /// This is a specialized version of clamp specifically designed for probability
    /// values, clamping them to [eps, 1-eps] to prevent CPUNumberal issues in
    /// logarithmic operations.
    ///
    /// # Arguments
    ///
    /// * `probabilities` - Input probability tensor
    /// * `eps` - Small epsilon value (default: 1e-8)
    pub fn clamp_probabilities(
        &mut self,
        probabilities: NodeId,
        eps: f64,
    ) -> Result<NodeId, String> {
        let eps_val =
            <T as CPUNumber>::from_f64(eps).ok_or("Failed to convert epsilon to tensor type")?;
        let one_minus_eps = <T as CPUNumber>::from_f64(1.0 - eps)
            .ok_or("Failed to convert 1-epsilon to tensor type")?;

        self.clamp(probabilities, eps_val, one_minus_eps)
    }

    /// Convenience method for computing L2 norm of a tensor.
    ///
    /// Computes the L2 (Euclidean) norm: sqrt(sum(x^2))
    pub fn l2_norm(&mut self, input: NodeId) -> Result<NodeId, String> {
        let squared = self.mul(input, input)?;
        let sum_squared = self.sum(squared, None)?;
        self.sqrt(sum_squared)
    }

    /// Convenience method for computing L1 norm of a tensor.
    ///
    /// Computes the L1 (Manhattan) norm: sum(abs(x))
    pub fn l1_norm(&mut self, input: NodeId) -> Result<NodeId, String> {
        let abs_values = self.abs(input)?;
        self.sum(abs_values, None)
    }
}
