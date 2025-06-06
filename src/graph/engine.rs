use super::op::{
    AddOp, AddScalarOp, BroadcastToOp, DivOp, ExpOp, LogOp, MatMulOp, MulOp, MulScalarOp, NegateOp,
    Operator, PowOp, ReLUOp, ReshapeOp, SumOp, SummationOp, TransposeOp,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::node::{Node, NodeId};
use crate::tensor::Tensor;

// Computational graph engine that manages all nodes and their relationships
#[derive(Debug)]
pub struct Engine {
    // Creating a graph is kind of boilerplate in Rust
    // Thing is i do not want to hhave to grab the graph mutably the graph every time I want to read an item
    // RefCell allows us to have interior mutability, which means we can mutate the node inside the Rc without having to grab a mutable reference to it.
    pub nodes: HashMap<NodeId, Rc<RefCell<Node>>>,
    pub gradients: HashMap<NodeId, Tensor>, // To store gradients for each node
}

impl Engine {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            gradients: HashMap::new(),
        }
    }

    // Add a Node to the graph
    pub fn add_node(&mut self, node: Node) -> NodeId {
        let id = node.id;
        self.nodes.insert(id, Rc::new(RefCell::new(node)));
        id
    }

    // Create a leaf node
    pub fn create_tensor(&mut self, data: Tensor, requires_grad: bool) -> NodeId {
        let node = Node::new(data, requires_grad);
        self.add_node(node)
    }

    // Create a tensor from data with default device
    pub fn tensor_from_vec(
        &mut self,
        data: Vec<f64>,
        shape: &[usize],
        requires_grad: bool,
    ) -> Result<NodeId, String> {
        let tensor = Tensor::from_vec(data, shape)?;
        Ok(self.create_tensor(tensor, requires_grad))
    }

    // Create zeros tensor
    pub fn zeros(&mut self, shape: &[usize], requires_grad: bool) -> NodeId {
        let tensor = Tensor::zeros(shape);
        self.create_tensor(tensor, requires_grad)
    }

    // Create ones tensor
    pub fn ones(&mut self, shape: &[usize], requires_grad: bool) -> NodeId {
        let tensor = Tensor::ones(shape);
        self.create_tensor(tensor, requires_grad)
    }

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
    pub fn add_scalar(&mut self, a: NodeId, scalar: f64) -> Result<NodeId, String> {
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
    pub fn mul_scalar(&mut self, a: NodeId, scalar: f64) -> Result<NodeId, String> {
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

    // Power operation
    pub fn pow(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();
        let data_b = self.nodes[&b].borrow().cached_data.clone();

        let op = Box::new(PowOp);
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

    // Negation operation
    pub fn negate(&mut self, a: NodeId) -> Result<NodeId, String> {
        let data_a = self.nodes[&a].borrow().cached_data.clone();

        let op = Box::new(NegateOp);
        let result_data = op.compute(&[data_a])?;

        let node = Node::from_op(op, vec![a], result_data);
        Ok(self.add_node(node))
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

    // Get data from a node
    pub fn get_data(&self, node_id: NodeId) -> Tensor {
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

    // Main gradient computation function. This can actually be called on any node,
    pub fn compute_gradient_of_variables(
        &mut self,
        output_tensor: NodeId,
        out_grad: Option<Tensor>,
    ) -> Result<(), String> {
        // Clear previous gradients
        self.gradients.clear();

        // Initialize gradient of output
        let output_grad = match out_grad {
            Some(grad) => grad,
            None => {
                let output_data = self.nodes[&output_tensor].borrow().cached_data.clone();
                Tensor::ones(output_data.shape())
            }
        };

        // Map from node to list of gradient contributions
        let mut node_to_output_grads_list: HashMap<NodeId, Vec<Tensor>> = HashMap::new();
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
                    let input_data: Vec<Tensor> = node
                        .inputs
                        .iter()
                        .map(|&input_id| self.nodes[&input_id].borrow().cached_data.clone())
                        .collect();

                    // Compute gradients for all inputs
                    let input_grads: Vec<Tensor> = op.gradient(&accumulated_grad, &input_data)?;

                    // Accumulate gradients for input nodes
                    for (i, &input_id) in node.inputs.iter().enumerate() {
                        if self.nodes[&input_id].borrow().requires_grad {
                            node_to_output_grads_list
                                .entry(input_id)
                                .or_insert_with(Vec::new)
                                .push(input_grads[i].clone());
                        }
                    }
                }
            }
        }

        Ok(())
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

    // Get gradient for a node
    pub fn get_gradient(&self, node_id: NodeId) -> Option<Tensor> {
        self.gradients.get(&node_id).cloned()
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
