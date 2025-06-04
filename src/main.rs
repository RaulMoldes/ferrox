#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use ndarray::ArrayD;

mod tensor;
use crate::tensor::Tensor;
mod op;
use crate::op::{Operator, AddOp, AddScalarOp, MulOp, MulScalarOp, DivOp, PowOp, MatMulOp, ReLUOp, ExpOp, LogOp, NegateOp, SumOp, SummationOp, TransposeOp, ReshapeOp, BroadcastToOp};
mod backend;
use backend::{Device, default_device, cpu};

// Unique ID generator for computational graph nodes
static NODE_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn next_node_id() -> usize {
    // The use of `Ordering::Relaxed` allows for relaxed consistency,
    // which provides better performance and is sufficient for generating unique IDs in this context.
    // You can look at `test_node_id_atomicity` for a detailed test to validate the atomicity 
    // of the incremental counter across multiple threads.
    NODE_COUNTER.fetch_add(1, Ordering::Relaxed)
}

// Id of the node in the computational graph.
type NodeId = usize;





// Represents a value in the computational graph
// A value can be either a leaf node (input tensor) or an intermediate result from an operation.
#[derive(Debug)]
struct Value {
    id: NodeId,
    op: Option<Box<dyn Operator>>, // Use Box here to allow dynamic dispatch for different operators
   // Apparently, dynamic dispatch is less efficient than static dispatch, but it allows us to use different operators in the same graph.
   // https://softwaremill.com/rust-static-vs-dynamic-dispatch/
   // The thing is that with static dispatch the compiler will perform monomorphization, which means that it will generate a new version of the code for each type used.
   // On the other hand, dynamic dispatch will use a vtable to call the methods of the trait, which is less efficient.
   // I an not sure which one is better for our use case, but I will stick with dynamic dispatch for now.
   // Static dispatch would consume more memory, but would also allow the compiler to perform more optimizations (by means of increasig the compile time).
    inputs: Vec<NodeId>,
    cached_data: Tensor,
    requires_grad: bool,
}

impl Value {
    // Create a new leaf node in the graph
    fn new(data: Tensor, requires_grad: bool) -> Self {
        Self {
            id: next_node_id(),
            op: None,
            inputs: Vec::new(),
            cached_data: data,
            requires_grad,
        }
    }
    
    // Create a new node from an operation
    fn from_op(op: Box<dyn Operator>, inputs: Vec<NodeId>, data: Tensor) -> Self {
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
    }
    
    // Check if this is a leaf node (no operation)
    fn is_leaf(&self) -> bool {
        self.op.is_none()
    }
    
    // Detach from computation graph
    fn detach(&self) -> Self {
        Self {
            id: next_node_id(),
            op: None,
            inputs: Vec::new(),
            cached_data: self.cached_data.detach(),
            requires_grad: false,
        }
    }
}

// Computational graph that manages all values and their relationships
#[derive(Debug)]
struct ComputationGraph {
    // Creating a graph is kind of boilerplate in Rust
    // Thing is i do not want to hhave to grab the graph mutably the graph every time I want to read an item
    // RefCell allows us to have interior mutability, which means we can mutate the value inside the Rc without having to grab a mutable reference to it.
    values: HashMap<NodeId, Rc<RefCell<Value>>>,
    gradients: HashMap<NodeId, Tensor>, // To store gradients for each node
}

impl ComputationGraph {
    fn new() -> Self {
        Self {
            values: HashMap::new(),
            gradients: HashMap::new(),
        }
    }
    
    // Add a value to the graph
    fn add_value(&mut self, value: Value) -> NodeId {
        let id = value.id;
        self.values.insert(id, Rc::new(RefCell::new(value)));
        id
    }
    
    // Create a leaf node
    fn create_tensor(&mut self, data: Tensor, requires_grad: bool) -> NodeId {
        let value = Value::new(data, requires_grad);
        self.add_value(value)
    }
    
    // Create a tensor from data with default device
    fn tensor_from_vec(&mut self, data: Vec<f64>, shape: &[usize], requires_grad: bool) -> Result<NodeId, String> {
        let tensor = Tensor::from_vec(data, shape)?;
        Ok(self.create_tensor(tensor, requires_grad))
    }
    
    // Create zeros tensor
    fn zeros(&mut self, shape: &[usize], requires_grad: bool) -> NodeId {
        let tensor = Tensor::zeros(shape);
        self.create_tensor(tensor, requires_grad)
    }
    
    // Create ones tensor
    fn ones(&mut self, shape: &[usize], requires_grad: bool) -> NodeId {
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
    /// 
    /// These operations are implemented as methods of the `ComputationGraph` struct.
    
    // Addition operation
    fn add(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        let data_a = self.values[&a].borrow().cached_data.clone();
        let data_b = self.values[&b].borrow().cached_data.clone();
        
        let op = Box::new(AddOp);
        let result_data = op.compute(&[data_a, data_b])?;
        
        let value = Value::from_op(op, vec![a, b], result_data);
        Ok(self.add_value(value))
    }
    
    // Scalar addition
    fn add_scalar(&mut self, a: NodeId, scalar: f64) -> Result<NodeId, String> {
        let data_a = self.values[&a].borrow().cached_data.clone();
        
        let op = Box::new(AddScalarOp::new(scalar));
        let result_data = op.compute(&[data_a])?;
        
        let value = Value::from_op(op, vec![a], result_data);
        Ok(self.add_value(value))
    }
    
    // Multiplication operation
    fn mul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        let data_a = self.values[&a].borrow().cached_data.clone();
        let data_b = self.values[&b].borrow().cached_data.clone();
        
        let op = Box::new(MulOp);
        let result_data = op.compute(&[data_a, data_b])?;
        
        let value = Value::from_op(op, vec![a, b], result_data);
        Ok(self.add_value(value))
    }
    
    // Scalar multiplication
    fn mul_scalar(&mut self, a: NodeId, scalar: f64) -> Result<NodeId, String> {
        let data_a = self.values[&a].borrow().cached_data.clone();
        
        let op = Box::new(MulScalarOp::new(scalar));
        let result_data = op.compute(&[data_a])?;
        
        let value = Value::from_op(op, vec![a], result_data);
        Ok(self.add_value(value))
    }
    
    // Division operation
    fn div(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        let data_a = self.values[&a].borrow().cached_data.clone();
        let data_b = self.values[&b].borrow().cached_data.clone();
        
        let op = Box::new(DivOp);
        let result_data = op.compute(&[data_a, data_b])?;
        
        let value = Value::from_op(op, vec![a, b], result_data);
        Ok(self.add_value(value))
    }
    
    // Power operation
    fn pow(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        let data_a = self.values[&a].borrow().cached_data.clone();
        let data_b = self.values[&b].borrow().cached_data.clone();
        
        let op = Box::new(PowOp);
        let result_data = op.compute(&[data_a, data_b])?;
        
        let value = Value::from_op(op, vec![a, b], result_data);
        Ok(self.add_value(value))
    }
    
    // Matrix multiplication operation
    fn matmul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        let data_a = self.values[&a].borrow().cached_data.clone();
        let data_b = self.values[&b].borrow().cached_data.clone();
        
        let op = Box::new(MatMulOp);
        let result_data = op.compute(&[data_a, data_b])?;
        
        let value = Value::from_op(op, vec![a, b], result_data);
        Ok(self.add_value(value))
    }
    
    // ReLU activation
    fn relu(&mut self, a: NodeId) -> Result<NodeId, String> {
        let data_a = self.values[&a].borrow().cached_data.clone();
        
        let op = Box::new(ReLUOp);
        let result_data = op.compute(&[data_a])?;
        
        let value = Value::from_op(op, vec![a], result_data);
        Ok(self.add_value(value))
    }
    
    // Exponential operation
    fn exp(&mut self, a: NodeId) -> Result<NodeId, String> {
        let data_a = self.values[&a].borrow().cached_data.clone();
        
        let op = Box::new(ExpOp);
        let result_data = op.compute(&[data_a])?;
        
        let value = Value::from_op(op, vec![a], result_data);
        Ok(self.add_value(value))
    }
    
    // Logarithm operation
    fn log(&mut self, a: NodeId) -> Result<NodeId, String> {
        let data_a = self.values[&a].borrow().cached_data.clone();
        
        let op = Box::new(LogOp);
        let result_data = op.compute(&[data_a])?;
        
        let value = Value::from_op(op, vec![a], result_data);
        Ok(self.add_value(value))
    }
    
    // Negation operation
    fn negate(&mut self, a: NodeId) -> Result<NodeId, String> {
        let data_a = self.values[&a].borrow().cached_data.clone();
        
        let op = Box::new(NegateOp);
        let result_data = op.compute(&[data_a])?;
        
        let value = Value::from_op(op, vec![a], result_data);
        Ok(self.add_value(value))
    }
    
    // Sum operation
    fn sum(&mut self, a: NodeId, axis: Option<usize>) -> Result<NodeId, String> {
        let data_a = self.values[&a].borrow().cached_data.clone();
        
        let op = Box::new(SumOp::new(axis));
        let result_data = op.compute(&[data_a])?;
        
        let value = Value::from_op(op, vec![a], result_data);
        Ok(self.add_value(value))
    }
    
    // Summation with multiple axes
    fn summation(&mut self, a: NodeId, axes: Option<Vec<usize>>) -> Result<NodeId, String> {
        let data_a = self.values[&a].borrow().cached_data.clone();
        
        let op = Box::new(SummationOp::new(axes));
        let result_data = op.compute(&[data_a])?;
        
        let value = Value::from_op(op, vec![a], result_data);
        Ok(self.add_value(value))
    }
    
    // Transpose operation
    fn transpose(&mut self, a: NodeId, axes: Option<Vec<usize>>) -> Result<NodeId, String> {
        let data_a = self.values[&a].borrow().cached_data.clone();
        
        let op = Box::new(TransposeOp::new(axes));
        let result_data = op.compute(&[data_a])?;
        
        let value = Value::from_op(op, vec![a], result_data);
        Ok(self.add_value(value))
    }
    
    // Reshape operation
    fn reshape(&mut self, a: NodeId, new_shape: Vec<usize>) -> Result<NodeId, String> {
        let data_a = self.values[&a].borrow().cached_data.clone();
        
        let op = Box::new(ReshapeOp::new(new_shape));
        let result_data = op.compute(&[data_a])?;
        
        let value = Value::from_op(op, vec![a], result_data);
        Ok(self.add_value(value))
    }
    
    // Broadcast operation
    fn broadcast_to(&mut self, a: NodeId, target_shape: Vec<usize>) -> Result<NodeId, String> {
        let data_a = self.values[&a].borrow().cached_data.clone();
        
        let op = Box::new(BroadcastToOp::new(target_shape));
        let result_data = op.compute(&[data_a])?;
        
        let value = Value::from_op(op, vec![a], result_data);
        Ok(self.add_value(value))
    }
    
    // Get data from a node
    fn get_data(&self, node_id: NodeId) -> Tensor {
        self.values[&node_id].borrow().cached_data.clone()
    }
    
    // Get shape of a node
    fn get_shape(&self, node_id: NodeId) -> Vec<usize> {
        self.values[&node_id].borrow().cached_data.shape().to_vec()
    }
    
    // Check if node requires gradient
    fn requires_grad(&self, node_id: NodeId) -> bool {
        self.values[&node_id].borrow().requires_grad
    }
    
    // Detach a node from the computation graph
    fn detach(&mut self, node_id: NodeId) -> NodeId {
        let detached_value = self.values[&node_id].borrow().detach();
        self.add_value(detached_value)
    }
    
    // Find topological sort - needed for backward pass
    fn find_topo_sort(&self, output_nodes: &[NodeId]) -> Vec<NodeId> {
        let mut visited = std::collections::HashSet::new();
        let mut topo_order = Vec::new();
        
        for &node_id in output_nodes {
            self.topo_sort_dfs(node_id, &mut visited, &mut topo_order);
        }
        
        topo_order
    }
    
    // Backward pass: compute gradients using reverse-mode automatic differentiation
    // I am using reverse mode as it is more efficent for neural networks and other models where the number of outputs is much smaller than the number of inputs.
    // In forward mode, we would compute the gradient of each output with respect to each input, which is not efficient for large models that can have thousands of inputs.
    fn backward(&mut self, output_node: NodeId) -> Result<(), String> {
        self.compute_gradient_of_variables(output_node, None)
    }
    
    // Backward pass with custom output gradient
    fn backward_with_grad(&mut self, output_node: NodeId, out_grad: Tensor) -> Result<(), String> {
        self.compute_gradient_of_variables(output_node, Some(out_grad))
    }
    
    // Main gradient computation function following the Python course structure
    fn compute_gradient_of_variables(&mut self, output_tensor: NodeId, out_grad: Option<Tensor>) -> Result<(), String> {
        // Clear previous gradients
        self.gradients.clear();
        
        // Initialize gradient of output
        let output_grad = match out_grad {
            Some(grad) => grad,
            None => {
                let output_data = self.values[&output_tensor].borrow().cached_data.clone();
                Tensor::ones(output_data.shape())
            }
        };
        
        // Map from node to list of gradient contributions
        let mut node_to_output_grads_list: HashMap<NodeId, Vec<Tensor>> = HashMap::new();
        node_to_output_grads_list.insert(output_tensor, vec![output_grad]);
        
        // Get reverse topological order
        let reverse_topo_order: Vec<NodeId> = self.find_topo_sort(&[output_tensor])
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
                
                let value = self.values[&node_id].borrow();
                
                // If this node has an operation, compute gradients for its inputs
                if let Some(ref op) = value.op {
                    // Collect input data for gradient computation
                    let input_data: Vec<Tensor> = value.inputs.iter()
                        .map(|&input_id| self.values[&input_id].borrow().cached_data.clone())
                        .collect();
                    
                    // Compute gradients for all inputs
                    let input_grads = op.gradient(&accumulated_grad, &input_data)?;
                    
                    // Accumulate gradients for input nodes
                    for (i, &input_id) in value.inputs.iter().enumerate() {
                        if self.values[&input_id].borrow().requires_grad {
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
    
    // Topological sort helper
    // This function traverses the graph in a depth-first manner to ensure that we process nodes in the correct order.
    // 
    // Parameters:
    // - `node_id`: The current node to process.
    // - `visited`: A set to keep track of visited nodes to avoid cycles.
    // - `topo_order`: A vector to store the nodes in topological order.
    //
    // It is implemented in a recursive manner, which is a common approach for topological sorting in directed acyclic graphs (DAGs)..
    // We should add some kind of cycle detection in the future, but for now we assume that the graph is acyclic.
    fn topological_sort(&self, node_id: NodeId, visited: &mut std::collections::HashSet<NodeId>, topo_order: &mut Vec<NodeId>) {
        if visited.contains(&node_id) {
            return;
        }
        
        visited.insert(node_id);
        
        let value = self.values[&node_id].borrow();
        for &input_id in &value.inputs {
            self.topological_sort(input_id, visited, topo_order);
        }
        
        topo_order.push(node_id);
    }
    
    // DFS helper for topological sort following Python course structure
    fn topo_sort_dfs(&self, node_id: NodeId, visited: &mut std::collections::HashSet<NodeId>, topo_order: &mut Vec<NodeId>) {
        if visited.contains(&node_id) {
            return;
        }
        
        visited.insert(node_id);
        
        let value = self.values[&node_id].borrow();
        for &input_id in &value.inputs {
            self.topo_sort_dfs(input_id, visited, topo_order);
        }
        
        topo_order.push(node_id);
    }
    
    // Get gradient for a node
    fn get_gradient(&self, node_id: NodeId) -> Option<Tensor> {
        self.gradients.get(&node_id).cloned()
    }
    
    // Helper function to sum a list of nodes (useful for loss functions)
    fn sum_node_list(&mut self, node_list: Vec<NodeId>) -> Result<NodeId, String> {
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

fn main() -> Result<(), String> {
    // Example usage of the automatic differentiation engine
    let mut graph = ComputationGraph::new();
    
    // Create input tensors using the new Tensor wrapper
    let x_data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    let w_data = Tensor::from_vec(vec![0.5, 0.2, -0.1, 0.3, -0.4, 0.6], &[3, 2])?;
    
    let x = graph.create_tensor(x_data, true);
    let w = graph.create_tensor(w_data, true);
    
    println!("Created tensors:");
    println!("X shape: {:?}", graph.get_data(x).shape());
    println!("W shape: {:?}", graph.get_data(w).shape());
    
    // Forward pass: y = ReLU(x @ w)
    let matmul_result = graph.matmul(x, w)?;
    let y = graph.relu(matmul_result)?;
    
    // Sum to get scalar loss
    let loss = graph.sum(y, None)?;
    
    println!("\nForward pass completed");
    println!("Loss shape: {:?}", graph.get_data(loss).shape());
    println!("Loss value: {:?}", graph.get_data(loss).data().iter().next().unwrap());
    
    // Backward pass
    graph.backward(loss)?;
    
    println!("\nBackward pass completed");
    
    if let Some(x_grad) = graph.get_gradient(x) {
        println!("Gradient w.r.t. x shape: {:?}", x_grad.shape());
        println!("Gradient w.r.t. x: {:?}", x_grad.data());
    }
    
    if let Some(w_grad) = graph.get_gradient(w) {
        println!("Gradient w.r.t. w shape: {:?}", w_grad.shape());
        println!("Gradient w.r.t. w: {:?}", w_grad.data());
    }
    
    // Demonstrate additional operations following the Python course structure
    println!("\n=== Additional Operations Demo ===");
    
    // Create some test tensors
    let a = graph.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true)?;
    let b = graph.tensor_from_vec(vec![2.0, 1.0, 0.5, 2.0], &[2, 2], true)?;
    
    // Test element-wise operations
    let add_result = graph.add(a, b)?;
    let mul_result = graph.mul(a, b)?;
    let div_result = graph.div(a, b)?;
    
    println!("Element-wise operations:");
    println!("A + B: {:?}", graph.get_data(add_result).data());
    println!("A * B: {:?}", graph.get_data(mul_result).data());
    println!("A / B: {:?}", graph.get_data(div_result).data());
    
    // Test scalar operations
    let scalar_add = graph.add_scalar(a, 10.0)?;
    let scalar_mul = graph.mul_scalar(a, 2.0)?;
    
    println!("\nScalar operations:");
    println!("A + 10: {:?}", graph.get_data(scalar_add).data());
    println!("A * 2: {:?}", graph.get_data(scalar_mul).data());
    
    // Test activations
    let exp_result = graph.exp(a)?;
    let log_result = graph.log(add_result)?; // log of positive values
    let neg_result = graph.negate(a)?;
    
    println!("\nActivations:");
    println!("exp(A): {:?}", graph.get_data(exp_result).data());
    println!("log(A+B): {:?}", graph.get_data(log_result).data());
    println!("-A: {:?}", graph.get_data(neg_result).data());
    
    // Test reshape and transpose
    let reshaped = graph.reshape(a, vec![4, 1])?;
    let transposed = graph.transpose(a, None)?;
    
    println!("\nShape operations:");
    println!("Original A shape: {:?}", graph.get_shape(a));
    println!("Reshaped A shape: {:?}", graph.get_shape(reshaped));
    println!("Transposed A shape: {:?}", graph.get_shape(transposed));
    
    // Test summation with axes
    let sum_all = graph.summation(a, None)?;
    let sum_axis0 = graph.summation(a, Some(vec![0]))?;
    let sum_axis1 = graph.summation(a, Some(vec![1]))?;
    
    println!("\nSummation operations:");
    println!("Sum all: {:?}", graph.get_data(sum_all).data());
    println!("Sum axis 0: {:?}", graph.get_data(sum_axis0).data());
    println!("Sum axis 1: {:?}", graph.get_data(sum_axis1).data());
    
    // Test backward pass on complex computation
    let complex_result = graph.add(mul_result, exp_result)?;
    let final_loss = graph.summation(complex_result, None)?;
    
    graph.backward(final_loss)?;
    
    println!("\nComplex computation gradients:");
    if let Some(a_grad) = graph.get_gradient(a) {
        println!("Gradient w.r.t. A: {:?}", a_grad.data());
    }
    if let Some(b_grad) = graph.get_gradient(b) {
        println!("Gradient w.r.t. B: {:?}", b_grad.data());
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::sync::Arc;
    use std::sync::Barrier;

    #[test]
    fn test_device_operations() {
        let device = cpu();
        
        let zeros = device.zeros(&[2, 3]);
        assert_eq!(zeros.shape(), &[2, 3]);
        assert!(zeros.iter().all(|&x| x == 0.0));
        
        let ones = device.ones(&[2, 3]);
        assert_eq!(ones.shape(), &[2, 3]);
        assert!(ones.iter().all(|&x| x == 1.0));
        
        let full = device.full(&[2, 2], 5.0);
        assert_eq!(full.shape(), &[2, 2]);
        assert!(full.iter().all(|&x| x == 5.0));
    }

    #[test]
    fn test_tensor_creation_with_device() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.len(), 4);
        assert_eq!(tensor.device(), &cpu());
        
        let zeros = Tensor::zeros_with_device(&[3, 3], cpu());
        assert_eq!(zeros.shape(), &[3, 3]);
    }

    #[test]
    fn test_tensor_operations() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        
        let sum = a.add(&b).unwrap();
        let expected = Tensor::from_vec(vec![3.0, 5.0, 7.0, 9.0], &[2, 2]).unwrap();
        assert_eq!(sum, expected);
        
        let mul = a.mul(&b).unwrap();
        let expected_mul = Tensor::from_vec(vec![2.0, 6.0, 12.0, 20.0], &[2, 2]).unwrap();
        assert_eq!(mul, expected_mul);
    }

    #[test]
    fn test_tensor_matmul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        let expected = Tensor::from_vec(vec![22.0, 28.0, 49.0, 64.0], &[2, 2]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_tensor_activations() {
        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        
        // Test ReLU
        let relu_result = input.relu();
        let expected_relu = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        assert_eq!(relu_result, expected_relu);
        
        // Test Sigmoid (should be between 0 and 1)
        let sigmoid_result = input.sigmoid();
        for &val in sigmoid_result.data().iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
        
        // Test Exp
        let exp_result = input.exp();
        assert!(exp_result.data().iter().all(|&x| x > 0.0));
        
        // Test Negate
        let neg_result = input.negate();
        let expected_neg = Tensor::from_vec(vec![2.0, 1.0, 0.0, -1.0, -2.0], &[5]).unwrap();
        assert_eq!(neg_result, expected_neg);
    }

    #[test]
    fn test_tensor_scalar_operations() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        
        let add_scalar = tensor.add_scalar(5.0);
        let expected_add = Tensor::from_vec(vec![6.0, 7.0, 8.0, 9.0], &[2, 2]).unwrap();
        assert_eq!(add_scalar, expected_add);
        
        let mul_scalar = tensor.mul_scalar(2.0);
        let expected_mul = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[2, 2]).unwrap();
        assert_eq!(mul_scalar, expected_mul);
        
        let div_scalar = tensor.div_scalar(2.0);
        let expected_div = Tensor::from_vec(vec![0.5, 1.0, 1.5, 2.0], &[2, 2]).unwrap();
        assert_eq!(div_scalar, expected_div);
    }

    #[test]
    fn test_tensor_transpose_comprehensive() {
        // Test 2D transpose
        let tensor_2d = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        
        // Default transpose (should swap axes)
        let transposed_default = tensor_2d.transpose(None).unwrap();
        assert_eq!(transposed_default.shape(), &[3, 2]);
        
        // Explicit axes transpose
        let transposed_explicit = tensor_2d.transpose(Some(&[1, 0])).unwrap();
        assert_eq!(transposed_explicit.shape(), &[3, 2]);
        assert_eq!(transposed_default.data(), transposed_explicit.data());
        
        // Test 3D transpose
        let tensor_3d = Tensor::from_vec(
            (0..24).map(|x| x as f64).collect(), 
            &[2, 3, 4]
        ).unwrap();
        
        // Default transpose (reverse all axes: [2,3,4] -> [4,3,2])
        let transposed_3d_default = tensor_3d.transpose(None).unwrap();
        assert_eq!(transposed_3d_default.shape(), &[4, 3, 2]);
        
        // Custom permutation: [2,3,4] -> [4,2,3] (axes [2,0,1])
        let transposed_3d_custom = tensor_3d.transpose(Some(&[2, 0, 1])).unwrap();
        assert_eq!(transposed_3d_custom.shape(), &[4, 2, 3]);
        
        // Test 1D and 0D tensors (should be unchanged)
        let tensor_1d = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let transposed_1d = tensor_1d.transpose(None).unwrap();
        assert_eq!(transposed_1d.shape(), &[3]);
        assert_eq!(transposed_1d.data(), tensor_1d.data());
        
        let tensor_0d = Tensor::from_vec(vec![42.0], &[]).unwrap();
        let transposed_0d = tensor_0d.transpose(None).unwrap();
        assert_eq!(transposed_0d.shape(), &[]);
        assert_eq!(transposed_0d.data(), tensor_0d.data());
    }

    #[test]
    fn test_transpose_error_cases() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        
        // Invalid axes length
        assert!(tensor.transpose(Some(&[0])).is_err());
        assert!(tensor.transpose(Some(&[0, 1, 2])).is_err());
        
        // Invalid axes values
        assert!(tensor.transpose(Some(&[0, 2])).is_err()); // 2 is out of bounds
        assert!(tensor.transpose(Some(&[0, 0])).is_err()); // duplicate axis
        assert!(tensor.transpose(Some(&[1, 1])).is_err()); // duplicate axis
    }

    #[test]
    fn test_transpose_gradient() {
        let mut graph = ComputationGraph::new();
        
        // Test 2D transpose gradient
        let a = graph.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true).unwrap();
        
        // Default transpose
        let b = graph.transpose(a, None).unwrap();
        let loss = graph.summation(b, None).unwrap();
        
        graph.backward(loss).unwrap();
        
        let grad = graph.get_gradient(a).unwrap();
        // Gradient should have same shape as original tensor
        assert_eq!(grad.shape(), &[2, 3]);
        // Since we're summing, gradient should be all ones
        assert!(grad.data().iter().all(|&x| (x - 1.0).abs() < 1e-6));
        
        // Test custom transpose gradient
        let mut graph2 = ComputationGraph::new();
        let a2 = graph2.tensor_from_vec(
            (0..24).map(|x| x as f64).collect(), 
            &[2, 3, 4], 
            true
        ).unwrap();
        
        let b2 = graph2.transpose(a2, Some(vec![2, 0, 1])).unwrap();
        let loss2 = graph2.summation(b2, None).unwrap();
        
        graph2.backward(loss2).unwrap();
        
        let grad2 = graph2.get_gradient(a2).unwrap();
        assert_eq!(grad2.shape(), &[2, 3, 4]);
        assert!(grad2.data().iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_node_id_generation() {
        let id1 = next_node_id();
        let id2 = next_node_id();
        assert_ne!(id1, id2);
        assert!(id2 > id1);
    }

    #[test]
    fn test_node_id_atomicity() {
        let num_threads = 10;
        let ids_per_thread = 100;
        let barrier = Arc::new(Barrier::new(num_threads));
        
        let mut handles = vec![];
        
        for _ in 0..num_threads {
            let barrier_clone = Arc::clone(&barrier);
            let handle = thread::spawn(move || {
                barrier_clone.wait();
                let mut ids = vec![];
                for _ in 0..ids_per_thread {
                    ids.push(next_node_id());
                }
                ids
            });
            handles.push(handle);
        }
        
        let mut all_ids = std::collections::HashSet::new();
        for handle in handles {
            let ids = handle.join().unwrap();
            for id in ids {
                assert!(all_ids.insert(id), "Duplicate ID found: {}", id);
            }
        }
        
        assert_eq!(all_ids.len(), num_threads * ids_per_thread);
    }

    #[test]
    fn test_computational_graph_basic() {
        let mut graph = ComputationGraph::new();
        
        let a_data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b_data = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        
        let a = graph.create_tensor(a_data, true);
        let b = graph.create_tensor(b_data, true);
        let c = graph.add(a, b).unwrap();
        
        let result = graph.get_data(c);
        let expected = Tensor::from_vec(vec![3.0, 5.0, 7.0, 9.0], &[2, 2]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_graph_scalar_operations() {
        let mut graph = ComputationGraph::new();
        
        let a = graph.tensor_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true).unwrap();
        
        let add_scalar = graph.add_scalar(a, 5.0).unwrap();
        let mul_scalar = graph.mul_scalar(a, 2.0).unwrap();
        
        let add_result = graph.get_data(add_scalar);
        let expected_add = Tensor::from_vec(vec![6.0, 7.0, 8.0, 9.0], &[2, 2]).unwrap();
        assert_eq!(add_result, expected_add);
        
        let mul_result = graph.get_data(mul_scalar);
        let expected_mul = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[2, 2]).unwrap();
        assert_eq!(mul_result, expected_mul);
    }

    #[test]
    fn test_graph_activations() {
        let mut graph = ComputationGraph::new();
        
        let a = graph.tensor_from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[2, 2], true).unwrap();
        
        let relu_result = graph.relu(a).unwrap();
        let exp_result = graph.exp(a).unwrap();
        let neg_result = graph.negate(a).unwrap();
        
        let relu_data = graph.get_data(relu_result);
        let expected_relu = Tensor::from_vec(vec![0.0, 0.0, 1.0, 2.0], &[2, 2]).unwrap();
        assert_eq!(relu_data, expected_relu);
        
        let neg_data = graph.get_data(neg_result);
        let expected_neg = Tensor::from_vec(vec![1.0, 0.0, -1.0, -2.0], &[2, 2]).unwrap();
        assert_eq!(neg_data, expected_neg);
        
        let exp_data = graph.get_data(exp_result);
        assert!(exp_data.data().iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_backward_pass_simple() {
        let mut graph = ComputationGraph::new();
        
        // Simple case: z = x * y
        let x = graph.tensor_from_vec(vec![3.0], &[1], true).unwrap();
        let y = graph.tensor_from_vec(vec![4.0], &[1], true).unwrap();
        let z = graph.mul(x, y).unwrap();
        
        graph.backward(z).unwrap();
        
        // dz/dx should be y = 4.0
        // dz/dy should be x = 3.0
        let x_grad = graph.get_gradient(x).unwrap();
        let y_grad = graph.get_gradient(y).unwrap();
        
        assert_eq!(x_grad, Tensor::from_vec(vec![4.0], &[1]).unwrap());
        assert_eq!(y_grad, Tensor::from_vec(vec![3.0], &[1]).unwrap());
    }

    #[test]
    fn test_backward_pass_scalar_ops() {
        let mut graph = ComputationGraph::new();
        
        let x = graph.tensor_from_vec(vec![2.0], &[1], true).unwrap();
        let y = graph.add_scalar(x, 3.0).unwrap(); // y = x + 3
        let z = graph.mul_scalar(y, 2.0).unwrap(); // z = 2 * (x + 3)
        
        graph.backward(z).unwrap();
        
        // dz/dx = 2
        let x_grad = graph.get_gradient(x).unwrap();
        assert_eq!(x_grad, Tensor::from_vec(vec![2.0], &[1]).unwrap());
    }

    #[test]
    fn test_neural_network_forward_backward() {
        let mut graph = ComputationGraph::new();
        
        // Simple neural network: y = ReLU(x @ w)
        let x = graph.tensor_from_vec(vec![1.0, 2.0], &[1, 2], true).unwrap();
        let w = graph.tensor_from_vec(vec![0.5, -0.3], &[2, 1], true).unwrap();
        
        let linear = graph.matmul(x, w).unwrap();
        let output = graph.relu(linear).unwrap();
        let loss = graph.sum(output, None).unwrap();
        
        // Forward pass should work
        let loss_data = graph.get_data(loss);
        assert!(loss_data.data().iter().next().unwrap() >= &0.0);
        
        // Backward pass should work
        graph.backward(loss).unwrap();
        
        let x_grad = graph.get_gradient(x);
        let w_grad = graph.get_gradient(w);
        
        assert!(x_grad.is_some());
        assert!(w_grad.is_some());
    }

    #[test]
    fn test_relu_gradient() {
        let mut graph = ComputationGraph::new();
        
        let input_data = Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], &[2, 2]).unwrap();
        let a = graph.create_tensor(input_data, true);
        let b = graph.relu(a).unwrap();
        let loss = graph.sum(b, None).unwrap();
        
        // Check forward pass
        let result = graph.get_data(b);
        let expected = Tensor::from_vec(vec![0.0, 2.0, 0.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(result, expected);
        
        // Check backward pass
        graph.backward(loss).unwrap();
        let grad = graph.get_gradient(a).unwrap();
        let expected_grad = Tensor::from_vec(vec![0.0, 1.0, 0.0, 1.0], &[2, 2]).unwrap();
        assert_eq!(grad, expected_grad);
    }

    #[test]
    fn test_sum_operation() {
        let mut graph = ComputationGraph::new();
        
        let input_data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let a = graph.create_tensor(input_data, true);
        
        // Sum all elements
        let sum_all = graph.sum(a, None).unwrap();
        let result = graph.get_data(sum_all);
        
        assert_eq!(result.data().iter().next().unwrap(), &21.0);
        
        // Test summation with axes
        let sum_axis0 = graph.summation(a, Some(vec![0])).unwrap();
        let sum_axis0_result = graph.get_data(sum_axis0);
        assert_eq!(sum_axis0_result.shape(), &[3]);
    }

    #[test]
    fn test_gradient_accumulation() {
        let mut graph = ComputationGraph::new();
        
        // Test that gradients accumulate correctly when a node is used multiple times
        let x = graph.tensor_from_vec(vec![2.0], &[1], true).unwrap();
        let y1 = graph.mul(x, x).unwrap(); // y1 = x^2
        let y2 = graph.mul(x, x).unwrap(); // y2 = x^2
        let z = graph.add(y1, y2).unwrap(); // z = 2x^2
        
        graph.backward(z).unwrap();
        
        // dz/dx = 4x = 8.0
        let grad = graph.get_gradient(x).unwrap();
        assert_eq!(grad, Tensor::from_vec(vec![8.0], &[1]).unwrap());
    }

    #[test]
    fn test_tensor_broadcasting() {
        let a = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let target_shape = &[2, 3];
        
        let broadcasted = a.broadcast_to(target_shape).unwrap();
        let expected = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[2, 3]).unwrap();
        assert_eq!(broadcasted, expected);
    }

    #[test]
    fn test_tensor_squeeze_unsqueeze() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
        
        // Test squeeze
        let squeezed = tensor.squeeze(Some(0)).unwrap();
        assert_eq!(squeezed.shape(), &[3]);
        
        // Test unsqueeze
        let unsqueezed = squeezed.unsqueeze(1);
        assert_eq!(unsqueezed.shape(), &[3, 1]);
    }

    #[test]
    fn test_complex_computation_graph() {
        let mut graph = ComputationGraph::new();
        
        // Test more complex computation: loss = sum(ReLU((x @ w1 + b1) @ w2))
        let x = graph.tensor_from_vec(vec![1.0, 2.0], &[1, 2], true).unwrap();
        let w1 = graph.tensor_from_vec(vec![0.5, 0.3, -0.2, 0.4], &[2, 2], true).unwrap();
        let w2 = graph.tensor_from_vec(vec![0.1, -0.3], &[2, 1], true).unwrap();
        
        // Forward pass
        let h1 = graph.matmul(x, w1).unwrap();
        let h1_relu = graph.relu(h1).unwrap();
        let output = graph.matmul(h1_relu, w2).unwrap();
        let loss = graph.sum(output, None).unwrap();
        
        // Backward pass
        graph.backward(loss).unwrap();
        
        // Check that all gradients exist
        assert!(graph.get_gradient(x).is_some());
        assert!(graph.get_gradient(w1).is_some());
        assert!(graph.get_gradient(w2).is_some());
        
        // Check gradient shapes
        assert_eq!(graph.get_gradient(x).unwrap().shape(), &[1, 2]);
        assert_eq!(graph.get_gradient(w1).unwrap().shape(), &[2, 2]);
        assert_eq!(graph.get_gradient(w2).unwrap().shape(), &[2, 1]);
    }

    #[test]
    fn test_tensor_error_handling() {
        // Test shape mismatch in addition
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        
        assert!(a.add(&b).is_err());
        
        // Test invalid matrix multiplication
        let c = Tensor::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
        let d = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
        
        assert!(c.matmul(&d).is_err());
    }

    #[test]
    fn test_tensor_reductions() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        
        // Test sum
        let sum_all = tensor.sum(None);
        assert_eq!(sum_all.data().iter().next().unwrap(), &21.0);
        
        // Test mean
        let mean_all = tensor.mean(None);
        assert_eq!(mean_all.data().iter().next().unwrap(), &3.5);
        
        // Test sum along axis
        let sum_axis0 = tensor.sum(Some(0));
        assert_eq!(sum_axis0.shape(), &[3]);
        
        let sum_axis1 = tensor.sum(Some(1));
        assert_eq!(sum_axis1.shape(), &[2]);
    }

    #[test]
    fn test_graph_convenience_methods() {
        let mut graph = ComputationGraph::new();
        
        // Test zeros and ones creation
        let zeros = graph.zeros(&[2, 3], false);
        let ones = graph.ones(&[2, 3], false);
        
        assert_eq!(graph.get_shape(zeros), vec![2, 3]);
        assert_eq!(graph.get_shape(ones), vec![2, 3]);
        
        // Test requires_grad
        assert!(!graph.requires_grad(zeros));
        assert!(!graph.requires_grad(ones));
        
        // Test detach
        let a = graph.tensor_from_vec(vec![1.0, 2.0], &[2], true).unwrap();
        let detached = graph.detach(a);
        
        assert!(graph.requires_grad(a));
        assert!(!graph.requires_grad(detached));
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = ComputationGraph::new();
        
        let a = graph.tensor_from_vec(vec![1.0], &[1], true).unwrap();
        let b = graph.tensor_from_vec(vec![2.0], &[1], true).unwrap();
        let c = graph.add(a, b).unwrap();
        let d = graph.mul(c, a).unwrap();
        
        let topo_order = graph.find_topo_sort(&[d]);
        
        // Check that nodes appear in correct order
        let a_pos = topo_order.iter().position(|&x| x == a).unwrap();
        let b_pos = topo_order.iter().position(|&x| x == b).unwrap();
        let c_pos = topo_order.iter().position(|&x| x == c).unwrap();
        let d_pos = topo_order.iter().position(|&x| x == d).unwrap();
        
        assert!(a_pos < c_pos);
        assert!(b_pos < c_pos);
        assert!(c_pos < d_pos);
    }
}