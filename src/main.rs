use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use ndarray::{ArrayD, Array, IxDyn, Axis};

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

// Tensor wrapper to handle dynamic arrays more elegantly
#[derive(Debug, Clone)]
pub struct Tensor {
    data: ArrayD<f64>,
}

impl Tensor {
    pub fn new(data: ArrayD<f64>) -> Self {
        Self { data }
    }
    
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::zeros(IxDyn(shape))
        }
    }
    
    pub fn ones(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::ones(IxDyn(shape))
        }
    }
    
    pub fn from_vec(data: Vec<f64>, shape: &[usize]) -> Result<Self, String> {
        let total_elements: usize = shape.iter().product();
        if data.len() != total_elements {
            return Err(format!(
                "Data length {} doesn't match shape {:?} (expected {})", 
                data.len(), shape, total_elements
            ));
        }
        
        match Array::from_shape_vec(IxDyn(shape), data) {
            Ok(array) => Ok(Self { data: array }),
            Err(e) => Err(format!("Failed to create tensor: {}", e)),
        }
    }
    
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }
    
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    pub fn data(&self) -> &ArrayD<f64> {
        &self.data
    }
    
    pub fn into_data(self) -> ArrayD<f64> {
        self.data
    }
    
    // Element-wise operations
    pub fn add(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape() != other.shape() {
            return Err(format!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape()));
        }
        Ok(Tensor::new(&self.data + &other.data))
    }
    
    pub fn mul(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape() != other.shape() {
            return Err(format!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape()));
        }
        Ok(Tensor::new(&self.data * &other.data))
    }
    
    // Matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err("Matrix multiplication requires 2D tensors".to_string());
        }
        
        let a_shape = self.shape();
        let b_shape = other.shape();
        
        if a_shape[1] != b_shape[0] {
            return Err(format!(
                "Matrix multiplication shape mismatch: ({}, {}) @ ({}, {})", 
                a_shape[0], a_shape[1], b_shape[0], b_shape[1]
            ));
        }
        

        // Convert a and b to 2D views.
        // In `ndarray` a view is a read-only reference to the data.
        // We use `into_dimensionality` to ensure the data is treated as 2D.
        let a = self.data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b = other.data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        

        // Dot product of two 2D arrays, gives the matrix multiplication result.
        // Look at `ndarray` documentation for more details on `dot`.
        // https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.dot
        let result = a.dot(&b);
        Ok(Tensor::new(result.into_dyn()))
    }
    
    // Activation functions
    pub fn relu(&self) -> Tensor {
        // ReLU activation function: max(0, x)
        Tensor::new(self.data.mapv(|x| x.max(0.0)))
    }
    
    pub fn sigmoid(&self) -> Tensor {
        // Euler's sigmoid function: 1 / (1 + exp(-x))
        // equivalent to sin(x)
        Tensor::new(self.data.mapv(|x| 1.0 / (1.0 + (-x).exp())))
    }
    
    // Reduction operations. 
    // As we do not know the shape of the tensor at compile time, we use `ndarray`'s dynamic arrays.
    // We can sum or mean over a specific axis or all elements, it is up to the user to provide the axis over which to perform the reduction operation.
    pub fn sum(&self, axis: Option<usize>) -> Tensor {
        match axis {
            Some(ax) => {
                let result = self.data.sum_axis(Axis(ax));
                Tensor::new(result)
            }
            None => {
                // If axis is not provided we just sum all elements
                let total_sum = self.data.sum();
                Tensor::new(ArrayD::from_elem(IxDyn(&[]), total_sum))
            }
        }
    }
    
    pub fn mean(&self, axis: Option<usize>) -> Tensor {
        match axis {
            Some(ax) => {
                let result = self.data.mean_axis(Axis(ax)).unwrap();
                Tensor::new(result)
            }
            None => {
                let total_mean = self.data.mean().unwrap();
                Tensor::new(ArrayD::from_elem(IxDyn(&[]), total_mean))
            }
        }
    }
    
    // Broadcasting for gradient computation
    // Broadcasting allows us to perform operations on tensors of different shapes.
    // As can be seen on Ndarray's docs, broadcast function returns None if the shapes of the tensors cannot be broadcasted together.
    // https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.broadcast
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Tensor, String> {
        match self.data.broadcast(target_shape) {
            Some(broadcasted) => Ok(Tensor::new(broadcasted.to_owned())),
            None => Err(format!("Cannot broadcast {:?} to {:?}", self.shape(), target_shape)),
        }
    }
    
    // Similar to tf.expand_dims, this function adds a new dimension at the specified axis.
    pub fn unsqueeze(&self, axis: usize) -> Tensor {
        let expanded = self.data.clone().insert_axis(Axis(axis));
        Tensor::new(expanded)
    }
    
    // Basically the opposite of unsqueeze, this function removes a dimension of size 1 from the tensor.
    // We need to check the size of the axis before removing it, as it is not possible to remove an axis with size greater than 1.
    // Imagine a tensor: [[[1, 3, 1, 5],[1,2,3,4]],[[1, 3, 1, 5],[1,2,3,4]],] if we try to squeeze axis 1, we would need to remove the two elements on that axis, 
    // which is not the purpose of the squeeze operation.
    pub fn squeeze(&self, axis: Option<usize>) -> Result<Tensor, String> {
        match axis {
            Some(ax) => {
                if self.shape()[ax] != 1 {
                    return Err(format!("Cannot squeeze axis {} with size {}", ax, self.shape()[ax]));
                }
                let squeezed = self.data.clone().remove_axis(Axis(ax));
                Ok(Tensor::new(squeezed))
            }
            None => {
                // Remove all dimensions of size 1
                let mut result = self.data.clone();
                let mut axis_to_remove = Vec::new();
                
                for (i, &size) in self.shape().iter().enumerate() {
                    if size == 1 {
                        axis_to_remove.push(i);
                    }
                }
                
                // Remove axes in reverse order to maintain indices
                for &ax in axis_to_remove.iter().rev() {
                    result = result.remove_axis(Axis(ax));
                }
                
                Ok(Tensor::new(result))
            }
        }
    }
}

// Implement equality for testing
impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for Tensor {}

trait Operator: std::fmt::Debug {
    // Defines the interface for operators in the computational graph
    // Compute function computes the output in the computational graph.
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String>;
    
    // Gradient function computes the gradient of the output with respect to the inputs.
    fn gradient(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String>;
    
    // Get number of inputs this operator expects
    fn num_inputs(&self) -> usize;
}

// Basic arithmetic operators
// Curently all operators are assuming that we are only going to use them with tensors.
// I will probably add some trait bounds to ensure that the inputs are tensors in the future.
#[derive(Debug, Clone)]
struct AddOp;

impl Operator for AddOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 2 {
            return Err("AddOp requires exactly 2 inputs".to_string());
        }
        inputs[0].add(&inputs[1])
    }
    
    fn gradient(&self, grad_output: &Tensor, _inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        // Gradient of addition: both inputs get the same gradient
        Ok(vec![grad_output.clone(), grad_output.clone()])
    }
    
    fn num_inputs(&self) -> usize { 2 }
}

#[derive(Debug, Clone)]
struct MulOp;

impl Operator for MulOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 2 {
            return Err("MulOp requires exactly 2 inputs".to_string());
        }
        inputs[0].mul(&inputs[1])
    }
    
    fn gradient(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        // Gradient of multiplication: d(a*b)/da = b, d(a*b)/db = a
        let grad_a = grad_output.mul(&inputs[1])?;
        let grad_b = grad_output.mul(&inputs[0])?;
        Ok(vec![grad_a, grad_b])
    }
    
    fn num_inputs(&self) -> usize { 2 }
}

#[derive(Debug, Clone)]
struct MatMulOp;

impl Operator for MatMulOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 2 {
            return Err("MatMulOp requires exactly 2 inputs".to_string());
        }
        inputs[0].matmul(&inputs[1])
    }
    
    fn gradient(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        // For C = A @ B:
        // dC/dA = grad_output @ B^T
        // dC/dB = A^T @ grad_output
        
        let a_data = inputs[0].data().view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b_data = inputs[1].data().view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let grad_2d = grad_output.data().view().into_dimensionality::<ndarray::Ix2>().unwrap();
        
        let grad_a = grad_2d.dot(&b_data.t());
        let grad_b = a_data.t().dot(&grad_2d);
        
        Ok(vec![
            Tensor::new(grad_a.into_dyn()),
            Tensor::new(grad_b.into_dyn())
        ])
    }
    
    fn num_inputs(&self) -> usize { 2 }
}

#[derive(Debug, Clone)]
struct ReLUOp;

impl Operator for ReLUOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 1 {
            return Err("ReLUOp requires exactly 1 input".to_string());
        }
        Ok(inputs[0].relu())
    }
    
    fn gradient(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        // Gradient of ReLU: 1 if input > 0, 0 otherwise
        let mask = Tensor::new(inputs[0].data().mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }));
        let grad = grad_output.mul(&mask)?;
        Ok(vec![grad])
    }
    
    fn num_inputs(&self) -> usize { 1 }
}

#[derive(Debug, Clone)]
struct SumOp {
    axis: Option<usize>,
}

impl SumOp {
    fn new(axis: Option<usize>) -> Self {
        Self { axis }
    }
}

impl Operator for SumOp {
    fn compute(&self, inputs: &[Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 1 {
            return Err("SumOp requires exactly 1 input".to_string());
        }
        Ok(inputs[0].sum(self.axis))
    }
    
    fn gradient(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        // Gradient of sum: broadcast gradient back to original shape
        let input_shape = inputs[0].shape();
        
        let broadcasted_grad = match self.axis {
            Some(axis) => {
                // For axis-specific sum, we need to expand dimensions back
                let expanded = grad_output.unsqueeze(axis);
                expanded.broadcast_to(input_shape)?
            }
            None => {
                // Sum over all dimensions: broadcast scalar gradient to input shape
                grad_output.broadcast_to(input_shape)?
            }
        };
        
        Ok(vec![broadcasted_grad])
    }
    
    fn num_inputs(&self) -> usize { 1 }
}

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

    /// GRAPH OPERATIONS:
    /// 
    /// Currently, implemented graph operations include:
    /// 
    /// - Addition
    /// - Multiplication
    /// - Matrix multiplication
    /// - ReLU activation
    /// - Sum operation
    /// 
    /// Operations in the TO-DO list include:
    /// 
    /// - Scalar multiplication
    /// - Scalar power
    /// - Scalar division
    /// - Tensor transpose
    /// - Tensor reshape
    /// - Exponential function
    /// - Logarithm function
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
    
    // Multiplication operation
    fn mul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        let data_a = self.values[&a].borrow().cached_data.clone();
        let data_b = self.values[&b].borrow().cached_data.clone();
        
        let op = Box::new(MulOp);
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
    
    // Sum operation
    fn sum(&mut self, a: NodeId, axis: Option<usize>) -> Result<NodeId, String> {
        let data_a = self.values[&a].borrow().cached_data.clone();
        
        let op = Box::new(SumOp::new(axis));
        let result_data = op.compute(&[data_a])?;
        
        let value = Value::from_op(op, vec![a], result_data);
        Ok(self.add_value(value))
    }
    
    // Get data from a node
    fn get_data(&self, node_id: NodeId) -> Tensor {
        self.values[&node_id].borrow().cached_data.clone()
    }
    
    // Backward pass: compute gradients using reverse-mode automatic differentiation
    // I am using reverse mode as it is more efficent for neural networks and other models where the number of outputs is much smaller than the number of inputs.
    // In forward mode, we would compute the gradient of each output with respect to each input, which is not efficient for large models that can have thousands of inputs.
    fn backward(&mut self, output_node: NodeId) -> Result<(), String> {
        // Clear previous gradients
        self.gradients.clear();
        
        // Initialize gradient of output to ones
        let output_data = self.values[&output_node].borrow().cached_data.clone();
        let output_shape = output_data.shape();
        self.gradients.insert(output_node, Tensor::ones(output_shape));
        
        // Topological sort to process nodes in reverse order
        let mut visited = std::collections::HashSet::new();
        let mut topo_order = Vec::new();
        self.topological_sort(output_node, &mut visited, &mut topo_order);
        
        // Process nodes in reverse topological order
        // This means we start from the leaf and porpoagate the gradients back to the inputs.
        for &node_id in topo_order.iter().rev() {
            if let Some(grad_output) = self.gradients.get(&node_id).cloned() {
                let value = self.values[&node_id].borrow();
                
                if let Some(ref op) = value.op {
                    // Collect input data for gradient computation
                    let input_data: Vec<Tensor> = value.inputs.iter()
                        .map(|&input_id| self.values[&input_id].borrow().cached_data.clone())
                        .collect();
                    
                    // Compute and accumulate gradients for all the inputs of this node.
                    let input_grads = op.gradient(&grad_output, &input_data)?;
                    
                
                    for (i, &input_id) in value.inputs.iter().enumerate() {
                        if self.values[&input_id].borrow().requires_grad {
                            // If the input requires gradient, accumulate the gradient
                            // If the input already has a gradient, we add the new gradient to it.
                            // This code lazily initializes the gradient for the input if it does not exist.
                            //I think this is a good way to handle the case where a node is used multiple times in the graph.
                            let entry = self.gradients.entry(input_id).or_insert_with(|| {
                                Tensor::zeros(self.values[&input_id].borrow().cached_data.shape())
                            });
                            *entry = entry.add(&input_grads[i])?;
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
    
    // Get gradient for a node
    fn get_gradient(&self, node_id: NodeId) -> Option<Tensor> {
        self.gradients.get(&node_id).cloned()
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
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::sync::Arc;
    use std::sync::Barrier;

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.len(), 4);
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
    fn test_backward_pass_simple() {
        let mut graph = ComputationGraph::new();
        
        // Simple case: z = x * y
        let x = graph.create_tensor(Tensor::from_vec(vec![3.0], &[1]).unwrap(), true);
        let y = graph.create_tensor(Tensor::from_vec(vec![4.0], &[1]).unwrap(), true);
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
    fn test_neural_network_forward_backward() {
        let mut graph = ComputationGraph::new();
        
        // Simple neural network: y = ReLU(x @ w)
        let x = graph.create_tensor(
            Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).unwrap(), 
            true
        );
        let w = graph.create_tensor(
            Tensor::from_vec(vec![0.5, -0.3], &[2, 1]).unwrap(), 
            true
        );
        
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
    }

    #[test]
    fn test_gradient_accumulation() {
        let mut graph = ComputationGraph::new();
        
        // Test that gradients accumulate correctly when a node is used multiple times
        let x = graph.create_tensor(Tensor::from_vec(vec![2.0], &[1]).unwrap(), true);
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
        let x = graph.create_tensor(
            Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).unwrap(), 
            true
        );
        let w1 = graph.create_tensor(
            Tensor::from_vec(vec![0.5, 0.3, -0.2, 0.4], &[2, 2]).unwrap(), 
            true
        );
        let w2 = graph.create_tensor(
            Tensor::from_vec(vec![0.1, -0.3], &[2, 1]).unwrap(), 
            true
        );
        
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
}