pub mod graphviz;

use crate::backend::{FerroxCudaF, Tensor};
use crate::ops::Operator;
#[allow(unused_imports)]
pub use graphviz::EngineVisualization;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
/// ATOMIC auto incrementing id for all nodes.
static NODE_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub usize);

impl NodeId {
    pub fn new() -> Self {
        let id = NODE_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self(id)
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NodeId({})", self.0)
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

///  Evaluation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationMode {
    /// Lazy evaluation
    Lazy,
    /// Eager evaluation evaulates inmediately after each operation
    Eager,
}

#[derive(Debug)]
pub enum NodeState<T>
where
    T: FerroxCudaF,
{
    /// Leaf node with materialized tensor.
    Leaf(Tensor<T>),

    /// Node computed but not evaluated yet (lazy mode).
    Pending {
        op: Box<dyn Operator<T>>,
        inputs: Vec<NodeId>,
    },

    /// Node evaluated and cached
    Evaluated {
        tensor: Tensor<T>,
        op: Option<Box<dyn Operator<T>>>, // Some para nodos computados, None para leaf
        inputs: Vec<NodeId>,
    },
}

impl<T> Clone for NodeState<T>
where
    T: FerroxCudaF,
{
    fn clone(&self) -> Self {
        match self {
            NodeState::Leaf(tensor) => NodeState::Leaf(tensor.clone()),

            NodeState::Pending { op, inputs } => NodeState::Pending {
                op: op.clone_op(),
                inputs: inputs.clone(),
            },

            NodeState::Evaluated { tensor, op, inputs } => NodeState::Evaluated {
                tensor: tensor.clone(),
                op: op.as_ref().map(|o| o.clone_op()),
                inputs: inputs.clone(),
            },
        }
    }
}

/// Computational graph node. Supports dual mode.
#[derive(Debug, Clone)]
pub struct Node<T>
where
    T: FerroxCudaF,
{
    pub id: NodeId,
    pub state: NodeState<T>,
    pub requires_grad: bool,
}

impl<T> Node<T>
where
    T: FerroxCudaF,
{
    pub fn new_leaf(tensor: Tensor<T>, requires_grad: bool) -> Self {
        Self {
            id: NodeId::new(),
            state: NodeState::Leaf(tensor),
            requires_grad,
        }
    }

    pub fn new_lazy(op: Box<dyn Operator<T>>, inputs: Vec<NodeId>, requires_grad: bool) -> Self {
        Self {
            id: NodeId::new(),
            state: NodeState::Pending { op, inputs },
            requires_grad,
        }
    }

    pub fn get_op(&self) -> Option<&dyn Operator<T>> {
        match &self.state {
            NodeState::Leaf(_) => None,
            NodeState::Pending { op, .. } => Some(op.as_ref()),
            NodeState::Evaluated { op, .. } => op.as_ref().map(|operation| operation.as_ref()),
        }
    }

    pub fn get_inputs(&self) -> Option<&Vec<NodeId>> {
        match &self.state {
            NodeState::Leaf(_) => None,
            NodeState::Pending { inputs, .. } => Some(inputs),
            NodeState::Evaluated { inputs, .. } => Some(inputs),
        }
    }

    pub fn new_evaluated(
        tensor: Tensor<T>,
        op: Option<Box<dyn Operator<T>>>,
        inputs: Vec<NodeId>,
        requires_grad: bool,
    ) -> Self {
        Self {
            id: NodeId::new(),
            state: NodeState::Evaluated { tensor, op, inputs },
            requires_grad,
        }
    }

    pub fn get_tensor(&self) -> Option<&Tensor<T>> {
        match &self.state {
            NodeState::Leaf(tensor) => Some(tensor),
            NodeState::Evaluated { tensor, .. } => Some(tensor),
            NodeState::Pending { .. } => None,
        }
    }

    pub fn set_tensor(&mut self, new_tensor: Tensor<T>) -> Result<(), String> {
        match &mut self.state {
            NodeState::Leaf(_) => {
                self.state = NodeState::Leaf(new_tensor);
                Ok(())
            }
            NodeState::Evaluated { op, inputs, .. } => {
                let operation = op.take().unwrap();

                self.state = NodeState::Evaluated {
                    op: Some(operation),
                    tensor: new_tensor,
                    inputs: inputs.to_vec(),
                };

                Ok(())
            }
            NodeState::Pending { .. } => {
                Err("Cannot set the tensor of a Pending Node!".to_string())
            }
        }
    }

    pub fn is_evaluated(&self) -> bool {
        !matches!(self.state, NodeState::Pending { .. })
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self.state, NodeState::Leaf(_))
    }
}

/// Main computational graph engine.
#[derive(Debug)]
pub struct AutoFerroxEngine<T>
where
    T: FerroxCudaF,
{
    nodes: HashMap<NodeId, Node<T>>,
    gradients: HashMap<NodeId, Tensor<T>>,
    training_mode: bool,
    evaluation_mode: EvaluationMode,
}

impl<T> Default for AutoFerroxEngine<T>
where
    T: FerroxCudaF,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AutoFerroxEngine<T>
where
    T: FerroxCudaF,
{
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            gradients: HashMap::new(),
            training_mode: true,
            evaluation_mode: EvaluationMode::Eager, // Eager by default as PyTorch
        }
    }

    pub fn get_node(&self, node_id: &NodeId) -> &Node<T> {
        if let Some(node) = self.nodes.get(node_id) {
            node
        } else {
            panic!("Node not found!")
        }
    }

    pub fn set_evaluation_mode(&mut self, mode: EvaluationMode) {
        self.evaluation_mode = mode;
    }

    pub fn get_evaluation_mode(&self) -> EvaluationMode {
        self.evaluation_mode
    }

    pub fn set_training(&mut self, training: bool) {
        self.training_mode = training;
    }

    pub fn lazy_mode(&mut self) {
        self.evaluation_mode = EvaluationMode::Lazy;
    }

    pub fn eager_mode(&mut self) {
        self.evaluation_mode = EvaluationMode::Eager;
    }

    pub fn is_training(&self) -> bool {
        self.training_mode
    }

    pub fn add_node(&mut self, node: Node<T>) -> NodeId {
        let id = node.id;
        self.nodes.insert(id, node);
        id
    }

    pub fn get_gradient(&self, node_id: NodeId) -> Option<&Tensor<T>> {
        self.gradients.get(&node_id)
    }

    pub fn set_gradient(&mut self, node_id: NodeId, grad: Tensor<T>) {
        self.gradients.insert(node_id, grad);
    }

    // Creates a new leaf node in the computational graph
    pub fn create_variable(&mut self, tensor: Tensor<T>, requires_grad: bool) -> NodeId {
        let node = Node::new_leaf(tensor, requires_grad);
        let id = node.id;
        self.nodes.insert(id, node);
        id
    }

    // Batch evaluation for lazy mode
    pub fn evaluate_all(&mut self) -> Result<(), String> {
        // Force evaluation of all pending nodes
        let pending_ids: Vec<NodeId> = self
            .nodes
            .iter()
            .filter_map(|(&id, node)| if !node.is_evaluated() { Some(id) } else { None })
            .collect();

        for id in pending_ids {
            self.evaluate(id)?;
        }
        Ok(())
    }

    fn validate_inputs(&self, op: &dyn Operator<T>, input_ids: &[NodeId]) -> Result<(), String> {
        // Verificar que existen
        for &input_id in input_ids {
            if !self.nodes.contains_key(&input_id) {
                return Err(format!("Input node {} not found", input_id.0));
            }
        }

        // Verificar número correcto
        if input_ids.len() != op.num_inputs() {
            return Err(format!(
                "Operation {} expects {} inputs, got {}",
                op.name(),
                op.num_inputs(),
                input_ids.len()
            ));
        }

        Ok(())
    }

    // Evaluates a single node and takes its computed tensor.
    pub fn evaluate(&mut self, node_id: NodeId) -> Result<&Tensor<T>, String> {
        self.evaluate_node(node_id)?;
        self.get_tensor(node_id)
            .ok_or_else(|| format!("Failed to evaluate node {}", node_id.0))
    }

    fn evaluate_node(&mut self, node_id: NodeId) -> Result<(), String> {
        // Si ya está evaluado, no hacer nada
        if self.is_evaluated(node_id) {
            return Ok(());
        }

        // Obtener información del nodo pending
        let (op, input_ids) = {
            let node = self
                .nodes
                .get(&node_id)
                .ok_or_else(|| format!("Node {} not found", node_id.0))?;

            match &node.state {
                NodeState::Pending { op, inputs } => (op.clone_op(), inputs.clone()),
                _ => return Ok(()), // Ya evaluado
            }
        };

        // Evaluar entradas recursivamente
        for &input_id in &input_ids {
            self.evaluate_node(input_id)?;
        }

        // Obtener tensores de entrada
        let res: Result<Vec<_>, String> = input_ids
            .iter()
            .map(|&input_id| {
                self.get_tensor(input_id)
                    .ok_or_else(|| format!("Input node {} not evaluated", input_id.0))
            })
            .collect();
        let mut input_tensors = res?;

        // Computar resultado
        let result_tensor = op.compute(&mut input_tensors)?;

        // Actualizar nodo a evaluado
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.state = NodeState::Evaluated {
                tensor: result_tensor,
                op: Some(op),
                inputs: input_ids,
            };
        }

        Ok(())
    }

    pub fn get_tensor(&self, node_id: NodeId) -> Option<&Tensor<T>> {
        self.nodes.get(&node_id)?.get_tensor()
    }

    /// Verificar si un nodo está evaluado
    pub fn is_evaluated(&self, node_id: NodeId) -> bool {
        self.nodes
            .get(&node_id)
            .is_some_and(|node| node.is_evaluated())
    }

    pub fn apply_operation(
        &mut self,
        op: Box<dyn Operator<T>>,
        input_ids: Vec<NodeId>,
    ) -> Result<NodeId, String> {
        self.validate_inputs(&op, &input_ids)?;

        match self.evaluation_mode {
            EvaluationMode::Lazy => {
                // Create pending node
                let node = Node::new_lazy(op, input_ids, true);
                let id = node.id;
                self.nodes.insert(id, node);

                Ok(id)
            }
            EvaluationMode::Eager => {
                for &input_id in &input_ids {
                    if !self.is_evaluated(input_id) {
                        self.evaluate_node(input_id)?;
                    }
                }

                // Obtain input tensors.
                let res: Result<Vec<_>, String> = input_ids
                    .iter()
                    .map(|&input_id| {
                        self.get_tensor(input_id)
                            .ok_or_else(|| format!("Input node {} not available", input_id.0))
                    })
                    .collect();

                let mut input_tensors = res?;

                // Evaluate inmediately
                let result_tensor = op.compute(&mut input_tensors)?;
                // Create evaluated node
                let node = Node::new_evaluated(
                    result_tensor,
                    Some(op), // Save the op for the backward pass
                    input_ids,
                    true,
                );
                let id = node.id;
                self.nodes.insert(id, node);

                Ok(id)
            }
        }
    }

    fn accumulate_gradient(&mut self, node_id: NodeId, grad: Tensor<T>) -> Result<(), String> {
        match self.gradients.remove(&node_id) {
            Some(existing_grad) => {
                let accumulated = existing_grad.add(&grad)?;
                self.gradients.insert(node_id, accumulated);
            }
            None => {
                self.gradients.insert(node_id, grad);
            }
        }
        Ok(())
    }

    pub fn backward(&mut self, loss_id: NodeId) -> Result<(), String> {
        if !self.training_mode {
            return Ok(());
        }

        // Verificar que el nodo de pérdida está evaluado
        if !self.is_evaluated(loss_id) {
            return Err(
                "Cannot run backward on unevaluated node. Call evaluate() first.".to_string(),
            );
        }

        // Inicializar gradiente de pérdida
        let loss_tensor = self.get_tensor(loss_id).ok_or("Loss node not found")?;
        println!("Loss {:?}", loss_tensor.clone().into_data()?.as_slice().unwrap());
        let ones_grad = Tensor::ones_with_device(loss_tensor.shape(), loss_tensor.device)?;
        println!("Ones grad {:?}", ones_grad.clone().into_data()?.as_slice().unwrap());
        self.gradients.insert(loss_id, ones_grad);

        // Ordenamiento topológico y propagación
        let mut visited = std::collections::HashSet::new();
        let mut topo_order = Vec::new();
        self.topological_sort(loss_id, &mut visited, &mut topo_order)?;

        topo_order.reverse();
        println!("Topo order reversed: {:?}", topo_order);
        // Propagar gradientes
        for &node_id in &topo_order {
            self.backward_node(node_id)?;
        }

        Ok(())
    }

    /// Backward para un solo nodo
    fn backward_node(&mut self, node_id: NodeId) -> Result<(), String> {

        println!("Backwarding node id {}", node_id);
        // Take ownership of the gradient data
        let grad_output = match self.gradients.remove(&node_id) {
            Some(grad) => grad,
            None => return Ok(()),
        };

        println!("Grad output: {:?}", grad_output.clone().into_data().unwrap().as_slice().unwrap());

        // Obtain node information.
        let node = self
            .nodes
            .get(&node_id)
            .cloned()
            .ok_or_else(|| format!("Node {} not found", node_id.0))?;

        match node.state {
            NodeState::Evaluated { op, inputs, tensor } => {
                // Obtain inputs.
                let res: Result<Vec<_>, String> = inputs
                    .iter()
                    .map(|&input_id| {
                        self.nodes
                            .get(&input_id)
                            .and_then(|node| node.get_tensor())
                            .ok_or_else(|| format!("Input node {} not evaluated", input_id.0))
                    })
                    .collect();
                let mut input_tensors = res?;

                // Compute input gradients.
                let input_grads = match op {
                    Some(o) => {
                        let grad = o.gradient(grad_output, &mut input_tensors, &tensor)?;
                        println!("{}", o.as_ref().clone().name());
                        for g in grad.iter() {
                        println!("Grad : {:?}", g.clone().into_data().unwrap().as_slice().unwrap());
                        }
                        grad
                    },
                    None => {
                        panic!(
                            "Cannot compute gradient. Operation not defined for node: {}",
                            node_id
                        )
                    }
                };

                // Acumulate gradients
                for (input_id, input_grad) in inputs.iter().zip(input_grads) {
                    self.accumulate_gradient(*input_id, input_grad)?;
                }
            }
            _ => {
                // Leaf nodes. The gradient stops here
                self.gradients.insert(node_id, grad_output);
            }
        }

        Ok(())
    }

    /// TOPOLOGICAL SORTING TO COMPUTE OPS IN ORDER
    pub fn topological_sort(
        &self,
        node_id: NodeId,
        visited: &mut std::collections::HashSet<NodeId>,
        topo_order: &mut Vec<NodeId>,
    ) -> Result<(), String> {
        if visited.contains(&node_id) {
            return Ok(());
        }

        visited.insert(node_id);

        if let Some(node) = self.nodes.get(&node_id) {
            if let NodeState::Evaluated { inputs, .. } = &node.state {
                for &input_id in inputs {
                    self.topological_sort(input_id, visited, topo_order)?;
                }
            }
        }

        topo_order.push(node_id);
        Ok(())
    }

    /// GRAPH STATISTICS
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_evaluated_nodes(&self) -> usize {
        self.nodes
            .values()
            .filter(|node| node.is_evaluated())
            .count()
    }

    pub fn num_pending_nodes(&self) -> usize {
        self.nodes
            .values()
            .filter(|node| !node.is_evaluated())
            .count()
    }

    /// Clean up gradients
    pub fn zero_gradients(&mut self) {
        self.gradients.clear();
    }

    /// METHODS REQUIRED FOR LEARNING (OPTIMIZER COMPATIBILITY)
    /// Update a parameter node's tensor data
    /// This is needed for optimizers to modify parameters
    pub fn update_parameter(
        &mut self,
        node_id: NodeId,
        new_tensor: Tensor<T>,
    ) -> Result<(), String> {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            // Check if this is a parameter node (leaf node with requires_grad)
            if node.get_op().is_none() && node.requires_grad {
                // Update the tensor data in the node
                node.set_tensor(new_tensor)?;
                Ok(())
            } else {
                Err("Can only update parameter nodes (leaf nodes with requires_grad)".to_string())
            }
        } else {
            Err(format!("Node {} not found", node_id.0))
        }
    }

    /// Clear gradient for a specific parameter
    pub fn clear_gradient(&mut self, node_id: NodeId) {
        self.gradients.remove(&node_id);
    }

    /// Get tensor data from a node (for reading current parameter values)
    pub fn get_tensor_data(&self, node_id: NodeId) -> Option<&Tensor<T>> {
        self.nodes.get(&node_id)?.get_tensor()
    }

    pub fn clip_gradient(&mut self, node_id: NodeId, clip_coef: T) -> Result<(), String> {
        if let Some(grad) = self.get_gradient(node_id) {
            let clipped_grad = grad.mul_scalar(clip_coef)?;
            self.set_gradient(node_id, clipped_grad);
        }
        Ok(())
    }
}

#[cfg(test)]
mod graph_tests {

    use crate::backend::manager::best_f32_device;
    use crate::backend::Tensor;
    use crate::graph::{AutoFerroxEngine, EvaluationMode};
    use crate::ops::*;

    // Helper functions to create test tensors
    fn create_tensor_2x2(data: [f32; 4]) -> Tensor<f32> {
        let device = best_f32_device();
        Tensor::from_vec_with_device(data.to_vec(), &[2, 2], device)
            .expect("Failed to create tensor")
    }

    fn create_tensor_1d(data: &[f32]) -> Tensor<f32> {
        let device = best_f32_device();
        Tensor::from_vec_with_device(data.to_vec(), &[data.len()], device)
            .expect("Failed to create tensor")
    }

    fn create_engine_with_training() -> AutoFerroxEngine<f32> {
        let mut engine = AutoFerroxEngine::<f32>::new();
        engine.set_training(true); // Enable gradients
        engine
    }

    // Helper to test operation forward pass, gradient computation and shapes
    fn test_operation_with_gradients<Op: Operator<f32> + 'static>(
        op: Op,
        inputs: Vec<Tensor<f32>>,
        expected_output_shape: &[usize],
        test_name: &str,
    ) {
        let mut engine = create_engine_with_training();

        // Create input nodes with gradients enabled
        let input_nodes: Vec<_> = inputs
            .iter()
            .map(|tensor| engine.create_variable(tensor.clone(), true))
            .collect();

        // Apply operation
        let result_node = engine
            .apply_operation(Box::new(op), input_nodes.clone())
            .expect("{test_name} operation failed");

        // Verify forward pass result
        let result_tensor = engine
            .get_tensor(result_node)
            .expect("{test_name} result tensor not found");
        assert_eq!(
            result_tensor.shape(),
            expected_output_shape,
            "{}: shape mismatch",
            test_name
        );

        // Test backward pass - create dummy loss and run backward
        let mean_loss = Box::new(Mean::new());
        let loss_node = engine
            .apply_operation(mean_loss, vec![result_node])
            .expect("{test_name} loss creation failed");

        engine
            .backward(loss_node)
            .expect("{test_name} backward pass failed");

        // Verify gradients exist for all inputs
        for (i, &input_node) in input_nodes.iter().enumerate() {
            let grad = engine.get_gradient(input_node);
            assert!(
                grad.is_some(),
                "{}: gradient missing for input {}",
                test_name,
                i
            );
            assert_eq!(
                grad.unwrap().shape(),
                inputs[i].shape(),
                "{}: gradient shape mismatch for input {}",
                test_name,
                i
            );
        }

        println!("{} test completed successfully", test_name);
    }

    // ========================= BASIC ARITHMETIC OPERATIONS =========================

    #[test]
    fn test_add_operation() {
        // Test element-wise addition with broadcasting support
        let inputs = vec![
            create_tensor_2x2([1.0, 2.0, 3.0, 4.0]),
            create_tensor_2x2([2.0, 3.0, 4.0, 5.0]),
        ];
        test_operation_with_gradients(Add::default(), inputs, &[2, 2], "Add");
    }

    #[test]
    fn test_sub_operation() {
        // Test element-wise subtraction
        let inputs = vec![
            create_tensor_2x2([5.0, 6.0, 7.0, 8.0]),
            create_tensor_2x2([1.0, 2.0, 3.0, 4.0]),
        ];
        test_operation_with_gradients(Sub, inputs, &[2, 2], "Sub");
    }

    #[test]
    fn test_mul_operation() {
        // Test element-wise multiplication
        let inputs = vec![
            create_tensor_1d(&[2.0, 3.0, 4.0]),
            create_tensor_1d(&[5.0, 6.0, 7.0]),
        ];
        test_operation_with_gradients(Mul, inputs, &[3], "Mul");
    }

    #[test]
    fn test_div_operation() {
        // Test element-wise division
        let inputs = vec![
            create_tensor_1d(&[10.0, 15.0, 20.0]),
            create_tensor_1d(&[2.0, 3.0, 4.0]),
        ];
        test_operation_with_gradients(Div, inputs, &[3], "Div");
    }

    // ========================= SCALAR OPERATIONS =========================

    #[test]
    fn test_add_scalar_operation() {
        // Test scalar addition: input + scalar
        let inputs = vec![create_tensor_1d(&[1.0, 2.0, 3.0])];
        test_operation_with_gradients(AddScalar::new(5.0f32), inputs, &[3], "AddScalar");
    }

    #[test]
    fn test_sub_scalar_operation() {
        // Test scalar subtraction: input - scalar
        let inputs = vec![create_tensor_1d(&[10.0, 20.0, 30.0])];
        test_operation_with_gradients(SubScalar::new(5.0f32), inputs, &[3], "SubScalar");
    }

    #[test]
    fn test_mul_scalar_operation() {
        // Test scalar multiplication: input * scalar
        let inputs = vec![create_tensor_1d(&[1.0, 2.0, 3.0])];
        test_operation_with_gradients(MulScalar::new(3.0f32), inputs, &[3], "MulScalar");
    }

    #[test]
    fn test_div_scalar_operation() {
        // Test scalar division: input / scalar
        let inputs = vec![create_tensor_1d(&[6.0, 9.0, 12.0])];
        test_operation_with_gradients(DivScalar::new(3.0f32), inputs, &[3], "DivScalar");
    }

    #[test]
    fn test_power_scalar_operation() {
        // Test scalar power: input ^ scalar
        let inputs = vec![create_tensor_1d(&[2.0, 3.0, 4.0])];
        test_operation_with_gradients(PowerScalar::new(2.0f32), inputs, &[3], "PowerScalar");
    }

    // ========================= MATRIX OPERATIONS =========================

    #[test]
    fn test_matmul_operation() {
        let device = best_f32_device();
        // Test matrix multiplication: A @ B
        let inputs = vec![
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], device)
                .expect("Create matrix A"),
            Tensor::from_vec_with_device(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], device)
                .expect("Create matrix B"),
        ];
        test_operation_with_gradients(MatMul, inputs, &[2, 2], "MatMul");
    }

    #[test]
    fn test_transpose_operation() {
        // Test matrix transpose: A^T
        let device = best_f32_device();
        let inputs =
            vec![
                Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device)
                    .expect("Create matrix"),
            ];
        test_operation_with_gradients(Transpose { axes: None }, inputs, &[3, 2], "Transpose");
    }

    // ========================= UNARY OPERATIONS =========================

    #[test]
    fn test_exp_operation() {
        // Test element-wise exponential: exp(input)
        let inputs = vec![create_tensor_1d(&[0.0, 1.0, 2.0])];
        test_operation_with_gradients(Exp, inputs, &[3], "Exp");
    }

    #[test]
    fn test_log_operation() {
        // Test element-wise natural logarithm: log(input)
        let inputs = vec![create_tensor_1d(&[1.0, 2.0, 3.0])]; // Positive values for log
        test_operation_with_gradients(Log, inputs, &[3], "Log");
    }

    #[test]
    fn test_sqrt_operation() {
        // Test element-wise square root: sqrt(input)
        let inputs = vec![create_tensor_1d(&[1.0, 4.0, 9.0])]; // Perfect squares
        test_operation_with_gradients(Sqrt, inputs, &[3], "Sqrt");
    }

    #[test]
    fn test_abs_operation() {
        // Test element-wise absolute value: abs(input)
        let inputs = vec![create_tensor_1d(&[-2.0, 0.0, 3.0])];
        test_operation_with_gradients(Abs, inputs, &[3], "Abs");
    }

    #[test]
    fn test_neg_operation() {
        // Test element-wise negation: -input
        let inputs = vec![create_tensor_1d(&[1.0, -2.0, 3.0])];
        test_operation_with_gradients(Neg, inputs, &[3], "Neg");
    }

    #[test]
    fn test_power_operation() {
        // Test element-wise power: input1 ^ input2
        let inputs = vec![
            create_tensor_1d(&[2.0, 3.0, 4.0]),
            create_tensor_1d(&[2.0, 2.0, 2.0]),
        ];
        test_operation_with_gradients(Power, inputs, &[3], "Power");
    }

    // ========================= ACTIVATION FUNCTIONS =========================

    #[test]
    fn test_relu_operation() {
        // Test ReLU activation: max(0, input)
        let inputs = vec![create_tensor_1d(&[-2.0, 0.0, 2.0])];
        test_operation_with_gradients(ReLU, inputs, &[3], "ReLU");
    }

    #[test]
    fn test_sigmoid_operation() {
        // Test sigmoid activation: 1 / (1 + exp(-input))
        let inputs = vec![create_tensor_1d(&[-1.0, 0.0, 1.0])];
        test_operation_with_gradients(Sigmoid, inputs, &[3], "Sigmoid");
    }

    #[test]
    fn test_tanh_operation() {
        // Test hyperbolic tangent activation: tanh(input)
        let inputs = vec![create_tensor_1d(&[-1.0, 0.0, 1.0])];
        test_operation_with_gradients(Tanh, inputs, &[3], "Tanh");
    }

    // ========================= REDUCTION OPERATIONS =========================

    #[test]
    fn test_sum_operation() {
        // Test sum reduction: sum(input, axes)
        let inputs = vec![create_tensor_2x2([1.0, 2.0, 3.0, 4.0])];
        test_operation_with_gradients(Sum::new(), inputs, &[], "Sum"); // Scalar result
    }

    #[test]
    fn test_mean_operation() {
        // Test mean reduction: mean(input, axes)
        let inputs = vec![create_tensor_2x2([2.0, 4.0, 6.0, 8.0])];
        test_operation_with_gradients(Mean::new(), inputs, &[], "Mean"); // Scalar result
    }

    #[test]
    fn test_max_operation() {
        // Test max reduction: max(input, axes)
        let inputs = vec![create_tensor_2x2([1.0, 4.0, 2.0, 3.0])];
        test_operation_with_gradients(Max::new(), inputs, &[], "Max"); // Scalar result
    }

    #[test]
    fn test_min_operation() {
        // Test min reduction: min(input, axes)
        let inputs = vec![create_tensor_2x2([3.0, 1.0, 4.0, 2.0])];
        test_operation_with_gradients(Min::new(), inputs, &[], "Min"); // Scalar result
    }

    // ========================= COMPARISON OPERATIONS =========================

    #[test]
    fn test_greater_operation() {
        // Test element-wise greater than: input1 > input2
        let inputs = vec![
            create_tensor_1d(&[1.0, 3.0, 5.0]),
            create_tensor_1d(&[2.0, 3.0, 4.0]),
        ];
        test_operation_with_gradients(Greater, inputs, &[3], "Greater");
    }

    #[test]
    fn test_greater_equal_operation() {
        // Test element-wise greater than or equal: input1 >= input2
        let inputs = vec![
            create_tensor_1d(&[1.0, 3.0, 5.0]),
            create_tensor_1d(&[2.0, 3.0, 4.0]),
        ];
        test_operation_with_gradients(GreaterEqual, inputs, &[3], "GreaterEqual");
    }

    #[test]
    fn test_less_operation() {
        // Test element-wise less than: input1 < input2
        let inputs = vec![
            create_tensor_1d(&[1.0, 3.0, 5.0]),
            create_tensor_1d(&[2.0, 3.0, 4.0]),
        ];
        test_operation_with_gradients(Less, inputs, &[3], "Less");
    }

    #[test]
    fn test_less_equal_operation() {
        // Test element-wise less than or equal: input1 <= input2
        let inputs = vec![
            create_tensor_1d(&[1.0, 3.0, 5.0]),
            create_tensor_1d(&[2.0, 3.0, 4.0]),
        ];
        test_operation_with_gradients(LessEqual, inputs, &[3], "LessEqual");
    }

    #[test]
    fn test_equal_operation() {
        // Test element-wise equality: input1 == input2
        let inputs = vec![
            create_tensor_1d(&[1.0, 3.0, 5.0]),
            create_tensor_1d(&[2.0, 3.0, 4.0]),
        ];
        test_operation_with_gradients(Equal, inputs, &[3], "Equal");
    }

    #[test]
    fn test_clamp_operation() {
        // Test clamp operation: clamp(input, min_val, max_val)
        let inputs = vec![create_tensor_1d(&[-5.0, 0.0, 10.0])];
        test_operation_with_gradients(Clamp::new(-2.0f32, 5.0f32), inputs, &[3], "Clamp");
    }

    // ========================= ELEMENTWISE OPERATIONS =========================

    #[test]
    fn test_max_elementwise_operation() {
        // Test element-wise maximum: max(input1, input2)
        let inputs = vec![
            create_tensor_1d(&[1.0, 5.0, 3.0]),
            create_tensor_1d(&[4.0, 2.0, 6.0]),
        ];
        test_operation_with_gradients(MaxElementwise, inputs, &[3], "MaxElementwise");
    }

    #[test]
    fn test_min_elementwise_operation() {
        // Test element-wise minimum: min(input1, input2)
        let inputs = vec![
            create_tensor_1d(&[1.0, 5.0, 3.0]),
            create_tensor_1d(&[4.0, 2.0, 6.0]),
        ];
        test_operation_with_gradients(MinElementwise, inputs, &[3], "MinElementwise");
    }

    #[test]
    fn test_reciprocal_operation() {
        // Test element-wise reciprocal: 1 / input
        let inputs = vec![create_tensor_1d(&[1.0, 2.0, 4.0])]; // Non-zero values
        test_operation_with_gradients(Reciprocal, inputs, &[3], "Reciprocal");
    }

    #[test]
    fn test_sign_operation() {
        // Test element-wise sign: sign(input)
        let inputs = vec![create_tensor_1d(&[-2.0, 0.0, 3.0])];
        test_operation_with_gradients(Sign, inputs, &[3], "Sign");
    }

    // ========================= RESHAPE OPERATIONS =========================

    #[test]
    fn test_reshape_operation() {
        let device = best_f32_device();
        // Test reshape operation: reshape(input, new_shape)
        let inputs =
            vec![
                Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device)
                    .expect("Create tensor"),
            ];
        test_operation_with_gradients(Reshape::new(vec![3, 2]), inputs, &[3, 2], "Reshape");
    }

    #[test]
    fn test_unsqueeze_operation() {
        // Test unsqueeze operation: unsqueeze(input, axis)
        let inputs = vec![create_tensor_1d(&[1.0, 2.0, 3.0])];
        test_operation_with_gradients(Unsqueeze::new(0), inputs, &[1, 3], "Unsqueeze");
    }

    #[test]
    fn test_squeeze_operation() {
        let device = best_f32_device();
        // Test squeeze operation: squeeze(input, axis)
        let inputs = vec![
            Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0], &[1, 3], device)
                .expect("Create tensor with size-1 dimension"),
        ];
        test_operation_with_gradients(Squeeze::at_axis(0), inputs, &[3], "Squeeze");
    }

    #[test]
    fn test_broadcast_to_operation() {
        let device = best_f32_device();
        // Test broadcast_to operation: broadcast_to(input, target_shape)
        let inputs = vec![
            Tensor::from_vec_with_device(vec![1.0, 2.0], &[2], device).expect("Create tensor")
        ];
        test_operation_with_gradients(BroadcastTo::new(vec![3, 2]), inputs, &[3, 2], "BroadcastTo");
    }

    // ========================= INTEGRATION TEST =========================
    #[test]
    fn test_full_computational_graph() {
        // Complete end-to-end test demonstrating all major operation categories
        // This simulates a realistic neural network computation with multiple operation types
        let device = best_f32_device();
        let mut engine = create_engine_with_training();

        // Create input data representing a small batch of features
        let input_data = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], device)
            .expect("Failed to create input");
        let weights = Tensor::from_vec_with_device(vec![0.5, 0.3, 0.7, 0.1], &[2, 2], device)
            .expect("Failed to create weights");
        let bias = 0.2;

        let input_node = engine.create_variable(input_data, true);
        let weights_node = engine.create_variable(weights, true);

        // Step 1: Linear transformation (MatMul + Add)
        let linear_op = Box::new(MatMul);
        let linear_result = engine
            .apply_operation(linear_op, vec![input_node, weights_node])
            .expect("Linear transformation failed");

        println!("Matmul operation computed");
        let add_bias_op = Box::new(AddScalar::new(bias));
        let biased_result = engine
            .apply_operation(add_bias_op, vec![linear_result])
            .expect("Bias addition failed");

        println!("Add scalar op computed");

        // Step 2: Apply activation function (ReLU)
        let relu_op = Box::new(ReLU);
        let activated_result = engine
            .apply_operation(relu_op, vec![biased_result])
            .expect("ReLU activation failed");

        // Step 3: Apply mathematical transformation (Sqrt of Abs)
        let abs_op = Box::new(Abs);
        let abs_result = engine
            .apply_operation(abs_op, vec![activated_result])
            .expect("Abs operation failed");

        let sqrt_op = Box::new(Sqrt);
        let sqrt_result = engine
            .apply_operation(sqrt_op, vec![abs_result])
            .expect("Sqrt operation failed");

        // Step 4: Scalar operations (multiply by learning rate, add regularization)
        let lr_mul_op = Box::new(MulScalar::new(0.01f32));
        let scaled_result = engine
            .apply_operation(lr_mul_op, vec![sqrt_result])
            .expect("Learning rate scaling failed");

        let reg_add_op = Box::new(AddScalar::new(1e-6f32));
        let regularized_result = engine
            .apply_operation(reg_add_op, vec![scaled_result])
            .expect("Regularization failed");

        // Step 5: Apply comparison and clamp (numerical stability)
        let clamp_op = Box::new(Clamp::new(1e-10f32, 1e10f32));
        let clamped_result = engine
            .apply_operation(clamp_op, vec![regularized_result])
            .expect("Clamping failed");

        // Step 6: Reduction to loss (Mean)
        let mean_op = Box::new(Mean::new());
        let loss_node = engine
            .apply_operation(mean_op, vec![clamped_result])
            .expect("Mean reduction failed");

        // Verify forward pass completed successfully
        let loss_tensor = engine.get_tensor(loss_node).expect("Loss tensor not found");
        assert_eq!(loss_tensor.shape(), &[]); // Scalar loss

        // Step 7: Test backward pass - compute all gradients
        engine.backward(loss_node).expect("Backward pass failed");

        // Verify gradients exist for all trainable parameters
        let input_grad = engine.get_gradient(input_node);
        let weights_grad = engine.get_gradient(weights_node);

        assert!(input_grad.is_some(), "Input gradient should exist");
        assert!(weights_grad.is_some(), "Weights gradient should exist");

        // Verify gradient shapes match parameter shapes
        assert_eq!(input_grad.unwrap().shape(), &[2, 2]);
        assert_eq!(weights_grad.unwrap().shape(), &[2, 2]);

        // Step 8: Test lazy vs eager evaluation modes
        engine.set_evaluation_mode(EvaluationMode::Lazy);

        // Create new computation graph in lazy mode
        let lazy_input = create_tensor_2x2([5.0, 6.0, 7.0, 8.0]);
        let lazy_input_node = engine.create_variable(lazy_input, false);

        // Chain multiple operations in lazy mode
        let exp_op = Box::new(Exp);
        let exp_result = engine
            .apply_operation(exp_op, vec![lazy_input_node])
            .expect("Lazy Exp failed");

        let log_op = Box::new(Log);
        let log_result = engine
            .apply_operation(log_op, vec![exp_result])
            .expect("Lazy Log failed");

        let sigmoid_op = Box::new(Sigmoid);
        let sigmoid_result = engine
            .apply_operation(sigmoid_op, vec![log_result])
            .expect("Lazy Sigmoid failed");

        // Operations should not be evaluated yet in lazy mode
        assert!(!engine.is_evaluated(exp_result), "Exp should be lazy");
        assert!(!engine.is_evaluated(log_result), "Log should be lazy");
        assert!(
            !engine.is_evaluated(sigmoid_result),
            "Sigmoid should be lazy"
        );

        // Force evaluation of entire lazy chain
        let final_result = engine
            .evaluate(sigmoid_result)
            .expect("Lazy evaluation failed");
        assert_eq!(final_result.shape(), &[2, 2]);

        // Now all nodes should be evaluated
        assert!(
            engine.is_evaluated(exp_result),
            "Exp should now be evaluated"
        );
        assert!(
            engine.is_evaluated(log_result),
            "Log should now be evaluated"
        );
        assert!(
            engine.is_evaluated(sigmoid_result),
            "Sigmoid should now be evaluated"
        );

        println!("Computational graph test with ALL operations completed successfully!");
    }
}
