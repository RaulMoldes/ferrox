pub mod graphviz;

use crate::backend::{FerroxCudaF, Tensor};
use crate::ops::Operator;
#[allow(unused_imports)]
pub use graphviz::EngineVisualization;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashSet;
/// ATOMIC auto incrementing id for all nodes.
static NODE_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
pub struct MemoryStats {
    pub total_nodes: usize,
    pub cleanable_nodes: usize,
    pub persistent_nodes: usize,
    pub gradient_count: usize,
}

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
        tensor: Option<Tensor<T>>, // Some OPS do not require the output to compute the gradient. This saves memory. In the ops that need it, there is a trade off between the cost of recomputation and the cost of caching (memory BW mainly), that we have to manage.
        op: Option<Box<dyn Operator<T>>>, // Some para nodos computados, None para leaf
        inputs: Vec<NodeId>,
    },
}

impl<T> NodeState<T>
where
    T: FerroxCudaF,
{
    fn into_state(self, new_state: NodeState<T>) -> NodeState<T> {
        new_state
    }

    fn into_data(self) -> Option<Tensor<T>> {
        match self {
            NodeState::Evaluated { tensor, .. } => tensor,
            NodeState::Leaf(tensor) => Some(tensor),
            NodeState::Pending { .. } => panic!("Cannot take the data of a pending tensor"),
        }
    }

    fn into_inputs(self) -> Vec<NodeId> {
        match self {
            NodeState::Evaluated { inputs, .. } => inputs,
            NodeState::Leaf(_) => panic!("Cannot take the  inputs of a leaf tensor"),
            NodeState::Pending { inputs, .. } => inputs,
        }
    }
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
    pub ref_count: usize,
    pub is_persistent: bool,
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
            ref_count: 0,
            is_persistent: requires_grad,
        }
    }

    pub fn new_lazy(op: Box<dyn Operator<T>>, inputs: Vec<NodeId>, requires_grad: bool) -> Self {
        Self {
            id: NodeId::new(),
            state: NodeState::Pending { op, inputs },
            requires_grad,
            ref_count: 0,
            is_persistent: false,
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

    pub fn into_inputs(self) -> Vec<NodeId> {
        self.state.into_inputs()
    }

    pub fn into_data(self) -> Option<Tensor<T>> {
        self.state.into_data()
    }

    pub fn new_evaluated(
        tensor: Option<Tensor<T>>,
        op: Option<Box<dyn Operator<T>>>,
        inputs: Vec<NodeId>,
        requires_grad: bool,
    ) -> Self {
        Self {
            id: NodeId::new(),
            state: NodeState::Evaluated { tensor, op, inputs },
            requires_grad,
            ref_count: 0,
            is_persistent: false,
        }
    }

    pub fn get_tensor(&self) -> Option<&Tensor<T>> {
        match &self.state {
            NodeState::Leaf(tensor) => Some(tensor),
            NodeState::Evaluated { tensor, .. } => Some(tensor.as_ref()?),
            NodeState::Pending { .. } => None,
        }
    }

    pub fn get_tensor_mut(&mut self) -> Option<&mut Tensor<T>> {
        match &mut self.state {
            NodeState::Leaf(ref mut tensor) => Some(tensor),
            NodeState::Evaluated { ref mut tensor, .. } => Some(tensor.as_mut()?),
            NodeState::Pending { .. } => None,
        }
    }

    pub fn into_tensor(self) -> Option<Tensor<T>> {
        match self.state {
            NodeState::Leaf(tensor) => Some(tensor),
            NodeState::Evaluated { tensor, .. } => tensor,
            NodeState::Pending { .. } => None,
        }
    }

    pub fn into_state(self) -> NodeState<T> {
        self.state
    }

    pub fn get_tensor_owned(&self) -> Option<Tensor<T>> {
        match &self.state {
            NodeState::Leaf(tensor) => Some(tensor.clone()),
            NodeState::Evaluated { tensor, .. } => tensor.clone(),
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
                    tensor: Some(new_tensor),
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

    /// Increment reference count
    pub fn increment_ref(&mut self) {
        self.ref_count += 1;
    }

    /// Decrement reference count and return true if can be cleaned
    pub fn decrement_ref(&mut self) -> bool {
        if self.ref_count > 0 {
            self.ref_count -= 1;
        }
        self.can_be_cleaned()
    }

    /// Check if this node can be safely removed from memory
    pub fn can_be_cleaned(&self) -> bool {
        !self.is_persistent && self.ref_count == 0
    }

    /// Mark this node as persistent (never auto-clean)
    pub fn mark_persistent(&mut self) {
        self.is_persistent = true;
    }

    pub fn is_parameter(&self) -> bool {
        self.is_leaf() && self.requires_grad
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
    cache_outputs: bool
}

impl<T> Default for AutoFerroxEngine<T>
where
    T: FerroxCudaF,
{
    fn default() -> Self {
        Self::new(false)
    }
}

impl<T> AutoFerroxEngine<T>
where
    T: FerroxCudaF,
{
    pub fn new(cache_outputs: bool) -> Self {
        Self {
            nodes: HashMap::new(),
            gradients: HashMap::new(),
            training_mode: true,
            evaluation_mode: EvaluationMode::Eager, // Eager by default as PyTorch
            cache_outputs
        }
    }

    //// UTILITY METHODS ////
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
        // Verify inputs exist
        for &input_id in input_ids {
            if !self.nodes.contains_key(&input_id) {
                return Err(format!("Input node {} not found", input_id.0));
            }
        }

        // Verify the number matches with the operation required inputs.
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
        self.safe_evaluate_node(node_id)?;
        self.get_tensor(node_id)
            .ok_or_else(|| format!("Failed to evaluate node {}", node_id.0))
    }

    // Checks if the node is evaluated.
    // If already evaluated, does nothing
    // If not, evaluates all inputs and computes, then frees up everything it does not need anymore.
    fn safe_evaluate_node(&mut self, node_id: NodeId) -> Result<(), String> {
        // Si ya está evaluado, no hacer nada
        if self.is_evaluated(node_id) {
            return Ok(());
        }

        // If not evaluated take the pending node.
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

        // Evaluate all inputs recursively
        for &input_id in &input_ids {
            self.safe_evaluate_node(input_id)?;
        }

        // Obtain input tensors.
        let res: Result<Vec<_>, String> = input_ids
            .iter()
            .map(|&input_id| {
                self.get_tensor(input_id)
                    .ok_or_else(|| format!("Input node {} not evaluated", input_id.0))
            })
            .collect();

        let mut input_tensors = res?;

        // Compute result
        let result_tensor = op.compute(&mut input_tensors)?;

        // Clean up memory from the input nodes for efficiency.
        for input_id in &input_ids {
            // Decrement reference count
            let can_cleanup = self.decrement_ref_count(*input_id);

            if can_cleanup {
                // Only cleanup if this node doesn't need gradients
                if let Some(node) = self.nodes.get(&input_id) {
                    // Safe to cleanup if:
                    // 1. Not a parameter (leaf node with requires_grad)
                    // 2. Not persistent
                    // 3. Ref count is zero
                    let is_parameter = node.is_leaf() && node.requires_grad;

                    if !is_parameter && !node.is_persistent {
                        let _ = self.try_clear_node(*input_id);
                    }
                }
            }
        }

        // Update current state to evaluated.
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.state = NodeState::Evaluated {
                tensor: Some(result_tensor),
                op: Some(op),
                inputs: input_ids,
            };
        }

        Ok(())
    }

    pub fn get_tensor(&self, node_id: NodeId) -> Option<&Tensor<T>> {
        self.nodes.get(&node_id)?.get_tensor()
    }

    pub fn get_op(&self, node_id: NodeId) -> Option<&dyn Operator<T>> {
        self.nodes.get(&node_id)?.get_op()
    }


    pub fn is_parameter(&self, node_id: NodeId) -> bool {
        self.nodes.get(&node_id).expect("Node {node_id} not found").is_parameter()
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
    //    cache_output: bool,
    ) -> Result<NodeId, String> {
        self.validate_inputs(&op, &input_ids)?;

        // Increment reference count for all input nodes
        // This indicates they are being used by this new operation
        for &input_id in &input_ids {
            self.increment_ref_count(input_id);
        }

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
                        self.safe_evaluate_node(input_id)?;
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

                let cached_tensor = if self.cache_outputs && op.cache_output() { // Si la operación no se va a beneficiar de cachear el resultado me da igual lo que diga el usuario.
                    Some(result_tensor)
                } else {
                    drop(result_tensor);
                    None // Liberar automaticamente la memoria
                };

                // Create evaluated node
                let node = Node::new_evaluated(
                    cached_tensor,
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


        if !self.is_parameter(node_id){
            // Liberar la memoria de este nodo
             self.try_clear_node(node_id);
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

        let ones_grad = Tensor::ones_with_device(loss_tensor.shape(), loss_tensor.device)?;

        self.gradients.insert(loss_id, ones_grad);

        // Ordenamiento topológico y propagación
        let mut visited = HashSet::new();
        let mut topo_order = Vec::new();

        self.topological_sort(loss_id, &mut visited, &mut topo_order)?;
        topo_order.reverse();

        // Propagar gradientes
        for &node_id in &topo_order {
            self.backward_node(node_id)?;
        }

        Ok(())
    }

    fn get_node_owned(&self, node_id: NodeId) -> Node<T> {
        if let Some(node) = self.nodes.get(&node_id) {
            node.clone()
        } else {
            panic!("Node not found {}", node_id)
        }
    }

    fn backward_node(&mut self, node_id: NodeId) -> Result<(), String> {
        // Take ownership of the gradient data
        let grad_output = match self.gradients.remove(&node_id) {
            Some(grad) => grad,
            None => return Ok(()),
        };

        // Obtain node information
        let node = self.get_node_owned(node_id);

        if node.is_leaf() {
            self.gradients.insert(node_id, grad_output);
            return Ok(());
        }

        // Get output tensor (could be None if not cached)
        let some_output = self.get_tensor(node_id);

        // Get operation
        let some_op = self
            .get_op(node_id)
            .ok_or_else(|| format!("Operation not defined for node {}", node_id))?;

        // Get input node IDs
        let input_ids = node
            .get_inputs()
            .ok_or_else(|| format!("No inputs found for node {}", node_id))?;

        // Collect input tensors
        let input_tensors: Result<Vec<&Tensor<T>>, String> = input_ids
            .iter()
            .map(|&input_id| {
                self.get_tensor(input_id)
                    .ok_or_else(|| format!("Input tensor not found for node {}", input_id.0))
            })
            .collect();
        let mut input_tensors = input_tensors?;

        // Compute gradients using the operation's gradient method
        let input_grads = some_op.gradient(grad_output, &mut input_tensors, some_output)?;

        // Accumulate gradients for input nodes
        for (&input_id, input_grad) in input_ids.iter().zip(input_grads) {
            self.accumulate_gradient(input_id, input_grad)?;
        }



        Ok(())
    }

    /// TOPOLOGICAL SORTING TO COMPUTE OPS IN ORDER
    pub fn topological_sort(
        &self,
        node_id: NodeId,
        visited: &mut HashSet<NodeId>,
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
    /// This must be used by the caller at the beginning of each training step
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

    pub fn get_node_shape(&self, node_id: &NodeId) -> Option<&[usize]> {
        if let Some(tensor) = self.nodes.get(node_id)?.get_tensor() {
            Some(tensor.shape())
        } else {
            None
        }
    }

    pub fn clip_gradient(&mut self, node_id: NodeId, clip_coef: T) -> Result<(), String> {
        if let Some(grad) = self.get_gradient(node_id) {
            let clipped_grad = grad.mul_scalar(clip_coef)?;
            self.set_gradient(node_id, clipped_grad);
        }
        Ok(())
    }

    /// Increment reference count for a node
    pub fn increment_ref_count(&mut self, node_id: NodeId) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.increment_ref();
        }
    }

    /// Decrement reference count and return if node can be cleaned
    pub fn decrement_ref_count(&mut self, node_id: NodeId) -> bool {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.decrement_ref()
        } else {
            false
        }
    }

    /// Mark a node as persistent (won't be auto-cleaned)
    pub fn mark_persistent(&mut self, node_id: NodeId) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.mark_persistent();
        }
    }

    /// Check if a node can be safely removed
    pub fn can_node_be_cleaned(&self, node_id: NodeId) -> bool {
        self.nodes
            .get(&node_id)
            .map(|node| node.can_be_cleaned())
            .unwrap_or(false)
    }

    /// Get reference count for debugging
    pub fn get_ref_count(&self, node_id: NodeId) -> Option<usize> {
        self.nodes.get(&node_id).map(|node| node.ref_count)
    }

    /// Remove a node from the graph (but keep its gradient if it's a parameter)
    pub fn try_clear_node(&mut self, node_id: NodeId) -> Result<bool, String> {
        if self.can_node_be_cleaned(node_id) {
            // Decrement references to input nodes
            let mut nodes_to_decrement: Vec<NodeId> = Vec::new();

            if let Some(node) = self.nodes.get(&node_id) {
                if let Some(inputs) = node.get_inputs() {
                    for input_id in inputs {
                        nodes_to_decrement.push(*input_id);
                    }
                }
            }

            for input_node_id in nodes_to_decrement {
                self.decrement_ref_count(input_node_id);
            }

            // Remove only the node (keep gradient if it's a parameter)
            if let Some(node) = self.nodes.remove(&node_id) {
                // Only remove gradient if it's NOT a parameter
                let is_parameter = node.is_leaf() && node.requires_grad;
                if !is_parameter {
                    self.gradients.remove(&node_id);
                }
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
    }

    /// Get statistics about memory usage
    pub fn get_memory_stats(&self) -> MemoryStats {
        let total_nodes = self.nodes.len();
        let cleanable_nodes = self
            .nodes
            .values()
            .filter(|node| node.can_be_cleaned())
            .count();
        let persistent_nodes = self
            .nodes
            .values()
            .filter(|node| node.is_persistent)
            .count();

        MemoryStats {
            total_nodes,
            cleanable_nodes,
            persistent_nodes,
            gradient_count: self.gradients.len(),
        }
    }


    pub fn print_stats(&self) {
        println!("=== AutoFerrox Engine Statistics ===");

        // Basic counts
        let total_nodes = self.nodes.len();
        let total_gradients = self.gradients.len();

        println!("Nodes: {}", total_nodes);
        println!("Gradients: {}", total_gradients);

        // Node type breakdown
        let mut leaf_nodes = 0;
        let mut evaluated_nodes = 0;
        let mut pending_nodes = 0;
        let mut parameter_nodes = 0;
        let mut persistent_nodes = 0;
        let mut cleanable_nodes = 0;

        // Reference count statistics
        let mut total_refs = 0;
        let mut max_refs = 0;
        let mut zero_ref_nodes = 0;

        for node in self.nodes.values() {
            // Node types
            if node.is_leaf() {
                leaf_nodes += 1;
            }
            if node.is_evaluated() {
                evaluated_nodes += 1;
            } else {
                pending_nodes += 1;
            }
            if node.requires_grad && node.is_leaf() {
                parameter_nodes += 1;
            }
            if node.is_persistent {
                persistent_nodes += 1;
            }
            if node.can_be_cleaned() {
                cleanable_nodes += 1;
            }

            // Reference counts
            total_refs += node.ref_count;
            max_refs = max_refs.max(node.ref_count);
            if node.ref_count == 0 {
                zero_ref_nodes += 1;
            }
        }

        println!("  - Leaf nodes: {}", leaf_nodes);
        println!("  - Parameter nodes: {}", parameter_nodes);
        println!("  - Evaluated nodes: {}", evaluated_nodes);
        println!("  - Pending nodes: {}", pending_nodes);
        println!("  - Persistent nodes: {}", persistent_nodes);
        println!("  - Cleanable nodes: {}", cleanable_nodes);

        // Reference statistics
        println!("References:");
        println!("  - Total refs: {}", total_refs);
        println!("  - Max refs on single node: {}", max_refs);
        println!("  - Nodes with zero refs: {}", zero_ref_nodes);

        if total_nodes > 0 {
            println!(
                "  - Avg refs per node: {:.2}",
                total_refs as f32 / total_nodes as f32
            );
        }

        // Memory analysis
        let memory_pressure = if total_nodes > 0 {
            cleanable_nodes as f32 / total_nodes as f32
        } else {
            0.0
        };

        println!("Memory:");
        println!("  - Memory pressure: {:.1}%", memory_pressure * 100.0);
        println!("  - Training mode: {}", self.training_mode);
        println!("  - Evaluation mode: {:?}", self.evaluation_mode);

        // Gradient analysis
        let mut param_gradients = 0;
        let mut intermediate_gradients = 0;

        for &node_id in self.gradients.keys() {
            if let Some(node) = self.nodes.get(&node_id) {
                if node.requires_grad && node.is_leaf() {
                    param_gradients += 1;
                } else {
                    intermediate_gradients += 1;
                }
            }
        }

        println!("Gradients breakdown:");
        println!("  - Parameter gradients: {}", param_gradients);
        println!("  - Intermediate gradients: {}", intermediate_gradients);

        // Memory recommendations
        if memory_pressure > 0.3 {
            println!("High memory pressure - consider calling cleanup_now()");
        }

        if intermediate_gradients > parameter_nodes * 2 {
            println!("Many intermediate gradients - consider clear_intermediate_gradients()");
        }

        if zero_ref_nodes > total_nodes / 4 {
            println!("Many unreferenced nodes - consider cleanup_unreferenced_nodes()");
        }

        println!("=====================================");
    }
}
