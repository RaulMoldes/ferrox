use crate::backend::{FerroxCudaF, Tensor};
use crate::ops::Operator;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// ATOMIC auto incrementing id for all nodes.
static NODE_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

    pub fn get_node(&self, node_id: NodeId) -> Option<&Node<T>> {
        self.nodes.get(&node_id)
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

    fn validate_inputs(
        &self,
        op: &Box<dyn Operator<T>>,
        input_ids: &[NodeId],
    ) -> Result<(), String> {
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
            let node = self.nodes.get(&node_id)
                .ok_or_else(|| format!("Node {} not found", node_id.0))?;

            match &node.state {
                NodeState::Pending { op, inputs } => {
                    (op.clone_op(), inputs.clone())
                }
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
        self.nodes.get(&node_id).is_some_and(|node| node.is_evaluated())
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
            return Err("Cannot run backward on unevaluated node. Call evaluate() first.".to_string());
        }

        // Inicializar gradiente de pérdida
        let loss_tensor = self.get_tensor(loss_id)
            .ok_or("Loss node not found")?;
        let ones_grad = Tensor::ones(loss_tensor.shape())?;
        self.gradients.insert(loss_id, ones_grad);

        // Ordenamiento topológico y propagación
        let mut visited = std::collections::HashSet::new();
        let mut topo_order = Vec::new();
        self.topological_sort(loss_id, &mut visited, &mut topo_order)?;

        topo_order.reverse();

        // Propagar gradientes
        for &node_id in &topo_order {
            self.backward_node(node_id)?;
        }

        Ok(())
    }

    /// Backward para un solo nodo
    fn backward_node(&mut self, node_id: NodeId) -> Result<(), String> {
        // Take ownership of the gradient data
        let grad_output = match self.gradients.remove(&node_id) {
            Some(grad) => grad,
            None => return Ok(()),
        };

        // Obtain node information.
        let node = self.nodes.get(&node_id).cloned()
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
                        o.gradient(grad_output, &mut input_tensors, &tensor)?
                    },
                    None => { panic!("Cannot compute gradient. Operation not defined for node: {}", node_id) }

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
    fn topological_sort(
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
        self.nodes.values()
            .filter(|node| node.is_evaluated())
            .count()
    }

    pub fn num_pending_nodes(&self) -> usize {
        self.nodes.values()
            .filter(|node| !node.is_evaluated())
            .count()
    }

    /// Clean up gradients
    pub fn zero_gradients(&mut self) {
        self.gradients.clear();
    }
}
