use std::collections::HashSet;
use std::fmt::Write;
use std::fs::File;
use std::io::Write as IoWrite;
use std::process::Command;

use super::engine::Engine;
use super::node::{Node, NodeId};
use crate::backend::number::FerroxCudaF;

/// Graph visualization module for the computational graph engine
pub struct GraphVisualizer {
    /// Optional styling configuration
    pub config: VisualizationConfig,
}

impl Default for GraphVisualizer {
    fn default() -> Self {
        Self::new()
    }
}
/// Configuration for graph visualization
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    pub show_shapes: bool,
    pub show_gradients: bool,
    pub show_values: bool,
    pub max_tensor_display: usize,
    pub node_color: String,
    pub op_color: String,
    pub gradient_color: String,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            show_shapes: true,
            show_gradients: true,
            show_values: false, // Don't show values by default to avoid clutter
            max_tensor_display: 5,
            node_color: "#E3F2FD".to_string(),
            op_color: "#FFF3E0".to_string(),
            gradient_color: "#E8F5E8".to_string(),
        }
    }
}

impl GraphVisualizer {
    pub fn new() -> Self {
        Self {
            config: VisualizationConfig::default(),
        }
    }

    pub fn with_config(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Generate DOT format representation of the computational graph
    pub fn to_dot<T>(&self, engine: &Engine<T>, output_nodes: &[NodeId]) -> String
    where
        T: FerroxCudaF
            + Clone
            + std::fmt::Debug
            + ndarray::LinalgScalar
            + ndarray::ScalarOperand
            + rand_distr::num_traits::FromPrimitive,
    {
        let mut dot = String::new();
        writeln!(dot, "digraph ComputationalGraph {{").unwrap();
        writeln!(dot, "    rankdir=TB;").unwrap();
        writeln!(dot, "    node [shape=box, style=filled];").unwrap();
        writeln!(dot, "    edge [color=gray];").unwrap();

        // Find all nodes in the subgraph
        let relevant_nodes = self.find_relevant_nodes(engine, output_nodes);

        // Add nodes
        for &node_id in &relevant_nodes {
            let node = engine.nodes[&node_id].borrow();
            let label = self.create_node_label(engine, node_id, &node);
            let color = self.get_node_color(&node);

            writeln!(
                dot,
                "    {node_id} [label=\"{label}\", fillcolor=\"{color}\"];"
            )
            .unwrap();
        }

        // Add edges
        for &node_id in &relevant_nodes {
            let node = engine.nodes[&node_id].borrow();
            for &input_id in &node.inputs {
                writeln!(dot, "    {input_id} -> {node_id};").unwrap();
            }
        }

        writeln!(dot, "}}").unwrap();
        dot
    }

    /// Find all nodes that are relevant to the given output nodes
    fn find_relevant_nodes<T>(&self, engine: &Engine<T>, output_nodes: &[NodeId]) -> Vec<NodeId>
    where
        T: FerroxCudaF,
    {
        let mut visited = HashSet::new();
        let mut relevant = Vec::new();

        for &output_id in output_nodes {
            self.collect_nodes_dfs(engine, output_id, &mut visited, &mut relevant);
        }

        relevant
    }

    /// DFS to collect all nodes in the computation graph
    #[allow(clippy::only_used_in_recursion)]
    fn collect_nodes_dfs<T>(
        &self,
        engine: &Engine<T>,
        node_id: NodeId,
        visited: &mut HashSet<NodeId>,
        relevant: &mut Vec<NodeId>,
    ) where
        T: FerroxCudaF,
    {
        if visited.contains(&node_id) {
            return;
        }

        visited.insert(node_id);
        relevant.push(node_id);

        let node = engine.nodes[&node_id].borrow();
        for &input_id in &node.inputs {
            self.collect_nodes_dfs(engine, input_id, visited, relevant);
        }
    }

    /// Create a descriptive label for a node
    fn create_node_label<T>(&self, engine: &Engine<T>, node_id: NodeId, node: &Node<T>) -> String
    where
        T: FerroxCudaF,
    {
        let mut label = String::new();

        // Node ID and type
        if let Some(ref op) = node.op {
            write!(label, "{}\\n{}", node_id, self.get_op_name(op.as_ref())).unwrap();
        } else {
            write!(label, "{node_id}\\nTensor").unwrap();
        }

        // Shape information
        if self.config.show_shapes {
            let shape = node.cached_data.shape();
            write!(label, "\\nShape: {shape:?}").unwrap();
        }

        // Gradient information
        if self.config.show_gradients && node.requires_grad {
            write!(label, "\\nRequires Grad: true").unwrap();
            if let Some(_grad) = engine.get_gradient(node_id) {
                write!(label, "\\nHas Gradient").unwrap();
            }
        }

        // Value preview (if enabled and tensor is small)
        if self.config.show_values {
            let data = &node.cached_data;
            if data.size() <= self.config.max_tensor_display {
                write!(label, "\\nData: {:?}", data.as_slice()).unwrap();
            }
        }

        label
    }

    /// Get appropriate color for a node based on its properties
    fn get_node_color<T>(&self, node: &Node<T>) -> &str
    where
        T: FerroxCudaF,
    {
        if node.op.is_some() {
            &self.config.op_color
        } else if node.requires_grad {
            &self.config.gradient_color
        } else {
            &self.config.node_color
        }
    }

    /// Get a human-readable name for an operation
    fn get_op_name<T>(&self, op: &dyn super::op::Operator<T>) -> String
    where
        T: FerroxCudaF,
    {
        // This is a very simplistic way to get the operation name.
        format!("{op:?}")
    }

    /// Save the graph as a DOT file
    pub fn save_dot<T>(
        &self,
        engine: &Engine<T>,
        output_nodes: &[NodeId],
        filename: &str,
    ) -> Result<(), std::io::Error>
    where
        T: FerroxCudaF
            + Clone
            + std::fmt::Debug
            + ndarray::LinalgScalar
            + ndarray::ScalarOperand
            + rand_distr::num_traits::FromPrimitive,
    {
        let dot_content = self.to_dot(engine, output_nodes);
        let mut file = File::create(filename)?;
        file.write_all(dot_content.as_bytes())?;
        Ok(())
    }

    /// Generate and save the graph as an image (requires Graphviz)
    pub fn save_image<T>(
        &self,
        engine: &Engine<T>,
        output_nodes: &[NodeId],
        filename: &str,
        format: &str,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        T: FerroxCudaF
            + Clone
            + std::fmt::Debug
            + ndarray::LinalgScalar
            + ndarray::ScalarOperand
            + rand_distr::num_traits::FromPrimitive,
    {
        let dot_content = self.to_dot(engine, output_nodes);

        // Write DOT content to a temporary file
        let temp_dot = format!("{filename}.dot");
        let mut file = File::create(&temp_dot)?;
        file.write_all(dot_content.as_bytes())?;
        drop(file); // Ensure file is closed

        // Use Graphviz to generate the image
        let output = Command::new("dot")
            .arg(format!("-T{format}"))
            .arg(&temp_dot)
            .arg("-o")
            .arg(filename)
            .output()?;

        if !output.status.success() {
            return Err(format!(
                "Graphviz failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }

        // Clean up temporary file
        std::fs::remove_file(&temp_dot)?;

        Ok(())
    }

    /// Print the graph to console (simple text representation)
    pub fn print_graph<T>(&self, engine: &Engine<T>, output_nodes: &[NodeId])
    where
        T: FerroxCudaF
            + Clone
            + std::fmt::Debug
            + ndarray::LinalgScalar
            + ndarray::ScalarOperand
            + rand_distr::num_traits::FromPrimitive,
    {
        println!("Computational Graph:");
        println!("===================");

        let relevant_nodes = self.find_relevant_nodes(engine, output_nodes);
        let topo_order = engine.find_topo_sort(output_nodes);

        for &node_id in &topo_order {
            if relevant_nodes.contains(&node_id) {
                let node = engine.nodes[&node_id].borrow();

                print!("Node {node_id}: ");

                if let Some(ref op) = node.op {
                    print!("{} ", self.get_op_name(op.as_ref()));
                } else {
                    print!("Tensor ");
                }

                if self.config.show_shapes {
                    print!("Shape: {:?} ", node.cached_data.shape());
                }

                if self.config.show_gradients && node.requires_grad {
                    print!("(requires_grad) ");
                }

                if !node.inputs.is_empty() {
                    print!("← {:?}", node.inputs);
                }

                println!();
            }
        }
    }
}

// Extension trait to add visualization methods directly to Engine
pub trait EngineVisualization<T>
where
    T: FerroxCudaF,
{
    fn visualize(&self) -> GraphVisualizer;
    fn plot_graph(&self, output_nodes: &[NodeId]);
    fn save_graph_image(
        &self,
        output_nodes: &[NodeId],
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>>;
    fn save_graph_dot(&self, output_nodes: &[NodeId], filename: &str)
    -> Result<(), std::io::Error>;
}

impl<T> EngineVisualization<T> for Engine<T>
where
    T: FerroxCudaF
        + Clone
        + std::fmt::Debug
        + ndarray::LinalgScalar
        + ndarray::ScalarOperand
        + rand_distr::num_traits::FromPrimitive,
{
    fn visualize(&self) -> GraphVisualizer {
        GraphVisualizer::new()
    }

    fn plot_graph(&self, output_nodes: &[NodeId]) {
        let visualizer = GraphVisualizer::new();
        visualizer.print_graph(self, output_nodes);
    }

    fn save_graph_image(
        &self,
        output_nodes: &[NodeId],
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let visualizer = GraphVisualizer::new();
        visualizer.save_image(self, output_nodes, filename, "png")
    }

    fn save_graph_dot(
        &self,
        output_nodes: &[NodeId],
        filename: &str,
    ) -> Result<(), std::io::Error> {
        let visualizer = GraphVisualizer::new();
        visualizer.save_dot(self, output_nodes, filename)
    }
}
