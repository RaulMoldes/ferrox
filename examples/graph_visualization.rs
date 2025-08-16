use ferrox::backend::{default_device, Tensor};
use ferrox::graph::graphviz::{GraphVisualizer, VisualizationConfig};
use ferrox::graph::{AutoFerroxEngine, EngineVisualization, EvaluationMode, NodeId};
use ferrox::ops::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Graph Visualization Examples ===\n");

    // Create imgs directory if it doesn't exist
    std::fs::create_dir_all("imgs").unwrap_or_else(|_| {
        println!("Note: imgs directory already exists or created");
    });

    // Example 1: Computational graph with default config
    graph_default_config()?;

    // Example 2: Computational graph with custom config
    graph_custom_config()?;

    Ok(())
}

/// Helper function to create the computational graph
/// Eliminates code duplication between both examples
/// Creates: f(x,y) = sigmoid(x² + y) * tanh(x - y) + sqrt(|x * y|)
fn create_graph(engine: &mut AutoFerroxEngine<f32>) -> Result<NodeId, Box<dyn std::error::Error>> {
    let device = default_device();

    // Create input tensors
    let x_data = Tensor::from_vec_with_device(vec![1.0, 2.0, 3.0], &[3], device)?;
    let y_data = Tensor::from_vec_with_device(vec![0.5, 1.5, 2.5], &[3], device)?;

    let x_node = engine.create_variable(x_data, true);
    let y_node = engine.create_variable(y_data, true);

    // Branch 1: sigmoid(x² + y)
    let square_op = Box::new(Power);
    let x_squared = engine.apply_operation(square_op, vec![x_node, x_node])?;

    let add_op1 = Box::new(Add::new());
    let x2_plus_y = engine.apply_operation(add_op1, vec![x_squared, y_node])?;

    let sigmoid_op = Box::new(Sigmoid);
    let sigmoid_branch = engine.apply_operation(sigmoid_op, vec![x2_plus_y])?;

    // Branch 2: tanh(x - y)
    let sub_op = Box::new(Sub);
    let x_minus_y = engine.apply_operation(sub_op, vec![x_node, y_node])?;

    let tanh_op = Box::new(Tanh);
    let tanh_branch = engine.apply_operation(tanh_op, vec![x_minus_y])?;

    // Branch 3: sqrt(|x * y|)
    let mul_op1 = Box::new(Mul);
    let x_times_y = engine.apply_operation(mul_op1, vec![x_node, y_node])?;

    let abs_op = Box::new(Abs);
    let abs_xy = engine.apply_operation(abs_op, vec![x_times_y])?;

    let sqrt_op = Box::new(Sqrt);
    let sqrt_branch = engine.apply_operation(sqrt_op, vec![abs_xy])?;

    // Combine branches: sigmoid_branch * tanh_branch + sqrt_branch
    let mul_op2 = Box::new(Mul);
    let combined_branches = engine.apply_operation(mul_op2, vec![sigmoid_branch, tanh_branch])?;

    let add_op2 = Box::new(Add::new());
    let final_result = engine.apply_operation(add_op2, vec![combined_branches, sqrt_branch])?;

    Ok(final_result)
}

/// Creates a cool computational graph and visualizes it with default configuration
fn graph_default_config() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Visualizing Graph with Default Configuration");

    let mut engine = AutoFerroxEngine::<f32>::new(true);
    engine.set_training(true);
    engine.set_evaluation_mode(EvaluationMode::Eager);

    // Create the computational graph using helper function
    let final_result = create_graph(&mut engine)?;

    // Execute the graph and validate results
    let result_tensor = engine
        .get_tensor(final_result)
        .expect("Failed to get result tensor");
    println!(
        "   Graph execution completed! Result shape: {:?}",
        result_tensor.shape()
    );

    // Test backward pass to validate gradient computation
    engine.backward(final_result).expect("Backward pass failed");
    println!("   Backward pass completed successfully");

    // Use default visualizer (from the engine's extension trait)
    println!("   Function: f(x,y) = sigmoid(x² + y) * tanh(x - y) + sqrt(|x * y|)");
    println!("   Console output:");
    engine.plot_graph(&[final_result]);

    // Save with default configuration
    engine.save_graph_dot(&[final_result], "imgs/graph_default.dot")?;
    println!("Saved graph_default.dot");

    match engine.save_graph_image(&[final_result], "imgs/graph_default.png") {
        Ok(_) => println!("imgs/graph_default.png"),
        Err(e) => println!("   ⚠ PNG save failed (install GraphViz): {}", e),
    }

    println!(
        "   Graph stats: {} total nodes, {} evaluated\n",
        engine.num_nodes(),
        engine.num_evaluated_nodes()
    );

    Ok(())
}

/// Creates the same graph but with custom visualization configuration
fn graph_custom_config() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Computational Graph with Custom Configuration");

    let mut engine = AutoFerroxEngine::<f32>::new(true);
    engine.set_training(true);
    engine.set_evaluation_mode(EvaluationMode::Lazy); // Different mode for variety

    // Create the same computational graph using helper function
    let final_result = create_graph(&mut engine)?;

    // Force evaluation in lazy mode to see the difference
    let result_tensor = engine.evaluate(final_result)?;
    println!(
        "   Lazy evaluation completed! Result shape: {:?}",
        result_tensor.shape()
    );

    // Test backward pass in lazy mode
    engine
        .backward(final_result)
        .expect("Backward pass failed in lazy mode");
    println!("   Backward pass completed successfully in lazy mode");

    // Create custom visualizer with different styling
    let custom_config = VisualizationConfig {
        show_shapes: true,
        show_gradients: true,
        show_values: true, // Show tensor values
        max_tensor_display: 8,
        node_color: "#FFE6E6".to_string(), // Light pink for tensors
        op_color: "#E6F3FF".to_string(),   // Light blue for operations
        gradient_color: "#E6FFE6".to_string(), // Light green for gradient nodes
    };

    let custom_visualizer = GraphVisualizer::with_config(custom_config);

    println!("   Custom config: pink tensors, blue ops, green gradients, shows values");
    println!("   Console output:");
    custom_visualizer.print_graph(&engine, &[final_result]);

    // Save with custom configuration
    custom_visualizer.save_dot(&engine, &[final_result], "imgs/graph_custom.dot")?;
    println!("Saved graph_custom.dot");

    match custom_visualizer.save_image(&engine, &[final_result], "imgs/graph_custom.png", "png") {
        Ok(_) => println!("Saved graph_custom.png with custom styling"),
        Err(e) => println!("   ⚠ PNG save failed: {}", e),
    }

    println!(
        "   Graph stats: {} total nodes, {} evaluated",
        engine.num_nodes(),
        engine.num_evaluated_nodes()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation_and_execution() {
        // Test that graph creation works correctly
        let mut engine = AutoFerroxEngine::<f32>::new(true);
        engine.set_training(true);

        let result_node = create_graph(&mut engine).expect("Graph creation failed");

        // Verify forward pass works
        let result_tensor = engine
            .get_tensor(result_node)
            .expect("Failed to get result tensor");
        assert_eq!(result_tensor.shape(), &[3]); // Should be 3-element vector

        // Verify backward pass works
        engine.backward(result_node).expect("Backward pass failed");

        println!("✓ Graph creation and execution test passed");
    }

    #[test]
    fn test_visualization_examples() {
        // Test that both visualization examples run without errors
        graph_default_config().expect("Default config example failed");
        graph_custom_config().expect("Custom config example failed");

        // Verify files were created
        assert!(
            std::path::Path::new("imgs/graph_default.dot").exists(),
            "Default DOT file not created"
        );
        assert!(
            std::path::Path::new("imgs/graph_custom.dot").exists(),
            "Custom DOT file not created"
        );

        println!("✓ Visualization examples test passed");
    }

    #[test]
    fn test_lazy_vs_eager_execution() {
        // Test that both evaluation modes produce equivalent results
        let mut eager_engine = AutoFerroxEngine::<f32>::new(true);
        eager_engine.set_training(false); // Disable gradients for cleaner comparison
        eager_engine.set_evaluation_mode(EvaluationMode::Eager);

        let mut lazy_engine = AutoFerroxEngine::<f32>::new(true);
        lazy_engine.set_training(false);
        lazy_engine.set_evaluation_mode(EvaluationMode::Lazy);

        // Create identical graphs
        let eager_result = create_graph(&mut eager_engine).expect("Eager graph creation failed");
        let lazy_result = create_graph(&mut lazy_engine).expect("Lazy graph creation failed");

        // Force evaluation of lazy graph
        lazy_engine
            .evaluate(lazy_result)
            .expect("Lazy evaluation failed");

        // Compare results (should be identical)
        let eager_tensor = eager_engine
            .get_tensor(eager_result)
            .expect("Eager result not found");
        let lazy_tensor = lazy_engine
            .get_tensor(lazy_result)
            .expect("Lazy result not found");

        assert_eq!(
            eager_tensor.shape(),
            lazy_tensor.shape(),
            "Result shapes don't match"
        );

        // Compare actual values (within floating point tolerance)
        let eager_data = eager_tensor.as_slice();
        let lazy_data = lazy_tensor.as_slice();

        for (i, (&eager_val, &lazy_val)) in eager_data.iter().zip(lazy_data.iter()).enumerate() {
            let diff = (eager_val - lazy_val).abs();
            assert!(
                diff < 1e-6,
                "Values differ at index {}: eager={}, lazy={}, diff={}",
                i,
                eager_val,
                lazy_val,
                diff
            );
        }

        println!("✓ Lazy vs eager execution test passed");
    }
}
