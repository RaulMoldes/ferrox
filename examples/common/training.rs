use super::configs::TrainingConfig;
use ferrox::backend::manager::show_memory_stats;
use ferrox::backend::FerroxCudaF;
use ferrox::dataset::BatchedDataset;
use ferrox::graph::{AutoFerroxEngine, NodeId};
use ferrox::nn::{
    losses::Loss,
    optim::{Adam, Optim, Optimizer, SGD},
    Module,
};
use ferrox::FerroxN;
/// Initialize model parameters in the computational graph
pub fn initialize_model<M, T, O>(
    model: &M,
    graph: &mut AutoFerroxEngine<T>,
    optimizer: &mut O,
) -> Result<(), String>
where
    T: FerroxCudaF,
    O: Optimizer<T>,
    M: Module<T>,
{
    let param_map = model.create_parameters_in_graph(graph);
    for (_, param_node) in param_map {
        optimizer.add_param(0, param_node);
    }
    Ok(())
}

/// Execute single training epoch
#[allow(clippy::too_many_arguments)]
pub fn train_epoch<T, M, O, L>(
    graph: &mut AutoFerroxEngine<T>,
    optimizer: &mut O,
    model: &mut M,
    loss_fn: &L,
    input_node: NodeId,
    target_node: NodeId,
    loss_history: &mut Vec<T>,
    epoch: usize,
) -> Result<(), String>
where
    T: FerroxCudaF,
    M: Module<T>,
    O: Optimizer<T>,
    L: Loss<T>,
{
    // Clear gradients
    graph.zero_gradients();

    // Forward pass
    let predictions = model.forward(graph, input_node)?;

    // Compute loss
    let loss_node = loss_fn.forward(graph, predictions, target_node)?;

    // Get loss value
    let loss_tensor = graph.get_tensor(loss_node).ok_or("Loss tensor not found")?;
    let loss_value = loss_tensor.clone().first()?;

    // Check for training instability
    if loss_value.is_nan() || loss_value.is_infinite() {
        return Err(format!(
            "Training unstable at epoch {}: loss = {}",
            epoch + 1,
            <T as FerroxN>::to_f64(loss_value)
        ));
    }

    loss_history.push(loss_value);

    // Backward pass
    graph.backward(loss_node)?;

    // Update parameters
    optimizer.step(graph).map_err(|e| e.to_string())?;

    Ok(())
}

/// Main training function
pub fn train<T, M, L>(
    model: &mut M,
    loss_fn: &L,
    data: BatchedDataset<T>,
    config: &TrainingConfig<T>,
) -> Result<AutoFerroxEngine<T>, String>
where
    T: FerroxCudaF,
    M: Module<T>,
    L: Loss<T>,
{
    //let mut graph = AutoFerroxEngine::new();

    // Initialize optimizer
    let mut optimizer = match config.optimizer {
        "SGD" => Optim::SGD(SGD::new(
            config.learning_rate.expect("Learning rate not provided"),
            config.momentum.expect("Momentum not provided"),
            config.decay.expect("Weight decay not provided"),
            config.nesterov,
        )),
        "Adam" => Optim::Adam(Adam::new(
            config.learning_rate.expect("Learning rate not provided"),
            config.beta1.expect("Beta1 not provided"),
            config.beta2.expect("Beta2 not provided"),
            config.eps.expect("Epsilon not provided"),
            config.decay.expect("Weight decay not provided"),
            config.amsgrad,
        )),
        _ => return Err("Invalid optimizer name! Must be Adam or SGD".to_string()),
    };

    let mut graph = AutoFerroxEngine::new(false);
    // Initialize model parameters
    initialize_model(model, &mut graph, &mut optimizer)?;

    let mut loss_history: Vec<T> = Vec::with_capacity(config.num_epochs);

    // Training loop
    for epoch in 0..config.num_epochs {
        for (inputs, targets) in &data {
            let input_node = graph.create_variable(inputs.clone(), false);
            let target_node = graph.create_variable(targets.clone(), false);

            train_epoch(
                &mut graph,
                &mut optimizer,
                model,
                loss_fn,
                input_node,
                target_node,
                &mut loss_history,
                epoch,
            )?;
        }

        if epoch % 20 == 0 {
            graph.print_stats();
        }

        if (epoch + 1) % 50 == 0 {
            show_memory_stats::<T>()?;
        }

        // Print progress
        if epoch % config.print_every == 0 {
            if let Some(&current_loss) = loss_history.last() {
                println!(
                    "[EPOCH {}] Loss: {:.6}",
                    epoch,
                    <T as FerroxN>::to_f64(current_loss)
                );
            }
        }
    }

    Ok(graph)
}
