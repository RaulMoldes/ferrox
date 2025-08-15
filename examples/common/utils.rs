use ferrox::backend::{FerroxCudaF, Tensor};
use ferrox::graph::{AutoFerroxEngine, NodeId};
use ferrox::nn::{losses::Loss, Module};
use std::time::Instant;

/// Benchmark function execution
#[allow(dead_code)]
pub fn with_benchmark<F, R>(func: F) -> impl FnOnce() -> R
where
    F: FnOnce() -> R,
{
    move || {
        let start_time = Instant::now();
        let result = func();
        let elapsed = start_time.elapsed();
        println!("[INFO] Execution time: {:?}", elapsed);
        result
    }
}

/// Run forward pass and return useful information
pub fn run_forward<M, T>(
    model: &M,
    graph: &mut AutoFerroxEngine<T>,
    inputs: &Tensor<T>,
) -> Result<(NodeId, Vec<usize>, usize), String>
where
    M: Module<T>,
    T: FerroxCudaF,
{
    let input_node = graph.create_variable(inputs.clone(), false);

    let predictions = model.forward(graph, input_node)?;

    let pred_tensor = graph
        .get_tensor(predictions)
        .ok_or("Predictions tensor not found")?;
    let pred_shape = pred_tensor.shape().to_vec();
    let num_samples = inputs.shape()[0];

    Ok((predictions, pred_shape, num_samples))
}

/// Compute loss value using any loss function
pub fn compute_loss<T, L>(
    graph: &mut AutoFerroxEngine<T>,
    loss_fn: L,
    predictions: NodeId,
    targets: &Tensor<T>,
) -> Result<T, String>
where
    T: FerroxCudaF,
    L: Loss<T>,
{
    let target_node = graph.create_variable(targets.clone(), false);
    let loss_node = loss_fn.forward(graph, predictions, target_node)?;

    let loss_tensor = graph.get_tensor(loss_node).ok_or("Loss tensor not found")?;
    loss_tensor.clone().first()
}
