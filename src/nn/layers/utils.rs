use crate::graph::{AutoFerroxEngine, NodeId};
use crate::ops::{BroadcastTo, Reshape};
use crate::FerroxCudaF;
// Utilities for implementing the layers go here.
// This function reshapes and broadcasts an scalar tensor (shape == [1] ) into any 2D or 4D target shape
pub fn reshape_and_broadcast<T>(
    input: NodeId,
    target: NodeId,
    graph: &mut AutoFerroxEngine<T>,
) -> Result<NodeId, String>
where
    T: FerroxCudaF,
{
    let target_shape = graph
        .get_node_shape(&target)
        .expect("Failed to get target shape for broadcasting")
        .to_vec();

    let input_shape = graph
        .get_node_shape(&input)
        .expect("Failed to get input shape for broadcasting")
        .to_vec();
    let res = if input_shape.len() == 1 {
        let new_shape = if target_shape.len() == 4 {
            vec![1, input_shape[0], 1, 1]
        } else {
            vec![1, input_shape[0]]
        };

        let reshape = Box::new(Reshape::new(new_shape));
        graph.apply_operation(reshape, vec![input])?
    } else {
        input
    };

    if input_shape != target_shape {
        let bcop = Box::new(BroadcastTo::<T>::new(target_shape));
        graph.apply_operation(bcop, vec![res])
    } else {
        Ok(res)
    }
}
