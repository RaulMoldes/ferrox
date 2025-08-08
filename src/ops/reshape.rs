use crate::backend::{FerroxCudaF, Tensor};
use crate::ops::basic::reduce_gradient_for_broadcasting;
use crate::ops::Operator;
/// Reshape operation: output = reshape(input, new_shape)
/// Changes tensor shape while preserving total number of elements
#[derive(Debug, Clone)]
pub struct Reshape {
    /// Target shape for the reshape operation
    pub new_shape: Vec<usize>,
}

impl<T> Operator<T> for Reshape
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Reshape operation requires exactly 1 input".to_string());
        }

        let mut result = inputs[0].clone();
        result.reshape(&self.new_shape)?;
        Ok(result)
    }

    fn gradient(
        &self,
        mut grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Reshape operation requires exactly 1 input".to_string());
        }

        // For reshape: gradient just needs to be reshaped back to input shape
        let input_shape = inputs[0].shape();

        println!("Initial shape: {:?}", input_shape);

        grad_output.reshape(input_shape)?;

        Ok(vec![grad_output])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

impl Reshape {
    /// Create a reshape operation with target shape
    pub fn new(new_shape: Vec<usize>) -> Self {
        Self { new_shape }
    }
}

/// Broadcast operation: output = broadcast_to(input, target_shape)
/// Expands tensor dimensions following NumPy broadcasting rules
#[derive(Debug, Clone)]
pub struct BroadcastTo<T: FerroxCudaF> {
    /// Target shape for broadcasting
    pub target_shape: Vec<usize>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Operator<T> for BroadcastTo<T>
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("BroadcastTo operation requires exactly 1 input".to_string());
        }

        let mut result = inputs[0].clone();
        result.broadcast_to(&self.target_shape)?;
        Ok(result)
    }

    fn gradient(
        &self,
        mut grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("BroadcastTo operation requires exactly 1 input".to_string());
        }

        // For broadcast: gradient needs to be reduced back to original shape
        // This is handled by the same logic as in Add operator
        let input_shape = inputs[0].shape();

        reduce_gradient_for_broadcasting(&mut grad_output, input_shape)?;

        Ok(vec![grad_output])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

impl<T> BroadcastTo<T>
where
    T: FerroxCudaF,
{
    /// Create a broadcast operation with target shape
    pub fn new(target_shape: Vec<usize>) -> Self {
        Self {
            target_shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Unsqueeze operation: output = unsqueeze(input, axis)
/// Adds a dimension of size 1 at the specified axis
#[derive(Debug, Clone)]
pub struct Unsqueeze {
    /// Axis where to insert the new dimension
    pub axis: usize,
}

impl<T> Operator<T> for Unsqueeze
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Unsqueeze operation requires exactly 1 input".to_string());
        }

        let mut result = inputs[0].clone();
        result.unsqueeze(self.axis)?;
        Ok(result)
    }

    fn gradient(
        &self,
        mut grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Unsqueeze operation requires exactly 1 input".to_string());
        }

        // For unsqueeze: gradient just needs to squeeze back the added dimension
        grad_output.squeeze(Some(self.axis))?;
        Ok(vec![grad_output])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

impl Unsqueeze {
    /// Create an unsqueeze operation at specified axis
    pub fn new(axis: usize) -> Self {
        Self { axis }
    }
}

/// Squeeze operation: output = squeeze(input, axis)
/// Removes dimensions of size 1, either all of them or at a specific axis
#[derive(Debug, Clone)]
pub struct Squeeze {
    /// Specific axis to squeeze. If None, removes all size-1 dimensions
    pub axis: Option<usize>,
}

impl<T> Operator<T> for Squeeze
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Squeeze operation requires exactly 1 input".to_string());
        }

        let mut result = inputs[0].clone();
        result.squeeze(self.axis)?;
        Ok(result)
    }

    fn gradient(
        &self,
        mut grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Squeeze operation requires exactly 1 input".to_string());
        }

        // For squeeze: gradient needs to be unsqueezed back to input shape
        // We need to restore the original shape by unsqueezing
        let input_shape = inputs[0].shape();

        // Find which dimensions were squeezed and restore them
        match self.axis {
            Some(axis) => {
                // Specific axis was squeezed - unsqueeze it back
                grad_output.unsqueeze(axis)?;
            }
            None => {
                // All size-1 dimensions were squeezed - restore by reshaping
                grad_output.reshape(input_shape)?;
            }
        }

        Ok(vec![grad_output])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

impl Squeeze {
    /// Create squeeze operation that removes all size-1 dimensions
    pub fn new() -> Self {
        Self { axis: None }
    }

    /// Create squeeze operation that removes size-1 dimension at specific axis
    pub fn at_axis(axis: usize) -> Self {
        Self { axis: Some(axis) }
    }
}

impl Default for Squeeze {
    fn default() -> Self {
        Self::new()
    }
}

/// Expand dimensions operation: output = expand_dims(input, axis)
/// Alias for Unsqueeze operation, following TensorFlow naming convention
pub type ExpandDims = Unsqueeze;
