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

#[cfg(test)]
mod reshape_ops_test {
    use crate::backend::manager::best_f32_device;
    use crate::backend::Tensor;
    use crate::ops::matrix::Transpose;
    use crate::ops::Operator;

    #[test]
    fn test_transpose() {
        let device = best_f32_device();

        // Create a weight matrix similar to what Linear layer uses: [out_features, in_features]
        let weight_data = vec![
            -0.124830216,
            0.22747543,
            0.57083875,
            -0.19728178,
            -0.1812076,
            0.42339596,
            0.37712204,
            0.015537508,
            -0.05177227,
            -0.4744638,
            0.033790253,
            0.39341143,
            -0.18968286,
            -0.26776776,
            0.16704325,
            0.13673706,
        ];

        let weights = Tensor::from_vec_with_device(
            weight_data.clone(),
            &[4, 4], // 4x4 matrix
            device,
        )
        .unwrap();

        // Test transpose operation
        let transpose_op = Transpose::new();
        let mut inputs = [&weights];

        // Perform transpose
        let transposed = transpose_op.compute(&mut inputs).unwrap();
        let transposed_data = transposed.to_vec().unwrap();

        // Check for NaN values in result
        let has_nan = transposed_data.iter().any(|&x: &f32| x.is_nan());
        if has_nan {
            println!("TRANSPOSE PRODUCES NaN VALUES!");
            let nan_positions: Vec<_> = transposed_data
                .iter()
                .enumerate()
                .filter(|(_, &val)| val.is_nan())
                .map(|(i, _)| i)
                .collect();
            println!("NaN found at positions: {:?}", nan_positions);
        } else {
            println!("✓ Transpose completed without NaN");
        }

        // Verify transpose is mathematically correct (before checking for NaN)
        // Element at position (i,j) in original should be at (j,i) in transposed
        for i in 0..4 {
            for j in 0..4 {
                let original_val: f32 = weight_data[i * 4 + j];
                let transposed_val: f32 = transposed_data[j * 4 + i];

                if !original_val.is_nan() && !transposed_val.is_nan() {
                    assert!(
                        (original_val - transposed_val).abs() < 1e-6,
                        "Transpose error at ({},{}): expected {}, got {}",
                        i,
                        j,
                        original_val,
                        transposed_val
                    );
                }
            }
        }

        assert!(
            !has_nan,
            "Transpose operation should not produce NaN values"
        );
    }
}
