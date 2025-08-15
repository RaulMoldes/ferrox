// reduction_ops.rs
// Reduction operations for the computational graph
// These wrap tensor API methods to enable automatic differentiation

use crate::backend::{FerroxCudaF, Tensor};
use crate::ops::Operator;
use crate::FerroxN;

fn expand_reduction<T>(
    mut grad_output: Tensor<T>,
    axes: &Option<Vec<usize>>,
    input_shape: &[usize],
) -> Result<Tensor<T>, String>
where
    T: FerroxCudaF,
{
    match axes {
        Some(reduction_axes) => {
            // Must unsqueeze to restore reduced dimensions as size 1
            // Sort axes to process in correct order
            let mut sorted_axes = reduction_axes.clone();
            sorted_axes.sort_unstable();

            if sorted_axes.is_empty() {
                // No axes to process, return as is
                return Ok(grad_output);
            }

            // Unsqueeze each reduced axis to add dimension of size 1
            for &axis in &sorted_axes {
                grad_output.unsqueeze(axis)?;
            }

            // Now grad_output should have same number of dims as input_shape
            // with 1s where reduction occurred. Broadcast to expand those 1s.
            if grad_output.shape() != input_shape {
                grad_output.broadcast_to(input_shape)?;
            }

            Ok(grad_output)
        }
        None => {
            // All axes reduced to scalar - must create full tensor with scalar value
            let device = grad_output.device();
            let scalar = grad_output.first()?;

            Tensor::full_with_device(input_shape, device, scalar)
        }
    }
}

/// Sum reduction: output = sum(input, axes)
/// Reduces tensor along specified axes or all elements if None
#[derive(Debug, Clone)]
pub struct Sum {
    /// Axes to reduce along. If None, reduces all elements to scalar
    pub axes: Option<Vec<usize>>,
    /// Whether to keep the reduced dimensions as size 1
    pub keep_dims: bool,
}

impl<T> Operator<T> for Sum
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Sum operation requires exactly 1 input".to_string());
        }

        // Backend now handles keep_dims properly
        inputs[0].sum(self.axes.as_deref(), self.keep_dims)
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Sum operation requires exactly 1 input".to_string());
        }

        let input_shape = inputs[0].shape();

        // If keep_dims=true, gradient already has correct shape, just broadcast
        // If keep_dims=false, need to expand dimensions using unsqueeze
        let mut exp = if self.keep_dims && self.axes.is_some() {
            grad_output
        } else {
            expand_reduction(grad_output, &self.axes, input_shape)?
        };

        if exp.shape() != input_shape {
            exp.broadcast_to(input_shape)?;
        }
        Ok(vec![exp])
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

impl Default for Sum {
    fn default() -> Self {
        Self::new(false)
    }
}
impl Sum {
    /// Create sum operation that reduces all elements to scalar
    pub fn new(keep_dims: bool) -> Self {
        Self {
            axes: None,
            keep_dims,
        }
    }

    /// Create sum operation along specific axes
    pub fn along_axes(axes: Vec<usize>, keep_dims: bool) -> Self {
        Self {
            axes: Some(axes),
            keep_dims,
        }
    }
}

/// Mean reduction: output = mean(input, axes)
/// Computes average along specified axes or all elements if None
#[derive(Debug, Clone)]
pub struct Mean {
    /// Axes to reduce along. If None, reduces all elements to scalar
    pub axes: Option<Vec<usize>>,
    /// Whether to keep the reduced dimensions as size 1
    pub keep_dims: bool,
}

impl<T> Operator<T> for Mean
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Mean operation requires exactly 1 input".to_string());
        }

        // Backend now handles keep_dims properly
        inputs[0].mean(self.axes.as_deref(), self.keep_dims)
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Mean operation requires exactly 1 input".to_string());
        }

        // For mean: gradient is grad_output / reduction_size, broadcasted to input shape
        let input_shape = inputs[0].shape();

        // Calculate reduction size (number of elements that were averaged)
        let reduction_size = if let Some(ref axes) = self.axes {
            axes.iter()
                .map(|&axis| input_shape[axis])
                .product::<usize>() as f64
        } else {
            input_shape.iter().product::<usize>() as f64
        };

        let scale = <T as FerroxN>::from_f64(1.0 / reduction_size)
            .ok_or("Failed to convert reduction scale to tensor type")?;
        // Scale the gradient first
        let scaled_grad = grad_output.mul_scalar(scale)?;

        // If keep_dims=true, gradient already has correct shape, just broadcast
        // If keep_dims=false, need to expand dimensions using unsqueeze
        let mut result = if self.keep_dims && self.axes.is_some() {
            scaled_grad
        } else {
            expand_reduction(scaled_grad, &self.axes, input_shape)?
        };

        if result.shape() != input_shape {
            result.broadcast_to(input_shape)?;
        }

        Ok(vec![result])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

impl Default for Mean {
    fn default() -> Self {
        Self::new()
    }
}

impl Mean {
    /// Create mean operation that reduces all elements to scalar
    pub fn new() -> Self {
        Self {
            axes: None,
            keep_dims: false,
        }
    }

    /// Create mean operation along specific axes
    pub fn along_axes(axes: Vec<usize>, keep_dims: bool) -> Self {
        Self {
            axes: Some(axes),
            keep_dims,
        }
    }
}

/// Max reduction: output = max(input, axes)
/// Finds maximum values along specified axes or all elements if None
#[derive(Debug, Clone)]
pub struct Max {
    /// Axes to reduce along. If None, reduces all elements to scalar
    pub axes: Option<Vec<usize>>,
    /// Whether to keep the reduced dimensions as size 1
    pub keep_dims: bool,
}

impl<T> Operator<T> for Max
where
    T: FerroxCudaF + rand_distr::num_traits::Zero + rand_distr::num_traits::FromPrimitive,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Max operation requires exactly 1 input".to_string());
        }

        // Backend now handles keep_dims properly
        inputs[0].max_reduce(self.axes.as_deref(), self.keep_dims)
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Max operation requires exactly 1 input".to_string());
        }

        let input_shape = inputs[0].shape();

        // Create max values for comparison - always use keep_dims=true for mask creation
        let max_values = inputs[0].max_reduce(self.axes.as_deref(), true)?;

        let mut restored_max = max_values;
        if restored_max.shape() != input_shape {
            restored_max.broadcast_to(input_shape)?;
        }

      

        // Create mask where input equals max values (gradient flows here)
        let mask = inputs[0].equal(&restored_max)?;

        // Handle gradient expansion based on keep_dims
        let mut result_grad = if self.keep_dims && self.axes.is_some() {
            grad_output
        } else {
            expand_reduction(grad_output, &self.axes, input_shape)?
        };

        // Broadcast gradient to input shape and apply mask
        if result_grad.shape() != input_shape {
            result_grad.broadcast_to(input_shape)?;
        }

        let result = result_grad.mul(&mask)?;
        Ok(vec![result])
    }
    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

impl Default for Max {
    fn default() -> Self {
        Self::new()
    }
}

impl Max {
    /// Create max operation that reduces all elements to scalar
    pub fn new() -> Self {
        Self {
            axes: None,
            keep_dims: false,
        }
    }

    /// Create max operation along specific axes
    pub fn along_axes(axes: Vec<usize>, keep_dims: bool) -> Self {
        Self {
            axes: Some(axes),
            keep_dims,
        }
    }
}

/// Min reduction: output = min(input, axes)
/// Finds minimum values along specified axes or all elements if None
#[derive(Debug, Clone)]
pub struct Min {
    /// Axes to reduce along. If None, reduces all elements to scalar
    pub axes: Option<Vec<usize>>,
    /// Whether to keep the reduced dimensions as size 1
    pub keep_dims: bool,
}

impl<T> Operator<T> for Min
where
    T: FerroxCudaF + rand_distr::num_traits::Zero + rand_distr::num_traits::FromPrimitive,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Min operation requires exactly 1 input".to_string());
        }

        // Backend now handles keep_dims properly
        inputs[0].min_reduce(self.axes.as_deref(), self.keep_dims)
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Min operation requires exactly 1 input".to_string());
        }

        let input_shape = inputs[0].shape();

        // Create min values for comparison - always use keep_dims=true for mask creation
        let min_values = inputs[0].min_reduce(self.axes.as_deref(), true)?;

        let mut restored_min = min_values;
        if restored_min.shape() != input_shape {
            restored_min.broadcast_to(input_shape)?;
        }

        // Create mask where input equals min values (gradient flows here)
        let mask = inputs[0].equal(&restored_min)?;

        // Handle gradient expansion based on keep_dims
        let mut result_grad = if self.keep_dims && self.axes.is_some() {
            grad_output
        } else {
            expand_reduction(grad_output, &self.axes, input_shape)?
        };

        // Broadcast gradient to input shape and apply mask
        if result_grad.shape() != input_shape {
            result_grad.broadcast_to(input_shape)?;
        }

        let result = result_grad.mul(&mask)?;
        Ok(vec![result])
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

impl Default for Min {
    fn default() -> Self {
        Self::new()
    }
}

impl Min {
    /// Create min operation that reduces all elements to scalar
    pub fn new() -> Self {
        Self {
            axes: None,
            keep_dims: false,
        }
    }

    /// Create min operation along specific axes
    pub fn along_axes(axes: Vec<usize>, keep_dims: bool) -> Self {
        Self {
            axes: Some(axes),
            keep_dims,
        }
    }
}
