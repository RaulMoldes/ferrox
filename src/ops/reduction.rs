// reduction_ops.rs
// Reduction operations for the computational graph
// These wrap tensor API methods to enable automatic differentiation

use crate::backend::{FerroxCudaF, Tensor};
use crate::ops::Operator;
use crate::{FerroxF, FerroxN};

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

        let result = inputs[0].sum(self.axes.as_deref())?;

        // Handle keep_dims if needed by reshaping result
        if self.keep_dims && self.axes.is_some() {
            let mut output = result;
            let input_shape = inputs[0].shape();
            let axes = self.axes.as_ref().unwrap();

            // Create new shape with reduced dimensions as 1
            let mut new_shape = input_shape.to_vec();
            for &axis in axes {
                new_shape[axis] = 1;
            }
            output.reshape(&new_shape)?;
            Ok(output)
        } else {
            Ok(result)
        }
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn gradient(
        &self,
        mut grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Sum operation requires exactly 1 input".to_string());
        }

        // For sum: gradient is broadcasted grad_output to input shape
        // Sum operation distributes gradient equally to all reduced elements
        let input_shape = inputs[0].shape();

        let result = if grad_output.shape().is_empty(){

            let scalar = grad_output.first()?;
            Tensor::full_with_device(input_shape, inputs[0].device(), scalar)?
        }
        else
        {
            // Broadcast gradient back to input shape
            grad_output.broadcast_to(input_shape)?;
            grad_output
        };

        Ok(vec![result])
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

impl Default for Sum {
    fn default() -> Self {
        Self::new()
    }
}
impl Sum {
    /// Create sum operation that reduces all elements to scalar
    pub fn new() -> Self {
        Self {
            axes: None,
            keep_dims: false,
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

        let result = inputs[0].mean(self.axes.as_deref())?;

        // Handle keep_dims if needed
        if self.keep_dims && self.axes.is_some() {
            let mut output = result;
            let input_shape = inputs[0].shape();
            let axes = self.axes.as_ref().unwrap();

            let mut new_shape = input_shape.to_vec();
            for &axis in axes {
                new_shape[axis] = 1;
            }
            output.reshape(&new_shape)?;
            Ok(output)
        } else {
            Ok(result)
        }
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

        // Scale and broadcast gradient back to input shape
        let mut result = grad_output.mul_scalar(scale)?;

        let output = if result.shape().is_empty(){

            let scalar = grad_output.first()?;
            Tensor::full_with_device(input_shape, inputs[0].device(), scalar)?
        }
        else
        {
            // Broadcast gradient back to input shape
            result.broadcast_to(input_shape)?;
            result
        };

        Ok(vec![output])
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

        let result = inputs[0].max_reduce(self.axes.as_deref())?;

        // Handle keep_dims if needed
        if self.keep_dims && self.axes.is_some() {
            let mut output = result;
            let input_shape = inputs[0].shape();
            let axes = self.axes.as_ref().unwrap();

            let mut new_shape = input_shape.to_vec();
            for &axis in axes {
                new_shape[axis] = 1;
            }
            output.reshape(&new_shape)?;
            Ok(output)
        } else {
            Ok(result)
        }
    }

    fn gradient(
        &self,
        mut grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Max operation requires exactly 1 input".to_string());
        }

        // For max: gradient flows only to the maximum elements
        // Create a mask where input equals the max values
        let mut broadcasted_max = inputs[0].max_reduce(self.axes.as_deref())?;

        // Broadcast max result back to input shape for comparison
        let out = if !broadcasted_max.shape().is_empty(){
            broadcasted_max.broadcast_to(inputs[0].shape())?;
            broadcasted_max
        }else{
            let item = broadcasted_max.first()?;
            let device = inputs[0].device();
            Tensor::full_with_device(inputs[0].shape(), device, item)?
        };

        // Create mask where input == max (gets gradient of 1, others get 0)
        let mask = inputs[0].equal(&out)?;

        // Broadcast grad_output to input shape and apply mask

        grad_output.broadcast_to(inputs[0].shape())?;
        let result = grad_output.mul(&mask)?;

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

        let mut result = inputs[0].min_reduce(self.axes.as_deref())?;

        // Handle keep_dims if needed
        if self.keep_dims && self.axes.is_some() {
            let input_shape = inputs[0].shape();
            let axes = self.axes.as_ref().unwrap();

            let mut new_shape = input_shape.to_vec();
            for &axis in axes {
                new_shape[axis] = 1;
            }
            result.reshape(&new_shape)?;
            Ok(result)
        } else {
            Ok(result)
        }
    }

    fn gradient(
        &self,
        mut grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 1 {
            return Err("Min operation requires exactly 1 input".to_string());
        }

        // For min: gradient flows only to the minimum elements
        // Create a mask where input equals the min values
        let mut broadcasted_min = inputs[0].min_reduce(self.axes.as_deref())?;

        // Broadcast min result back to input shape for comparison

        let out = if !broadcasted_min.shape().is_empty(){
            broadcasted_min.broadcast_to(inputs[0].shape())?;
            broadcasted_min
        }else{
            let item = broadcasted_min.first()?;
            let device = inputs[0].device();
            Tensor::full_with_device(inputs[0].shape(), device, item)?
        };


        // Create mask where input == min (gets gradient of 1, others get 0)
        let mask = inputs[0].equal(&out)?;

        // Broadcast grad_output to input shape and apply mask
        grad_output.broadcast_to(inputs[0].shape())?;
        let result = grad_output.mul(&mask)?;

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
