// src/backend/storage/cpu.rs
use super::StorageBackend;
use crate::backend::{FerroxCudaF, FerroxCudaN, FerroxF};
use crate::FerroxN;
use ndarray::{concatenate, ArrayD, ArrayViewD, Axis, IxDyn, SliceInfoElem};
use rand::Rng;
use rand_distr::StandardUniform;
use std::any::Any;
#[derive(Debug, Clone)]
pub struct CPUStorage<T: Clone> {
    data: ArrayD<T>,
}

impl<T: Clone> CPUStorage<T> {
    pub fn new(data: ArrayD<T>) -> Self {
        Self { data }
    }

    /// Creates a new owned storage
    pub fn new_owned(data: ArrayD<T>) -> Self {
        Self { data }
    }

    pub fn from_view<'a>(view: &ArrayViewD<'a, T>) -> Self {
        // Always clone the data to avoid lifetime issues
        Self {
            data: view.to_owned(),
        }
    }

    pub fn from_array_ref(array: &ArrayD<T>) -> Self {
        Self {
            data: array.clone(),
        }
    }

    pub fn view(&self) -> ArrayViewD<'_, T> {
        self.data.view()
    }

    pub fn view_mut(&mut self) -> ndarray::ArrayViewMutD<'_, T> {
        self.data.view_mut()
    }

    pub fn array_ref(&self) -> &ArrayD<T> {
        &self.data
    }

    pub fn array_mut(&mut self) -> &mut ArrayD<T> {
        &mut self.data
    }
}


impl<T> CPUStorage<T>
where
    T: FerroxCudaN,
{
    // Move the generic reduce method to the concrete implementation
    // This avoids making StorageBackend non-dyn-compatible
    fn reduce<F>(
        &self,
        axes: Option<&[usize]>,
        keep_dims: bool, // Add keep_dims parameter
        reduction_fn: F,
    ) -> Result<CPUStorage<T>, String>
    where
        F: Fn(&ndarray::ArrayD<T>, ndarray::Axis) -> ndarray::ArrayD<T>,
    {
        match axes {
            Some(axes_list) => {
                // Validate axes bounds before processing
                for &ax in axes_list {
                    if ax >= self.data.ndim() {
                        return Err(format!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            ax,
                            self.data.ndim()
                        ));
                    }
                }

                let mut result = self.data.clone();
                let original_shape = self.shape().to_vec();

                // Sort in descending order to prevent index shifting during reduction
                let mut sorted_axes = axes_list.to_vec();
                sorted_axes.sort_unstable();
                sorted_axes.reverse();
                sorted_axes.dedup();

                // Apply reduction sequentially along each axis
                for &ax in &sorted_axes {
                    println!("Reduciendo eje {}", ax);
                    result = reduction_fn(&result, ndarray::Axis(ax));
                }

                // Handle keep_dims: restore reduced dimensions as size 1
                if keep_dims {
                    let mut final_shape = original_shape;
                    for &ax in axes_list {
                        final_shape[ax] = 1;
                    }
                    result = result
                        .into_shape_with_order(final_shape)
                        .map_err(|e| format!("Failed to reshape for keep_dims: {}", e))?;
                }

                Ok(CPUStorage::new(result))
            }
            None => {
                // Reduce across all dimensions to get scalar result
                let mut result = self.data.clone();
                let original_ndim = result.ndim();

                for dim in (0..original_ndim).rev() {
                    result = reduction_fn(&result, ndarray::Axis(dim));
                }

                // Handle keep_dims for full reduction: create shape with all 1s
                if keep_dims {
                    let final_shape = vec![1; original_ndim];
                    result = result
                        .into_shape_with_order(final_shape)
                        .map_err(|e| format!("Failed to reshape for keep_dims: {}", e))?;
                }

                Ok(CPUStorage::new(result))
            }
        }
    }

    /// Generic comparison method using ndarray::Zip for efficiency
    /// This is a helper method to avoid code duplication in comparison operations
    fn compare<F>(
        &self,
        other: &dyn StorageBackend<T>,
        comparison_fn: F,
    ) -> Result<CPUStorage<T>, String>
    where
        F: Fn(&T, &T) -> T,
    {
        let other_data = other.cpu_data()?;

        if self.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for comparison: {:?} vs {:?}",
                self.shape(),
                other_data.shape()
            ));
        }

        // Use ndarray's Zip for efficient element-wise comparison
        let result_data = ndarray::Zip::from(&self.data)
            .and(other_data)
            .map_collect(|&a, &b| comparison_fn(&a, &b));

        // Return the owned storage result
        Ok(CPUStorage::new(result_data))
    }

    pub fn zeros(shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
        T: rand_distr::num_traits::Zero,
    {
        // Create ndarray with zeros directly - more efficient than device layer
        let data = ndarray::ArrayD::zeros(ndarray::IxDyn(shape));
        Ok(Box::new(CPUStorage::new(data)))
    }

    pub fn ones(shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
        T: rand_distr::num_traits::One,
    {
        // Create ndarray with ones directly
        let data = ndarray::ArrayD::ones(ndarray::IxDyn(shape));
        Ok(Box::new(CPUStorage::new(data)))
    }

    pub fn full(shape: &[usize], value: T) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
    {
        // Create ndarray filled with specific value
        let data = ndarray::ArrayD::from_elem(ndarray::IxDyn(shape), value);
        Ok(Box::new(CPUStorage::new(data)))
    }

    pub fn randn(shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
        StandardUniform: rand_distr::Distribution<T>,
    {
        let mut rng = rand::rng();
        let total_elements: usize = shape.iter().product();
        let two = FerroxN::from_f64(2.0).expect("Cannot cast from f64");
        let one = FerroxN::one();
        let data: Vec<T> = (0..total_elements)
            .map(|_| rng.random::<T>() * two - one) // Simple random between -1 and 1
            .collect();
        let data_array = ArrayD::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| format!("Failed to create array from data: {e}"))?;
        Ok(Box::new(CPUStorage::new(data_array)))
    }

    /// Conditional selection operation
    pub fn where_condition(
        condition: &dyn StorageBackend<T>,
        true_vals: &dyn StorageBackend<T>,
        false_vals: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxCudaF,
        Self: Sized,
    {
        // Get data from all three storages
        let condition_data = condition.cpu_data()?;
        let true_data = true_vals.cpu_data()?;
        let false_data = false_vals.cpu_data()?;

        // Validate shapes match
        if condition_data.shape() != true_data.shape()
            || condition_data.shape() != false_data.shape()
        {
            return Err("Shape mismatch in where_condition".to_string());
        }

        // Element-wise selection using iterator zip
        let result_data: Vec<T> = condition_data
            .iter()
            .zip(true_data.iter())
            .zip(false_data.iter())
            .map(|((&cond, &true_val), &false_val)| {
                if cond > FerroxN::zero() {
                    true_val
                } else {
                    false_val
                }
            })
            .collect();

        // Create result array with same shape
        let result_array = ndarray::Array::from_shape_vec(condition_data.raw_dim(), result_data)
            .map_err(|e| format!("Failed to create result: {e}"))?;

        Ok(Box::new(CPUStorage::new(result_array)))
    }
}

impl<T> StorageBackend<T> for CPUStorage<T>
where
    T: FerroxCudaN,
{
    fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn into_any(self: Box<Self>) -> Box<dyn Any + 'static> {
        self as Box<dyn std::any::Any + 'static>
    }

    fn ndim(&self) -> usize {
        self.data.ndim()
    }

    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    fn is_gpu(&self) -> bool {
        false
    }

    fn cpu_data(&self) -> Result<&ArrayD<T>, String> {
        Ok(&self.data)
    }

    fn cpu_data_mut(&mut self) -> Result<&mut ArrayD<T>, String> {
        Ok(&mut self.data)
    }

    fn owns_data(&self) -> bool {
        true // Always owns data in this simplified version
    }

    fn clone_storage(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        Ok(Box::new(self.clone()))
    }

    fn reshape(&mut self, new_shape: &[usize]) -> Result<(), String> {
        // Validate total elements remain the same
        let current_size: usize = self.data.shape().iter().product();
        let new_size: usize = new_shape.iter().product();

        if current_size != new_size {
            return Err(format!(
                "Cannot change shape: current size {current_size} != new size {new_size}",
            ));
        }

        // Use into_shape_with_order for in-place reshape
        self.data = self
            .data
            .clone()
            .into_shape_with_order(IxDyn(new_shape))
            .map_err(|e| format!("Shape change failed: {e}"))?;

        Ok(())
    }

    // Efficient in-place operations using helper method
    fn broadcast_to(&mut self, target_shape: &[usize]) -> Result<(), String> {
        let current_shape = self.shape();

        // Validate broadcasting rules
        if target_shape.len() < current_shape.len() {
            return Err("Cannot broadcast to smaller number of dimensions".to_string());
        }

        // Check if shapes are compatible for broadcasting
        let offset = target_shape.len() - current_shape.len();
        for (i, (&current_dim, &target_dim)) in current_shape
            .iter()
            .zip(target_shape[offset..].iter())
            .enumerate()
        {
            if current_dim != 1 && current_dim != target_dim {
                return Err(format!(
                    "Cannot broadcast dimension {i} from size {current_dim} to size {target_dim}",
                ));
            }
        }

        // Create broadcasted array by cloning data
        let view = self.data.view();
        let broadcasted_view = view.broadcast(target_shape).ok_or("Broadcast failed")?;

        // Convert to owned data
        self.data = broadcasted_view.to_owned();
        Ok(())
    }

    fn transpose(&mut self, axes: Option<&[usize]>) -> Result<(), String> {
        match axes {
            Some(order) => {
                if order.len() != self.ndim() || {
                    let mut sorted = order.to_vec();
                    sorted.sort_unstable();
                    sorted != (0..self.ndim()).collect::<Vec<_>>()
                } {
                    return Err("Invalid axes permutation".into());
                }
                self.data = self.data.view().permuted_axes(order).to_owned();
            }
            None => {
                if self.ndim() > 1 {
                    self.data = self.data.t().to_owned();
                }
            }
        }
        Ok(())
    }

    fn unsqueeze(&mut self, axis: usize) -> Result<(), String> {
        let current_shape = self.shape();

        if axis > current_shape.len() {
            return Err(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                current_shape.len()
            ));
        }

        // Insert dimension of size 1 at specified axis
        let mut new_shape = current_shape.to_vec();
        new_shape.insert(axis, 1);

        // Use helper method for in-place operation
        self.reshape(&new_shape)
    }

    fn squeeze(&mut self, axis: Option<usize>) -> Result<(), String> {
        let current_shape = self.data.shape();

        let new_shape = match axis {
            Some(ax) => {
                if ax >= current_shape.len() {
                    return Err(format!(
                        "Axis {} out of bounds for tensor with {} dimensions",
                        ax,
                        current_shape.len()
                    ));
                }

                if current_shape[ax] != 1 {
                    return Err(format!(
                        "Cannot squeeze axis {} with size {}",
                        ax, current_shape[ax]
                    ));
                }

                // Remove the specified axis
                let mut new_shape = current_shape.to_vec();
                new_shape.remove(ax);
                new_shape
            }
            None => {
                // Remove all dimensions of size 1
                let new_shape: Vec<usize> = current_shape
                    .iter()
                    .filter(|&&size| size != 1)
                    .cloned()
                    .collect();

                // If all dimensions were size 1, keep at least one dimension
                if new_shape.is_empty() {
                    vec![1]
                } else {
                    new_shape
                }
            }
        };

        // Use helper method for in-place operation
        self.reshape(&new_shape)
    }

    fn add(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Get other's CPU data - this handles CPU-CPU operations
        let other_data = other.cpu_data()?;

        // Shape broadcasting check - ndarray handles this automatically but we validate first
        if self.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for addition: {:?} vs {:?}",
                self.shape(),
                other_data.shape()
            ));
        }

        // Element-wise addition using ndarray's efficient implementation
        let result = &self.data + other_data;
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn sub(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for subtraction: {:?} vs {:?}",
                self.shape(),
                other_data.shape()
            ));
        }

        let result = &self.data - other_data;
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn mul(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for multiplication: {:?} vs {:?}",
                self.shape(),
                other_data.shape()
            ));
        }

        let result = &self.data * other_data;
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn div(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for division: {:?} vs {:?}",
                self.shape(),
                other_data.shape()
            ));
        }

        let result = &self.data / other_data;
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn min(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for min operation: {:?} vs {:?}",
                self.shape(),
                other_data.shape()
            ));
        }

        // Use flat iteration for efficiency - works with any dimensional tensor
        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| if a <= b { a } else { b })
            .collect();

        let result_array = ndarray::Array::from_shape_vec(self.data.raw_dim(), result_data)
            .map_err(|e| format!("Failed to create result tensor: {e}"))?;

        Ok(Box::new(CPUStorage::new(result_array)))
    }

    fn max(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for max operation: {:?} vs {:?}",
                self.shape(),
                other_data.shape()
            ));
        }

        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| if a >= b { a } else { b })
            .collect();

        let result_array = ndarray::Array::from_shape_vec(self.data.raw_dim(), result_data)
            .map_err(|e| format!("Failed to create result tensor: {e}"))?;

        Ok(Box::new(CPUStorage::new(result_array)))
    }

    fn add_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        // ndarray's scalar operations are very efficient - no broadcasting overhead
        let result = &self.data + scalar;
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn mul_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = &self.data * scalar;
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn sub_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        // ndarray's scalar operations are very efficient - no broadcasting overhead
        let result = &self.data - scalar;
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn div_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = &self.data / scalar;
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn neg(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Unary negation - ndarray handles this efficiently
        let result = self.data.mapv(|x| -x);
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn abs(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Element-wise absolute value using mapv for efficiency
        let result_data: Vec<T> = self.data.iter().map(|&x| x.abs()).collect();

        let result_array = ndarray::Array::from_shape_vec(self.data.raw_dim(), result_data)
            .map_err(|e| format!("Failed to create result tensor: {e}"))?;

        Ok(Box::new(CPUStorage::new(result_array)))
    }

    fn reciprocal(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_data: Vec<T> = self
            .data
            .iter()
            .map(|&x| <T as FerroxN>::one() / x)
            .collect();

        let result_array = ndarray::Array::from_shape_vec(self.data.raw_dim(), result_data)
            .map_err(|e| format!("Failed to create result tensor: {e}"))?;

        Ok(Box::new(CPUStorage::new(result_array)))
    }

    fn clamp(&self, min_val: T, max_val: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Element-wise clamping using mapv - efficient vectorized operation
        let result_data = self.data.mapv(|x| {
            if x < min_val {
                min_val
            } else if x > max_val {
                max_val
            } else {
                x
            }
        });

        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn sqrt(&self) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxF,
    {
        let result_data = self.data.mapv(|x| x.sqrt());
        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn greater_equal(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        let storage = self.compare(other, |&a, &b| {
            if a >= b {
                FerroxN::one()
            } else {
                FerroxN::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn greater_equal_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_array = self.data.mapv(|x| {
            if x >= scalar {
                FerroxN::one()
            } else {
                FerroxN::zero()
            }
        });
        Ok(Box::new(CPUStorage::new(result_array)))
    }

    fn less_equal(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        let storage = self.compare(other, |&a, &b| {
            if a <= b {
                FerroxN::one()
            } else {
                FerroxN::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn less_equal_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_array = self.data.mapv(|x| {
            if x <= scalar {
                FerroxN::one()
            } else {
                FerroxN::zero()
            }
        });
        Ok(Box::new(CPUStorage::new(result_array)))
    }

    fn greater(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let storage = self.compare(other, |&a, &b| {
            if a > b {
                FerroxN::one()
            } else {
                FerroxN::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn greater_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_array = self.data.mapv(|x| {
            if x > scalar {
                FerroxN::one()
            } else {
                FerroxN::zero()
            }
        });
        Ok(Box::new(CPUStorage::new(result_array)))
    }

    fn less(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let storage = self.compare(other, |&a, &b| {
            if a < b {
                FerroxN::one()
            } else {
                FerroxN::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn less_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_array = self.data.mapv(|x| {
            if x < scalar {
                FerroxN::one()
            } else {
                FerroxN::zero()
            }
        });
        Ok(Box::new(CPUStorage::new(result_array)))
    }

    fn equal(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let storage = self.compare(other, |&a, &b| {
            if a == b {
                FerroxN::one()
            } else {
                FerroxN::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn logical_not(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Flip 0s to 1s and non-zeros to 0s
        let result_data = self.data.mapv(|x| {
            if x == <T as FerroxN>::zero() {
                <T as FerroxN>::one()
            } else {
                <T as FerroxN>::zero()
            }
        });

        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn in_range(&self, min_val: T, max_val: T) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxN,
    {
        // Check if values are in range [min_val, max_val]
        let result_data = self.data.mapv(|x| {
            if x >= min_val && x <= max_val {
                <T as FerroxN>::one()
            } else {
                <T as FerroxN>::zero()
            }
        });

        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn sign(&self) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxN,
    {
        // Return 1 for positive, -1 for negative, 0 for zero
        let result_data = self.data.mapv(|x| {
            if x > <T as FerroxN>::zero() {
                <T as FerroxN>::one()
            } else if x < <T as FerroxN>::zero() {
                -<T as FerroxN>::one()
            } else {
                <T as FerroxN>::zero()
            }
        });

        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn matmul(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: Clone + ndarray::LinalgScalar,
    {
        let other_data = other.cpu_data()?;

        if self.data.ndim() != 2_usize || other_data.ndim() != 2_usize {
            return Err("Matrix multiplication requires 2D tensors".to_string());
        }

        let a_shape = self.shape();
        let b_shape = other_data.shape();

        if a_shape[1] != b_shape[0] {
            return Err(format!(
                "Matrix multiplication shape mismatch: ({}, {}) @ ({}, {})",
                a_shape[0], a_shape[1], b_shape[0], b_shape[1]
            ));
        }

        // Convert to 2D views for matrix multiplication
        let a: ndarray::ArrayView2<T> = self
            .data
            .view()
            .into_dimensionality()
            .map_err(|e| format!("Failed to convert to 2D view: {e}"))?;
        let b: ndarray::ArrayView2<T> = other_data
            .view()
            .into_dimensionality()
            .map_err(|e| format!("Failed to convert to 2D view: {e}"))?;

        // Perform matrix multiplication using ndarray's dot product
        let result = a.dot(&b);

        Ok(Box::new(CPUStorage::new(result.into_dyn())))
    }

    fn sigmoid(&self) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxF,
    {
        // Sigmoid function: 1 / (1 + exp(-x))
        let result_data = self.data.mapv(|x| {
            let one = <T as FerroxN>::one();
            let neg_x = -x;
            one / (one + neg_x.exp())
        });

        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn softmax(&self) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxF,
    {
        // Softmax function: exp(x_i - max(x)) / sum(exp(x_j - max(x)))
        let max_val = self
            .data
            .fold(FerroxN::min_value(), |a, &b| if b > a { b } else { a });

        // Substract the max value
        let exp_shifted = self.data.mapv(|x| (x - max_val).exp());
        // Sum all exponents
        let sum_exp = exp_shifted.sum();

        // Divide each value by the total sum.
        let result_data = exp_shifted.mapv(|x| x / sum_exp);
        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn softmax_batched(&self, axis: usize) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxCudaF,
    {
        if axis >= self.data.ndim() {
            return Err(format!(
                "Softmax axis {} out of bounds for tensor with {} dimensions",
                axis,
                self.data.ndim()
            ));
        }


        let axis_obj = Axis(axis);

        let max_values = self
            .data
            .fold_axis(axis_obj, FerroxN::min_value(), |&acc, &x| {
                if x > acc {
                    x
                } else {
                    acc
                }
            });
        let mut expanded_max = max_values.clone();
        expanded_max = expanded_max.insert_axis(axis_obj);
        let shifted = &self.data - &expanded_max;
        let exp_values = shifted.mapv(|x| x.exp());
        let sum_exp = exp_values.sum_axis(axis_obj);
        let expanded_sum = sum_exp.insert_axis(axis_obj);
        let result = &exp_values / &expanded_sum;
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn relu(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // ReLU activation: max(0, x)
        let result_data = self.data.mapv(|x| {
            let zero = <T as FerroxN>::zero();
            if x > zero {
                x
            } else {
                zero
            }
        });

        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn exp(&self) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxCudaF,
    {
        // Element-wise exponential
        let result_data = self.data.mapv(|x| x.exp());
        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn log(&self) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxCudaF,
    {
        // Element-wise natural logarithm
        let result_data = self.data.mapv(|x| x.ln());
        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn tanh(&self) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxCudaF,
    {
        // Numerically stable hyperbolic tangent implementation
        // Avoids overflow for large positive/negative values
        let result_data = self.data.mapv(|x| {
            // For numerical stability, we use different formulas based on the sign of x
            // This prevents overflow when x is very large or very small

            if x > FerroxN::from_f64(20.0).unwrap() {
                // For large positive x, tanh(x) ≈ 1
                // Avoids computing exp(x) which would overflow
                FerroxN::one()
            } else if x < FerroxN::from_f64(-20.0).unwrap() {
                // For large negative x, tanh(x) ≈ -1
                // Avoids computing exp(-x) which would overflow
                FerroxN::from_f64(-1.0).unwrap()
            } else if x >= FerroxN::zero() {
                // For non-negative x, use: tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
                // This formulation is stable for positive x
                let exp_neg_2x = (-x * FerroxN::from_f64(2.0).unwrap()).exp();
                let numerator = <T as FerroxF>::one() - exp_neg_2x;
                let denominator = <T as FerroxF>::one() + exp_neg_2x;

                // Check for potential division by zero (shouldn't happen with this formulation)
                if denominator == FerroxN::zero() {
                    FerroxN::one() // Return 1 as limit
                } else {
                    numerator / denominator
                }
            } else {
                // For negative x, use: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
                // This formulation is stable for negative x
                let exp_2x = (x * FerroxN::from_f64(2.0).unwrap()).exp();
                let numerator = exp_2x - FerroxN::one();
                let denominator = exp_2x + FerroxN::one();

                // Check for potential division by zero (shouldn't happen with this formulation)
                if denominator == FerroxN::zero() {
                    FerroxN::from_f64(-1.0).unwrap() // Return -1 as limit
                } else {
                    numerator / denominator
                }
            }
        });

        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn powf(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxCudaF,
    {
        let other_data = other.cpu_data()?;

        if self.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for powf: {:?} vs {:?}",
                self.shape(),
                other_data.shape()
            ));
        }

        // Element-wise power using ndarray's Zip
        let result_data = ndarray::Zip::from(&self.data)
            .and(other_data)
            .map_collect(|&a, &b| a.powf(b));

        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn power_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxCudaF,
    {
        // Scalar power operation
        let result_data = self.data.mapv(|x| x.powf(scalar));
        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn sum(
        &self,
        axes: Option<&[usize]>,
        keep_dims: bool,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Use reduce_axes with ndarray's sum_axis function
        let result = self.reduce(axes, keep_dims, |array, ax| array.sum_axis(ax))?;
        Ok(Box::new(result) as Box<dyn StorageBackend<T>>)
    }

    fn mean(
        &self,
        axes: Option<&[usize]>,
        keep_dims: bool,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        // First compute sum using reduce_axes
        let sum_result = self.sum(axes, keep_dims)?;

        // Calculate the number of elements being averaged over
        let divisor = match axes {
            Some(axes_list) => {
                // Product of dimensions being reduced
                axes_list
                    .iter()
                    .map(|&ax| self.shape()[ax])
                    .product::<usize>() as f64
            }
            None => {
                // All elements if no axes specified
                self.data.len() as f64
            }
        };

        // Convert divisor to tensor type and divide
        let divisor_scalar =
            FerroxN::from_f64(1.0 / divisor).ok_or("Failed to convert divisor to tensor type")?;

        sum_result.mul_scalar(divisor_scalar)
    }

    fn max_reduce(
        &self,
        axes: Option<&[usize]>,
        keep_dims: bool,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Use reduce_axes with custom max reduction function
        // ndarray doesn't have a direct max_axis function, so we implement our own
        let result = self.reduce(axes, keep_dims, |array, ax| {
            let first = FerroxN::min_value();

            // Fold along the specified axis to find maximum values
            array.fold_axis(ax, first, |&acc, &x| if x > acc { x } else { acc })
        })?;

        Ok(Box::new(result) as Box<dyn StorageBackend<T>>)
    }

    fn min_reduce(
        &self,
        axes: Option<&[usize]>,
        keep_dims: bool,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Use reduce_axes with custom min reduction function
        // Similar to max_axes but finding minimum values
        let result = self.reduce(axes, keep_dims, |array, ax| {
            let first = FerroxN::max_value();
            // Fold along the specified axis to find minimum values
            array.fold_axis(ax, first, |&acc, &x| if x < acc { x } else { acc })
        })?;
        Ok(Box::new(result) as Box<dyn StorageBackend<T>>)
    }

    fn conv2d(
        &self,
        filter: &dyn StorageBackend<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Ensure filter is also CPU storage
        let filter_data = filter.cpu_data()?;

        let input_shape = self.shape();
        let filter_shape = filter.shape();

        // Validate input dimensions for conv2d
        if input_shape.len() != 4 || filter_shape.len() != 4 {
            return Err("Conv2D requires 4D tensors [batch, channels, height, width]".to_string());
        }

        let result = self.conv2d_impl(filter_data, stride, padding)?;
        Ok(Box::new(CPUStorage::new(result)))
    }


    fn deconv2d(
        &self,
        input: &dyn StorageBackend<T>,
        filter: &dyn StorageBackend<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Ensure filter is also CPU storage
        let filter_data = filter.cpu_data()?;
        let filter_shape = filter.shape();

        let input_shape = input.shape();
        // Validate input dimensions for conv2d
        if input_shape.len() != 4 || filter_shape.len() != 4 || self.shape().len() != 4 {
            return Err("Deconv2D requires 4D tensors [batch, channels, height, width]".to_string());
        }
        let result = self.deconv2d_impl( filter_data, input_shape, stride, padding)?;
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn cross_correlation(
            &self,
            other: &dyn StorageBackend<T>,
            output_shape: &[usize],
            stride: (usize, usize),
            padding: (usize, usize),
        ) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;
        let other_shape = other.shape();
        if self.shape().len() != 4 || other_shape.len() != 4 {
            return Err("Cross correlation requires 4D tensors [batch, channels, height, width]".to_string());
        }
        let result = self.cross_correlation_impl(other_data,  output_shape, stride, padding)?;
        Ok(Box::new(CPUStorage::new(result)))


    }



    fn iter_values(&self) -> Result<Vec<T>, String> {
        // Efficient cloning of all values for iteration
        Ok(self.data.iter().cloned().collect())
    }

    fn get_flat(&self, index: usize) -> Result<Option<T>, String> {
        // Flat indexing using ndarray's slice functionality
        match self.data.as_slice() {
            Some(slice) => {
                if index < slice.len() {
                    Ok(Some(slice[index]))
                } else {
                    Ok(None)
                }
            }
            None => Err("Data is not contiguous for flat indexing".to_string()),
        }
    }

    fn get_multi(&self, indices: &[usize]) -> Result<Option<T>, String> {
        // Multi-dimensional indexing using ndarray
        if indices.len() != self.data.ndim() {
            return Ok(None);
        }

        // Check bounds before accessing
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape()[i] {
                return Ok(None);
            }
        }

        Ok(Some(self.data[ndarray::IxDyn(indices)]))
    }

    /*  fn execute_custom_op<R>(&self, op: Box<dyn CustomOperation<T, R>>) -> Result<R, String> {
        // Execute custom operation with immutable access to ArrayD data
        // Operation creates new results following existing storage pattern
        op.execute_cpu(&self.data)
    }*/
}
