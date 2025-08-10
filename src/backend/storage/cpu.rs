// src/backend/storage/cpu.rs
use super::StorageBackend;
use crate::backend::{FerroxCudaF, FerroxF};
use ndarray::{ArrayD, ArrayViewD, IxDyn};
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
    T: crate::backend::number::FerroxCudaF + Clone,
{
    /// Convert image patches to column matrix (im2col) - reused from original impl
    /// This transforms 4D convolution into efficient 2D matrix multiplication
    fn im2col(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<T>, String> {
        let input_shape = self.shape();
        let (batch, channels, in_h, in_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (kernel_h, kernel_w) = kernel_size;

        let out_h = (in_h + 2 * padding.0 - kernel_h) / stride.0 + 1;
        let out_w = (in_w + 2 * padding.1 - kernel_w) / stride.1 + 1;

        let col_height = channels * kernel_h * kernel_w;
        let col_width = batch * out_h * out_w;

        let mut col_data = vec![FerroxF::zero(); col_height * col_width];

        // Use effective data access for logical views
        let input_data = if let Some(data) = self.data.as_slice() {
            data
        } else {
            return Err("Input data is empty or not contiguous!".to_string());
        };

        // Extract patches and arrange them as columns for matrix multiplication
        for b in 0..batch {
            for c in 0..channels {
                for ky in 0..kernel_h {
                    for kx in 0..kernel_w {
                        let col_row = c * kernel_h * kernel_w + ky * kernel_w + kx;

                        for out_y in 0..out_h {
                            for out_x in 0..out_w {
                                let in_y = out_y * stride.0 + ky;
                                let in_x = out_x * stride.1 + kx;
                                let col_col = b * out_h * out_w + out_y * out_w + out_x;

                                // Handle padding by checking bounds
                                if in_y >= padding.0
                                    && in_y < in_h + padding.0
                                    && in_x >= padding.1
                                    && in_x < in_w + padding.1
                                {
                                    let actual_y = in_y - padding.0;
                                    let actual_x = in_x - padding.1;

                                    if actual_y < in_h && actual_x < in_w {
                                        let input_idx = b * (channels * in_h * in_w)
                                            + c * (in_h * in_w)
                                            + actual_y * in_w
                                            + actual_x;
                                        col_data[col_row * col_width + col_col] =
                                            input_data[input_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        ArrayD::from_shape_vec(IxDyn(&[col_height, col_width]), col_data)
            .map_err(|e| format!("Failed to create im2col matrix: {e}"))
    }

    /// Standard 2D convolution implementation using im2col + GEMM
    fn conv2d_impl(
        &self,
        filter: &ArrayD<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<T>, String> {
        let input_shape = self.shape();
        let filter_shape = filter.shape();

        let (batch, in_channels, in_h, in_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (out_channels, _, kernel_h, kernel_w) = (
            filter_shape[0],
            filter_shape[1],
            filter_shape[2],
            filter_shape[3],
        );

        let out_h = (in_h + 2 * padding.0 - kernel_h) / stride.0 + 1;
        let out_w = (in_w + 2 * padding.1 - kernel_w) / stride.1 + 1;

        // Transform input to column matrix for efficient GEMM
        let col_matrix = self.im2col((kernel_h, kernel_w), stride, padding)?;

        // Reshape filter for matrix multiplication
        let filter_reshaped = filter
            .clone()
            .into_shape_with_order(IxDyn(&[out_channels, in_channels * kernel_h * kernel_w]))
            .map_err(|e| format!("Filter reshape failed: {e}"))?;

        // Perform convolution as matrix multiplication: filter @ col_matrix
        let im2col_view: ndarray::ArrayView2<T> = col_matrix
            .view()
            .into_dimensionality()
            .map_err(|e| format!("Shape error: {e}"))?;

        let filter_view: ndarray::ArrayView2<T> = filter_reshaped
            .view()
            .into_dimensionality()
            .map_err(|e| format!("Shape error: {e}"))?;

        let output_2d = filter_view.dot(&im2col_view);

        // Transpose result from [out_channels, batch * out_h * out_w] to [batch, out_channels, out_h, out_w]
        let output_data: Vec<T> = if let Some(out_slice) = output_2d.as_slice() {
            out_slice.to_vec()
        } else {
            return Err("Failed to get contiguous output data".to_string());
        };
        let mut final_output = vec![FerroxF::zero(); batch * out_channels * out_h * out_w];

        for out_c in 0..out_channels {
            for b in 0..batch {
                for y in 0..out_h {
                    for x in 0..out_w {
                        let src_idx =
                            out_c * (batch * out_h * out_w) + b * (out_h * out_w) + y * out_w + x;
                        let dst_idx = b * (out_channels * out_h * out_w)
                            + out_c * (out_h * out_w)
                            + y * out_w
                            + x;
                        final_output[dst_idx] = output_data[src_idx];
                    }
                }
            }
        }

        ArrayD::from_shape_vec(IxDyn(&[batch, out_channels, out_h, out_w]), final_output)
            .map_err(|e| format!("Failed to create output tensor: {e}"))
    }
}

impl<T> CPUStorage<T>
where
    T: FerroxCudaF,
{
    // Move the generic reduce method to the concrete implementation
    // This avoids making StorageBackend non-dyn-compatible
    fn reduce<F>(&self, axes: Option<&[usize]>, reduction_fn: F) -> Result<CPUStorage<T>, String>
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
                // Sort in descending order to prevent index shifting during reduction
                let mut sorted_axes = axes_list.to_vec();
                sorted_axes.sort_unstable();
                sorted_axes.reverse();
                sorted_axes.dedup();

                // Apply reduction sequentially along each axis
                for &ax in &sorted_axes {
                    result = reduction_fn(&result, ndarray::Axis(ax));
                }

                Ok(CPUStorage::new(result))
            }
            None => {
                // Reduce across all dimensions to get scalar result
                let mut result = self.data.clone();
                for dim in (0..result.ndim()).rev() {
                    result = reduction_fn(&result, ndarray::Axis(dim));
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
        let two = FerroxF::from_f64(2.0).expect("Cannot cast from f64");
        let one = FerroxF::one();
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
                if cond > FerroxF::zero() {
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
    T: FerroxCudaF,
{
    fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn into_any(self: Box<Self>) -> Box<dyn Any + 'static> {
        Box::new(self)
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
        let current_ndim = self.ndim();

        match axes {
            Some(axes_order) => {
                // Validate axes
                if axes_order.len() != current_ndim {
                    return Err(format!(
                        "Axes length {} doesn't match tensor dimensions {}",
                        axes_order.len(),
                        current_ndim
                    ));
                }

                // Validate permutation - check all axes are unique and in valid range
                let mut sorted_axes = axes_order.to_vec();
                sorted_axes.sort_unstable();
                let expected: Vec<usize> = (0..current_ndim).collect();
                if sorted_axes != expected {
                    return Err(format!("Invalid axes permutation: {axes_order:?}"));
                }

                // Apply transpose with specified axes
                let transposed_view = self.data.view().permuted_axes(axes_order);
                self.data = transposed_view.to_owned();
            }
            None => {
                // Default transpose - reverse all axes
                if current_ndim <= 1 {
                    // No change needed for 0D or 1D tensors
                    return Ok(());
                }

                let axes_order: Vec<usize> = (0..current_ndim).rev().collect();
                let transposed_view = self.data.view().permuted_axes(axes_order);
                self.data = transposed_view.to_owned();
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
            .map(|&x| <T as FerroxF>::one() / x)
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

    fn sqrt(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_data = self.data.mapv(|x| x.sqrt());
        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn greater_equal(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        let storage = self.compare(other, |&a, &b| {
            if a >= b {
                FerroxF::one()
            } else {
                FerroxF::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn greater_equal_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_array = self.data.mapv(|x| {
            if x >= scalar {
                FerroxF::one()
            } else {
                FerroxF::zero()
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
                FerroxF::one()
            } else {
                FerroxF::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn less_equal_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_array = self.data.mapv(|x| {
            if x <= scalar {
                FerroxF::one()
            } else {
                FerroxF::zero()
            }
        });
        Ok(Box::new(CPUStorage::new(result_array)))
    }

    fn greater(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let storage = self.compare(other, |&a, &b| {
            if a > b {
                FerroxF::one()
            } else {
                FerroxF::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn greater_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_array = self.data.mapv(|x| {
            if x > scalar {
                FerroxF::one()
            } else {
                FerroxF::zero()
            }
        });
        Ok(Box::new(CPUStorage::new(result_array)))
    }

    fn less(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let storage = self.compare(other, |&a, &b| {
            if a < b {
                FerroxF::one()
            } else {
                FerroxF::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn less_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_array = self.data.mapv(|x| {
            if x < scalar {
                FerroxF::one()
            } else {
                FerroxF::zero()
            }
        });
        Ok(Box::new(CPUStorage::new(result_array)))
    }

    fn equal(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let storage = self.compare(other, |&a, &b| {
            if a == b {
                FerroxF::one()
            } else {
                FerroxF::zero()
            }
        })?;
        Ok(Box::new(storage))
    }

    fn logical_not(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Flip 0s to 1s and non-zeros to 0s
        let result_data = self.data.mapv(|x| {
            if x == <T as crate::backend::number::FerroxF>::zero() {
                <T as crate::backend::number::FerroxF>::one()
            } else {
                <T as crate::backend::number::FerroxF>::zero()
            }
        });

        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn in_range(&self, min_val: T, max_val: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Check if values are in range [min_val, max_val]
        let result_data = self.data.mapv(|x| {
            if x >= min_val && x <= max_val {
                <T as crate::backend::number::FerroxF>::one()
            } else {
                <T as crate::backend::number::FerroxF>::zero()
            }
        });

        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn sign(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Return 1 for positive, -1 for negative, 0 for zero
        let result_data = self.data.mapv(|x| {
            if x > <T as crate::backend::number::FerroxF>::zero() {
                <T as crate::backend::number::FerroxF>::one()
            } else if x < <T as crate::backend::number::FerroxF>::zero() {
                -<T as crate::backend::number::FerroxF>::one()
            } else {
                <T as crate::backend::number::FerroxF>::zero()
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
        println!("Result = {result}");
        Ok(Box::new(CPUStorage::new(result.into_dyn())))
    }

    fn sigmoid(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Sigmoid function: 1 / (1 + exp(-x))
        let result_data = self.data.mapv(|x| {
            let one = <T as crate::backend::number::FerroxF>::one();
            let neg_x = -x;
            one / (one + neg_x.exp())
        });

        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn relu(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // ReLU activation: max(0, x)
        let result_data = self.data.mapv(|x| {
            let zero = <T as crate::backend::number::FerroxF>::zero();
            if x > zero {
                x
            } else {
                zero
            }
        });

        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn exp(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Element-wise exponential
        let result_data = self.data.mapv(|x| x.exp());
        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn log(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Element-wise natural logarithm
        let result_data = self.data.mapv(|x| x.ln());
        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn tanh(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Numerically stable hyperbolic tangent implementation
        // Avoids overflow for large positive/negative values
        let result_data = self.data.mapv(|x| {
            // For numerical stability, we use different formulas based on the sign of x
            // This prevents overflow when x is very large or very small

            if x > FerroxF::from_f64(20.0).unwrap() {
                // For large positive x, tanh(x) ≈ 1
                // Avoids computing exp(x) which would overflow
                FerroxF::one()
            } else if x < FerroxF::from_f64(-20.0).unwrap() {
                // For large negative x, tanh(x) ≈ -1
                // Avoids computing exp(-x) which would overflow
                FerroxF::from_f64(-1.0).unwrap()
            } else if x >= FerroxF::zero() {
                // For non-negative x, use: tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
                // This formulation is stable for positive x
                let exp_neg_2x = (-x * FerroxF::from_f64(2.0).unwrap()).exp();
                let numerator = <T as FerroxF>::one() - exp_neg_2x;
                let denominator = <T as FerroxF>::one() + exp_neg_2x;

                // Check for potential division by zero (shouldn't happen with this formulation)
                if denominator == FerroxF::zero() {
                    FerroxF::one() // Return 1 as limit
                } else {
                    numerator / denominator
                }
            } else {
                // For negative x, use: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
                // This formulation is stable for negative x
                let exp_2x = (x * FerroxF::from_f64(2.0).unwrap()).exp();
                let numerator = exp_2x - FerroxF::one();
                let denominator = exp_2x + FerroxF::one();

                // Check for potential division by zero (shouldn't happen with this formulation)
                if denominator == FerroxF::zero() {
                    FerroxF::from_f64(-1.0).unwrap() // Return -1 as limit
                } else {
                    numerator / denominator
                }
            }
        });

        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn powf(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
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

    fn power_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Scalar power operation
        let result_data = self.data.mapv(|x| x.powf(scalar));
        Ok(Box::new(CPUStorage::new(result_data)))
    }

    fn sum(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Use reduce_axes with ndarray's sum_axis function
        let result = self.reduce(axes, |array, ax| array.sum_axis(ax))?;
        Ok(Box::new(result) as Box<dyn StorageBackend<T>>)
    }

    fn mean(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        // First compute sum using reduce_axes
        let sum_result = self.sum(axes)?;

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
            FerroxF::from_f64(1.0 / divisor).ok_or("Failed to convert divisor to tensor type")?;

        sum_result.mul_scalar(divisor_scalar)
    }

    fn max_reduce(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Use reduce_axes with custom max reduction function
        // ndarray doesn't have a direct max_axis function, so we implement our own
        let result = self.reduce(axes, |array, ax| {
            let first = if let Some(f) = array.first() {
                *f
            } else {
                panic!("Array is empty! Cannot reduce over empty array");
            };

            // Fold along the specified axis to find maximum values
            array.fold_axis(ax, first, |&acc, &x| if x > acc { x } else { acc })
        })?;

        Ok(Box::new(result) as Box<dyn StorageBackend<T>>)
    }

    fn min_reduce(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Use reduce_axes with custom min reduction function
        // Similar to max_axes but finding minimum values
        let result = self.reduce(axes, |array, ax| {
            let first = if let Some(f) = array.first() {
                *f
            } else {
                panic!("Array is empty! Cannot reduce over empty array");
            };
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
