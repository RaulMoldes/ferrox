use crate::backend::number::{CPUNumber, GPUFloat, GPUNumber};
use crate::backend::{Device, default_device};
use ndarray::{Array, ArrayD, Axis, IxDyn};
use std::ops::{Index, IndexMut};

// Tensor wrapper to handle dynamic arrays more elegantly
#[derive(Debug, Clone)]
pub struct CPUTensor<T>
where
    T: GPUNumber,
{
    // This `data` field is the main data storage of the tensor on CPU.
    pub data: ArrayD<T>, // As I documented in the device module, this will be changed toa generic type <T>
    // This way I will be able to use different data types in the future.
    // For now, we will keep it as f64 for simplicity.
    pub device: Device,
}

// Main implementation block with basic operations
impl<T> CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    // Basically a constructor for the Tensor struct.
    // It takes an ArrayD<f64> and a Device, and returns a Tensor.
    // The device is set to the default device if not provided.
    // This is similar to how PyTorch and TensorFlow work, where the device is set to the default device if not specified.
    // We could also take advantage of Rust's default trait to set the device to the default device if not provided.
    // In the course that I am following (dlsyscourse.org), the Tensor is actually inheriting from a graph node.
    // In rust we do not have inheritance, so we will just use composition.
    // Additionally, I decided to reverse the composition hierarchy as opposite to the course,
    // so the tensor is used to represent data, and is the main layer of abstraction over the device.
    // The graph node is a separate struct (check src/graph/node.rs), which indeed has a data property, that in ferrrox will be a tensor.
    // In the course, the graph node was the main layer of abstraction over the device, and the tensor inherits from it. The data field of the
    pub fn new(data: ArrayD<T>) -> Self {
        Self {
            data,
            device: default_device(),
        }
    }

    // Creates a new tensor with the given data and device.
    // This is useful when you want to create a tensor with a specific device, for example, when you want to use a GPU.
    // In the future, we could also add a method to create a tensor with a specific data type, but for now we will keep it simple.
    // The device is set to the default device if not provided.
    // This is similar to how PyTorch and TensorFlow work, where the device is set to the default device if not specified.
    // Ideally,we should not be bound to ndarray backend here because it defaults to CPU, but it is okay for now as i prefer to focus more on the automatic differentiation engine thing.
    pub fn new_with_device(data: ArrayD<T>, device: Device) -> Self {
        Self { data, device }
    }

    // Random numbers
    pub fn randn(shape: &[usize]) -> Self {
        let device = default_device();
        let data_f64 = device.randn(shape);
        let data = data_f64.mapv(|x| <T as CPUNumber>::from_f64(x).unwrap());
        Self { data, device }
    }

    pub fn randn_with_device(shape: &[usize], device: Device) -> Self {
        // Generates a tensor with random numbers from a normal distribution.
        let data_f64 = device.randn(shape);
        let data = data_f64.mapv(|x| <T as CPUNumber>::from_f64(x).unwrap());
        Self { data, device }
    }

    // Random numbers
    pub fn randint(shape: &[usize]) -> Self {
        let device = default_device();
        let data_i64 = device.randint(shape);
        let data = data_i64.mapv(|x| <T as CPUNumber>::from_i64(x).unwrap());
        Self { data, device }
    }

    pub fn randint_with_device(shape: &[usize], device: Device) -> Self {
        // Generates a tensor with random integer numbers.
        let data_i64 = device.randint(shape);
        let data = data_i64.mapv(|x| <T as CPUNumber>::from_i64(x).unwrap());
        Self { data, device }
    }

    // Creates a tensor from a Rust vector. Again we are bound to ndarray backend here, but it is okay for now.
    // This function takes a vector of f64 and a shape, and returns a tensor with the given shape.
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self, String> {
        let total_elements: usize = shape.iter().product();
        if data.len() != total_elements {
            return Err(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                total_elements
            ));
        }

        match Array::from_shape_vec(IxDyn(shape), data) {
            Ok(array) => Ok(Self::new(array)),
            Err(e) => Err(format!("Failed to create tensor: {}", e)),
        }
    }

    // I decided not to implement the empty() function as it is useless in practice.
    // The empty function in numpy creates an uninitialized array, which is unsafe in Rust.
    // Instead, we will use the zeros() function to create a tensor with zeroes.
    // If you want to use uninitialized arrays, you can use `Array::uninit` but it is unsafe.

    // Some utility functions to get information about the tensor.
    // These functions are similar to the ones in PyTorch and TensorFlow, and they return the shape, number of dimensions, length, data, and device of the tensor.
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &ArrayD<T> {
        &self.data
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn into_data(self) -> ArrayD<T> {
        self.data
    }
    // TODO: Modify this section to have a more generic way to operate between cuda and cpu tensors.

    // Element-wise operations.
    // These are operations that are applied to each element of the tensor.
    // They are easily parallelizable and can be implemented using ndarray's mapv method.
    // The mapv method applies a function to each element of the array and returns a new array with the results.
    // The mapv operation does not actually parallelize by itself, but it is much more efficient th
    // - add
    // - multiply
    pub fn add(&self, other: &CPUTensor<T>) -> Result<CPUTensor<T>, String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }
        Ok(CPUTensor::new_with_device(
            &self.data + &other.data,
            self.device.clone(),
        ))
    }

    pub fn mul(&self, other: &CPUTensor<T>) -> Result<CPUTensor<T>, String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }
        Ok(CPUTensor::new_with_device(
            &self.data * &other.data,
            self.device.clone(),
        ))
    }

    pub fn negate(&self) -> CPUTensor<T> {
        CPUTensor::new_with_device(self.data.mapv(|x| -x), self.device.clone())
    }

    pub fn div(&self, other: &CPUTensor<T>) -> Result<CPUTensor<T>, String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }
        Ok(CPUTensor::new_with_device(
            &self.data / &other.data,
            self.device.clone(),
        ))
    }

    /// Element-wise minimum between two tensors
    /// Returns a new tensor with the minimum value at each position
    pub fn min(&self, other: &Self) -> Result<Self, String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Shape mismatch in min operation: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }

        // Use flat iteration for efficiency - works with any dimensional tensor
        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| if a <= b { a } else { b })
            .collect();

        let result_array = ndarray::Array::from_shape_vec(self.data.raw_dim(), result_data)
            .map_err(|e| format!("Failed to create result tensor: {}", e))?;

        Ok(Self::new_with_device(result_array, self.device.clone()))
    }

    /// Element-wise maximum between two tensors  
    /// Returns a new tensor with the maximum value at each position
    pub fn max(&self, other: &Self) -> Result<Self, String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Shape mismatch in max operation: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }

        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| if a >= b { a } else { b })
            .collect();

        let result_array = ndarray::Array::from_shape_vec(self.data.raw_dim(), result_data)
            .map_err(|e| format!("Failed to create result tensor: {}", e))?;

        Ok(Self::new_with_device(result_array, self.device.clone()))
    }

    /// Element-wise absolute value
    /// Returns a new tensor with absolute values
    pub fn abs(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.abs()).collect();

        let result_array = ndarray::Array::from_shape_vec(self.data.raw_dim(), result_data)
            .expect("Shape should match original tensor");

        Self::new_with_device(result_array, self.device.clone())
    }

    // Clamping operation.Clamps the values of the tensor to be within the specified range.
    pub fn clamp(&self, min_val: T, max_val: T) -> Self {
        let result_data = self.data.mapv(|x| {
            if x < min_val {
                min_val
            } else if x > max_val {
                max_val
            } else {
                x
            }
        });
        Self::new_with_device(result_data, self.device.clone())
    }

    /// Find maximum values along a specific dimension
    /// Reduces the tensor by one dimension
    pub fn max_along_dim(&self, dim: usize) -> Result<Self, String> {
        let shape = self.shape();

        if dim >= shape.len() {
            return Err(format!(
                "Dimension {} out of bounds for tensor with {} dimensions",
                dim,
                shape.len()
            ));
        }

        // Calculate the output shape (remove the specified dimension)
        let mut output_shape = shape.to_vec();
        output_shape.remove(dim);

        // If we're reducing all dimensions, result is a scalar
        if output_shape.is_empty() {
            output_shape.push(1);
        }

        let output_size: usize = output_shape.iter().product();
        let mut result_data = vec![T::min_value(); output_size];

        // Calculate strides for efficient indexing
        let input_strides = self.calculate_strides();
        let output_strides = Self::calculate_strides_for_shape(&output_shape);

        // Iterate through all elements and find maximum along the specified dimension
        for input_idx in 0..self.data.len() {
            // Convert flat index to multi-dimensional coordinates
            let coords = self.flat_to_coords(input_idx, &input_strides);

            // Convert back to flat index in output tensor
            let output_idx = Self::coords_to_flat(&coords, &output_strides);

            let current_value = self.data.as_slice().unwrap()[input_idx];
            if current_value > result_data[output_idx] {
                result_data[output_idx] = current_value;
            }
        }

        let result_array = ndarray::Array::from_shape_vec(output_shape, result_data)
            .map_err(|e| format!("Failed to create result tensor: {}", e))?;

        Ok(Self::new_with_device(
            result_array.into_dyn(),
            self.device.clone(),
        ))
    }

    // Helper method to calculate strides for a given shape
    fn calculate_strides_for_shape(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    // Helper method to calculate strides for current tensor
    fn calculate_strides(&self) -> Vec<usize> {
        Self::calculate_strides_for_shape(self.shape())
    }

    // Helper method to convert flat index to coordinates
    fn flat_to_coords(&self, mut flat_idx: usize, strides: &[usize]) -> Vec<usize> {
        let mut coords = vec![0; strides.len()];
        for i in 0..strides.len() {
            coords[i] = flat_idx / strides[i];
            flat_idx %= strides[i];
        }
        coords
    }

    // Helper method to convert coordinates to flat index
    fn coords_to_flat(coords: &[usize], strides: &[usize]) -> usize {
        coords
            .iter()
            .zip(strides.iter())
            .map(|(coord, stride)| coord * stride)
            .sum()
    }

    // -------------------------------------------------------------------
    //  Other utility common tensor operations
    // -------------------------------------------------------------------
    // Element-wise greater than or equal comparison
    pub fn greater_equal(&self, other: &Self) -> Result<CPUTensor<T>, String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Shape mismatch for greater_equal: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }

        let result_data = ndarray::Zip::from(&self.data)
            .and(&other.data)
            .map_collect(|&a, &b| {
                if a >= b {
                    <T as CPUNumber>::one()
                } else {
                    <T as CPUNumber>::zero()
                }
            });

        Ok(Self::new_with_device(result_data, self.device.clone()))
    }

    // Element-wise less than or equal comparison
    pub fn less_equal(&self, other: &Self) -> Result<CPUTensor<T>, String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Shape mismatch for less_equal: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }

        let result_data = ndarray::Zip::from(&self.data)
            .and(&other.data)
            .map_collect(|&a, &b| {
                if a <= b {
                    <T as CPUNumber>::one()
                } else {
                    <T as CPUNumber>::zero()
                }
            });

        Ok(Self::new_with_device(result_data, self.device.clone()))
    }

    // Element-wise equality comparison
    pub fn equal(&self, other: &Self) -> Result<CPUTensor<T>, String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Shape mismatch for equal: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }

        let result_data = ndarray::Zip::from(&self.data)
            .and(&other.data)
            .map_collect(|&a, &b| {
                if a == b {
                    <T as CPUNumber>::one()
                } else {
                    <T as CPUNumber>::zero()
                }
            });

        Ok(Self::new_with_device(result_data, self.device.clone()))
    }

    // Logical NOT operation (flips 0s and 1s)
    pub fn logical_not(&self) -> Result<CPUTensor<T>, String> {
        let result_data = self.data.mapv(|x| {
            if x == <T as CPUNumber>::zero() {
                <T as CPUNumber>::one()
            } else {
                <T as CPUNumber>::zero()
            }
        });

        Ok(Self::new_with_device(result_data, self.device.clone()))
    }

    // Check if values are in range [min_val, max_val]
    pub fn in_range(&self, min_val: T, max_val: T) -> Result<CPUTensor<T>, String> {
        let result_data = self.data.mapv(|x| {
            if x >= min_val && x <= max_val {
                <T as CPUNumber>::one()
            } else {
                <T as CPUNumber>::zero()
            }
        });

        Ok(Self::new_with_device(result_data, self.device.clone()))
    }

    // Expand dimensions by adding a new axis at the specified position
    pub fn expand_dims(&self, axis: usize) -> Result<CPUTensor<T>, String> {
        if axis > self.ndim() {
            return Err(format!(
                "Cannot insert axis {} for tensor with {} dimensions",
                axis,
                self.ndim()
            ));
        }

        let expanded = self.data.clone().insert_axis(ndarray::Axis(axis));
        Ok(Self::new_with_device(expanded, self.device.clone()))
    }

    // Returns 1 if the value is positive, 0 if it is zero, and -1 if it is negative.
    pub fn sign(&self) -> CPUTensor<T> {
        let result_data = self.data.mapv(|x| {
            if x > <T as CPUNumber>::zero() {
                <T as CPUNumber>::one()
            } else if x < <T as CPUNumber>::zero() {
                -<T as CPUNumber>::one()
            } else {
                <T as CPUNumber>::zero()
            }
        });

        Self::new_with_device(result_data, self.device.clone())
    }

    // Detach operation - creates a new tensor that shares data but detaches from graph
    // Need to check if this is the right way to do it.
    // In Pytorch i think the detach operation sets the requires_grad flag to false, but we don't have that concept at the tensor level.
    // We can just return a new tensor with the same data and device, but without any gradient tracking.
    pub fn detach(&self) -> CPUTensor<T> {
        CPUTensor::new_with_device(self.data.clone(), self.device.clone())
    }

    /// Returns an iterator over elements in row-major order
    pub fn iter(&self) -> ndarray::iter::Iter<'_, T, ndarray::IxDyn> {
        self.data.iter()
    }

    /// Returns a mutable iterator over elements in row-major order
    pub fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, T, ndarray::IxDyn> {
        self.data.iter_mut()
    }

    /// Returns an iterator over elements with their indices
    pub fn indexed_iter(&self) -> ndarray::iter::IndexedIter<'_, T, ndarray::IxDyn> {
        self.data.indexed_iter()
    }

    /// Returns a mutable iterator over elements with their indices
    pub fn indexed_iter_mut(&mut self) -> ndarray::iter::IndexedIterMut<'_, T, ndarray::IxDyn> {
        self.data.indexed_iter_mut()
    }

    /// Collect all elements into a Vec in row-major order.
    /// Works for both CPU and CUDA tensors.
    pub fn to_vec(&self) -> Result<Vec<T>, String> {
        Ok(self.data.iter().copied().collect())
    }

    /// Conditional selection: where condition is true, use true_vals, else false_vals
    pub fn where_condition(
        condition: &CPUTensor<T>,
        true_vals: &CPUTensor<T>,
        false_vals: &CPUTensor<T>,
    ) -> Result<CPUTensor<T>, String> {
        let condition_vec = condition.to_vec()?;
        let true_vec = true_vals.to_vec()?;
        let false_vec = false_vals.to_vec()?;

        let result_vec: Vec<T> = condition_vec
            .iter()
            .zip(true_vec.iter())
            .zip(false_vec.iter())
            .map(|((&cond, &true_val), &false_val)| {
                if cond > <T as CPUNumber>::zero() {
                    true_val
                } else {
                    false_val
                }
            })
            .collect();

        CPUTensor::from_vec(result_vec, condition.shape())
    }
}

impl<T> CPUTensor<T>
where
    T: GPUNumber,
{
    pub fn matmul(&self, other: &CPUTensor<T>) -> Result<CPUTensor<T>, String>
    where
        T: Clone + ndarray::LinalgScalar,
    {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err("Matrix multiplication requires 2D tensors".to_string());
        }

        let a_shape = self.shape();
        let b_shape = other.shape();

        if a_shape[1] != b_shape[0] {
            return Err(format!(
                "Matrix multiplication shape mismatch: ({}, {}) @ ({}, {})",
                a_shape[0], a_shape[1], b_shape[0], b_shape[1]
            ));
        }

        // Be more explicit about the types
        let a: ndarray::ArrayView2<T> = self.data.view().into_dimensionality().unwrap();
        let b: ndarray::ArrayView2<T> = other.data.view().into_dimensionality().unwrap();

        // Dot product of two 2D arrays, gives the matrix multiplication result.
        // Look at `ndarray` documentation for more details on `dot`.
        // https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.dot
        let result = a.dot(&b);

        Ok(CPUTensor::new_with_device(
            result.into_dyn(),
            self.device.clone(),
        ))
    }
}

// Implementation for operations requiring ScalarOperand
impl<T> CPUTensor<T>
where
    T: GPUNumber,
{
    // Scalar operations
    pub fn add_scalar(&self, scalar: T) -> CPUTensor<T> {
        CPUTensor::new_with_device(&self.data + scalar, self.device.clone())
    }

    pub fn mul_scalar(&self, scalar: T) -> CPUTensor<T> {
        CPUTensor::new_with_device(&self.data * scalar, self.device.clone())
    }

    pub fn div_scalar(&self, scalar: T) -> CPUTensor<T> {
        CPUTensor::new_with_device(&self.data / scalar, self.device.clone())
    }
}

// Implementation for floating-point operations
impl<T> CPUTensor<T>
where
    T: GPUFloat,
{
    pub fn sigmoid(&self) -> CPUTensor<T> {
        // Euler's sigmoid function: 1 / (1 + exp(-x))
        CPUTensor::new_with_device(
            self.data.mapv(|x| {
                let one = <T as CPUNumber>::one();
                let neg_x = -x;
                one / (one + neg_x.exp())
            }),
            self.device.clone(),
        )
    }

    /// Element-wise square root
    /// Returns a new tensor with square root values
    /// Validates that all values are non-negative
    pub fn sqrt(&self) -> Result<Self, String> {
        // Check for negative values first
        let has_negative = self.data.iter().any(|&x| x < <T as CPUNumber>::zero());
        if has_negative {
            return Err("Cannot compute square root of negative values".to_string());
        }

        let result_data: Vec<T> = self.data.iter().map(|&x| x.sqrt()).collect();

        let result_array = ndarray::Array::from_shape_vec(self.data.raw_dim(), result_data)
            .map_err(|e| format!("Failed to create result tensor: {}", e))?;

        Ok(Self::new_with_device(result_array, self.device.clone()))
    }

    // Activation functions
    pub fn relu(&self) -> CPUTensor<T> {
        // ReLU activation function: max(0, x)
        CPUTensor::new_with_device(
            self.data.mapv(|x| {
                let zero = <T as CPUNumber>::zero();
                if x > zero { x } else { zero }
            }),
            self.device.clone(),
        )
    }

    // Additional activation functions
    pub fn exp(&self) -> CPUTensor<T> {
        CPUTensor::new_with_device(self.data.mapv(|x| x.exp()), self.device.clone())
    }

    pub fn log(&self) -> CPUTensor<T> {
        CPUTensor::new_with_device(self.data.mapv(|x| x.ln()), self.device.clone())
    }
    pub fn tanh(&self) -> CPUTensor<T> {
        CPUTensor::new_with_device(
            self.data.mapv(|x| {
                let e_x = x.exp();
                let e_neg_x = (-x).exp();
                (e_x - e_neg_x) / (e_x + e_neg_x)
            }),
            self.device.clone(),
        )
    }

    pub fn powf(&self, other: &CPUTensor<T>) -> Result<CPUTensor<T>, String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }
        Ok(CPUTensor::new_with_device(
            ndarray::Zip::from(&self.data)
                .and(&other.data)
                .map_collect(|&a, &b| a.powf(b)),
            self.device.clone(),
        ))
    }

    pub fn power_scalar(&self, scalar: T) -> CPUTensor<T> {
        CPUTensor::new_with_device(self.data.mapv(|x| x.powf(scalar)), self.device.clone())
    }
}

// Implementation for reduction operations and tensor manipulations
impl<T> CPUTensor<T>
where
    T: GPUNumber + Clone + rand_distr::num_traits::Zero + rand_distr::num_traits::FromPrimitive,
{
    // Reduction operations.
    // As we do not know the shape of the tensor at compile time, we use `ndarray`'s dynamic arrays.
    // We can sum or mean over a specific axis or all elements, it is up to the user to provide the axis over which to perform the reduction operation.
    pub fn sum(&self, axis: Option<usize>) -> CPUTensor<T> {
        match axis {
            Some(ax) => {
                let result = self.data.sum_axis(Axis(ax));
                CPUTensor::new_with_device(result, self.device.clone())
            }
            None => {
                // If axis is not provided we just sum all elements
                let total_sum = self.data.sum();
                CPUTensor::new_with_device(
                    ArrayD::from_elem(IxDyn(&[]), total_sum),
                    self.device.clone(),
                )
            }
        }
    }

    // Sum over the specified axes in reverse sorted order.
    // This function allows summing over multiple axes at once, which is useful for reducing the dimensionality of the tensor.
    // Example:
    // Given tensor [2, 2, 2] with data [1,2,3,4,5,6,7,8]:
    // [[[1, 2],     <- indices [0,0,:]
    // [3, 4]],    <- indices [0,1,:]
    // [[5, 6],     <- indices [1,0,:]
    // [7, 8]]]    <- indices [1,1,:]
    // If we sum over axes [0, 2], we get a tensor with shape [2] and data [14, 22]
    // - Step 1: Sum over axis 2:
    // [1,2] → 3
    // [3,4] → 7
    // [5,6] → 11
    // [7,8] → 15
    // Result: [[3,7], [11,15]] with shape [2,2]
    // - Step 2: Sum over axis 0:
    // [3,11] → 14
    // [7,15] → 22
    // Result: [14, 22] with shape [2]
    pub fn sum_axes(&self, axes: Option<&[usize]>) -> CPUTensor<T> {
        match axes {
            Some(axes_list) => {
                // Validate axes are within bounds
                for &ax in axes_list {
                    if ax >= self.ndim() {
                        panic!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            ax,
                            self.ndim()
                        );
                    }
                }

                let mut result = self.data.clone();

                // Sort axes in descending order to avoid index shifting issues
                let mut sorted_axes = axes_list.to_vec();
                sorted_axes.sort_unstable();
                sorted_axes.reverse();

                // Remove duplicates to avoid summing the same axis twice
                sorted_axes.dedup();

                for &ax in &sorted_axes {
                    result = result.sum_axis(Axis(ax));
                }

                CPUTensor::new_with_device(result, self.device.clone())
            }
            None => self.sum(None),
        }
    }

    pub fn mean(&self, axis: Option<usize>) -> CPUTensor<T> {
        match axis {
            Some(ax) => {
                let result = self.data.mean_axis(Axis(ax)).unwrap();
                CPUTensor::new_with_device(result, self.device.clone())
            }
            None => {
                let total_mean = self.data.mean().unwrap();
                CPUTensor::new_with_device(
                    ArrayD::from_elem(IxDyn(&[]), total_mean),
                    self.device.clone(),
                )
            }
        }
    }

    // Broadcasting for gradient computation
    // Broadcasting allows us to perform operations on tensors of different shapes.
    // As can be seen on Ndarray's docs, broadcast function returns None if the shapes of the tensors cannot be broadcasted together.
    // https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.broadcast
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<CPUTensor<T>, String> {
        match self.data.broadcast(target_shape) {
            Some(broadcasted) => Ok(CPUTensor::new_with_device(
                broadcasted.to_owned(),
                self.device.clone(),
            )),
            None => Err(format!(
                "Cannot broadcast {:?} to {:?}",
                self.shape(),
                target_shape
            )),
        }
    }

    // Similar to tf.expand_dims, this function adds a new dimension at the specified axis.
    pub fn unsqueeze(&self, axis: usize) -> CPUTensor<T> {
        let expanded = self.data.clone().insert_axis(Axis(axis));
        CPUTensor::new_with_device(expanded, self.device.clone())
    }

    // Basically the opposite of unsqueeze, this function removes a dimension of size 1 from the tensor.
    // We need to check the size of the axis before removing it, as it is not possible to remove an axis with size greater than 1.
    // Imagine a tensor: [[[1, 3, 1, 5],[1,2,3,4]],[[1, 3, 1, 5],[1,2,3,4]],] if we try to squeeze axis 1, we would need to remove the two elements on that axis,
    // which is not the purpose of the squeeze operation.
    pub fn squeeze(&self, axis: Option<usize>) -> Result<CPUTensor<T>, String> {
        match axis {
            Some(ax) => {
                if self.shape()[ax] != 1 {
                    return Err(format!(
                        "Cannot squeeze axis {} with size {}",
                        ax,
                        self.shape()[ax]
                    ));
                }
                let squeezed = self.data.clone().remove_axis(Axis(ax));
                Ok(CPUTensor::new_with_device(squeezed, self.device.clone()))
            }
            None => {
                // Remove all dimensions of size 1
                let mut result = self.data.clone();
                let mut axis_to_remove = Vec::new();

                for (i, &size) in self.shape().iter().enumerate() {
                    if size == 1 {
                        axis_to_remove.push(i);
                    }
                }

                // Remove axes in reverse order to maintain indices
                for &ax in axis_to_remove.iter().rev() {
                    result = result.remove_axis(Axis(ax));
                }

                Ok(CPUTensor::new_with_device(result, self.device.clone()))
            }
        }
    }

    // Reshape operation
    pub fn reshape(&self, new_shape: &[usize]) -> Result<CPUTensor<T>, String> {
        let total_elements: usize = self.shape().iter().product();
        let new_total_elements: usize = new_shape.iter().product();

        if total_elements != new_total_elements {
            return Err(format!(
                "Cannot reshape tensor with {} elements to shape with {} elements",
                total_elements, new_total_elements
            ));
        }

        match self.data.clone().into_shape_with_order(IxDyn(new_shape)) {
            Ok(reshaped) => Ok(CPUTensor::new_with_device(reshaped, self.device.clone())),
            Err(e) => Err(format!("Failed to reshape tensor: {}", e)),
        }
    }

    // Transpose operation is one of the other things that is quite tricky to implement from scratch.
    // It requires permuting the axes of the CPUTensor, which can be done using ndarray's `permuted_axes` method.
    pub fn transpose(&self, axes: Option<&[usize]>) -> Result<CPUTensor<T>, String> {
        match axes {
            Some(axes_order) => {
                if axes_order.len() != self.ndim() {
                    return Err(format!(
                        "Axes length {} doesn't match CPUTensor dimensions {}",
                        axes_order.len(),
                        self.ndim()
                    ));
                }
                // Validate that axes_order is a valid permutation
                let mut sorted_axes = axes_order.to_vec();
                sorted_axes.sort_unstable(); // Sort unstable is more performant than actual sort, 
                // the drawback is it does not preserve the
                // order of equal elements, which is fine for our use case.
                // Basically if you had ("a", 1) and ("a", 2) in the vector,
                // they could be swapped in the sorted vector.
                // We expect the sorted axes to be 0..ndim()
                let expected: Vec<usize> = (0..self.ndim()).collect();
                if sorted_axes != expected {
                    return Err(format!(
                        "Invalid axes permutation: {:?}. Must be a permutation of 0..{}",
                        axes_order,
                        self.ndim()
                    ));
                }
                // Create the transposed array by permuting axes
                let transposed = self.data.clone().permuted_axes(axes_order);
                Ok(CPUTensor::new_with_device(transposed, self.device.clone()))
            }
            // If no axes are provided, we perform the default transpose operation,
            // which as we know is to reverse the order of the axes.
            None => {
                // Default transpose: reverse all axes
                match self.ndim() {
                    0 | 1 => {
                        // 0D and 1D tensors are unchanged by transpose
                        Ok(self.clone())
                    }
                    2 => {
                        // For 2D arrays, swap the two axes
                        let transposed = self.data.clone().reversed_axes();
                        Ok(CPUTensor::new_with_device(transposed, self.device.clone()))
                    }
                    _ => {
                        // Till now it has been easy. Now we need to handle higher dimensional arrays.
                        // Everybody gangsta until they have to transpose a 3D or higher tensor.
                        // For higher dimensional arrays, reverse the order of all axes
                        let axes_order: Vec<usize> = (0..self.ndim()).rev().collect();
                        // Convert Vec<usize> to &[usize] for permuted_axes.
                        // This is required because the dimension of the Vec is not known at compile time.
                        // We can use `as_slice()` to convert it to a slice.
                        let transposed = self.data.clone().permuted_axes(axes_order.as_slice());
                        Ok(CPUTensor::new_with_device(transposed, self.device.clone()))
                    }
                }
            }
        }
    }
}

// Implementation for tensor creation with Zero trait
impl<T> CPUTensor<T>
where
    T: GPUNumber + Clone + rand_distr::num_traits::Zero,
{
    // Initialization functions for creating tensors with specific shapes.
    // They all have a `_with_device` variant that allows specifying the device.
    // Zeroes
    pub fn zeros(shape: &[usize]) -> Self {
        let device = default_device();
        Self {
            data: device.zeros(shape),
            device,
        }
    }

    pub fn zeros_with_device(shape: &[usize], device: Device) -> Self {
        Self {
            data: device.zeros(shape),
            device,
        }
    }
}

impl<T> CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    /// Check if tensor is on CUDA
    pub fn is_cuda(&self) -> bool {
        self.device.is_cuda()
    }
}

// Implementation for tensor creation with One trait
impl<T> CPUTensor<T>
where
    T: GPUNumber,
{
    // Ones
    pub fn ones(shape: &[usize]) -> Self {
        let device = default_device();
        Self {
            data: device.ones(shape),
            device,
        }
    }

    pub fn ones_with_device(shape: &[usize], device: Device) -> Self {
        Self {
            data: device.ones(shape),
            device,
        }
    }
}

// Implement equality for testing, and because will be useful in the future.
impl<T> PartialEq for CPUTensor<T>
where
    T: GPUNumber + Clone + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.device == other.device
    }
}

impl<T> Eq for CPUTensor<T> where T: GPUNumber + Clone + PartialEq {}

pub struct CPUTensorIterator<T> {
    data: ndarray::ArrayD<T>,
    index: usize,
}

impl<T> Iterator for CPUTensorIterator<T>
where
    T: Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.data.len() {
            let item = self.data.as_slice().unwrap()[self.index];
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.data.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for CPUTensorIterator<T>
where
    T: Copy,
{
    fn len(&self) -> usize {
        self.data.len().saturating_sub(self.index)
    }
}

// Implementation for owned Tensor (consumes the tensor)
impl<T> IntoIterator for CPUTensor<T>
where
    T: GPUNumber + Clone + Copy,
{
    type Item = T;
    type IntoIter = CPUTensorIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        CPUTensorIterator {
            data: self.data,
            index: 0,
        }
    }
}

// Implementation for borrowed Tensor (&Tensor)
impl<'a, T> IntoIterator for &'a CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    type Item = &'a T;
    type IntoIter = ndarray::iter::Iter<'a, T, ndarray::IxDyn>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

// Implementation for mutable borrowed Tensor (&mut Tensor)
impl<'a, T> IntoIterator for &'a mut CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    type Item = &'a mut T;
    type IntoIter = ndarray::iter::IterMut<'a, T, ndarray::IxDyn>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

// Implementation for single usize index (flat indexing for any dimensional tensor)
// This attempts to mimic the behavior of NumPy's flat indexing,
// therefore you could access elements in a multi-dimensional tensor as it was a flat array.
impl<T> Index<usize> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        // Convert flat index to multi-dimensional coordinates
        let flat_data = self
            .data
            .as_slice()
            .expect("Tensor data should be contiguous");
        &flat_data[index]
    }
}

impl<T> IndexMut<usize> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // Convert flat index to multi-dimensional coordinates
        let flat_data = self
            .data
            .as_slice_mut()
            .expect("Tensor data should be contiguous");
        &mut flat_data[index]
    }
}

// Implementation for slice of usize (multi-dimensional indexing)
impl<T> Index<&[usize]> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize]> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

// Implementation for Vec<usize> (convenient alternative to slice)
impl<T> Index<Vec<usize>> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    type Output = T;

    fn index(&self, indices: Vec<usize>) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<Vec<usize>> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    fn index_mut(&mut self, indices: Vec<usize>) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

// Implementation for arrays of different sizes (up to 6D for common use cases)
impl<T> Index<[usize; 1]> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    type Output = T;

    fn index(&self, indices: [usize; 1]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<[usize; 1]> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    fn index_mut(&mut self, indices: [usize; 1]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

impl<T> Index<[usize; 2]> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    type Output = T;

    fn index(&self, indices: [usize; 2]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<[usize; 2]> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    fn index_mut(&mut self, indices: [usize; 2]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

impl<T> Index<[usize; 3]> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    type Output = T;

    fn index(&self, indices: [usize; 3]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<[usize; 3]> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    fn index_mut(&mut self, indices: [usize; 3]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

impl<T> Index<[usize; 4]> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    type Output = T;

    fn index(&self, indices: [usize; 4]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<[usize; 4]> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    fn index_mut(&mut self, indices: [usize; 4]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

// Implementation for tuples (more ergonomic for 2D and 3D)
impl<T> Index<(usize, usize)> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.data[[i, j]]
    }
}

impl<T> IndexMut<(usize, usize)> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.data[[i, j]]
    }
}

impl<T> Index<(usize, usize, usize)> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    type Output = T;

    fn index(&self, (i, j, k): (usize, usize, usize)) -> &Self::Output {
        &self.data[[i, j, k]]
    }
}

impl<T> IndexMut<(usize, usize, usize)> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    fn index_mut(&mut self, (i, j, k): (usize, usize, usize)) -> &mut Self::Output {
        &mut self.data[[i, j, k]]
    }
}

impl<T> Index<(usize, usize, usize, usize)> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    type Output = T;

    fn index(&self, (i, j, k, l): (usize, usize, usize, usize)) -> &Self::Output {
        &self.data[[i, j, k, l]]
    }
}

impl<T> IndexMut<(usize, usize, usize, usize)> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    fn index_mut(&mut self, (i, j, k, l): (usize, usize, usize, usize)) -> &mut Self::Output {
        &mut self.data[[i, j, k, l]]
    }
}

// Implementation for references to arrays of different sizes
impl<T> Index<&[usize; 1]> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize; 1]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize; 1]> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    fn index_mut(&mut self, indices: &[usize; 1]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

impl<T> Index<&[usize; 2]> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize; 2]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize; 2]> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    fn index_mut(&mut self, indices: &[usize; 2]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

impl<T> Index<&[usize; 3]> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize; 3]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize; 3]> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    fn index_mut(&mut self, indices: &[usize; 3]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

impl<T> Index<&[usize; 4]> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize; 4]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize; 4]> for CPUTensor<T>
where
    T: GPUNumber + Clone,
{
    fn index_mut(&mut self, indices: &[usize; 4]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}
