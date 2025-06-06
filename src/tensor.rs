use crate::backend::{Device, default_device};
use ndarray::{Array, ArrayD, Axis, IxDyn};
use std::ops::{Index, IndexMut};
// Tensor wrapper to handle dynamic arrays more elegantly
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: ArrayD<f64>, // As I documented in the device module, this will be changed toa generic type <T>
    // This way I will be able to use different data types in the future.
    // For now, we will keep it as f64 for simplicity.
    pub device: Device,
}

impl Tensor {
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
    pub fn new(data: ArrayD<f64>) -> Self {
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
    pub fn new_with_device(data: ArrayD<f64>, device: Device) -> Self {
        Self { data, device }
    }

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

    // Random numbers
    pub fn randn(shape: &[usize]) -> Self {
        let device = default_device();
        Self {
            data: device.randn(shape),
            device,
        }
    }

    pub fn randn_with_device(shape: &[usize], device: Device) -> Self {
        // Generates a tensor with random numbers from a normal distribution.
        Self {
            data: device.randn(shape),
            device,
        }
    }

    // Creates a tensor from a Rust vector. Again we are bound to ndarray backend here, but it is okay for now.
    // This function takes a vector of f64 and a shape, and returns a tensor with the given shape.
    pub fn from_vec(data: Vec<f64>, shape: &[usize]) -> Result<Self, String> {
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

    pub fn data(&self) -> &ArrayD<f64> {
        &self.data
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn into_data(self) -> ArrayD<f64> {
        self.data
    }

    // Element-wise operations.
    // These are operations that are applied to each element of the tensor.
    // They are easily parallelizable and can be implemented using ndarray's mapv method.
    // The mapv method applies a function to each element of the array and returns a new array with the results.
    // The mapv operation does not actually parallelize by itself, but it is much more efficient th
    // - add
    // - multiply
    pub fn add(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }
        Ok(Tensor::new_with_device(
            &self.data + &other.data,
            self.device.clone(),
        ))
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }
        Ok(Tensor::new_with_device(
            &self.data * &other.data,
            self.device.clone(),
        ))
    }

    // Matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, String> {
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

        // Convert a and b to 2D views.
        // In `ndarray` a view is a read-only reference to the data.
        // We use `into_dimensionality` to ensure the data is treated as 2D.
        let a = self
            .data
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let b = other
            .data
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();

        // Dot product of two 2D arrays, gives the matrix multiplication result.
        // Look at `ndarray` documentation for more details on `dot`.
        // https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.dot
        let result = a.dot(&b);
        Ok(Tensor::new_with_device(
            result.into_dyn(),
            self.device.clone(),
        ))
    }

    // Activation functions
    pub fn relu(&self) -> Tensor {
        // ReLU activation function: max(0, x)
        Tensor::new_with_device(self.data.mapv(|x| x.max(0.0)), self.device.clone())
    }

    pub fn sigmoid(&self) -> Tensor {
        // Euler's sigmoid function: 1 / (1 + exp(-x))
        // equivalent to sin(x)
        Tensor::new_with_device(
            self.data.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            self.device.clone(),
        )
    }

    // Additional activation functions
    pub fn exp(&self) -> Tensor {
        Tensor::new_with_device(self.data.mapv(|x| x.exp()), self.device.clone())
    }

    pub fn log(&self) -> Tensor {
        Tensor::new_with_device(self.data.mapv(|x| x.ln()), self.device.clone())
    }

    pub fn negate(&self) -> Tensor {
        Tensor::new_with_device(self.data.mapv(|x| -x), self.device.clone())
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }
        Ok(Tensor::new_with_device(
            &self.data / &other.data,
            self.device.clone(),
        ))
    }

    pub fn pow(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }
        Ok(Tensor::new_with_device(
            ndarray::Zip::from(&self.data)
                .and(&other.data)
                .map_collect(|&a, &b| a.powf(b)),
            self.device.clone(),
        ))
    }

    // Scalar operations
    pub fn add_scalar(&self, scalar: f64) -> Tensor {
        Tensor::new_with_device(&self.data + scalar, self.device.clone())
    }

    pub fn mul_scalar(&self, scalar: f64) -> Tensor {
        Tensor::new_with_device(&self.data * scalar, self.device.clone())
    }

    pub fn div_scalar(&self, scalar: f64) -> Tensor {
        Tensor::new_with_device(&self.data / scalar, self.device.clone())
    }

    pub fn power_scalar(&self, scalar: f64) -> Tensor {
        Tensor::new_with_device(self.data.mapv(|x| x.powf(scalar)), self.device.clone())
    }

    // Reduction operations.
    // As we do not know the shape of the tensor at compile time, we use `ndarray`'s dynamic arrays.
    // We can sum or mean over a specific axis or all elements, it is up to the user to provide the axis over which to perform the reduction operation.
    pub fn sum(&self, axis: Option<usize>) -> Tensor {
        match axis {
            Some(ax) => {
                let result = self.data.sum_axis(Axis(ax));
                Tensor::new_with_device(result, self.device.clone())
            }
            None => {
                // If axis is not provided we just sum all elements
                let total_sum = self.data.sum();
                Tensor::new_with_device(
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
    pub fn sum_axes(&self, axes: Option<&[usize]>) -> Tensor {
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

                Tensor::new_with_device(result, self.device.clone())
            }
            None => self.sum(None),
        }
    }

    pub fn mean(&self, axis: Option<usize>) -> Tensor {
        match axis {
            Some(ax) => {
                let result = self.data.mean_axis(Axis(ax)).unwrap();
                Tensor::new_with_device(result, self.device.clone())
            }
            None => {
                let total_mean = self.data.mean().unwrap();
                Tensor::new_with_device(
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
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Tensor, String> {
        match self.data.broadcast(target_shape) {
            Some(broadcasted) => Ok(Tensor::new_with_device(
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
    pub fn unsqueeze(&self, axis: usize) -> Tensor {
        let expanded = self.data.clone().insert_axis(Axis(axis));
        Tensor::new_with_device(expanded, self.device.clone())
    }

    // Basically the opposite of unsqueeze, this function removes a dimension of size 1 from the tensor.
    // We need to check the size of the axis before removing it, as it is not possible to remove an axis with size greater than 1.
    // Imagine a tensor: [[[1, 3, 1, 5],[1,2,3,4]],[[1, 3, 1, 5],[1,2,3,4]],] if we try to squeeze axis 1, we would need to remove the two elements on that axis,
    // which is not the purpose of the squeeze operation.
    pub fn squeeze(&self, axis: Option<usize>) -> Result<Tensor, String> {
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
                Ok(Tensor::new_with_device(squeezed, self.device.clone()))
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

                Ok(Tensor::new_with_device(result, self.device.clone()))
            }
        }
    }

    // Reshape operation
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Tensor, String> {
        let total_elements: usize = self.shape().iter().product();
        let new_total_elements: usize = new_shape.iter().product();

        if total_elements != new_total_elements {
            return Err(format!(
                "Cannot reshape tensor with {} elements to shape with {} elements",
                total_elements, new_total_elements
            ));
        }

        match self.data.clone().into_shape_with_order(IxDyn(new_shape)) {
            Ok(reshaped) => Ok(Tensor::new_with_device(reshaped, self.device.clone())),
            Err(e) => Err(format!("Failed to reshape tensor: {}", e)),
        }
    }

    // Transpose operation is one of the other things that is quite tricky to implement from scratch.
    // It requires permuting the axes of the tensor, which can be done using ndarray's `permuted_axes` method.
    pub fn transpose(&self, axes: Option<&[usize]>) -> Result<Tensor, String> {
        match axes {
            Some(axes_order) => {
                if axes_order.len() != self.ndim() {
                    return Err(format!(
                        "Axes length {} doesn't match tensor dimensions {}",
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
                Ok(Tensor::new_with_device(transposed, self.device.clone()))
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
                        Ok(Tensor::new_with_device(transposed, self.device.clone()))
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
                        Ok(Tensor::new_with_device(transposed, self.device.clone()))
                    }
                }
            }
        }
    }

    // Detach operation - creates a new tensor that shares data but detaches from graph
    // Need to check if this is the right way to do it.
    // In Pytorch i think the detach operation sets the requires_grad flag to false, but we don't have that concept at the tensor level.
    // We can just return a new tensor with the same data and device, but without any gradient tracking.
    pub fn detach(&self) -> Tensor {
        Tensor::new_with_device(self.data.clone(), self.device.clone())
    }

    /// Returns an iterator over elements in row-major order
    pub fn iter(&self) -> ndarray::iter::Iter<'_, f64, ndarray::IxDyn> {
        self.data.iter()
    }

    /// Returns a mutable iterator over elements in row-major order
    pub fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, f64, ndarray::IxDyn> {
        self.data.iter_mut()
    }

    /// Returns an iterator over elements with their indices
    pub fn indexed_iter(&self) -> ndarray::iter::IndexedIter<'_, f64, ndarray::IxDyn> {
        self.data.indexed_iter()
    }

    /// Returns a mutable iterator over elements with their indices
    pub fn indexed_iter_mut(&mut self) -> ndarray::iter::IndexedIterMut<'_, f64, ndarray::IxDyn> {
        self.data.indexed_iter_mut()
    }

    /// Collect all elements into a Vec in row-major order
    pub fn to_vec(&self) -> Vec<f64> {
        self.data.iter().copied().collect()
    }
}

// Implement equality for testing, and because will be useful in the future.
impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.device == other.device
    }
}

impl Eq for Tensor {}

// Iterator struct that holds the state of iteration
pub struct TensorIterator {
    iter: ndarray::iter::Iter<'static, f64, ndarray::IxDyn>,
}

impl Iterator for TensorIterator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().copied()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl ExactSizeIterator for TensorIterator {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

// Implementation for owned Tensor (consumes the tensor)
impl IntoIterator for Tensor {
    type Item = f64;
    type IntoIter = TensorIterator;

    fn into_iter(self) -> Self::IntoIter {
        // We need to leak the data to get a 'static reference for the iterator
        // This is safe because we're consuming the tensor
        let leaked_data = Box::leak(Box::new(self.data));
        TensorIterator {
            iter: leaked_data.iter(),
        }
    }
}

// Implementation for borrowed Tensor (&Tensor)
impl<'a> IntoIterator for &'a Tensor {
    type Item = &'a f64;
    type IntoIter = ndarray::iter::Iter<'a, f64, ndarray::IxDyn>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

// Implementation for mutable borrowed Tensor (&mut Tensor)
impl<'a> IntoIterator for &'a mut Tensor {
    type Item = &'a mut f64;
    type IntoIter = ndarray::iter::IterMut<'a, f64, ndarray::IxDyn>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

// Implementation for single usize index (flat indexing for any dimensional tensor)
// This attempts to mimic the behavior of NumPy's flat indexing,
// therefore you could access elements in a multi-dimensional tensor as it was a flat array.
impl Index<usize> for Tensor {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        // Convert flat index to multi-dimensional coordinates
        let flat_data = self
            .data
            .as_slice()
            .expect("Tensor data should be contiguous");
        &flat_data[index]
    }
}

impl IndexMut<usize> for Tensor {
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
impl Index<&[usize]> for Tensor {
    type Output = f64;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl IndexMut<&[usize]> for Tensor {
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

// Implementation for Vec<usize> (convenient alternative to slice)
impl Index<Vec<usize>> for Tensor {
    type Output = f64;

    fn index(&self, indices: Vec<usize>) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl IndexMut<Vec<usize>> for Tensor {
    fn index_mut(&mut self, indices: Vec<usize>) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

// Implementation for arrays of different sizes (up to 6D for common use cases)
impl Index<[usize; 1]> for Tensor {
    type Output = f64;

    fn index(&self, indices: [usize; 1]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl IndexMut<[usize; 1]> for Tensor {
    fn index_mut(&mut self, indices: [usize; 1]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

impl Index<[usize; 2]> for Tensor {
    type Output = f64;

    fn index(&self, indices: [usize; 2]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl IndexMut<[usize; 2]> for Tensor {
    fn index_mut(&mut self, indices: [usize; 2]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

impl Index<[usize; 3]> for Tensor {
    type Output = f64;

    fn index(&self, indices: [usize; 3]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl IndexMut<[usize; 3]> for Tensor {
    fn index_mut(&mut self, indices: [usize; 3]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

impl Index<[usize; 4]> for Tensor {
    type Output = f64;

    fn index(&self, indices: [usize; 4]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl IndexMut<[usize; 4]> for Tensor {
    fn index_mut(&mut self, indices: [usize; 4]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

// Implementation for tuples (more ergonomic for 2D and 3D)
impl Index<(usize, usize)> for Tensor {
    type Output = f64;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.data[[i, j]]
    }
}

impl IndexMut<(usize, usize)> for Tensor {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.data[[i, j]]
    }
}

impl Index<(usize, usize, usize)> for Tensor {
    type Output = f64;

    fn index(&self, (i, j, k): (usize, usize, usize)) -> &Self::Output {
        &self.data[[i, j, k]]
    }
}

impl IndexMut<(usize, usize, usize)> for Tensor {
    fn index_mut(&mut self, (i, j, k): (usize, usize, usize)) -> &mut Self::Output {
        &mut self.data[[i, j, k]]
    }
}

impl Index<(usize, usize, usize, usize)> for Tensor {
    type Output = f64;

    fn index(&self, (i, j, k, l): (usize, usize, usize, usize)) -> &Self::Output {
        &self.data[[i, j, k, l]]
    }
}

impl IndexMut<(usize, usize, usize, usize)> for Tensor {
    fn index_mut(&mut self, (i, j, k, l): (usize, usize, usize, usize)) -> &mut Self::Output {
        &mut self.data[[i, j, k, l]]
    }
}

// Implementation for references to arrays of different sizes
impl Index<&[usize; 1]> for Tensor {
    type Output = f64;

    fn index(&self, indices: &[usize; 1]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl IndexMut<&[usize; 1]> for Tensor {
    fn index_mut(&mut self, indices: &[usize; 1]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

impl Index<&[usize; 2]> for Tensor {
    type Output = f64;

    fn index(&self, indices: &[usize; 2]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl IndexMut<&[usize; 2]> for Tensor {
    fn index_mut(&mut self, indices: &[usize; 2]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

impl Index<&[usize; 3]> for Tensor {
    type Output = f64;

    fn index(&self, indices: &[usize; 3]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl IndexMut<&[usize; 3]> for Tensor {
    fn index_mut(&mut self, indices: &[usize; 3]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

impl Index<&[usize; 4]> for Tensor {
    type Output = f64;

    fn index(&self, indices: &[usize; 4]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl IndexMut<&[usize; 4]> for Tensor {
    fn index_mut(&mut self, indices: &[usize; 4]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Device, default_device};
    use ndarray::{Array, ArrayD, IxDyn};
    use std::f64::EPSILON;

    // Helper function to compare tensors with floating point tolerance
    fn tensors_approx_equal(a: &Tensor, b: &Tensor, tolerance: f64) -> bool {
        if a.shape() != b.shape() {
            return false;
        }

        for (val_a, val_b) in a.data().iter().zip(b.data().iter()) {
            if (val_a - val_b).abs() > tolerance {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_tensor_creation() {
        // Test basic tensor creation
        let data =
            Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tensor = Tensor::new(data);

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.len(), 6);
        assert_eq!(tensor.size(), 6);
    }

    #[test]
    fn test_tensor_with_device() {
        let data = Array::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let device = default_device();
        let tensor = Tensor::new_with_device(data, device.clone());

        assert_eq!(tensor.device(), &device);
        assert_eq!(tensor.shape(), &[2, 2]);
    }

    #[test]
    fn test_zeros() {
        let tensor = Tensor::zeros(&[3, 2]);
        assert_eq!(tensor.shape(), &[3, 2]);

        for &val in tensor.data().iter() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_zeros_with_device() {
        let device = default_device();
        let tensor = Tensor::zeros_with_device(&[2, 3], device.clone());
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.device(), &device);

        for &val in tensor.data().iter() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_ones() {
        let tensor = Tensor::ones(&[2, 2]);
        assert_eq!(tensor.shape(), &[2, 2]);

        for &val in tensor.data().iter() {
            assert_eq!(val, 1.0);
        }
    }

    #[test]
    fn test_ones_with_device() {
        let device = default_device();
        let tensor = Tensor::ones_with_device(&[3, 1], device.clone());
        assert_eq!(tensor.shape(), &[3, 1]);
        assert_eq!(tensor.device(), &device);

        for &val in tensor.data().iter() {
            assert_eq!(val, 1.0);
        }
    }

    #[test]
    fn test_randn() {
        let tensor = Tensor::randn(&[100, 100]);
        assert_eq!(tensor.shape(), &[100, 100]);

        // Check that values are not all the same (very unlikely with random numbers)
        let values: Vec<f64> = tensor.data().iter().cloned().collect();
        let first_val = values[0];
        let all_same = values.iter().all(|&x| (x - first_val).abs() < EPSILON);
        assert!(
            !all_same,
            "Random tensor should not have all identical values"
        );
    }

    #[test]
    fn test_from_vec_success() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data.clone(), &[2, 3]).unwrap();

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.to_vec(), data);
    }

    #[test]
    fn test_from_vec_failure() {
        let data = vec![1.0, 2.0, 3.0];
        let result = Tensor::from_vec(data, &[2, 3]); // Should need 6 elements

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("Data length 3 doesn't match shape")
        );
    }

    #[test]
    fn test_element_wise_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let result = a.add(&b).unwrap();
        let expected = vec![6.0, 8.0, 10.0, 12.0];

        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_element_wise_add_shape_mismatch() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();

        let result = a.add(&b);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Shape mismatch"));
    }

    #[test]
    fn test_element_wise_multiply() {
        let a = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let result = a.mul(&b).unwrap();
        let expected = vec![2.0, 6.0, 12.0, 20.0];

        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_matrix_multiplication() {
        // 2x3 * 3x2 = 2x2
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();

        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        // Expected: [[58, 64], [139, 154]]
        let expected = vec![58.0, 64.0, 139.0, 154.0];
        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_matrix_multiplication_shape_error() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();

        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("Matrix multiplication shape mismatch")
        );
    }

    #[test]
    fn test_matrix_multiplication_not_2d() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("Matrix multiplication requires 2D tensors")
        );
    }

    #[test]
    fn test_relu_activation() {
        let tensor = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let result = tensor.relu();
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];

        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_sigmoid_activation() {
        let tensor = Tensor::from_vec(vec![0.0], &[1]).unwrap();
        let result = tensor.sigmoid();

        // sigmoid(0) = 0.5
        assert!((result.to_vec()[0] - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_exp() {
        let tensor = Tensor::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();
        let result = tensor.exp();
        let expected = vec![
            1.0,
            std::f64::consts::E,
            std::f64::consts::E * std::f64::consts::E,
        ];

        for (actual, expected) in result.to_vec().iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_log() {
        let tensor = Tensor::from_vec(vec![1.0, std::f64::consts::E, 10.0], &[3]).unwrap();
        let result = tensor.log();
        let expected = vec![0.0, 1.0, 10.0_f64.ln()];

        for (actual, expected) in result.to_vec().iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_negate() {
        let tensor = Tensor::from_vec(vec![1.0, -2.0, 3.0, -4.0], &[4]).unwrap();
        let result = tensor.negate();
        let expected = vec![-1.0, 2.0, -3.0, 4.0];

        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_division() {
        let a = Tensor::from_vec(vec![6.0, 8.0, 10.0, 12.0], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![2.0, 4.0, 5.0, 3.0], &[2, 2]).unwrap();

        let result = a.div(&b).unwrap();
        let expected = vec![3.0, 2.0, 2.0, 4.0];

        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_power() {
        let base = Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3]).unwrap();
        let exp = Tensor::from_vec(vec![2.0, 3.0, 2.0], &[3]).unwrap();

        let result = base.pow(&exp).unwrap();
        let expected = vec![4.0, 27.0, 16.0];

        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_scalar_operations() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        // Add scalar
        let add_result = tensor.add_scalar(5.0);
        assert_eq!(add_result.to_vec(), vec![6.0, 7.0, 8.0, 9.0]);

        // Multiply scalar
        let mul_result = tensor.mul_scalar(2.0);
        assert_eq!(mul_result.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);

        // Divide scalar
        let div_result = tensor.div_scalar(2.0);
        assert_eq!(div_result.to_vec(), vec![0.5, 1.0, 1.5, 2.0]);

        // Power scalar
        let pow_result = tensor.power_scalar(2.0);
        assert_eq!(pow_result.to_vec(), vec![1.0, 4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_sum_operations() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Sum all elements
        let total_sum = tensor.sum(None);
        assert_eq!(total_sum.to_vec(), vec![21.0]);

        // Sum along axis 0
        let sum_axis_0 = tensor.sum(Some(0));
        assert_eq!(sum_axis_0.to_vec(), vec![5.0, 7.0, 9.0]);

        // Sum along axis 1
        let sum_axis_1 = tensor.sum(Some(1));
        assert_eq!(sum_axis_1.to_vec(), vec![6.0, 15.0]);
    }

    #[test]
    fn test_sum_multiple_axes() {
        let tensor =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();

        let result = tensor.sum_axes(Some(&[0, 2]));
        println!("Tensor: {:?}", tensor);
        println!("Result: {:?}", result);
        // Should sum over axes 0 and 2, leaving axis 1
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.to_vec(), vec![14.0, 22.0]);
    }

    #[test]
    fn test_mean_operations() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Mean of all elements
        let total_mean = tensor.mean(None);
        assert_eq!(total_mean.to_vec(), vec![3.5]);

        // Mean along axis 0
        let mean_axis_0 = tensor.mean(Some(0));
        assert_eq!(mean_axis_0.to_vec(), vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_reshape() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let reshaped = tensor.reshape(&[3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reshape_invalid() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let result = tensor.reshape(&[3, 2]); // 4 elements can't fit in 6 slots
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("Cannot reshape tensor with 4 elements")
        );
    }

    #[test]
    fn test_unsqueeze() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        let unsqueezed = tensor.unsqueeze(0);
        assert_eq!(unsqueezed.shape(), &[1, 3]);

        let unsqueezed2 = tensor.unsqueeze(1);
        assert_eq!(unsqueezed2.shape(), &[3, 1]);
    }

    #[test]
    fn test_squeeze() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3, 1]).unwrap();

        // Squeeze specific axis
        let squeezed = tensor.squeeze(Some(0)).unwrap();
        assert_eq!(squeezed.shape(), &[3, 1]);

        // Squeeze all singleton dimensions
        let squeezed_all = tensor.squeeze(None).unwrap();
        assert_eq!(squeezed_all.shape(), &[3]);
    }

    #[test]
    fn test_squeeze_invalid() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let result = tensor.squeeze(Some(0)); // Cannot squeeze axis with size > 1
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("Cannot squeeze axis 0 with size 2")
        );
    }

    #[test]
    fn test_transpose_2d() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let transposed = tensor.transpose(None).unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_with_axes() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let transposed = tensor.transpose(Some(&[1, 0])).unwrap();
        assert_eq!(transposed.shape(), &[2, 2]);
        assert_eq!(transposed.to_vec(), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_transpose_invalid_axes() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let result = tensor.transpose(Some(&[0, 2])); // Invalid axes for 2D tensor
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid axes permutation"));
    }

    #[test]
    fn test_broadcast_to() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();

        let broadcasted = tensor.broadcast_to(&[3, 2]).unwrap();
        assert_eq!(broadcasted.shape(), &[3, 2]);
        assert_eq!(broadcasted.to_vec(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_broadcast_invalid() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        let result = tensor.broadcast_to(&[2]); // Cannot broadcast [3] to [2]
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Cannot broadcast"));
    }

    #[test]
    fn test_detach() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let detached = tensor.detach();

        assert_eq!(tensor.to_vec(), detached.to_vec());
        assert_eq!(tensor.shape(), detached.shape());
        assert_eq!(tensor.device(), detached.device());
    }

    #[test]
    fn test_indexing() {
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Test single index access

        assert_eq!(tensor[0], 1.0);

        assert_eq!(tensor[5], 6.0);

        // Test 2D tuple indexing
        assert_eq!(tensor[(0, 0)], 1.0);
        assert_eq!(tensor[(1, 2)], 6.0);

        // Test slice indexing
        assert_eq!(tensor[&[0, 1]], 2.0);
        assert_eq!(tensor[&[1, 2]], 6.0);

        // Test mutable indexing
        tensor[0] = 10.0;
        assert_eq!(tensor[0], 10.0);

        tensor[(1, 1)] = 20.0;
        assert_eq!(tensor[(1, 1)], 20.0);
    }

    #[test]
    fn test_iterators() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        // Test borrowed iterator
        let values: Vec<&f64> = (&tensor).into_iter().collect();
        assert_eq!(values.len(), 4);
        assert_eq!(*values[0], 1.0);

        // Test owned iterator
        let owned_values: Vec<f64> = tensor.clone().into_iter().collect();
        assert_eq!(owned_values, vec![1.0, 2.0, 3.0, 4.0]);

        // Test to_vec method
        assert_eq!(tensor.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_mutable_iterator() {
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        // Modify values using mutable iterator
        for val in &mut tensor {
            *val *= 2.0;
        }

        assert_eq!(tensor.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_tensor_equality() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let c = Tensor::from_vec(vec![1.0, 2.0, 4.0], &[3]).unwrap();

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_edge_cases() {
        // Test empty tensor
        let empty = Tensor::zeros(&[0]);
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.shape(), &[0]);

        // Test scalar tensor (0D)
        let scalar_data = Array::from_shape_vec(IxDyn(&[]), vec![42.0]).unwrap();
        let scalar = Tensor::new(scalar_data);
        assert_eq!(scalar.ndim(), 0);
        assert_eq!(scalar.len(), 1);

        // Test 1D tensor transpose (should be unchanged)
        let tensor_1d = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let transposed_1d = tensor_1d.transpose(None).unwrap();
        assert_eq!(tensor_1d.shape(), transposed_1d.shape());
        assert_eq!(tensor_1d.to_vec(), transposed_1d.to_vec());
    }

    #[test]
    fn test_large_tensor_operations() {
        let size = 1000;
        let a = Tensor::ones(&[size]);
        let b = Tensor::ones(&[size]);

        let result = a.add(&b).unwrap();

        // All values should be 2.0
        for &val in result.data().iter() {
            assert_eq!(val, 2.0);
        }
    }
}
