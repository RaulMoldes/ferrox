#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use ndarray::{ArrayD, Array, IxDyn, Axis};
use crate::backend::{Device, default_device};

// Tensor wrapper to handle dynamic arrays more elegantly
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: ArrayD<f64>,
    pub device: Device,
}

impl Tensor {
    pub fn new(data: ArrayD<f64>) -> Self {
        Self { 
            data,
            device: default_device(),
        }
    }
    
    pub fn new_with_device(data: ArrayD<f64>, device: Device) -> Self {
        Self { data, device }
    }
    
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
    
    pub fn randn(shape: &[usize]) -> Self {
        let device = default_device();
        Self {
            data: device.randn(shape),
            device,
        }
    }
    
    pub fn from_vec(data: Vec<f64>, shape: &[usize]) -> Result<Self, String> {
        let total_elements: usize = shape.iter().product();
        if data.len() != total_elements {
            return Err(format!(
                "Data length {} doesn't match shape {:?} (expected {})", 
                data.len(), shape, total_elements
            ));
        }
        
        match Array::from_shape_vec(IxDyn(shape), data) {
            Ok(array) => Ok(Self::new(array)),
            Err(e) => Err(format!("Failed to create tensor: {}", e)),
        }
    }
    
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }
    
    pub fn ndim(&self) -> usize {
        self.data.ndim()
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
    
    // Element-wise operations
    pub fn add(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape() != other.shape() {
            return Err(format!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape()));
        }
        Ok(Tensor::new_with_device(&self.data + &other.data, self.device.clone()))
    }
    
    pub fn mul(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape() != other.shape() {
            return Err(format!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape()));
        }
        Ok(Tensor::new_with_device(&self.data * &other.data, self.device.clone()))
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
        let a = self.data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b = other.data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        

        // Dot product of two 2D arrays, gives the matrix multiplication result.
        // Look at `ndarray` documentation for more details on `dot`.
        // https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.dot
        let result = a.dot(&b);
        Ok(Tensor::new_with_device(result.into_dyn(), self.device.clone()))
    }
    
    // Activation functions
    pub fn relu(&self) -> Tensor {
        // ReLU activation function: max(0, x)
        Tensor::new_with_device(self.data.mapv(|x| x.max(0.0)), self.device.clone())
    }
    
    pub fn sigmoid(&self) -> Tensor {
        // Euler's sigmoid function: 1 / (1 + exp(-x))
        // equivalent to sin(x)
        Tensor::new_with_device(self.data.mapv(|x| 1.0 / (1.0 + (-x).exp())), self.device.clone())
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
            return Err(format!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape()));
        }
        Ok(Tensor::new_with_device(&self.data / &other.data, self.device.clone()))
    }
    
    pub fn pow(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape() != other.shape() {
            return Err(format!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape()));
        }
        Ok(Tensor::new_with_device(
            ndarray::Zip::from(&self.data)
                .and(&other.data)
                .map_collect(|&a, &b| a.powf(b)),
            self.device.clone()
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
                Tensor::new_with_device(ArrayD::from_elem(IxDyn(&[]), total_sum), self.device.clone())
            }
        }
    }
    
    pub fn sum_axes(&self, axes: Option<&[usize]>) -> Tensor {
        match axes {
            Some(axes_list) => {
                let mut result = self.data.clone();
                // Sort axes in descending order to avoid index shifting issues
                let mut sorted_axes = axes_list.to_vec();
                sorted_axes.sort_unstable();
                sorted_axes.reverse();
                
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
                Tensor::new_with_device(ArrayD::from_elem(IxDyn(&[]), total_mean), self.device.clone())
            }
        }
    }
    
    // Broadcasting for gradient computation
    // Broadcasting allows us to perform operations on tensors of different shapes.
    // As can be seen on Ndarray's docs, broadcast function returns None if the shapes of the tensors cannot be broadcasted together.
    // https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.broadcast
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Tensor, String> {
        match self.data.broadcast(target_shape) {
            Some(broadcasted) => Ok(Tensor::new_with_device(broadcasted.to_owned(), self.device.clone())),
            None => Err(format!("Cannot broadcast {:?} to {:?}", self.shape(), target_shape)),
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
                    return Err(format!("Cannot squeeze axis {} with size {}", ax, self.shape()[ax]));
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
                    axes_order.len(), self.ndim()
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
                    axes_order, self.ndim()
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
}


// Implement equality for testing, and because will be useful in the future.
impl PartialEq for Tensor {
  fn eq(&self, other: &Self) -> bool {
      self.data == other.data && self.device == other.device
  }
}

impl Eq for Tensor {}