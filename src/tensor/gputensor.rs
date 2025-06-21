use crate::backend::manager::get_backend;
use crate::backend::number::{CPUFloat, CPUNumber, GPUFloat, GPUNumber};
use crate::backend::{Device, default_device};
use ndarray::{Array, ArrayD, Axis, IxDyn};
use std::ops::{BitOr, Index};

#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaTensor;

#[cfg(feature = "cuda")]
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct GPUTensor<T>
where
    T: GPUNumber,
{
    pub data: ArrayD<T>,
    device: Device,
    cuda_storage: Option<CudaTensor<T>>,
}

#[cfg(feature = "cuda")]
impl<T> GPUTensor<T>
where
    T: GPUNumber,
{
    // Basic constructor - creates a GPU-capable tensor with CPU data initially
    // Similar to CPUTensor but this one can be moved to GPU when needed
    pub fn new(data: ArrayD<T>) -> Self {
        Self {
            data,
            device: default_device(),
            cuda_storage: None,
        }
    }

    // Constructor with explicit device specification
    // If device is CUDA, we'll need to move data there later via to_cuda()
    pub fn new_with_device(data: ArrayD<T>, device: Device) -> Self {
        Self {
            data,
            device,
            cuda_storage: None,
        }
    }

    /// Helper to eliminate repeated backend access pattern.
    /// This removes the need to repeatedly call `get_backend()` in every method.
    /// Allows us to execute a closure with the CUDA backend.
    fn with_cuda_backend<F, R>(&self, f: F) -> Result<R, String>
    where
        F: FnOnce(&crate::backend::cuda::CudaBackend) -> Result<R, String>,
    {
        use crate::backend::manager::get_backend;
        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;
        f(cuda_backend)
    }

    // Create tensor from Vec - this is the main way users will create tensors
    // Works exactly like CPUTensor but the result can be moved to GPU
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

    // Shape getter - works for both CPU and CUDA tensors
    // If we have CUDA storage, get shape from there, otherwise from CPU data
    pub fn shape(&self) -> &[usize] {
        if let Some(cuda_tensor) = &self.cuda_storage {
            &cuda_tensor.shape
        } else {
            self.data.shape()
        }
    }

    // Number of dimensions - works for both CPU and CUDA
    pub fn ndim(&self) -> usize {
        if let Some(cuda_tensor) = &self.cuda_storage {
            cuda_tensor.shape.len()
        } else {
            self.data.ndim()
        }
    }

    // Total number of elements
    pub fn size(&self) -> usize {
        if let Some(cuda_tensor) = &self.cuda_storage {
            cuda_tensor.shape.iter().product()
        } else {
            self.data.len()
        }
    }

    // Check if this tensor is currently on CUDA
    pub fn is_cuda(&self) -> bool {
        self.device.is_cuda()
    }

    // Get device reference
    pub fn device(&self) -> &Device {
        &self.device
    }

    // Convert to Vec - handles both CPU and CUDA tensors
    pub fn to_vec(&self) -> Result<Vec<T>, String> {
        if let Some(cuda_tensor) = &self.cuda_storage {
            // If we have CUDA data, copy it back to CPU first
            use crate::backend::manager::get_backend;
            let backend = get_backend();
            let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;
            let memory_manager = cuda_backend.memory_manager();

            memory_manager.device_to_host(&cuda_tensor.data)
        } else {
            // If it's CPU data, just clone it
            Ok(self.data.iter().cloned().collect())
        }
    }

    pub fn len(&self) -> usize {
        self.size()
    }

    /// Conditional selection: where condition is true, use true_vals, else false_vals (CPU only for now)
    pub fn where_condition(
        condition: &GPUTensor<T>,
        true_vals: &GPUTensor<T>,
        false_vals: &GPUTensor<T>,
    ) -> Result<GPUTensor<T>, String> {
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

        GPUTensor::from_vec(result_vec, condition.shape())
    }

    // Helper method to perform CPU operations on this GPU-capable tensor
    // This allows us to fall back to CPU when CUDA fails
    fn add_cpu(&self, other: &Self) -> Result<GPUTensor<T>, String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }
        Ok(Self::new_with_device(
            &self.data + &other.data,
            self.device.clone(),
        ))
    }

    // Helper method to perform CPU multiplication
    fn mul_cpu(&self, other: &Self) -> Result<GPUTensor<T>, String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }
        Ok(Self::new_with_device(
            &self.data * &other.data,
            self.device.clone(),
        ))
    }

    // Helper method to perform CPU division
    fn div_cpu(&self, other: &Self) -> Result<GPUTensor<T>, String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }
        Ok(Self::new_with_device(
            &self.data / &other.data,
            self.device.clone(),
        ))
    }


    // Smart addition - this is the main API users will use
    // Automatically chooses between CPU and CUDA based on device
    pub fn add(&self, other: &Self) -> Result<GPUTensor<T>, String> {
        // Check device compatibility first
        if self.device != other.device {
            return Err("Tensors must be on the same device for operation".to_string());
        }

        match &self.device {
            Device::CPU => {
                // If both tensors are on CPU, just do CPU operation
                self.add_cpu(other)
            }
            Device::CUDA(_) => {
                // Try CUDA operation first, fall back to CPU if it fails
                self.add_cuda(other).or_else(|_| {
                    println!("CUDA add failed, falling back to CPU");
                    self.add_cpu(other)
                })
            }
        }
    }

    // Smart multiplication with automatic device selection
    pub fn mul(&self, other: &Self) -> Result<GPUTensor<T>, String> {
        if self.device != other.device {
            return Err("Tensors must be on the same device for operation".to_string());
        }

        match &self.device {
            Device::CPU => self.mul_cpu(other),
            Device::CUDA(_) => self.mul_cuda(other).or_else(|_| {
                println!("CUDA mul failed, falling back to CPU");
                self.mul_cpu(other)
            }),
        }
    }

    // Smart division with automatic device selection
    pub fn div(&self, other: &Self) -> Result<GPUTensor<T>, String> {
        if self.device != other.device {
            return Err("Tensors must be on the same device for operation".to_string());
        }

        match &self.device {
            Device::CPU => self.div_cpu(other),
            Device::CUDA(_) => self.div_cuda(other).or_else(|_| {
                println!("CUDA div failed, falling back to CPU");
                self.div_cpu(other)
            }),
        }
    }

    // Detach operation - creates a new tensor without gradient tracking
    // Useful for autograd system
    pub fn detach(&self) -> Self {
        Self::new_with_device(self.data.clone(), self.device.clone())
    }

    // Iterator methods - these work on CPU data
    pub fn iter(&self) -> ndarray::iter::Iter<'_, T, ndarray::IxDyn> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, T, ndarray::IxDyn> {
        self.data.iter_mut()
    }

    // Random tensor creation methods
    pub fn randn(shape: &[usize]) -> Self {
        let device = default_device();
        let data_f64 = device.randn(shape);
        let data = data_f64.mapv(|x| <T as CPUNumber>::from_f64(x).unwrap());
        Self {
            data,
            device,
            cuda_storage: None,
        }
    }

    pub fn randn_with_device(shape: &[usize], device: Device) -> Self {
        let data_f64 = device.randn(shape);
        let data = data_f64.mapv(|x| <T as CPUNumber>::from_f64(x).unwrap());
        Self {
            data,
            device,
            cuda_storage: None,
        }
    }

    // Helper methods for CUDA operations
    fn get_or_create_cuda_tensor(
        &self,
        cuda_backend: &crate::backend::cuda::CudaBackend,
    ) -> Result<crate::backend::cuda::CudaTensor<T>, String> {
        if let Some(cuda_tensor) = &self.cuda_storage {
            Ok(cuda_tensor.clone())
        } else {
            // Convert CPU data to CUDA
            let host_data: Vec<T> = self.data.iter().cloned().collect();
            let shape: Vec<usize> = self.shape().to_vec();
            let cuda_data = cuda_backend.memory_manager().host_to_device(host_data)?;
            Ok(crate::backend::cuda::CudaTensor::new(cuda_data, shape))
        }
    }
}

///  GPU TENSOR REPRESENTS A TENSOR THAT CAN BE MOVED TO CUDA
/// It is a sibling of CPUTensor, but it can be moved to CUDA and run kernels on it.
#[cfg(feature = "cuda")]
impl<T> GPUTensor<T>
where
    T: GPUNumber,
{
    fn matmul_cpu(&self, other: &Self) -> Result<GPUTensor<T>, String> {
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

        let a: ndarray::ArrayView2<T> = self.data.view().into_dimensionality().unwrap();
        let b: ndarray::ArrayView2<T> = other.data.view().into_dimensionality().unwrap();
        let result = a.dot(&b);

        Ok(Self::new_with_device(
            result.into_dyn(),
            self.device.clone(),
        ))
    }

    pub fn matmul(&self, other: &Self) -> Result<GPUTensor<T>, String> {
        if self.device != other.device {
            return Err("Tensors must be on the same device for matrix multiplication".to_string());
        }

        match &self.device {
            Device::CPU => self.matmul_cpu(other),
            Device::CUDA(_) => self.matmul_cuda(other).or_else(|_| {
                println!("CUDA matmul failed, falling back to CPU");
                self.matmul_cpu(other)
            }),
        }
    }
    //---------------------------------------------------------
    // CUDA matrix multiplication implementation using tiled matmul kernel
    // (see matmul.ptx)
    // This function performs matrix multiplication on CUDA tensors
    //----------------------------------------------------------
    fn matmul_cuda(&self, other: &Self) -> Result<GPUTensor<T>, String> {
        use crate::backend::manager::get_backend;

        // Basic validation - ensure we have 2D tensors
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err("CUDA matrix multiplication requires 2D tensors".to_string());
        }

        let a_shape = self.shape();
        let b_shape = other.shape();

        // Check dimension compatibility: (M, K) @ (K, N) -> (M, N)
        if a_shape[1] != b_shape[0] {
            return Err(format!(
                "CUDA matrix multiplication shape mismatch: ({}, {}) @ ({}, {})",
                a_shape[0], a_shape[1], b_shape[0], b_shape[1]
            ));
        }

        
        self.with_cuda_backend(|cuda_backend| {
            // Get or create CUDA tensors for both operands
            let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
            let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;
            
            // Get CUDA operations handle
            let cuda_ops = cuda_backend.ops();
            
            // Use the high-level matmul operation instead of direct kernel launch
            let result_cuda = cuda_ops.matmul(&cuda_a, &cuda_b)?;
            
            // Convert the CUDA result back to GPUTensor
            self.create_tensor_from_cuda_result(result_cuda)
        })
}
}

#[cfg(feature = "cuda")]
impl<T> GPUTensor<T>
where
    T: GPUNumber,
{
    // Summation operations for GPUTensor
    // These methods handle summation along specified axes or over the entire tensor.
    fn sum_cpu(&self, axis: Option<usize>) -> GPUTensor<T> {
        match axis {
            Some(ax) => {
                let result = self.data.sum_axis(ndarray::Axis(ax));
                Self::new_with_device(result, self.device.clone())
            }
            None => {
                // Sum all elements
                let total_sum = self.data.sum();
                let result_array = ArrayD::from_elem(IxDyn(&[]), total_sum);
                Self::new_with_device(result_array, self.device.clone())
            }
        }
    }

    // This uses a parallel reduction on CUDA to sum the tensor.
    // It can sum along a specific axis or over the entire tensor.
    fn sum_cuda(&self, axis: Option<usize>) -> Result<GPUTensor<T>, String> {

        self.with_cuda_backend(|cuda_backend| {
            // Get or create CUDA tensor
            let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
            let cuda_ops = cuda_backend.ops();

            // Perform summation operation
            let result_cuda = match axis {
                // Use sum_axis for summing along a specific axis
                Some(ax) => cuda_ops.sum_axis(&cuda_tensor, ax, false)?,
                // Use sum_all for summing over the entire tensor
                None => cuda_ops.sum_all(&cuda_tensor)?,
            };

            // Convert the CUDA result back to GPUTensor
            self.create_tensor_from_cuda_result(result_cuda)
        })
       
    }

    /// Smart sum operation
    pub fn sum(&self, axis: Option<usize>) -> GPUTensor<T> {
        match &self.device {
            Device::CPU => self.sum_cpu(axis),
            Device::CUDA(_) => self.sum_cuda(axis).unwrap_or_else(|_| {
                println!("CUDA sum failed, falling back to CPU");
                self.sum_cpu(axis)
            }),
        }
    }

    // For mean operation, we can use the same logic as sum,
    // but we need to divide by the number of elements along the specified axis.
    // TODO: Add a kernel for mean operation on CUDA.
    pub fn mean(&self, axis: Option<usize>) -> Self {
        match axis {
            Some(ax) => {
                let result = self.data.mean_axis(Axis(ax)).unwrap();
                Self::new_with_device(result, self.device.clone())
            }
            None => {
                let total_mean = self.data.mean().unwrap();
                Self::new_with_device(
                    ArrayD::from_elem(IxDyn(&[]), total_mean),
                    self.device.clone(),
                )
            }
        }
    }

    // Broadcasting for gradient computation - now returns GPUTensor
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Self, String> {
        match self.data.broadcast(target_shape) {
            Some(broadcasted) => Ok(Self::new_with_device(
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
    pub fn unsqueeze(&self, axis: usize) -> Self {
        let expanded = self.data.clone().insert_axis(Axis(axis));
        Self::new_with_device(expanded, self.device.clone())
    }

    // Squeeze operation - remove dimensions of size 1
    pub fn squeeze(&self, axis: Option<usize>) -> Result<Self, String> {
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
                Ok(Self::new_with_device(squeezed, self.device.clone())) // Return Self, not CPUTensor
            }
            None => {
                let mut result = self.data.clone();
                let axes_to_remove: Vec<usize> = self
                    .shape()
                    .iter()
                    .enumerate()
                    .filter(|&(_, &size)| size == 1)
                    .map(|(i, _)| i)
                    .collect();

                for &ax in axes_to_remove.iter().rev() {
                    result = result.remove_axis(Axis(ax));
                }

                Ok(Self::new_with_device(result, self.device.clone())) // Return Self, not CPUTensor
            }
        }
    }

    // Reshape operation - change tensor shape while preserving total elements
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, String> {
        let total_elements: usize = self.shape().iter().product();
        let new_total_elements: usize = new_shape.iter().product();

        if total_elements != new_total_elements {
            return Err(format!(
                "Cannot reshape tensor with {} elements to shape with {} elements",
                total_elements, new_total_elements
            ));
        }

        match self.data.clone().into_shape_with_order(IxDyn(new_shape)) {
            Ok(reshaped) => Ok(Self::new_with_device(reshaped, self.device.clone())),
            Err(e) => Err(format!("Failed to reshape tensor: {}", e)),
        }
    }

    // Efficient 2D transpose operation using CUDA
    // This is a specialized operation for 2D tensors, leveraging CUDA capabilities
    // It assumes the tensor is 2D and transposes it by swapping rows and columns, in parallel
    fn transpose_2D_cuda(&self) -> Result<Self, String> {
        if self.ndim() != 2 {
            return Err("Transpose is only supported for 2D tensors".to_string());
        }

        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.transpose_2d(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

    // Transpose operation - permute the axes of the tensor
    // This is tricky to implement from scratch but ndarray handles it for us
    pub fn transpose(&self, axes: Option<&[usize]>) -> Result<Self, String> {
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
                Ok(Self::new_with_device(transposed, self.device.clone()))
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
                        // For 2D tensors, we can use specialized methods
                        // to perform the transpose efficiently.
                        match &self.device {
                            Device::CPU => {
                                let transposed = self.data.clone().reversed_axes();
                                Ok(Self::new_with_device(transposed, self.device.clone()))
                            }
                            Device::CUDA(_) => self.transpose_2D_cuda(),
                        }
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
                        Ok(Self::new_with_device(transposed, self.device.clone()))
                    }
                }
            }
        }
    }

    /// Sum along multiple axes (matches CPU tensor functionality)
    fn sum_axes_cpu(&self, axes: Option<&[usize]>) -> GPUTensor<T> {
        match axes {
            Some(axes_list) => {
                // Validate axes are within bounds
                for &ax in axes_list {
                    if ax >= self.ndim() {
                        panic!(
                            "Axis {} out of bounds for tensor with {} dimensions",
                            ax,
                            self.ndim()
                        );
                    }
                }

                // Sort axes in reverse order to maintain correct indexing
                let mut sorted_axes = axes_list.to_vec();
                sorted_axes.sort_by(|a, b| b.cmp(a));

                let mut current = self.clone();
                for &axis in &sorted_axes {
                    current = current.sum(Some(axis));
                }
                current
            }
            None => self.sum(None),
        }
    }

    fn sum_axes_cuda(&self, axes: Option<&[usize]>) -> Result<GPUTensor<T>, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();

        let result_cuda = match axes {
            Some(axes_list) => cuda_ops.sum_axes(&cuda_tensor, axes_list, false)?,
            None => cuda_ops.sum_all(&cuda_tensor)?,
        };

        self.create_tensor_from_cuda_result(result_cuda)
    }

    /// Smart sum_axes operation
    pub fn sum_axes(&self, axes: Option<&[usize]>) -> GPUTensor<T> {
        match &self.device {
            Device::CPU => self.sum_axes_cpu(axes),
            Device::CUDA(_) => self.sum_axes_cuda(axes).unwrap_or_else(|_| {
                println!("CUDA sum_axes failed, falling back to CPU");
                self.sum_axes_cpu(axes)
            }),
        }
    }
}

// Zero initialization for GPU tensors
#[cfg(feature = "cuda")]
impl<T> GPUTensor<T>
where
    T: GPUNumber,
{
    pub fn zeros(shape: &[usize]) -> Self {
        let device = default_device();
        Self {
            data: device.zeros(shape),
            device,
            cuda_storage: None,
        }
    }

    pub fn zeros_with_device(shape: &[usize], device: Device) -> Self {
        Self {
            data: device.zeros(shape),
            device,
            cuda_storage: None,
        }
    }
}

// One initialization for GPU tensors
#[cfg(feature = "cuda")]
impl<T> GPUTensor<T>
where
    T: GPUNumber,
{
    pub fn ones(shape: &[usize]) -> Self {
        let device = default_device();
        Self {
            data: device.ones(shape),
            device,
            cuda_storage: None,
        }
    }

    pub fn ones_with_device(shape: &[usize], device: Device) -> Self {
        Self {
            data: device.ones(shape),
            device,
            cuda_storage: None,
        }
    }
}

/// CUDA SPECIFIC IMPLEMENTATIONS
#[cfg(feature = "cuda")]
impl<T> GPUTensor<T>
where
    T: GPUNumber,
{
    pub fn to_cuda(&self) -> Result<Self, String> {
        if self.is_cuda() {
            return Ok(self.clone());
        }

        use crate::backend::manager::get_backend;
        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let shape: Vec<usize> = self.shape().to_vec();
        let host_data: Vec<T> = self.data.iter().cloned().collect();
        let cuda_tensor = CudaTensor::from_vec(&cuda_backend.memory_manager(), host_data, shape)?;

        Ok(Self {
            data: self.data.clone(),
            device: Device::CUDA(0),
            cuda_storage: Some(cuda_tensor),
        })
    }

    /// Transfer tensor to specified device
    pub fn to_device(&self, device: Device) -> Result<Self, String> {
        match device {
            Device::CPU => self.to_cpu(),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.to_cuda(),
        }
    }

    /// Move tensor to CPU
    pub fn to_cpu(&self) -> Result<Self, String> {
        if !self.is_cuda() {
            return Ok(self.clone());
        }

        #[cfg(feature = "cuda")]
        {
            if let Some(cuda_tensor) = &self.cuda_storage {
                // Get the memory manager from backend
                use crate::backend::manager::get_backend;
                let backend = get_backend();
                let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;
                let memory_manager = cuda_backend.memory_manager();

                // Pass memory manager to to_vec()
                let host_data = cuda_tensor.to_vec(&memory_manager)?;

                let cpu_array = ArrayD::from_shape_vec(IxDyn(cuda_tensor.shape()), host_data)
                    .map_err(|e| format!("Failed to create CPU array: {}", e))?;

                Ok(Self {
                    data: cpu_array,
                    device: Device::CPU,
                    cuda_storage: None,
                })
            } else {
                Ok(self.clone())
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(self.clone())
        }
    }

    /// ------------------------------------------------------------
    /// CUDA - BASED ELEMENTWISE ARITHMETIC OPERATIONS
    /// -------------------------------------------------------------

    pub fn add_cuda(&self, other: &GPUTensor<T>) -> Result<Self, String> {
        self.with_cuda_backend(|cuda_backend| {
            // Ensure both tensors are on CUDA
            if !self.is_cuda() || !other.is_cuda() {
                return Err("Both tensors must be on CUDA for add operation".to_string());
            }

            // Convert tensors to CUDA if needed
            let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
            let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

            // Perform CUDA operation
            let cuda_ops = cuda_backend.ops();
            let result_cuda = cuda_ops.add(&cuda_a, &cuda_b)?;

            // Create result tensor with CUDA storage
            self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    /// Element-wise multiplication on CUDA tensors
    pub fn mul_cuda(&self, other: &GPUTensor<T>) -> Result<GPUTensor<T>, String> {
       self.with_cuda_backend(|cuda_backend| {
        // Ensure both tensors are on CUDA
        if !self.is_cuda() || !other.is_cuda() {
            return Err("Both tensors must be on CUDA for mul operation".to_string());
        }

        let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.mul(&cuda_a, &cuda_b)?;

        self.create_tensor_from_cuda_result(result_cuda)
       })
    }
    /// Element-wise division on CUDA tensors
    pub fn div_cuda(&self, other: &GPUTensor<T>) -> Result<Self, String> {
        self.with_cuda_backend(|cuda_backend| {
            // Ensure both tensors are on CUDA
            if !self.is_cuda() || !other.is_cuda() {
                return Err("Both tensors must be on CUDA for div operation".to_string());
            }

        let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.div(&cuda_a, &cuda_b)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    // ------------------------------------------------------------
    // MAX-MIN-ABS OPERATIONS (ELEMENTWISE)
    // -------------------------------------------------------------

    /// Element-wise minimum between two tensors
    /// Supports both CPU and CUDA execution
    pub fn min(&self, other: &Self) -> Result<Self, String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Shape mismatch in min operation: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }

        match &self.device {
            Device::CPU => self.min_cpu(other),
            Device::CUDA(_) => match self.min_cuda(other) {
                Ok(result) => Ok(result),
                Err(_) => {
                    println!("CUDA min failed, falling back to CPU");
                    self.min_cpu(other)
                }
            },
        }
    }

    /// Element-wise maximum between two tensors
    /// Supports both CPU and CUDA execution  
    pub fn max(&self, other: &Self) -> Result<Self, String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Shape mismatch in max operation: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }

        match &self.device {
            Device::CPU => self.max_cpu(other),
            Device::CUDA(_) => match self.max_cuda(other) {
                Ok(result) => Ok(result),
                Err(_) => {
                    println!("CUDA max failed, falling back to CPU");
                    self.max_cpu(other)
                }
            },
        }
    }

    /// Element-wise absolute value
    /// Supports both CPU and CUDA execution
    pub fn abs(&self) -> Self {
        match &self.device {
            Device::CPU => self.abs_cpu(),
            Device::CUDA(_) => match self.abs_cuda() {
                Ok(result) => result,
                Err(_) => {
                    println!("CUDA abs failed, falling back to CPU");
                    self.abs_cpu()
                }
            },
        }
    }

    // CPU fallback implementations
    fn min_cpu(&self, other: &Self) -> Result<Self, String> {
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

    fn max_cpu(&self, other: &Self) -> Result<Self, String> {
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

    fn abs_cpu(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.abs()).collect();

        let result_array = ndarray::Array::from_shape_vec(self.data.raw_dim(), result_data)
            .expect("Shape should match original tensor");

        Self::new_with_device(result_array, self.device.clone())
    }

    // CUDA implementations
    fn min_cuda(&self, other: &Self) -> Result<Self, String> {
        self.with_cuda_backend(|cuda_backend| {
        // Ensure both tensors are on CUDA
        if !self.is_cuda() || !other.is_cuda() {
            return Err("Both tensors must be on CUDA for min operation".to_string());
        }

        let self_cuda = self.get_or_create_cuda_tensor(cuda_backend)?;
        let other_cuda = other.get_or_create_cuda_tensor(cuda_backend)?;

        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.min_elementwise(&self_cuda, &other_cuda)?;

        self.create_tensor_from_cuda_result(result_cuda)
    })
    }

    fn max_cuda(&self, other: &Self) -> Result<Self, String> {
        self.with_cuda_backend(|cuda_backend| {
            // Ensure both tensors are on CUDA
            if !self.is_cuda() || !other.is_cuda() {
                return Err("Both tensors must be on CUDA for max operation".to_string());
            }

        let self_cuda = self.get_or_create_cuda_tensor(cuda_backend)?;
        let other_cuda = other.get_or_create_cuda_tensor(cuda_backend)?;

        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.max_elementwise(&self_cuda, &other_cuda)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    fn abs_cuda(&self) -> Result<Self, String> {
        self.with_cuda_backend(|cuda_backend| {
        // Ensure tensor is on CUDA
        if !self.is_cuda() {
            return Err("Tensor must be on CUDA for abs operation".to_string());
        }

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.abs(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }
    /// ------------------------------------------------------------
    /// CLAMP OPERATION
    /// -------------------------------------------------------------

    /// Clamp values within range
    pub fn clamp(&self, min_val: T, max_val: T) -> Self {
        match &self.device {
            Device::CPU => self.clamp_cpu(min_val, max_val),
            Device::CUDA(_) => self.clamp_cuda(min_val, max_val).unwrap_or_else(|_| {
                println!("CUDA clamp failed, falling back to CPU");
                self.clamp_cpu(min_val, max_val)
            }),
        }
    }

    fn clamp_cpu(&self, min_val: T, max_val: T) -> Self {
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

    fn clamp_cuda(&self, min_val: T, max_val: T) -> Result<Self, String> {
        
        self.with_cuda_backend(|cuda_backend| {
        // Ensure tensor is on CUDA
        if !self.is_cuda() {
            return Err("Tensor must be on CUDA for clamp operation".to_string());
        }
        // Get or create CUDA tensor
        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();

        let result_cuda = cuda_ops.clamp(&cuda_tensor, min_val, max_val)?;
        self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    /// -------------------------------------------------------------
    /// MAX ALONG DIMENSION
    /// -------------------------------------------------------------

    /// Find maximum values along a specific dimension
    /// Supports both CPU and CUDA execution
    pub fn max_along_dim(&self, dim: usize) -> Result<Self, String> {
        let shape = self.shape();

        if dim >= shape.len() {
            return Err(format!(
                "Dimension {} out of bounds for tensor with {} dimensions",
                dim,
                shape.len()
            ));
        }

        match &self.device {
            Device::CPU => self.max_along_dim_cpu(dim),
            Device::CUDA(_) => match self.max_along_dim_cuda(dim) {
                Ok(result) => Ok(result),
                Err(err) => {
                    println!("CUDA max_along_dim failed ({}), falling back to CPU", err);
                    self.max_along_dim_cpu(dim)
                }
            },
        }
    }

    fn max_along_dim_cpu(&self, dim: usize) -> Result<Self, String> {
        let shape = self.shape();

        // Calculate the output shape (remove the specified dimension)
        let mut output_shape = shape.to_vec();
        output_shape.remove(dim);

        // If we're reducing all dimensions, result is a scalar
        if output_shape.is_empty() {
            output_shape.push(1);
        }

        let output_size: usize = output_shape.iter().product();
        let mut result_data = vec![<T as CPUNumber>::min_value(); output_size];

        // Use the same helper methods as CPUTensor
        let input_strides = Self::calculate_strides_for_shape(shape);
        let output_strides = Self::calculate_strides_for_shape(&output_shape);

        // Iterate through all elements and find maximum along the specified dimension
        for input_idx in 0..self.data.len() {
            let mut coords = Self::flat_to_coords(input_idx, &input_strides, shape);
            coords.remove(dim);

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

    fn max_along_dim_cuda(&self, dim: usize) -> Result<Self, String> {
        self.with_cuda_backend(|cuda_backend| {
        // Ensure tensor is on CUDA
        if !self.is_cuda() {
            return Err("Tensor must be on CUDA for max_along_dim operation".to_string());
        }

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.max_along_dim(&cuda_tensor, dim)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    // Helper methods (similar to CPUTensor)
    fn calculate_strides_for_shape(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    fn flat_to_coords(mut flat_idx: usize, strides: &[usize], shape: &[usize]) -> Vec<usize> {
        let mut coords = vec![0; strides.len()];
        for i in 0..strides.len() {
            coords[i] = flat_idx / strides[i];
            flat_idx %= strides[i];
        }
        coords
    }

    fn coords_to_flat(coords: &[usize], strides: &[usize]) -> usize {
        coords
            .iter()
            .zip(strides.iter())
            .map(|(coord, stride)| coord * stride)
            .sum()
    }

    //-------------------------------------------------------------
    // COMPARISON AND EQUALITY
    //-------------------------------------------------------------

    // Element-wise greater than or equal comparison
    pub fn greater_equal(&self, other: &Self) -> Result<GPUTensor<T>, String> {
        if self.device != other.device {
            return Err("Tensors must be on the same device".to_string());
        }

        match &self.device {
            Device::CPU => self.greater_equal_cpu(other),
            Device::CUDA(_) => self.greater_equal_cuda(other).or_else(|_| {
                println!("CUDA greater_equal failed, falling back to CPU");
                self.greater_equal_cpu(other)
            }),
        }
    }

    // CPU implementation for greater_equal
    fn greater_equal_cpu(&self, other: &Self) -> Result<GPUTensor<T>, String> {
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

    // CUDA implementation for greater_equal
    fn greater_equal_cuda(&self, other: &Self) -> Result<GPUTensor<T>, String> {
        
        self.with_cuda_backend(|cuda_backend| {
        // Ensure both tensors are on CUDA
        if !self.is_cuda() || !other.is_cuda() {
            return Err("Both tensors must be on CUDA for greater_equal operation".to_string());
        }

        // Get or create CUDA tensors for both operands
        let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.greater_equal(&cuda_a, &cuda_b)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    // Less than or equal comparison
    pub fn less_equal(&self, other: &Self) -> Result<GPUTensor<T>, String> {
        if self.device != other.device {
            return Err("Tensors must be on the same device".to_string());
        }

        match &self.device {
            Device::CPU => self.less_equal_cpu(other),
            Device::CUDA(_) => self.less_equal_cuda(other).or_else(|_| {
                println!("CUDA less_equal failed, falling back to CPU");
                self.less_equal_cpu(other)
            }),
        }
    }

    fn less_equal_cpu(&self, other: &Self) -> Result<GPUTensor<T>, String> {
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

    fn less_equal_cuda(&self, other: &Self) -> Result<GPUTensor<T>, String> {
        self.with_cuda_backend(|cuda_backend| {
            // Ensure both tensors are on CUDA
            if !self.is_cuda() || !other.is_cuda() {
                return Err("Both tensors must be on CUDA for less_equal operation".to_string());
            }

        let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.less_equal(&cuda_a, &cuda_b)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    // Equality comparison
    pub fn equal(&self, other: &Self) -> Result<GPUTensor<T>, String> {
        if self.device != other.device {
            return Err("Tensors must be on the same device".to_string());
        }

        match &self.device {
            Device::CPU => self.equal_cpu(other),
            Device::CUDA(_) => self.equal_cuda(other).or_else(|_| {
                println!("CUDA equal failed, falling back to CPU");
                self.equal_cpu(other)
            }),
        }
    }

    fn equal_cpu(&self, other: &Self) -> Result<GPUTensor<T>, String> {
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

    fn equal_cuda(&self, other: &Self) -> Result<GPUTensor<T>, String> {
        self.with_cuda_backend(|cuda_backend| {
            // Ensure both tensors are on CUDA
            if !self.is_cuda() || !other.is_cuda() {
                return Err("Both tensors must be on CUDA for equal operation".to_string());
            }

        let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.equal(&cuda_a, &cuda_b)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    // Logical NOT operation
    pub fn logical_not(&self) -> Result<GPUTensor<T>, String> {
        match &self.device {
            Device::CPU => self.logical_not_cpu(),
            Device::CUDA(_) => self.logical_not_cuda().or_else(|_| {
                println!("CUDA logical_not failed, falling back to CPU");
                self.logical_not_cpu()
            }),
        }
    }

    fn logical_not_cpu(&self) -> Result<GPUTensor<T>, String> {
        let result_data = self.data.mapv(|x| {
            if x == <T as CPUNumber>::zero() {
                <T as CPUNumber>::one()
            } else {
                <T as CPUNumber>::zero()
            }
        });

        Ok(Self::new_with_device(result_data, self.device.clone()))
    }

    fn logical_not_cuda(&self) -> Result<GPUTensor<T>, String> {
        self.with_cuda_backend(|cuda_backend| {
            // Ensure both tensors are on CUDA
            if !self.is_cuda() || !other.is_cuda() {
                return Err("Both tensors must be on CUDA for logical not operation".to_string());
            }

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.logical_not(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    // Check if values are in range
    pub fn in_range(&self, min_val: T, max_val: T) -> Result<GPUTensor<T>, String> {
        match &self.device {
            Device::CPU => self.in_range_cpu(min_val, max_val),
            Device::CUDA(_) => self.in_range_cuda(min_val, max_val).or_else(|_| {
                println!("CUDA in_range failed, falling back to CPU");
                self.in_range_cpu(min_val, max_val)
            }),
        }
    }

    fn in_range_cpu(&self, min_val: T, max_val: T) -> Result<GPUTensor<T>, String> {
        let result_data = self.data.mapv(|x| {
            if x >= min_val && x <= max_val {
                <T as CPUNumber>::one()
            } else {
                <T as CPUNumber>::zero()
            }
        });

        Ok(Self::new_with_device(result_data, self.device.clone()))
    }

    fn in_range_cuda(&self, min_val: T, max_val: T) -> Result<GPUTensor<T>, String> {
        self.with_cuda_backend(|cuda_backend| {
            // Ensure both tensors are on CUDA
            if !self.is_cuda() || !other.is_cuda() {
                return Err("Both tensors must be on CUDA for in range operation".to_string());
            }

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.in_range(&cuda_tensor, min_val, max_val)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    // Expand dimensions
    pub fn expand_dims(&self, axis: usize) -> Result<GPUTensor<T>, String> {
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

    /// SIGN METHOD
    pub fn sign(&self) -> GPUTensor<T> {
        match &self.device {
            Device::CPU => self.sign_cpu(),
            Device::CUDA(_) => self.sign_cuda().unwrap_or_else(|_| {
                println!("CUDA sign failed, falling back to CPU");
                self.sign_cpu()
            }),
        }
    }

    fn sign_cpu(&self) -> GPUTensor<T> {
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

    fn sign_cuda(&self) -> Result<GPUTensor<T>, String> {
        self.with_cuda_backend(|cuda_backend| {
            // Ensure both tensors are on CUDA
            if !self.is_cuda() {
                return Err("Tensor must be on CUDA for sign operation".to_string());
            }

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.sign(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    fn create_tensor_from_cuda_result(
        &self,
        cuda_result: crate::backend::cuda::CudaTensor<T>,
    ) -> Result<Self, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;
        let memory_manager = cuda_backend.memory_manager();

        // Transfer CUDA result back to CPU
        let result_cpu = cuda_result.to_cpu(&memory_manager)?;
        let shape = cuda_result.shape();

        // Create a new ArrayD from the CPU result
        let result_array = ArrayD::from_shape_vec(IxDyn(shape), result_cpu)
            .map_err(|e| format!("Failed to create array from CUDA result: {}", e))?;
        let mut result_tensor = GPUTensor::new_with_device(result_array, Device::CUDA(0));
        result_tensor.cuda_storage = Some(cuda_result);
        Ok(result_tensor)
    }
}

#[cfg(feature = "cuda")]
impl<T> GPUTensor<T>
where
    T: GPUNumber,
{
    // CPU scalar operations for fallback
    fn add_scalar_cpu(&self, scalar: T) -> GPUTensor<T> {
        Self::new_with_device(&self.data + scalar, self.device.clone())
    }

    fn mul_scalar_cpu(&self, scalar: T) -> GPUTensor<T> {
        Self::new_with_device(&self.data * scalar, self.device.clone())
    }

    fn div_scalar_cpu(&self, scalar: T) -> GPUTensor<T> {
        Self::new_with_device(&self.data / scalar, self.device.clone())
    }

    // Smart scalar operations
    pub fn add_scalar(&self, scalar: T) -> GPUTensor<T> {
        match &self.device {
            Device::CPU => self.add_scalar_cpu(scalar),
            Device::CUDA(_) => self.add_scalar_cuda(scalar).unwrap_or_else(|_| {
                println!("CUDA add_scalar failed, falling back to CPU");
                self.add_scalar_cpu(scalar)
            }),
        }
    }

    pub fn mul_scalar(&self, scalar: T) -> GPUTensor<T> {
        match &self.device {
            Device::CPU => self.mul_scalar_cpu(scalar),
            Device::CUDA(_) => self.mul_scalar_cuda(scalar).unwrap_or_else(|_| {
                println!("CUDA mul_scalar failed, falling back to CPU");
                self.mul_scalar_cpu(scalar)
            }),
        }
    }

    pub fn div_scalar(&self, scalar: T) -> GPUTensor<T> {
        match &self.device {
            Device::CPU => self.div_scalar_cpu(scalar),
            Device::CUDA(_) => self.div_scalar_cuda(scalar).unwrap_or_else(|_| {
                println!("CUDA div_scalar failed, falling back to CPU");
                self.div_scalar_cpu(scalar)
            }),
        }
    }

    // ------------------------------------------
    // CUDA-based scalar operations
    // ------------------------------------------

    pub fn add_scalar_cuda(&self, scalar: T) -> Result<Self, String> {
        self.with_cuda_backend(|cuda_backend| {
             // Ensure both tensors are on CUDA
             if !self.is_cuda() {
                return Err("Tensor must be on CUDA for add scalar operation".to_string());
            }

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.add_scalar(&cuda_tensor, scalar)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    pub fn mul_scalar_cuda(&self, scalar: T) -> Result<GPUTensor<T>, String> {
        self.with_cuda_backend(|cuda_backend| {
             // Ensure both tensors are on CUDA
             if !self.is_cuda() {
                return Err("Tensor must be on CUDA for mul scalar operation".to_string());
            }

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.mul_scalar(&cuda_tensor, scalar)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    pub fn div_scalar_cuda(&self, scalar: T) -> Result<GPUTensor<T>, String> {
        self.with_cuda_backend(|cuda_backend| {
             // Ensure both tensors are on CUDA
             if !self.is_cuda() {
                return Err("Tensor must be on CUDA for div scalar operation".to_string());
            }

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.div_scalar(&cuda_tensor, scalar)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }
}

#[cfg(feature = "cuda")]
impl<T> GPUTensor<T>
where
    T: GPUFloat,
{
    // CPU activation functions for fallback
    fn relu_cpu(&self) -> GPUTensor<T> {
        Self::new_with_device(
            self.data.mapv(|x| {
                let zero = <T as CPUNumber>::zero();
                if x > zero { x } else { zero }
            }),
            self.device.clone(),
        )
    }

    fn exp_cpu(&self) -> GPUTensor<T> {
        Self::new_with_device(self.data.mapv(|x| x.exp()), self.device.clone())
    }

    // Smart activation functions
    pub fn relu(&self) -> GPUTensor<T> {
        match &self.device {
            Device::CPU => self.relu_cpu(),
            Device::CUDA(_) => self.relu_cuda().unwrap_or_else(|_| {
                println!("CUDA relu failed, falling back to CPU");
                self.relu_cpu()
            }),
        }
    }

    pub fn exp(&self) -> GPUTensor<T> {
        match &self.device {
            Device::CPU => self.exp_cpu(),
            Device::CUDA(_) => self.exp_cuda().unwrap_or_else(|_| {
                println!("CUDA exp failed, falling back to CPU");
                self.exp_cpu()
            }),
        }
    }

    // Sigmoid activation function
    pub fn sigmoid_cpu(&self) -> GPUTensor<T> {
        Self::new_with_device(
            self.data.mapv(|x| {
                let one = <T as CPUNumber>::one();
                let neg_x = -x;
                one / (one + neg_x.exp())
            }),
            self.device.clone(),
        )
    }

    // Sigmoid activation function with CUDA support and fallback
    pub fn sigmoid(&self) -> GPUTensor<T> {
        match &self.device {
            Device::CPU => self.sigmoid_cpu(),
            Device::CUDA(_) => self.sigmoid_cuda().unwrap_or_else(|_| {
                println!("CUDA sigmoid failed, falling back to CPU");
                self.sigmoid_cpu()
            }),
        }
    }

    /// Element-wise power operation with CUDA support
    /// No unwrapping here, we handle errors gracefully
    pub fn powf(&self, other: &Self) -> Result<GPUTensor<T>, String> {
        // Ensure tensors are on the same device
        if self.device != other.device {
            return Err("Power operation requires tensors on the same device".to_string());
        }

        match &self.device {
            Device::CPU => self.powf_cpu(other),
            Device::CUDA(_) => {
                if let Ok(result) = self.powf_cuda(other) {
                    Ok(result)
                } else {
                    println!("CUDA power operation failed, falling back to CPU");
                    self.powf_cpu(other)
                }
            }
        }
    }

    /// Scalar power operation with CUDA support
    pub fn power_scalar(&self, scalar: T) -> GPUTensor<T> {
        match &self.device {
            Device::CPU => self.power_scalar_cpu(scalar),
            Device::CUDA(_) => self.power_scalar_cuda(scalar).unwrap_or_else(|_| {
                println!("CUDA scalar power failed, falling back to CPU");
                self.power_scalar_cpu(scalar)
            }),
        }
    }

    // CPU implementations
    fn powf_cpu(&self, other: &Self) -> Result<GPUTensor<T>, String> {
        if self.shape() != other.shape() {
            return Err(format!(
                "Shape mismatch for power operation: {:?} vs {:?}",
                self.shape(),
                other.shape()
            ));
        }

        Ok(Self::new_with_device(
            ndarray::Zip::from(&self.data)
                .and(&other.data)
                .map_collect(|&a, &b| a.powf(b)),
            self.device.clone(),
        ))
    }

    fn power_scalar_cpu(&self, scalar: T) -> GPUTensor<T> {
        Self::new_with_device(self.data.mapv(|x| x.powf(scalar)), self.device.clone())
    }

    /// Natural logarithm with CUDA support and CPU fallback
    pub fn log(&self) -> GPUTensor<T> {
        match &self.device {
            Device::CPU => self.log_cpu(),
            Device::CUDA(_) => self.log_cuda().unwrap_or_else(|_| {
                println!("CUDA log failed, falling back to CPU");
                self.log_cpu()
            }),
        }
    }

    /// Hyperbolic tangent with CUDA support and CPU fallback  
    pub fn tanh(&self) -> GPUTensor<T> {
        match &self.device {
            Device::CPU => self.tanh_cpu(),
            Device::CUDA(_) => self.tanh_cuda().unwrap_or_else(|_| {
                println!("CUDA tanh failed, falling back to CPU");
                self.tanh_cpu()
            }),
        }
    }

    // CPU log implementation for fallback
    fn log_cpu(&self) -> GPUTensor<T> {
        Self::new_with_device(self.data.mapv(|x| x.ln()), self.device.clone())
    }

    fn tanh_cpu(&self) -> GPUTensor<T> {
        Self::new_with_device(
            self.data.mapv(|x| {
                let e_x = x.exp();
                let e_neg_x = (-x).exp();
                (e_x - e_neg_x) / (e_x + e_neg_x)
            }),
            self.device.clone(),
        )
    }

    // ------------------------------------------------------------
    // CUDA - BASED ACTIVATION FUNCTIONS
    // -------------------------------------------------------------
    pub fn relu_cuda(&self) -> Result<GPUTensor<T>, String> {
        self.with_cuda_backend(|cuda_backend| {
             // Ensure both tensors are on CUDA
             if !self.is_cuda() {
                return Err("Tensor must be on CUDA for relu operation".to_string());
            }

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.relu(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    pub fn exp_cuda(&self) -> Result<GPUTensor<T>, String> {
        self.with_cuda_backend(|cuda_backend| {
             // Ensure both tensors are on CUDA
             if !self.is_cuda() {
                return Err("Tensor must be on CUDA for exp operation".to_string());
            }

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.exp(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    fn powf_cuda(&self, other: &Self) -> Result<GPUTensor<T>, String> {
        self.with_cuda_backend(|cuda_backend| {
             // Ensure both tensors are on CUDA
             if !self.is_cuda() {
                return Err("Tensor must be on CUDA for pow operation".to_string());
            }

        // Convert both tensors to CUDA
        let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.power(&cuda_a, &cuda_b)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    fn power_scalar_cuda(&self, scalar: T) -> Result<GPUTensor<T>, String> {
        self.with_cuda_backend(|cuda_backend| {
             // Ensure both tensors are on CUDA
             if !self.is_cuda() {
                return Err("Tensor must be on CUDA for power scalar operation".to_string());
            }

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.power_scalar(&cuda_tensor, scalar)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    fn log_cuda(&self) -> Result<GPUTensor<T>, String> {
        self.with_cuda_backend(|cuda_backend| {
             // Ensure both tensors are on CUDA
             if !self.is_cuda() {
                return Err("Tensor must be on CUDA for log operation".to_string());
            }

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.log(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    fn tanh_cuda(&self) -> Result<GPUTensor<T>, String> {
        self.with_cuda_backend(|cuda_backend| {
             // Ensure both tensors are on CUDA
             if !self.is_cuda() {
                return Err("Tensor must be on CUDA for tanh operation".to_string());
            }

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.tanh(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    fn sigmoid_cuda(&self) -> Result<GPUTensor<T>, String> {
        self.with_cuda_backend(|cuda_backend| {
        // Ensure tensor is on CUDA
        if !self.is_cuda() {
            return Err("Tensor must be on CUDA for sigmoid operation".to_string());
        }

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.sigmoid(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)

        })
    }

    // ------------------------------------------------------------
    // SQRT OPS
    // -------------------------------------------------------------

    /// Element-wise square root
    /// Supports both CPU and CUDA execution
    /// Validates that all values are non-negative
    pub fn sqrt(&self) -> Result<Self, String> {
        match &self.device {
            Device::CPU => self.sqrt_cpu(),
            Device::CUDA(_) => match self.sqrt_cuda() {
                Ok(result) => Ok(result),
                Err(err) => {
                    println!("CUDA sqrt failed ({}), falling back to CPU", err);
                    self.sqrt_cpu()
                }
            },
        }
    }

    // CPU fallback implementations
    fn sqrt_cpu(&self) -> Result<Self, String> {
        // Check for negative values first
        let has_negative = self.data.iter().any(|&x| x < <T as CPUFloat>::zero());
        if has_negative {
            return Err("Cannot compute square root of negative values".to_string());
        }

        let result_data: Vec<T> = self.data.iter().map(|&x| x.sqrt()).collect();

        let result_array = ndarray::Array::from_shape_vec(self.data.raw_dim(), result_data)
            .map_err(|e| format!("Failed to create result tensor: {}", e))?;

        Ok(Self::new_with_device(result_array, self.device.clone()))
    }

    // CUDA implementations
    fn sqrt_cuda(&self) -> Result<Self, String> {
        self.with_cuda_backend(|cuda_backend| {
        
        // Ensure tensor is on CUDA
        if !self.is_cuda() {
            return Err("Tensor must be on CUDA for sqrt operation".to_string());
        }
        // Get or create CUDA tensor
        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.sqrt(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }
}

// Additional utility methods for better testing support
#[cfg(feature = "cuda")]
impl<T> GPUTensor<T>
where
    T: GPUNumber,
{
    // Method to access data field for tests (similar to the errors we saw)
    pub fn data(&self) -> &ArrayD<T> {
        &self.data
    }

    // Negate operation for GPUTensor
    fn negate_cpu(&self) -> GPUTensor<T> {
        Self::new_with_device(self.data.mapv(|x| -x), self.device.clone())
    }

    fn negate_cuda(&self) -> Result<GPUTensor<T>, String> {
        self.with_cuda_backend(|cuda_backend| {
        // Ensure tensor is on CUDA
        if !self.is_cuda() {
            return Err("Tensor must be on CUDA for negate operation".to_string());
        }

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.negate(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
        })
    }

    /// Smart negate operation
    pub fn negate(&self) -> GPUTensor<T> {
        match &self.device {
            Device::CPU => self.negate_cpu(),
            Device::CUDA(_) => self.negate_cuda().unwrap_or_else(|_| {
                println!("CUDA negate failed, falling back to CPU");
                self.negate_cpu()
            }),
        }
    }
}

// PartialEq implementation to fix comparison errors in tests
#[cfg(feature = "cuda")]
impl<T> PartialEq for GPUTensor<T>
where
    T: GPUNumber + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        // Compare shapes first
        if self.shape() != other.shape() || self.device != other.device {
            return false;
        }

        // Not the best implementation as we should compare the CUDA storage
        // But for now, we compare the data directly
        self.data == other.data
    }
}

// Add indexing support for GPUTensor
#[cfg(feature = "cuda")]
impl<T> Index<usize> for GPUTensor<T>
where
    T: GPUNumber,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.size() {
            panic!(
                "Index {} out of bounds for tensor with {} elements",
                index,
                self.size()
            );
        }

        // Convert flat index to multi-dimensional coordinates
        let shape = self.data.shape();
        let mut coords = vec![0; shape.len()];
        let mut remaining = index;

        for i in (0..shape.len()).rev() {
            coords[i] = remaining % shape[i];
            remaining /= shape[i];
        }

        // Access using multi-dimensional indexing
        // Aparently this t¡is not the safest way to do this, but it is the fastest.
        // This returns a reference to the element at the specified index.
        &self.data[&coords[..]]
    }
}
