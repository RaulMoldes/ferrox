use crate::backend::manager::get_backend;
use crate::backend::number::{CPUNumber, Float, GPUNumber};
use crate::backend::{Device, default_device};
use ndarray::{Array, ArrayD, Axis, IxDyn};

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

    // Helper method to wrap CPU result back into GPUTensor
    // This is needed when we fall back to CPU operations
    fn wrap_cpu_result(&self, cpu_result: Self) -> Self {
        cpu_result
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
        let data = data_f64.mapv(|x| T::from_f64(x).unwrap());
        Self {
            data,
            device,
            cuda_storage: None,
        }
    }

    pub fn randn_with_device(shape: &[usize], device: Device) -> Self {
        let data_f64 = device.randn(shape);
        let data = data_f64.mapv(|x| T::from_f64(x).unwrap());
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

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        // Get or create CUDA tensors for both operands
        let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

        // Extract matrix dimensions
        let m = a_shape[0] as i32; // rows of A
        let k = a_shape[1] as i32; // cols of A / rows of B  
        let n = b_shape[1] as i32; // cols of B

        // Get CUDA operations handle
        let cuda_ops = cuda_backend.ops();

        // Create result tensor on GPU
        let result_shape = vec![m as usize, n as usize];
        let mut result_cuda = crate::backend::cuda::CudaTensor::zeros(
            &cuda_backend.memory_manager(),
            result_shape.clone(),
        )?;

        // Calculate optimal launch configuration for the kernel
        // For matmul, we typically use 2D thread blocks
        let block_size = 16; // 16x16 = 256 threads per block (good for most GPUs)
        let grid_x = (n + block_size - 1) / block_size as i32;
        let grid_y = (m + block_size - 1) / block_size as i32;

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (block_size as u32, block_size as u32, 1),
            shared_mem_bytes: 0,
        };

        // Launch the matmul kernel
        // The kernel should compute: C[i,j] = sum_k(A[i,k] * B[k,j])
        cuda_ops.kernels().launch_matmul(
            cfg,
            &cuda_a.data,          // input matrix A
            &cuda_b.data,          // input matrix B
            &mut result_cuda.data, // output matrix C
            m,                     // rows of A
            n,                     // cols of B
            k,                     // inner dimension
        )?;

        // Convert the CUDA result back to GPUTensor
        self.create_tensor_from_cuda_result(result_cuda)
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
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();

        let result_cuda = match axis {
            Some(ax) => cuda_ops.sum_axis(&cuda_tensor, ax, false)?,
            None => cuda_ops.sum_all(&cuda_tensor)?,
        };

        self.create_tensor_from_cuda_result(result_cuda)
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
    /// CUDA - BASED ARITHMETIC OPERATIONS
    /// -------------------------------------------------------------

    pub fn add_cuda(&self, other: &Tensor<T>) -> Result<Self, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        // Convert tensors to CUDA if needed
        let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

        // Perform CUDA operation
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.add(&cuda_a, &cuda_b)?;

        // Create result tensor with CUDA storage
        self.create_tensor_from_cuda_result(result_cuda)
    }

    pub fn mul_cuda(&self, other: &Tensor<T>) -> Result<Tensor<T>, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.mul(&cuda_a, &cuda_b)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

    pub fn div_cuda(&self, other: &Tensor<T>) -> Result<Self, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.div(&cuda_a, &cuda_b)?;

        self.create_tensor_from_cuda_result(result_cuda)
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
        let mut result_tensor = Tensor::new_with_device(result_array, Device::CUDA(0));
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
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.add_scalar(&cuda_tensor, scalar)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

    pub fn mul_scalar_cuda(&self, scalar: T) -> Result<GPUTensor<T>, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.mul_scalar(&cuda_tensor, scalar)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

    pub fn div_scalar_cuda(&self, scalar: T) -> Result<GPUTensor<T>, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.div_scalar(&cuda_tensor, scalar)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }
}

#[cfg(feature = "cuda")]
impl<T> GPUTensor<T>
where
    T: GPUNumber,
{
    // CPU activation functions for fallback
    fn relu_cpu(&self) -> GPUTensor<T> {
        Self::new_with_device(
            self.data.mapv(|x| {
                let zero = T::zero();
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
                let one = T::one();
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
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.relu(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

    pub fn exp_cuda(&self) -> Result<GPUTensor<T>, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.exp(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

    fn powf_cuda(&self, other: &Self) -> Result<GPUTensor<T>, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        // Convert both tensors to CUDA
        let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.power(&cuda_a, &cuda_b)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

    fn power_scalar_cuda(&self, scalar: T) -> Result<GPUTensor<T>, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.power_scalar(&cuda_tensor, scalar)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

    fn log_cuda(&self) -> Result<GPUTensor<T>, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.log(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

    fn tanh_cuda(&self) -> Result<GPUTensor<T>, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.tanh(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

    fn sigmoid_cuda(&self) -> Result<GPUTensor<T>, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.sigmoid(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
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
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.negate(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
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
