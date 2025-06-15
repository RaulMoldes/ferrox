use crate::backend::numeric::{Float, Numeric};
use crate::backend::{Device, default_device};
use ndarray::{Array, ArrayD, Axis, IxDyn};
use std::ops::{Index, IndexMut};
use crate::backend::manager::get_backend;

#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaTensor;



// Tensor wrapper to handle dynamic arrays more elegantly
#[derive(Debug, Clone)]
pub struct Tensor<T>
where
    T: Numeric + Clone,
{   

    // This `data` field is the main data storage of the tensor on CPU.
    pub data: ArrayD<T>, // As I documented in the device module, this will be changed toa generic type <T>
    // This way I will be able to use different data types in the future.
    // For now, we will keep it as f64 for simplicity.
    pub device: Device,

    // Optional CUDA storage for GPU tensors
    #[cfg(feature = "cuda")]
    cuda_storage: Option<crate::backend::cuda::CudaTensor<T>>,
}

// Main implementation block with basic operations
impl<T> Tensor<T>
where
    T: Numeric + Clone,
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
            #[cfg(feature = "cuda")]
            cuda_storage: None,
        }
    }

    // Creates a new tensor with the given data and device.
    // This is useful when you want to create a tensor with a specific device, for example, when you want to use a GPU.
    // In the future, we could also add a method to create a tensor with a specific data type, but for now we will keep it simple.
    // The device is set to the default device if not provided.
    // This is similar to how PyTorch and TensorFlow work, where the device is set to the default device if not specified.
    // Ideally,we should not be bound to ndarray backend here because it defaults to CPU, but it is okay for now as i prefer to focus more on the automatic differentiation engine thing.
    pub fn new_with_device(data: ArrayD<T>, device: Device) -> Self {
        Self { data, device,
            #[cfg(feature = "cuda")]
            cuda_storage: None, }
    }

    // Creates a new tensor with the given data and automatically selects the best device.
    pub fn new_auto(data: ArrayD<T>) -> Result<Self, String> {
        let backend = get_backend();
        let best_device = backend.best_device();
        
        let mut tensor = Self::new_with_device(data, best_device.clone());
        
        // If CUDA is available and best device is CUDA, move data there
        #[cfg(feature = "cuda")]
        {
            if let Device::CUDA(_) = best_device {
                if let Some(cuda_backend) = backend.cuda_backend() {
                    tensor = tensor.to_cuda(cuda_backend)?;
                }
            }
        }
        
        Ok(tensor)
    }


    /// Get the underlying CUDA tensor if available
    #[cfg(feature = "cuda")]
    pub fn cuda_tensor(&self) -> Option<&crate::backend::cuda::CudaTensor<T>> {
        self.cuda_storage.as_ref()
    }


    // Random numbers
    pub fn randn(shape: &[usize]) -> Self {
        let device = default_device();
        let data_f64 = device.randn(shape);
        let data = data_f64.mapv(|x| T::from_f64(x).unwrap());
        Self { data, device }
    }

    pub fn randn_with_device(shape: &[usize], device: Device) -> Self {
        // Generates a tensor with random numbers from a normal distribution.
        let data_f64 = device.randn(shape);
        let data = data_f64.mapv(|x| T::from_f64(x).unwrap());
        Self { data, device }
    }

    // Random numbers
    pub fn randint(shape: &[usize]) -> Self {
        let device = default_device();
        let data_i64 = device.randint(shape);
        let data = data_i64.mapv(|x| T::from_i64(x).unwrap());
        Self { data, device }
    }

    pub fn randint_with_device(shape: &[usize], device: Device) -> Self {
        // Generates a tensor with random integer numbers.
        let data_i64 = device.randint(shape);
        let data = data_i64.mapv(|x| T::from_i64(x).unwrap());
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

    /// Auto from_vec - uses best device
    pub fn from_vec_auto(data: Vec<T>, shape: &[usize]) -> Result<Self, String> {
        let cpu_tensor = Self::from_vec(data, shape)?;
        Self::new_auto(cpu_tensor.data)
    }

    // I decided not to implement the empty() function as it is useless in practice.
    // The empty function in numpy creates an uninitialized array, which is unsafe in Rust.
    // Instead, we will use the zeros() function to create a tensor with zeroes.
    // If you want to use uninitialized arrays, you can use `Array::uninit` but it is unsafe.

    // Some utility functions to get information about the tensor.
    // These functions are similar to the ones in PyTorch and TensorFlow, and they return the shape, number of dimensions, length, data, and device of the tensor.
    pub fn shape(&self) -> &[usize] {
        #[cfg(feature = "cuda")]
        {
            if let Some(cuda_tensor) = &self.cuda_storage {
                return cuda_tensor.shape();
            }
        }
        self.data.shape()
    }

    pub fn ndim(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            if let Some(cuda_tensor) = &self.cuda_storage {
                return cuda_tensor.ndim();
            }
        }
        self.data.ndim()
    }

    pub fn size(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            if let Some(cuda_tensor) = &self.cuda_storage {
                return cuda_tensor.size();
            }
        }
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
    pub fn add_cpu(&self, other: &Tensor<T>) -> Result<Tensor<T>, String> {
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

    // Smart addition function that checks the device of the tensors and calls the appropriate method.
    pub fn add(&self, other: &Tensor<T>) -> Result<Tensor<T>, String> {
        // Check device compatibility
        if self.device != other.device {
            return Err("Tensors must be on the same device for operation".to_string());
        }

        match self.device {
            Device::CPU => self.add_cpu(other),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.add_cuda(other),
            _ => Err("Unsupported device for addition".to_string()),
        }
    }

    pub fn mul_cpu(&self, other: &Tensor<T>) -> Result<Tensor<T>, String> {
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
    /// Smart multiplication
    pub fn mul(&self, other: &Tensor<T>) -> Result<Tensor<T>, String> {
        if self.device != other.device {
            return Err("Tensors must be on the same device for operation".to_string());
        }

        match self.device {
            Device::CPU => self.mul_cpu(other),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.mul_cuda(other),
            
        }
    }

    

    pub fn matmul_cpu(&self, other: &Tensor<T>) -> Result<Tensor<T>, String>
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

        Ok(Tensor::new_with_device(
            result.into_dyn(),
            self.device.clone(),
        ))
    }

    pub fn matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>, String>
    where
        T: Clone + ndarray::LinalgScalar,
    {
        #[cfg(feature = "cuda")]
        {
            if self.is_cuda() && other.is_cuda() {
                return self.cuda_tensor().unwrap().matmul(other.cuda_tensor().unwrap());
            }
        }
        self.matmul_cpu(other)
    }

    pub fn negate(&self) -> Tensor<T> {
        Tensor::new_with_device(self.data.mapv(|x| -x), self.device.clone())
    }

    pub fn div_cpu(&self, other: &Tensor<T>) -> Result<Tensor<T>, String> {
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
    
    // I think this repetition could be handled better via a macro but is not a priority right now.
    /// Smart division
    pub fn div(&self, other: &Tensor<T>) -> Result<Tensor<T>, String> {
        if self.device != other.device {
            return Err("Tensors must be on the same device for operation".to_string());
        }

        match self.device {
            Device::CPU => self.div_cpu(other),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.div_cuda(other),
           
        }
    }

    // Detach operation - creates a new tensor that shares data but detaches from graph
    // Need to check if this is the right way to do it.
    // In Pytorch i think the detach operation sets the requires_grad flag to false, but we don't have that concept at the tensor level.
    // We can just return a new tensor with the same data and device, but without any gradient tracking.
    pub fn detach(&self) -> Tensor<T> {
        Tensor::new_with_device(self.data.clone(), self.device.clone())
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
    pub fn to_vec(&self) -> Vec<T> {
        #[cfg(feature = "cuda")]
        {
            if let Some(cuda_tensor) = &self.cuda_storage {
                return cuda_tensor.to_vec();
            }
        }
        self.data.iter().copied().collect()
    }


    /// ------------------------------------------------------------
    /// CUDA - BASED ARITHMETIC OPERATIONS
    /// -------------------------------------------------------------
    
    #[cfg(feature = "cuda")]
    pub fn add_cuda(&self, other: &Tensor<T>) -> Result<Tensor<T>, String> {
        use crate::backend::manager::get_backend;
        
        let backend = get_backend();
        let cuda_backend = backend.cuda_backend()
            .ok_or("CUDA backend not available")?;

        // Convert tensors to CUDA if needed
        let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

        // Perform CUDA operation
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.add(&cuda_a, &cuda_b)?;

        // Create result tensor with CUDA storage
        self.create_tensor_from_cuda_result(result_cuda)
    }

    #[cfg(feature = "cuda")]
    pub fn mul_cuda(&self, other: &Tensor<T>) -> Result<Tensor<T>, String> {
        use crate::backend::manager::get_backend;
        
        let backend = get_backend();
        let cuda_backend = backend.cuda_backend()
            .ok_or("CUDA backend not available")?;

        let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.mul(&cuda_a, &cuda_b)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

    #[cfg(feature = "cuda")]
    pub fn div_cuda(&self, other: &Tensor<T>) -> Result<Tensor<T>, String> {
        use crate::backend::manager::get_backend;
        
        let backend = get_backend();
        let cuda_backend = backend.cuda_backend()
            .ok_or("CUDA backend not available")?;

        let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.div(&cuda_a, &cuda_b)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

    // Helper methods
    #[cfg(feature = "cuda")]
    fn get_or_create_cuda_tensor(&self, cuda_backend: &crate::backend::cuda::CudaBackend) -> Result<crate::backend::cuda::CudaTensor<T>, String> {
        match &self.cuda_storage {
            Some(cuda_tensor) => Ok(cuda_tensor.clone()),
            None => {
                let storage = TensorStorage::Cpu(self.data.clone());
                storage.to_cuda(cuda_backend)
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn create_tensor_from_cuda_result(&self, cuda_result: crate::backend::cuda::CudaTensor<T>) -> Result<Tensor<T>, String> {
        let result_cpu = cuda_result.to_cpu()?;
        let mut result_tensor = Tensor::new_with_device(result_cpu, Device::CUDA(0));
        result_tensor.cuda_storage = Some(cuda_result);
        Ok(result_tensor)
    }
}

// Implementation for operations requiring ScalarOperand
impl<T> Tensor<T>
where
    T: Numeric + Clone + ndarray::ScalarOperand,
{
    // Scalar operations
    pub fn add_scalar_cpu(&self, scalar: T) -> Tensor<T> {
        Tensor::new_with_device(&self.data + scalar, self.device.clone())
    }

    pub fn mul_scalar_cpu(&self, scalar: T) -> Tensor<T> {
        Tensor::new_with_device(&self.data * scalar, self.device.clone())
    }

    pub fn div_scalar_cpu(&self, scalar: T) -> Tensor<T> {
        Tensor::new_with_device(&self.data / scalar, self.device.clone())
    }

    // ------------------------------------------
    // CUDA-based scalar operations
    // ------------------------------------------


    #[cfg(feature = "cuda")]
    pub fn add_scalar_cuda(&self, scalar: T) -> Result<Tensor<T>, String> {
        use crate::backend::manager::get_backend;
        
        let backend = get_backend();
        let cuda_backend = backend.cuda_backend()
            .ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.add_scalar(&cuda_tensor, scalar)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

    #[cfg(feature = "cuda")]
    pub fn mul_scalar_cuda(&self, scalar: T) -> Result<Tensor<T>, String> {
        use crate::backend::manager::get_backend;
        
        let backend = get_backend();
        let cuda_backend = backend.cuda_backend()
            .ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.mul_scalar(&cuda_tensor, scalar)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

    #[cfg(feature = "cuda")]
    pub fn div_scalar_cuda(&self, scalar: T) -> Result<Tensor<T>, String> {
        use crate::backend::manager::get_backend;
        
        let backend = get_backend();
        let cuda_backend = backend.cuda_backend()
            .ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.div_scalar(&cuda_tensor, scalar)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }


     // Smart scalar operations with automatic device selection
     pub fn add_scalar(&self, scalar: T) -> Tensor<T> {
        match self.device {
            Device::CPU => self.add_scalar_cpu(scalar),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.add_scalar_cuda(scalar).unwrap_or_else(|_| {
                // Fallback to CPU if CUDA fails
                self.add_scalar_cpu(scalar)
            })
        }
    }

    pub fn mul_scalar(&self, scalar: T) -> Tensor<T> {
        match self.device {
            Device::CPU => self.mul_scalar_cpu(scalar),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.mul_scalar_cuda(scalar).unwrap_or_else(|_| {
                self.mul_scalar_cpu(scalar)
            })
        }
    }

    pub fn div_scalar(&self, scalar: T) -> Tensor<T> {
        match self.device {
            Device::CPU => self.div_scalar_cpu(scalar),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.div_scalar_cuda(scalar).unwrap_or_else(|_| {
                self.div_scalar_cpu(scalar)
            })
           
        }
    }
}

// Implementation for floating-point operations
impl<T> Tensor<T>
where
    T: Float + Clone + ndarray::ScalarOperand,
{
    pub fn sigmoid(&self) -> Tensor<T> {
        // Euler's sigmoid function: 1 / (1 + exp(-x))
        Tensor::new_with_device(
            self.data.mapv(|x| {
                let one = T::one();
                let neg_x = -x;
                one / (one + neg_x.exp())
            }),
            self.device.clone(),
        )
    }

    // Activation functions
    pub fn relu_cpu(&self) -> Tensor<T> {
        // ReLU activation function: max(0, x)
        Tensor::new_with_device(
            self.data.mapv(|x| {
                let zero = T::zero();
                if x > zero { x } else { zero }
            }),
            self.device.clone(),
        )
    }

    // Additional activation functions
    pub fn exp_cpu(&self) -> Tensor<T> {
        Tensor::new_with_device(self.data.mapv(|x| x.exp()), self.device.clone())
    }


    pub fn log(&self) -> Tensor<T> {
        Tensor::new_with_device(self.data.mapv(|x| x.ln()), self.device.clone())
    }

    pub fn powf(&self, other: &Tensor<T>) -> Result<Tensor<T>, String> {
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

    pub fn power_scalar(&self, scalar: T) -> Tensor<T> {
        Tensor::new_with_device(self.data.mapv(|x| x.powf(scalar)), self.device.clone())
    }


    pub fn relu(&self) -> Tensor<T> {
        match self.device {
            Device::CPU => self.relu_cpu(),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.relu_cuda().unwrap_or_else(|_| {
                self.relu_cpu()
            }),
            
        }
    }

    pub fn exp(&self) -> Tensor<T> {
        match self.device {
            Device::CPU => self.exp_cpu(),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.exp_cuda().unwrap_or_else(|_| {
                self.exp_cpu()
            })
        }
    }


    // ------------------------------------------------------------
    // CUDA - BASED ACTIVATION FUNCTIONS
    // -------------------------------------------------------------
    #[cfg(feature = "cuda")]
    pub fn relu_cuda(&self) -> Result<Tensor<T>, String> {
        use crate::backend::manager::get_backend;
        
        let backend = get_backend();
        let cuda_backend = backend.cuda_backend()
            .ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.relu(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

    #[cfg(feature = "cuda")]
    pub fn exp_cuda(&self) -> Result<Tensor<T>, String> {
        use crate::backend::manager::get_backend;
        
        let backend = get_backend();
        let cuda_backend = backend.cuda_backend()
            .ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.exp(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

}

// Implementation for reduction operations and tensor manipulations
impl<T> Tensor<T>
where
    T: Numeric + Clone + rand_distr::num_traits::Zero + rand_distr::num_traits::FromPrimitive,
{
    // Reduction operations.
    // As we do not know the shape of the tensor at compile time, we use `ndarray`'s dynamic arrays.
    // We can sum or mean over a specific axis or all elements, it is up to the user to provide the axis over which to perform the reduction operation.
    pub fn sum(&self, axis: Option<usize>) -> Tensor<T> {
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
    pub fn sum_axes(&self, axes: Option<&[usize]>) -> Tensor<T> {
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

    pub fn mean(&self, axis: Option<usize>) -> Tensor<T> {
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
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Tensor<T>, String> {
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
    pub fn unsqueeze(&self, axis: usize) -> Tensor<T> {
        let expanded = self.data.clone().insert_axis(Axis(axis));
        Tensor::new_with_device(expanded, self.device.clone())
    }

    // Basically the opposite of unsqueeze, this function removes a dimension of size 1 from the tensor.
    // We need to check the size of the axis before removing it, as it is not possible to remove an axis with size greater than 1.
    // Imagine a tensor: [[[1, 3, 1, 5],[1,2,3,4]],[[1, 3, 1, 5],[1,2,3,4]],] if we try to squeeze axis 1, we would need to remove the two elements on that axis,
    // which is not the purpose of the squeeze operation.
    pub fn squeeze(&self, axis: Option<usize>) -> Result<Tensor<T>, String> {
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
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Tensor<T>, String> {
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
    pub fn transpose(&self, axes: Option<&[usize]>) -> Result<Tensor<T>, String> {
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
}

// Implementation for tensor creation with Zero trait
impl<T> Tensor<T>
where
    T: Numeric + Clone + rand_distr::num_traits::Zero,
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

    /// Auto zeros - uses best device
    pub fn zeros_auto(shape: &[usize]) -> Result<Self, String> {
        let data = default_device().zeros(shape);
        Self::new_auto(data)
    }

    pub fn zeros_with_device(shape: &[usize], device: Device) -> Self {
        Self {
            data: device.zeros(shape),
            device,
        }
    }
}


impl<T> Tensor<T>
where
    T: Numeric + Clone,
{
    /// Check if tensor is on CUDA
    pub fn is_cuda(&self) -> bool {
        self.device.is_cuda()
    }

    /// Move tensor to CPU
    pub fn to_cpu(&self) -> Result<Self, String> {
        if !self.is_cuda() {
            return Ok(self.clone());
        }

        #[cfg(feature = "cuda")]
        {
            if let Some(cuda_tensor) = &self.cuda_storage {
                let host_data = cuda_tensor.to_vec()?;
                let cpu_array = ArrayD::from_shape_vec(
                    IxDyn(cuda_tensor.shape()),
                    host_data
                ).map_err(|e| format!("Failed to create CPU array: {}", e))?;

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

    /// Move tensor to CUDA
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&self) -> Result<Self, String> {
        if self.is_cuda() {
            return Ok(self.clone());
        }

        use crate::backend::manager::get_backend;
        
        let backend = get_backend();
        let cuda_backend = backend.cuda_backend()
            .ok_or("CUDA backend not available")?;

        let shape = self.shape();
        let host_data: Vec<T> = self.data.iter().cloned().collect();
        let cuda_tensor = CudaTensor::from_vec(
            cuda_backend.memory_manager(),
            host_data,
            shape
        )?;

        Ok(Self {
            data: self.data.clone(), // Keep CPU copy for compatibility
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
}

// Implementation for tensor creation with One trait
impl<T> Tensor<T>
where
    T: Numeric + Clone + rand_distr::num_traits::One,
{
    // Ones
    pub fn ones(shape: &[usize]) -> Self {
        let device = default_device();
        Self {
            data: device.ones(shape),
            device,
        }
    }

    /// Auto ones - uses best device
    pub fn ones_auto(shape: &[usize]) -> Result<Self, String> {
        let data = default_device().ones(shape);
        Self::new_auto(data)
    }

    pub fn ones_with_device(shape: &[usize], device: Device) -> Self {
        Self {
            data: device.ones(shape),
            device,
        }
    }
}

// Implement equality for testing, and because will be useful in the future.
impl<T> PartialEq for Tensor<T>
where
    T: Numeric + Clone + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.device == other.device
    }
}

impl<T> Eq for Tensor<T> where T: Numeric + Clone + PartialEq {}

pub struct TensorIterator<T> {
    data: ndarray::ArrayD<T>,
    index: usize,
}

impl<T> Iterator for TensorIterator<T>
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

impl<T> ExactSizeIterator for TensorIterator<T>
where
    T: Copy,
{
    fn len(&self) -> usize {
        self.data.len().saturating_sub(self.index)
    }
}

// Implementation for owned Tensor (consumes the tensor)
impl<T> IntoIterator for Tensor<T>
where
    T: Numeric + Clone + Copy,
{
    type Item = T;
    type IntoIter = TensorIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        TensorIterator {
            data: self.data,
            index: 0,
        }
    }
}

// Implementation for borrowed Tensor (&Tensor)
impl<'a, T> IntoIterator for &'a Tensor<T>
where
    T: Numeric + Clone,
{
    type Item = &'a T;
    type IntoIter = ndarray::iter::Iter<'a, T, ndarray::IxDyn>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

// Implementation for mutable borrowed Tensor (&mut Tensor)
impl<'a, T> IntoIterator for &'a mut Tensor<T>
where
    T: Numeric + Clone,
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
impl<T> Index<usize> for Tensor<T>
where
    T: Numeric + Clone,
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

impl<T> IndexMut<usize> for Tensor<T>
where
    T: Numeric + Clone,
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
impl<T> Index<&[usize]> for Tensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize]> for Tensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

// Implementation for Vec<usize> (convenient alternative to slice)
impl<T> Index<Vec<usize>> for Tensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: Vec<usize>) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<Vec<usize>> for Tensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, indices: Vec<usize>) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

// Implementation for arrays of different sizes (up to 6D for common use cases)
impl<T> Index<[usize; 1]> for Tensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: [usize; 1]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<[usize; 1]> for Tensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, indices: [usize; 1]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

impl<T> Index<[usize; 2]> for Tensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: [usize; 2]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<[usize; 2]> for Tensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, indices: [usize; 2]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

impl<T> Index<[usize; 3]> for Tensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: [usize; 3]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<[usize; 3]> for Tensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, indices: [usize; 3]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

impl<T> Index<[usize; 4]> for Tensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: [usize; 4]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<[usize; 4]> for Tensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, indices: [usize; 4]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

// Implementation for tuples (more ergonomic for 2D and 3D)
impl<T> Index<(usize, usize)> for Tensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.data[[i, j]]
    }
}

impl<T> IndexMut<(usize, usize)> for Tensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.data[[i, j]]
    }
}

impl<T> Index<(usize, usize, usize)> for Tensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, (i, j, k): (usize, usize, usize)) -> &Self::Output {
        &self.data[[i, j, k]]
    }
}

impl<T> IndexMut<(usize, usize, usize)> for Tensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, (i, j, k): (usize, usize, usize)) -> &mut Self::Output {
        &mut self.data[[i, j, k]]
    }
}

impl<T> Index<(usize, usize, usize, usize)> for Tensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, (i, j, k, l): (usize, usize, usize, usize)) -> &Self::Output {
        &self.data[[i, j, k, l]]
    }
}

impl<T> IndexMut<(usize, usize, usize, usize)> for Tensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, (i, j, k, l): (usize, usize, usize, usize)) -> &mut Self::Output {
        &mut self.data[[i, j, k, l]]
    }
}

// Implementation for references to arrays of different sizes
impl<T> Index<&[usize; 1]> for Tensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize; 1]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize; 1]> for Tensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, indices: &[usize; 1]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

impl<T> Index<&[usize; 2]> for Tensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize; 2]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize; 2]> for Tensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, indices: &[usize; 2]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

impl<T> Index<&[usize; 3]> for Tensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize; 3]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize; 3]> for Tensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, indices: &[usize; 3]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

impl<T> Index<&[usize; 4]> for Tensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize; 4]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize; 4]> for Tensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, indices: &[usize; 4]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu;

    #[test]
    fn test_tensor_creation_with_device() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.len(), 4);
        assert_eq!(tensor.device(), &cpu());

        let zeros = Tensor::<f64>::zeros_with_device(&[3, 3], cpu());
        assert_eq!(zeros.shape(), &[3, 3]);
    }

    #[test]
    fn test_tensor_operations() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();

        let sum = a.add(&b).unwrap();
        let expected = Tensor::from_vec(vec![3.0, 5.0, 7.0, 9.0], &[2, 2]).unwrap();
        assert_eq!(sum, expected);

        let mul = a.mul(&b).unwrap();
        let expected_mul = Tensor::from_vec(vec![2.0, 6.0, 12.0, 20.0], &[2, 2]).unwrap();
        assert_eq!(mul, expected_mul);
    }

    #[test]
    fn test_tensor_matmul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();

        let result = a.matmul(&b).unwrap();
        let expected = Tensor::from_vec(vec![22.0, 28.0, 49.0, 64.0], &[2, 2]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_tensor_activations() {
        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

        // Test ReLU
        let relu_result = input.relu();
        let expected_relu = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        assert_eq!(relu_result, expected_relu);

        // Test Sigmoid (should be between 0 and 1)
        let sigmoid_result = input.sigmoid();
        for &val in sigmoid_result.data().iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }

        // Test Exp
        let exp_result = input.exp();
        assert!(exp_result.data().iter().all(|&x| x > 0.0));

        // Test Negate
        let neg_result = input.negate();
        let expected_neg = Tensor::from_vec(vec![2.0, 1.0, 0.0, -1.0, -2.0], &[5]).unwrap();
        assert_eq!(neg_result, expected_neg);
    }

    #[test]
    fn test_tensor_scalar_operations() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let add_scalar = tensor.add_scalar(5.0);
        let expected_add = Tensor::from_vec(vec![6.0, 7.0, 8.0, 9.0], &[2, 2]).unwrap();
        assert_eq!(add_scalar, expected_add);

        let mul_scalar = tensor.mul_scalar(2.0);
        let expected_mul = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[2, 2]).unwrap();
        assert_eq!(mul_scalar, expected_mul);

        let div_scalar = tensor.div_scalar(2.0);
        let expected_div = Tensor::from_vec(vec![0.5, 1.0, 1.5, 2.0], &[2, 2]).unwrap();
        assert_eq!(div_scalar, expected_div);
    }

    #[test]
    fn test_tensor_transpose_comprehensive() {
        // Test 2D transpose
        let tensor_2d = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Default transpose (should swap axes)
        let transposed_default = tensor_2d.transpose(None).unwrap();
        assert_eq!(transposed_default.shape(), &[3, 2]);

        // Explicit axes transpose
        let transposed_explicit = tensor_2d.transpose(Some(&[1, 0])).unwrap();
        assert_eq!(transposed_explicit.shape(), &[3, 2]);
        assert_eq!(transposed_default.data(), transposed_explicit.data());

        // Test 3D transpose
        let tensor_3d = Tensor::from_vec((0..24).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();

        // Default transpose (reverse all axes: [2,3,4] -> [4,3,2])
        let transposed_3d_default = tensor_3d.transpose(None).unwrap();
        assert_eq!(transposed_3d_default.shape(), &[4, 3, 2]);

        // Custom permutation: [2,3,4] -> [4,2,3] (axes [2,0,1])
        let transposed_3d_custom = tensor_3d.transpose(Some(&[2, 0, 1])).unwrap();
        assert_eq!(transposed_3d_custom.shape(), &[4, 2, 3]);

        // Test 1D and 0D tensors (should be unchanged)
        let tensor_1d = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let transposed_1d = tensor_1d.transpose(None).unwrap();
        assert_eq!(transposed_1d.shape(), &[3]);
        assert_eq!(transposed_1d.data(), tensor_1d.data());

        let tensor_0d = Tensor::from_vec(vec![42.0], &[]).unwrap();
        let transposed_0d = tensor_0d.transpose(None).unwrap();
        assert_eq!(transposed_0d.shape(), &[]);
        assert_eq!(transposed_0d.data(), tensor_0d.data());
    }

    #[test]
    fn test_transpose_error_cases() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Invalid axes length
        assert!(tensor.transpose(Some(&[0])).is_err());
        assert!(tensor.transpose(Some(&[0, 1, 2])).is_err());

        // Invalid axes values
        assert!(tensor.transpose(Some(&[0, 2])).is_err()); // 2 is out of bounds
        assert!(tensor.transpose(Some(&[0, 0])).is_err()); // duplicate axis
        assert!(tensor.transpose(Some(&[1, 1])).is_err()); // duplicate axis
    }

    #[test]
    fn test_tensor_broadcasting() {
        let a = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let target_shape = &[2, 3];

        let broadcasted = a.broadcast_to(target_shape).unwrap();
        let expected = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[2, 3]).unwrap();
        assert_eq!(broadcasted, expected);
    }

    #[test]
    fn test_tensor_squeeze_unsqueeze() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();

        // Test squeeze
        let squeezed = tensor.squeeze(Some(0)).unwrap();
        assert_eq!(squeezed.shape(), &[3]);

        // Test unsqueeze
        let unsqueezed = squeezed.unsqueeze(1);
        assert_eq!(unsqueezed.shape(), &[3, 1]);
    }

    #[test]
    fn test_tensor_error_handling() {
        // Test shape mismatch in addition
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        assert!(a.add(&b).is_err());

        // Test invalid matrix multiplication
        let c = Tensor::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
        let d = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();

        assert!(c.matmul(&d).is_err());
    }

    #[test]
    fn test_tensor_reductions() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Test sum
        let sum_all = tensor.sum(None);
        assert_eq!(sum_all.data().iter().next().unwrap(), &21.0);

        // Test mean
        let mean_all = tensor.mean(None);
        assert_eq!(mean_all.data().iter().next().unwrap(), &3.5);

        // Test sum along axis
        let sum_axis0 = tensor.sum(Some(0));
        assert_eq!(sum_axis0.shape(), &[3]);

        let sum_axis1 = tensor.sum(Some(1));
        assert_eq!(sum_axis1.shape(), &[2]);
    }
}
