use crate::backend::manager::get_backend;
use crate::backend::numeric::{Float, Numeric, NumericCuda};
use crate::backend::{Device, default_device};
use ndarray::{Array, ArrayD, Axis, IxDyn};

#[cfg(feature = "cuda")]
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

use std::ops::{Index, IndexMut};

#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaTensor;



// Or simply:
#[cfg(feature = "cuda")]
pub type Tensor<T> = GPUTensor<T>;

#[cfg(not(feature = "cuda"))]
pub type Tensor<T> = CPUTensor<T>;

// Tensor wrapper to handle dynamic arrays more elegantly
#[derive(Debug, Clone)]
pub struct CPUTensor<T>
where
    T: Numeric + Clone,
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
            
        }
    }

    // Creates a new tensor with the given data and device.
    // This is useful when you want to create a tensor with a specific device, for example, when you want to use a GPU.
    // In the future, we could also add a method to create a tensor with a specific data type, but for now we will keep it simple.
    // The device is set to the default device if not provided.
    // This is similar to how PyTorch and TensorFlow work, where the device is set to the default device if not specified.
    // Ideally,we should not be bound to ndarray backend here because it defaults to CPU, but it is okay for now as i prefer to focus more on the automatic differentiation engine thing.
    pub fn new_with_device(data: ArrayD<T>, device: Device) -> Self {
        Self {
            data,
            device,
           
        }
    }


    

    // Random numbers
    pub fn randn(shape: &[usize]) -> Self {
        let device = default_device();
        let data_f64 = device.randn(shape);
        let data = data_f64.mapv(|x| T::from_f64(x).unwrap());
        Self { data, device,
            #[cfg(feature = "cuda")] // Cuda storage is always initialized to None
            // As the tensor must be created on cpu first, we do not need to initialize it here.
        cuda_storage: None, }
    }

    pub fn randn_with_device(shape: &[usize], device: Device) -> Self {
        // Generates a tensor with random numbers from a normal distribution.
        let data_f64 = device.randn(shape);
        let data = data_f64.mapv(|x| T::from_f64(x).unwrap());
        Self {
            data,
            device,
          
        }
    }

    // Random numbers
    pub fn randint(shape: &[usize]) -> Self {
        let device = default_device();
        let data_i64 = device.randint(shape);
        let data = data_i64.mapv(|x| T::from_i64(x).unwrap());
        Self { data, device,
             }
    }

    pub fn randint_with_device(shape: &[usize], device: Device) -> Self {
        // Generates a tensor with random integer numbers.
        let data_i64 = device.randint(shape);
        let data = data_i64.mapv(|x| T::from_i64(x).unwrap());
        Self {
            data,
            device
        }
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

   

    
}


#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct GPUTensor<T>
where
   T: Numeric + Clone + DeviceRepr + ValidAsZeroBits + Unpin,
{
   data: ArrayD<T>,
   device: Device,
   cuda_storage: Option<CudaTensor<T>>,
}

#[cfg(feature = "cuda")]
impl<T> GPUTensor<T>
where
   T: Numeric + Clone + DeviceRepr + ValidAsZeroBits + Unpin,
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
           let memory_manager = cuda_backend.memory();
           
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
   fn add_cpu(&self, other: &Self) -> Result<Self, String> {
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
   fn mul_cpu(&self, other: &Self) -> Result<Self, String> {
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
   fn div_cpu(&self, other: &Self) -> Result<Self, String> {
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
   pub fn add(&self, other: &Self) -> Result<Self, String> {
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
   pub fn mul(&self, other: &Self) -> Result<Self, String> {
       if self.device != other.device {
           return Err("Tensors must be on the same device for operation".to_string());
       }

       match &self.device {
           Device::CPU => {
               self.mul_cpu(other)
           }
           Device::CUDA(_) => {
               self.mul_cuda(other).or_else(|_| {
                   println!("CUDA mul failed, falling back to CPU");
                   self.mul_cpu(other)
               })
           }
       }
   }

   // Smart division with automatic device selection
   pub fn div(&self, other: &Self) -> Result<Self, String> {
       if self.device != other.device {
           return Err("Tensors must be on the same device for operation".to_string());
       }

       match &self.device {
           Device::CPU => {
               self.div_cpu(other)
           }
           Device::CUDA(_) => {
               self.div_cuda(other).or_else(|_| {
                   println!("CUDA div failed, falling back to CPU");
                   self.div_cpu(other)
               })
           }
       }
   }

   // Element-wise negation - works on both CPU and CUDA
   pub fn negate(&self) -> Self {
       Self::new_with_device(self.data.mapv(|x| -x), self.device.clone())
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

   // The actual CUDA operations - these are already implemented in your existing code
   // I'm just referencing them here so the smart methods can call them
   fn add_cuda(&self, other: &Self) -> Result<Self, String> {
       use crate::backend::manager::get_backend;

       let backend = get_backend();
       let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

       let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
       let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

       let cuda_ops = cuda_backend.create_ops();
       let result_cuda = cuda_ops.add(&cuda_a, &cuda_b)?;

       self.create_tensor_from_cuda_result(result_cuda)
   }

   fn mul_cuda(&self, other: &Self) -> Result<Self, String> {
       use crate::backend::manager::get_backend;

       let backend = get_backend();
       let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

       let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
       let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

       let cuda_ops = cuda_backend.create_ops();
       let result_cuda = cuda_ops.mul(&cuda_a, &cuda_b)?;

       self.create_tensor_from_cuda_result(result_cuda)
   }

   fn div_cuda(&self, other: &Self) -> Result<Self, String> {
       use crate::backend::manager::get_backend;

       let backend = get_backend();
       let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

       let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
       let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

       let cuda_ops = cuda_backend.create_ops();
       let result_cuda = cuda_ops.div(&cuda_a, &cuda_b)?;

       self.create_tensor_from_cuda_result(result_cuda)
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
           let cuda_data = cuda_backend.memory().host_to_device(host_data)?;
           Ok(crate::backend::cuda::CudaTensor::new(cuda_data, shape))
       }
   }

   
 

   // Move tensor back to CPU
   pub fn to_cpu(&self) -> Result<Self, String> {
       if !self.is_cuda() {
           return Ok(self.clone());
       }

       if let Some(cuda_tensor) = &self.cuda_storage {
           let host_data = self.to_vec()?;
           let cpu_array = ArrayD::from_shape_vec(IxDyn(&cuda_tensor.shape), host_data)
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
}

#[cfg(feature = "cuda")]
impl<T> GPUTensor<T>
where
    T: Numeric + Clone + DeviceRepr + ValidAsZeroBits + Unpin + ndarray::LinalgScalar,
{
    fn matmul_cpu(&self, other: &Self) -> Result<Self, String> {
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

        Ok(Self::new_with_device(result.into_dyn(), self.device.clone()))
    }

    pub fn matmul(&self, other: &Self) -> Result<Self, String> {
        if self.device != other.device {
            return Err("Tensors must be on the same device for matrix multiplication".to_string());
        }

        match &self.device {
            Device::CPU => self.matmul_cpu(other),
            Device::CUDA(_) => {
                self.matmul_cuda(other).or_else(|_| {
                    println!("CUDA matmul failed, falling back to CPU");
                    self.matmul_cpu(other)
                })
            }
        }
    }

    fn matmul_cuda(&self, other: &Self) -> Result<Self, String> {
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
        let m = a_shape[0] as i32;  // rows of A
        let k = a_shape[1] as i32;  // cols of A / rows of B  
        let n = b_shape[1] as i32;  // cols of B
    
        // Get CUDA operations handle
        let cuda_ops = cuda_backend.create_ops();
        
        // Create result tensor on GPU
        let result_shape = vec![m as usize, n as usize];
        let mut result_cuda = crate::backend::cuda::CudaTensor::zeros(
            cuda_backend.memory(), 
            result_shape.clone()
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
            &cuda_a.data,      // input matrix A
            &cuda_b.data,      // input matrix B  
            &mut result_cuda.data, // output matrix C
            m,                 // rows of A
            n,                 // cols of B
            k,                 // inner dimension
        )?;
    
        // Convert the CUDA result back to GPUTensor
        self.create_tensor_from_cuda_result(result_cuda)
    }
}

#[cfg(feature = "cuda")]
impl<T> GPUTensor<T>
where
    T: Numeric + Clone + DeviceRepr + ValidAsZeroBits + Unpin + rand_distr::num_traits::Zero + rand_distr::num_traits::FromPrimitive,
{
    pub fn sum(&self, axis: Option<usize>) -> Self {
        match axis {
            Some(ax) => {
                let result = self.data.sum_axis(Axis(ax));
                Self::new_with_device(result, self.device.clone())
            }
            None => {
                let total_sum = self.data.sum();
                Self::new_with_device(
                    ArrayD::from_elem(IxDyn(&[]), total_sum),
                    self.device.clone(),
                )
            }
        }
    }

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
                Ok(Self::new_with_device(squeezed, self.device.clone()))
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

                Ok(Self::new_with_device(result, self.device.clone()))
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
                        // For 2D arrays, swap the two axes
                        let transposed = self.data.clone().reversed_axes();
                        Ok(Self::new_with_device(transposed, self.device.clone()))
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

    // Sum over multiple axes - useful for gradient computation
    pub fn sum_axes(&self, axes: Option<&[usize]>) -> Self {
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

                Self::new_with_device(result, self.device.clone())
            }
            None => self.sum(None),
        }
    }
}

// Zero initialization for GPU tensors
#[cfg(feature = "cuda")]
impl<T> GPUTensor<T>
where
   T: Numeric + Clone + DeviceRepr + ValidAsZeroBits + Unpin + rand_distr::num_traits::Zero,
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
   T: Numeric + Clone + DeviceRepr + ValidAsZeroBits + Unpin + rand_distr::num_traits::One,
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
    T: Numeric + Clone + DeviceRepr + ValidAsZeroBits + Unpin,
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
                let host_data = cuda_tensor.to_vec()?;
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

    #[cfg(feature = "cuda")]
    pub fn add(&self, other: &Tensor<T>) -> Result<Self, String> {
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

    #[cfg(feature = "cuda")]
    pub fn mul(&self, other: &Tensor<T>) -> Result<Tensor<T>, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.mul(&cuda_a, &cuda_b)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

    #[cfg(feature = "cuda")]
    pub fn div(&self, other: &Tensor<T>) -> Result<Self, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.div(&cuda_a, &cuda_b)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }


   

    /// CUDA-based matrix multiplication
    #[cfg(feature = "cuda")]
    pub fn matmul_cuda(&self, other: &Tensor<T>) -> Result<Self, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        // Get or create CUDA tensors for both operands
        let cuda_a = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_b = other.get_or_create_cuda_tensor(cuda_backend)?;

        // Perform matrix multiplication using CUDA operations
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.matmul(&cuda_a, &cuda_b)?;

        // Convert result back to Tensor and return
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
        let result_array = ArrayD::from_shape_vec(IxDyn(shape), result_cpu).map_err(|e| format!("Failed to create array from CUDA result: {}", e))?;
        let mut result_tensor = Tensor::new_with_device(result_array, Device::CUDA(0));
        result_tensor.cuda_storage = Some(cuda_result);
        Ok(result_tensor)
    }
}

#[cfg(feature = "cuda")]
impl<T> GPUTensor<T>
where
    T: Numeric + Clone + DeviceRepr + ValidAsZeroBits + Unpin + ndarray::ScalarOperand,
{
    // CPU scalar operations for fallback
    fn add_scalar_cpu(&self, scalar: T) -> Self {
        Self::new_with_device(&self.data + scalar, self.device.clone())
    }

    fn mul_scalar_cpu(&self, scalar: T) -> Self {
        Self::new_with_device(&self.data * scalar, self.device.clone())
    }

    fn div_scalar_cpu(&self, scalar: T) -> Self {
        Self::new_with_device(&self.data / scalar, self.device.clone())
    }

    // Smart scalar operations
    pub fn add_scalar(&self, scalar: T) -> Result<Self, String> {
        match &self.device {
            Device::CPU => Ok(self.add_scalar_cpu(scalar)),
            Device::CUDA(_) => {
                self.add_scalar_cuda(scalar).or_else(|_| {
                    println!("CUDA add_scalar failed, falling back to CPU");
                    Ok(self.add_scalar_cpu(scalar))
                })
            }
        }
    }

    pub fn mul_scalar(&self, scalar: T) -> Result<Self, String> {
        match &self.device {
            Device::CPU => Ok(self.mul_scalar_cpu(scalar)),
            Device::CUDA(_) => {
                self.mul_scalar_cuda(scalar).or_else(|_| {
                    println!("CUDA mul_scalar failed, falling back to CPU");
                    Ok(self.mul_scalar_cpu(scalar))
                })
            }
        }
    }

    pub fn div_scalar(&self, scalar: T) -> Result<Self, String> {
        match &self.device {
            Device::CPU => Ok(self.div_scalar_cpu(scalar)),
            Device::CUDA(_) => {
                self.div_scalar_cuda(scalar).or_else(|_| {
                    println!("CUDA div_scalar failed, falling back to CPU");
                    Ok(self.div_scalar_cpu(scalar))
                })
            }
        }
    }

     // ------------------------------------------
    // CUDA-based scalar operations
    // ------------------------------------------

    #[cfg(feature = "cuda")]
    pub fn add_scalar_cuda(&self, scalar: T) -> Result<Self, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.add_scalar(&cuda_tensor, scalar)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

    #[cfg(feature = "cuda")]
    pub fn mul_scalar_cuda(&self, scalar: T) -> Result<Self, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.mul_scalar(&cuda_tensor, scalar)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

    #[cfg(feature = "cuda")]
    pub fn div_scalar_cuda(&self, scalar: T) -> Result<Self, String> {
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
    T: Numeric + Clone + DeviceRepr + ValidAsZeroBits + Unpin + Float + ndarray::ScalarOperand,
{
    // CPU activation functions for fallback
    fn relu_cpu(&self) -> Self {
        Self::new_with_device(
            self.data.mapv(|x| {
                let zero = T::zero();
                if x > zero { x } else { zero }
            }),
            self.device.clone(),
        )
    }

    fn exp_cpu(&self) -> Self {
        Self::new_with_device(self.data.mapv(|x| x.exp()), self.device.clone())
    }

    // Smart activation functions
    pub fn relu(&self) -> Result<Self, String> {
        match &self.device {
            Device::CPU => Ok(self.relu_cpu()),
            Device::CUDA(_) => {
                self.relu_cuda().or_else(|_| {
                    println!("CUDA relu failed, falling back to CPU");
                    Ok(self.relu_cpu())
                })
            }
        }
    }

    pub fn exp(&self) -> Result<Self, String> {
        match &self.device {
            Device::CPU => Ok(self.exp_cpu()),
            Device::CUDA(_) => {
                self.exp_cuda().or_else(|_| {
                    println!("CUDA exp failed, falling back to CPU");
                    Ok(self.exp_cpu())
                })
            }
        }
    }

    // Other activation functions
    pub fn sigmoid(&self) -> Self {
        Self::new_with_device(
            self.data.mapv(|x| {
                let one = T::one();
                let neg_x = -x;
                one / (one + neg_x.exp())
            }),
            self.device.clone(),
        )
    }

    // ------------------------------------------------------------
    // CUDA - BASED ACTIVATION FUNCTIONS
    // -------------------------------------------------------------
    #[cfg(feature = "cuda")]
    pub fn relu_cuda(&self) -> Result<Self, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.relu(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

    #[cfg(feature = "cuda")]
    pub fn exp_cuda(&self) -> Result<Self, String> {
        use crate::backend::manager::get_backend;

        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;

        let cuda_tensor = self.get_or_create_cuda_tensor(cuda_backend)?;
        let cuda_ops = cuda_backend.ops();
        let result_cuda = cuda_ops.exp(&cuda_tensor)?;

        self.create_tensor_from_cuda_result(result_cuda)
    }

}

impl<T> CPUTensor<T>
where
    T: Numeric + Clone + ndarray::LinalgScalar,
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
    T: Numeric + Clone + ndarray::ScalarOperand,
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
    T: Float + Clone + ndarray::ScalarOperand,
{
    pub fn sigmoid(&self) -> CPUTensor<T> {
        // Euler's sigmoid function: 1 / (1 + exp(-x))
        CPUTensor::new_with_device(
            self.data.mapv(|x| {
                let one = T::one();
                let neg_x = -x;
                one / (one + neg_x.exp())
            }),
            self.device.clone(),
        )
    }

    // Activation functions
    pub fn relu(&self) -> CPUTensor<T> {
        // ReLU activation function: max(0, x)
        CPUTensor::new_with_device(
            self.data.mapv(|x| {
                let zero = T::zero();
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
    T: Numeric + Clone + rand_distr::num_traits::Zero + rand_distr::num_traits::FromPrimitive,
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
    pub fn unsqueeze(&self, axis: usize) -> CPUTensor<T> {
        let expanded = self.data.clone().insert_axis(Axis(axis));
        CPUTensor::new_with_device(expanded, self.device.clone())
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

    

    pub fn zeros_with_device(shape: &[usize], device: Device) -> Self {
        Self {
            data: device.zeros(shape),
            device,
        
        }
    }
}

impl<T> CPUTensor<T>
where
    T: Numeric + Clone,
{
    /// Check if tensor is on CUDA
    pub fn is_cuda(&self) -> bool {
        self.device.is_cuda()
    }
}



// Implementation for tensor creation with One trait
impl<T> CPUTensor<T>
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

   

    pub fn ones_with_device(shape: &[usize], device: Device) -> Self {
        Self {
            data: device.ones(shape),
            device,
            #[cfg(feature = "cuda")]
            cuda_storage: None,
        }
    }
}

// Implement equality for testing, and because will be useful in the future.
impl<T> PartialEq for CPUTensor<T>
where
    T: Numeric + Clone + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.device == other.device
    }
}

impl<T> Eq for CPUTensor<T> where T: Numeric + Clone + PartialEq {}

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
    T: Numeric + Clone + Copy,
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
    T: Numeric + Clone,
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
impl<T> Index<usize> for CPUTensor<T>
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

impl<T> IndexMut<usize> for CPUTensor<T>
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
impl<T> Index<&[usize]> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize]> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

// Implementation for Vec<usize> (convenient alternative to slice)
impl<T> Index<Vec<usize>> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: Vec<usize>) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<Vec<usize>> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, indices: Vec<usize>) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

// Implementation for arrays of different sizes (up to 6D for common use cases)
impl<T> Index<[usize; 1]> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: [usize; 1]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<[usize; 1]> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, indices: [usize; 1]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

impl<T> Index<[usize; 2]> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: [usize; 2]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<[usize; 2]> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, indices: [usize; 2]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

impl<T> Index<[usize; 3]> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: [usize; 3]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<[usize; 3]> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, indices: [usize; 3]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

impl<T> Index<[usize; 4]> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: [usize; 4]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<[usize; 4]> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, indices: [usize; 4]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

// Implementation for tuples (more ergonomic for 2D and 3D)
impl<T> Index<(usize, usize)> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.data[[i, j]]
    }
}

impl<T> IndexMut<(usize, usize)> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.data[[i, j]]
    }
}

impl<T> Index<(usize, usize, usize)> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, (i, j, k): (usize, usize, usize)) -> &Self::Output {
        &self.data[[i, j, k]]
    }
}

impl<T> IndexMut<(usize, usize, usize)> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, (i, j, k): (usize, usize, usize)) -> &mut Self::Output {
        &mut self.data[[i, j, k]]
    }
}

impl<T> Index<(usize, usize, usize, usize)> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, (i, j, k, l): (usize, usize, usize, usize)) -> &Self::Output {
        &self.data[[i, j, k, l]]
    }
}

impl<T> IndexMut<(usize, usize, usize, usize)> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, (i, j, k, l): (usize, usize, usize, usize)) -> &mut Self::Output {
        &mut self.data[[i, j, k, l]]
    }
}

// Implementation for references to arrays of different sizes
impl<T> Index<&[usize; 1]> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize; 1]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize; 1]> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, indices: &[usize; 1]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

impl<T> Index<&[usize; 2]> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize; 2]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize; 2]> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, indices: &[usize; 2]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

impl<T> Index<&[usize; 3]> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize; 3]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize; 3]> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    fn index_mut(&mut self, indices: &[usize; 3]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

impl<T> Index<&[usize; 4]> for CPUTensor<T>
where
    T: Numeric + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize; 4]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize; 4]> for CPUTensor<T>
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
