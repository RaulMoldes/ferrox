// src/backend/cuda/ops.rs
// This module contains abstarctions for CUDA tensor operations that mirror existent CPU tensor operations
use super::kernels::CudaKernels;
use super::memory::{CudaMemoryManager, CudaTensor};
use cudarc::driver::LaunchConfig;

// The lifetime parameter `'a` allows CudaOps to borrow CudaKernels and CudaMemoryManager,
// ensuring that the operations do not outlive the resources they depend on.
pub struct CudaOps<'a> {
    kernels: &'a CudaKernels,
    memory: &'a CudaMemoryManager,
}

// Element-wise operations on CUDA tensors
impl<'a> CudaOps<'a> {
    pub fn new(kernels: &'a CudaKernels, memory: &'a CudaMemoryManager) -> Self {
        Self { kernels, memory }
    }

    pub fn kernels(&self) -> &CudaKernels {
        self.kernels
    }

    /// Calculate optimal launch configuration for given size
    fn get_launch_config(&self, size: usize) -> LaunchConfig {
        let block_size = 256;
        let grid_size = (size + block_size - 1) / block_size;
        LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Element-wise addition: a + b
    pub fn add<T>(&self, a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        if a.shape != b.shape {
            return Err("Shape mismatch for addition".to_string());
        }

        let size = a.size();
        let mut result = CudaTensor::zeros(self.memory, a.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_add(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// Element-wise multiplication: a * b
    pub fn mul<T>(&self, a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        if a.shape != b.shape {
            return Err("Shape mismatch for multiplication".to_string());
        }

        let size = a.size();
        let mut result = CudaTensor::zeros(self.memory, a.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_mul(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// Scalar addition: a + scalar
    pub fn add_scalar<T>(&self, a: &CudaTensor<T>, scalar: T) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        // We'll need a separate kernel for scalar operations
        // For now, create a tensor filled with the scalar value
        let scalar_tensor = self.full(&a.shape, scalar)?;
        self.add(a, &scalar_tensor)
    }

    /// Create tensor filled with given value. This is quite handly becauase very easily we can create
    /// tensors filled with a constant value and use it in operations like addition or multiplication
    /// without creating a separate kernel for each operation. Might not be the most efficient way, but it is simple and works.
    pub fn full<T>(&self, shape: &[usize], value: T) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let size = shape.iter().product();
        let host_data = vec![value; size];
        let gpu_data = self.memory.host_to_device(host_data)?;
        Ok(CudaTensor::new(gpu_data, shape.to_vec()))
    }

    /// Element-wise division: a / b
    pub fn div<T>(&self, a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        if a.shape != b.shape {
            return Err("Shape mismatch for division".to_string());
        }

        let size = a.size();
        let mut result = CudaTensor::zeros(self.memory, a.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_div(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// Scalar multiplication: a * scalar
    pub fn mul_scalar<T>(&self, a: &CudaTensor<T>, scalar: T) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let scalar_tensor = self.full(&a.shape, scalar)?;
        self.mul(a, &scalar_tensor)
    }

    /// Scalar division: a / scalar
    pub fn div_scalar<T>(&self, a: &CudaTensor<T>, scalar: T) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let scalar_tensor = self.full(&a.shape, scalar)?;
        self.div(a, &scalar_tensor)
    }

    /// ReLU activation function
    pub fn relu<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let size = input.size();
        let mut result = CudaTensor::zeros(self.memory, input.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_relu(cfg, &input.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// Exponential function
    pub fn exp<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let size = input.size();
        let mut result = CudaTensor::zeros(self.memory, input.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_activation("exp", cfg, &input.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    pub fn matmul<T>(&self, a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + Copy + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        // Check dimensions
        if a.ndim() != 2 || b.ndim() != 2 {
            return Err("Matrix multiplication requires 2D tensors".to_string());
        }

        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape[1] != b_shape[0] {
            return Err(format!(
                "Matrix multiplication shape mismatch: ({}, {}) @ ({}, {})",
                a_shape[0], a_shape[1], b_shape[0], b_shape[1]
            ));
        }

        // Result shape will be [a_shape[0], b_shape[1]]
        let result_shape = vec![a_shape[0], b_shape[1]];
        let result_size = result_shape.iter().product();

        // Allocate result tensor
        let result_data = self.memory.alloc_zeros::<T>(result_size)?;
        let mut result = CudaTensor::new(result_data, result_shape);

        // Try to get matmul kernel

        let cfg = self.get_launch_config(result_size);
        self.kernels.launch_matmul(
            cfg,
            &a.data,
            &b.data,
            &mut result.data,
            a_shape[0] as i32,
            a_shape[1] as i32,
            b_shape[1] as i32,
        )?;
        Ok(result)
    }

    /// Applies natural logarithm element-wise using CUDA kernel
    pub fn log<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr
            + Clone
            + cudarc::driver::ValidAsZeroBits
            + std::marker::Unpin,
    {
      

        // Calculate total number of elements
       
        let mut output = self.memory.alloc_zeros::<T>(size)?;

        // Configure CUDA launch parameters
        let threads_per_block = 256;
        let blocks = (size + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch the log kernel
        self.kernels
            .launch_activation("log", cfg, &input.data, &mut output.data, size)?;

        Ok(output)
    }

    /// Applies hyperbolic tangent element-wise using CUDA kernel
    pub fn tanh<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr
            + Clone
            + cudarc::driver::ValidAsZeroBits
            + std::marker::Unpin,
    {
        

        // Calculate total number of elements
        let size = input.shape.iter().product::<usize>() as i32;
        
        // Create output tensor with same shape as input
        let mut output = self.memory.alloc_zeros::<T>(size.try_into().unwrap())?;

        // Configure CUDA launch parameters
        let threads_per_block = 256;
        let blocks = (size + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch the tanh kernel
        self.kernels
            .launch_activation("tanh", cfg, &input.data, &mut output.data, size)?;

        Ok(output)
    }

    /// Element-wise power operation using CUDA kernel: a^b
    pub fn power<T>(&self, a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr
            + Clone
            + cudarc::driver::ValidAsZeroBits
            + std::marker::Unpin,
    {
        // Validate tensor shapes match
        if a.shape != b.shape {
            return Err(format!(
                "Power operation requires tensors with matching shapes: {:?} vs {:?}",
                a.shape, b.shape
            ));
        }

        // Calculate total number of elements
        let size = a.shape.iter().product::<usize>() as i32;

        // Create output tensor with same shape as inputs
        let mut output = self.memory.alloc_zeros::<T>(size.try_into().unwrap())?;


        // Configure CUDA launch parameters
        let threads_per_block = 256;
        let blocks = (size + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch the power kernel
        self.kernels
            .launch_power(cfg, &a.data, &b.data, &mut output.data, size)?;

        Ok(output)
    }

    /// Power with scalar exponent using CUDA
    pub fn power_scalar<T>(
        &self,
        base: &CudaTensor<T>,
        exponent: T,
    ) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr
            + Clone
            + cudarc::driver::ValidAsZeroBits
            + std::marker::Unpin,
    {
        // Create a tensor filled with the scalar exponent
        let exponent_tensor = self.full(base.shape.clone(), exponent)?;

        // Use the regular power operation
        self.power(base, &exponent_tensor)
    }

    /// Sigmoid activation function using CUDA kernel
    pub fn sigmoid<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr
            + Clone
            + cudarc::driver::ValidAsZeroBits
            + std::marker::Unpin,
    {
        // Create output tensor with same shape as input
        

        // Calculate total number of elements
        let size = input.shape.iter().product::<usize>() as i32;
        let mut output = self.memory.alloc_zeros::<T>(size.try_into().unwrap())?;

        // Configure CUDA launch parameters
        let threads_per_block = 256;
        let blocks = (size + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch the sigmoid kernel using the generic activation launcher
        self.kernels
            .launch_activation("sigmoid", cfg, &input.data, &mut output.data, size)?;

        Ok(output)
    }
}
