// src/backend/cuda/ops.rs
// This module contains abstarctions for CUDA tensor operations that mirror existent CPU tensor operations
use super::memory::{CudaMemoryManager, CudaTensor};
use super::kernels::CudaKernels;
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
    pub fn add(&self, a: &CudaTensor<f32>, b: &CudaTensor<f32>) -> Result<CudaTensor<f32>, String> {
        if a.shape != b.shape {
            return Err("Shape mismatch for addition".to_string());
        }

        let size = a.size();
        let mut result = CudaTensor::zeros(self.memory, a.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_add(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// Element-wise multiplication: a * b
    pub fn mul(&self, a: &CudaTensor<f32>, b: &CudaTensor<f32>) -> Result<CudaTensor<f32>, String> {
        if a.shape != b.shape {
            return Err("Shape mismatch for multiplication".to_string());
        }

        let size = a.size();
        let mut result = CudaTensor::zeros(self.memory, a.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_mul(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// Scalar addition: a + scalar
    pub fn add_scalar(&self, a: &CudaTensor<f32>, scalar: f32) -> Result<CudaTensor<f32>, String> {
      // We'll need a separate kernel for scalar operations
      // For now, create a tensor filled with the scalar value
      let scalar_tensor = self.full(&a.shape, scalar)?;
      self.add(a, &scalar_tensor)
  }

  /// Create tensor filled with given value. This is quite handly becauase very easily we can create
  /// tensors filled with a constant value and use it in operations like addition or multiplication 
  /// without creating a separate kernel for each operation. Might not be the most efficient way, but it is simple and works.
  pub fn full(&self, shape: &[usize], value: f32) -> Result<CudaTensor<f32>, String> {
      let size = shape.iter().product();
      let host_data = vec![value; size];
      let gpu_data = self.memory.host_to_device(host_data)?;
      Ok(CudaTensor::new(gpu_data, shape.to_vec()))
  }

    /// Element-wise division: a / b
    pub fn div(&self, a: &CudaTensor<f32>, b: &CudaTensor<f32>) -> Result<CudaTensor<f32>, String> {
        if a.shape != b.shape {
            return Err("Shape mismatch for division".to_string());
        }

        let size = a.size();
        let mut result = CudaTensor::zeros(self.memory, a.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_div(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// Scalar multiplication: a * scalar
    pub fn mul_scalar(&self, a: &CudaTensor<f32>, scalar: f32) -> Result<CudaTensor<f32>, String> {
        let scalar_tensor = self.full(&a.shape, scalar)?;
        self.mul(a, &scalar_tensor)
    }

    /// Scalar division: a / scalar
    pub fn div_scalar(&self, a: &CudaTensor<f32>, scalar: f32) -> Result<CudaTensor<f32>, String> {
        let scalar_tensor = self.full(&a.shape, scalar)?;
        self.div(a, &scalar_tensor)
    }


  /// ReLU activation function
  pub fn relu(&self, input: &CudaTensor<f32>) -> Result<CudaTensor<f32>, String> {
      let size = input.size();
      let mut result = CudaTensor::zeros(self.memory, input.shape.clone())?;
      let cfg = self.get_launch_config(size);

      self.kernels.launch_relu(cfg, &input.data, &mut result.data, size as i32)?;
      Ok(result)
  }

  /// Exponential function
  pub fn exp(&self, input: &CudaTensor<f32>) -> Result<CudaTensor<f32>, String> {
      let size = input.size();
      let mut result = CudaTensor::zeros(self.memory, input.shape.clone())?;
      let cfg = self.get_launch_config(size);

      self.kernels.launch_activation("exp", cfg, &input.data, &mut result.data, size as i32)?;
      Ok(result)
  }


  /// Matrix multiplication: C = A @ B
    /// A is (m x k), B is (k x n), C is (m x n)
    pub fn matmul(&self, a: &CudaTensor<f32>, b: &CudaTensor<f32>) -> Result<CudaTensor<f32>, String> {
        // Validate that both tensors are 2D
        if a.ndim() != 2 || b.ndim() != 2 {
            return Err("Matrix multiplication requires 2D tensors".to_string());
        }

        let a_shape = a.shape();
        let b_shape = b.shape();

        // Check dimension compatibility: A(m,k) @ B(k,n) = C(m,n)
        if a_shape[1] != b_shape[0] {
            return Err(format!(
                "Matrix multiplication shape mismatch: ({}, {}) @ ({}, {})",
                a_shape[0], a_shape[1], b_shape[0], b_shape[1]
            ));
        }

        let m = a_shape[0] as i32;  // rows of A and C
        let k = a_shape[1] as i32;  // cols of A, rows of B  
        let n = b_shape[1] as i32;  // cols of B and C

        // Create result tensor with shape (m, n)
        let result_shape = vec![m as usize, n as usize];
        let mut result = CudaTensor::zeros(self.memory, result_shape)?;

        // Calculate optimal launch configuration for matrix multiplication
        let cfg = self.get_matmul_launch_config(m as usize, n as usize);

        // Launch the matrix multiplication kernel
        self.kernels.launch_matmul(
            cfg,
            &a.data,
            &b.data, 
            &mut result.data,
            m,
            n,
            k
        )?;

        Ok(result)
    }

    /// Calculate optimal launch configuration for matrix multiplication
    /// This uses 2D blocks optimized for matrix operations
    fn get_matmul_launch_config(&self, m: usize, n: usize) -> LaunchConfig {
        // Use 16x16 thread blocks for better memory coalescing
        let block_dim_x = 16;
        let block_dim_y = 16;

        // Calculate grid dimensions to cover the entire result matrix
        let grid_dim_x = (n + block_dim_x - 1) / block_dim_x;
        let grid_dim_y = (m + block_dim_y - 1) / block_dim_y;

        LaunchConfig {
            grid_dim: (grid_dim_x as u32, grid_dim_y as u32, 1),
            block_dim: (block_dim_x as u32, block_dim_y as u32, 1),
            shared_mem_bytes: 0, // Could be optimized with shared memory later
        }
    }
}