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
        // Create output tensor with same shape as input
        let mut output = CudaTensor::zeros(self.memory, input.shape.clone())?;
        // Calculate total number of elements
        let size = input.shape.iter().product::<usize>() as i32;

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
        // Create output tensor with same shape as input
        let mut output = CudaTensor::zeros(self.memory, input.shape.clone())?;

        // Calculate total number of elements
        let size = input.shape.iter().product::<usize>() as i32;

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

        // Create output tensor with same shape as input
        let mut output = CudaTensor::zeros(self.memory, a.shape.clone())?;

        // Calculate total number of elements
        let size = a.shape.iter().product::<usize>() as i32;

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
        let exponent_tensor = self.full(&base.shape, exponent)?;

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
        let mut output = CudaTensor::zeros(self.memory, input.shape.clone())?;

        // Calculate total number of elements
        let size = input.shape.iter().product::<usize>() as i32;

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

    /// Element-wise subtraction: a - b
    pub fn sub<T>(&self, a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        if a.shape != b.shape {
            return Err("Shape mismatch for subtraction".to_string());
        }

        let size = a.size();
        let mut result = CudaTensor::zeros(self.memory, a.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_sub(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// Scalar subtraction: a - scalar
    pub fn sub_scalar<T>(&self, a: &CudaTensor<T>, scalar: T) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let scalar_tensor = self.full(&a.shape, scalar)?;
        self.sub(a, &scalar_tensor)
    }

    /// Clamp tensor values between min and max
    pub fn clamp<T>(
        &self,
        input: &CudaTensor<T>,
        min_val: T,
        max_val: T,
    ) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let size = input.size();
        let mut result = CudaTensor::zeros(self.memory, input.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_clamp(
            cfg,
            &input.data,
            &mut result.data,
            min_val,
            max_val,
            size as i32,
        )?;
        Ok(result)
    }

    /// Element-wise minimum operation
    pub fn min_elementwise<T>(&self, a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        if a.shape != b.shape {
            return Err("Shape mismatch for min operation".to_string());
        }

        let size = a.size();
        let mut result = CudaTensor::zeros(self.memory, a.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_min_elementwise(
            cfg,
            &a.data,
            &b.data,
            &mut result.data,
            size as i32,
        )?;
        
        Ok(result)
    }

    /// Element-wise maximum operation
    pub fn max_elementwise<T>(&self, a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        if a.shape != b.shape {
            return Err("Shape mismatch for max operation".to_string());
        }

        let size = a.size();
        let mut result = CudaTensor::zeros(self.memory, a.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_max_elementwise(
            cfg,
            &a.data,
            &b.data,
            &mut result.data,
            size as i32,
        )?;
        
        Ok(result)
    }

    /// Element-wise absolute value
    pub fn abs<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let size = input.size();
        let mut result = CudaTensor::zeros(self.memory, input.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_abs(
            cfg,
            &input.data,
            &mut result.data,
            size as i32,
        )?;
        
        Ok(result)
    }

    /// Element-wise square root
    pub fn sqrt<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let size = input.size();
        let mut result = CudaTensor::zeros(self.memory, input.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_sqrt(
            cfg,
            &input.data,
            &mut result.data,
            size as i32,
        )?;
        
        Ok(result)
    }

    /// Launch maximum reduction along dimension kernel
    /// This is more complex than element-wise operations as it involves reduction
    pub fn max_along_dim<T>(
        &self,
        input: &CudaTensor<T>,
        dim: usize,
    ) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let input_shape = &input.shape;
    
        // Validate dimension
        if dim >= input_shape.len() {
            return Err(format!("Dimension {} out of bounds for tensor with {} dimensions", dim, input_shape.len()));
        }
    
        // Calculate output shape
        let mut output_shape = input_shape.clone();
        output_shape.remove(dim);
    
        // Calculate dimensions for simpler kernel
        let outer_size = input_shape[..dim].iter().product::<usize>() as i32;
        let axis_size = input_shape[dim] as i32;
        let inner_size = input_shape[dim + 1..].iter().product::<usize>() as i32;
    
        // Create output tensor
        let mut result = CudaTensor::zeros(self.memory, output_shape)?;
    
        // Configure kernel launch
        let cfg = LaunchConfig {
            grid_dim: (((outer_size * inner_size + 255) / 256).try_into().unwrap(), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
    
        self.kernels.launch_max_along_dim(
            cfg,
            &input.data,
            &mut result.data,
            outer_size,
            axis_size,
            inner_size,
        )?;
    
        Ok(result)
    }

    /// Sum tensor along specified axis
    pub fn sum_axis<T>(
        &self,
        input: &CudaTensor<T>,
        axis: usize,
        keep_dims: bool,
    ) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let input_shape = &input.shape;

        // Validate axis
        if axis >= input_shape.len() {
            return Err(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                input_shape.len()
            ));
        }

        // Calculate output shape
        let mut output_shape = input_shape.clone();
        if keep_dims {
            output_shape[axis] = 1;
        } else {
            output_shape.remove(axis);
        }

        // Calculate dimensions for the kernel
        let outer_size = input_shape[..axis].iter().product::<usize>() as i32;
        let axis_size = input_shape[axis] as i32;
        let inner_size = input_shape[axis + 1..].iter().product::<usize>() as i32;

        // Create output tensor
        let mut result = CudaTensor::zeros(self.memory, output_shape)?;

        // Configure kernel launch - each thread processes one inner element
        let total_output_elements = (outer_size * inner_size) as usize;
        let cfg = LaunchConfig {
            grid_dim: (outer_size as u32, 1, 1),
            block_dim: (inner_size.min(1024) as u32, 1, 1), // Limit block size to 1024
            shared_mem_bytes: 0,
        };

        self.kernels.launch_sum_axis(
            cfg,
            &input.data,
            &mut result.data,
            outer_size,
            axis_size,
            inner_size,
        )?;

        Ok(result)
    }

    /// Convenience method: sum all elements in tensor (equivalent to flatten + sum)
    pub fn sum_all<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        // Sum along each axis sequentially
        let mut current = input.clone();
        for axis in (0..input.shape.len()).rev() {
            current = self.sum_axis(&current, axis, false)?;
        }
        Ok(current)
    }

    /// ReLU with configurable clamp max (Leaky ReLU when min_val > 0)
    pub fn relu_clamp<T>(
        &self,
        input: &CudaTensor<T>,
        min_val: T,
        max_val: T,
    ) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        // First apply ReLU, then clamp the result
        let relu_result = self.relu(input)?;
        self.clamp(&relu_result, min_val, max_val)
    }

    /// Unary negation: -tensor
    pub fn negate<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let size = input.size();
        let mut result = CudaTensor::zeros(self.memory, input.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_negate(cfg, &input.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// 2D Matrix transpose
    /// Essential for neural networks (weight matrices, etc.)
    pub fn transpose_2d<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        // Validate that input is 2D
        if input.shape.len() != 2 {
            return Err("Transpose operation requires 2D tensor".to_string());
        }

        let rows = input.shape[0] as i32;
        let cols = input.shape[1] as i32;

        // Create output tensor with transposed shape
        let output_shape = vec![input.shape[1], input.shape[0]];
        let mut result = CudaTensor::zeros(self.memory, output_shape)?;

        // Configure 2D grid for transpose
        let block_size = 16; // 16x16 thread blocks work well for transpose
        let grid_x = (cols + block_size - 1) / block_size;
        let grid_y = (rows + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (block_size as u32, block_size as u32, 1),
            shared_mem_bytes: 0,
        };

        self.kernels
            .launch_transpose_2d(cfg, &input.data, &mut result.data, rows, cols)?;
        Ok(result)
    }


    /// Mean along axis (uses sum_axis + division)
    /// This reuses existing kernels rather than creating a new one
    pub fn mean_axis<T>(
        &self,
        input: &CudaTensor<T>,
        axis: usize,
        keep_dims: bool,
    ) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin + From<f32>,
    {
        // First sum along the axis
        let sum_result = self.sum_axis(input, axis, keep_dims)?;

        // Get the size of the reduced dimension
        let axis_size = input.shape[axis];
        let divisor = T::from(axis_size as f32);

        // Create a tensor filled with the divisor
        let divisor_tensor = self.full(&sum_result.shape, divisor)?;

        // Divide sum by count to get mean
        self.div(&sum_result, &divisor_tensor)
    }

    /// Mean of all elements (sum_all / total_count)
    pub fn mean_all<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin + From<f32>,
    {
        let sum_result = self.sum_all(input)?;
        let total_elements = input.size();
        let divisor = T::from(total_elements as f32);

        // Create scalar tensor with divisor
        let divisor_tensor = self.full(&[], divisor)?; // Scalar tensor

        self.div(&sum_result, &divisor_tensor)
    }

    /// Sum along multiple axes (matches CPU tensor sum_axes functionality)
    pub fn sum_axes<T>(
        &self,
        input: &CudaTensor<T>,
        axes: &[usize],
        keep_dims: bool,
    ) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        // Sort axes in descending order to maintain correct indexing
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_by(|a, b| b.cmp(a));

        let mut current = input.clone();

        // Sum along each axis sequentially
        for &axis in &sorted_axes {
            current = self.sum_axis(&current, axis, keep_dims)?;
        }

        Ok(current)
    }

    /// COMPARISON OPERATIONS
     // Comparison operations using CUDA kernels
     pub fn greater_equal<T>(&self, a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
     where
         T: GPUNumber,
     {
         if a.shape != b.shape {
             return Err("Shape mismatch for greater_equal operation".to_string());
         }
 
         let total_elements = a.shape.iter().product::<usize>();
         let mut result = CudaTensor::zeros(&self.memory_manager, a.shape.clone())?;
 
         let block_size = 256;
         let grid_size = (total_elements + block_size - 1) / block_size;
 
         let cfg = LaunchConfig {
             grid_dim: (grid_size as u32, 1, 1),
             block_dim: (block_size as u32, 1, 1),
             shared_mem_bytes: 0,
         };
 
         // Choose kernel based on type
         if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
             self.kernels.launch_greater_equal_f32(
                 cfg,
                 &a.data,
                 &b.data,
                 &mut result.data,
                 total_elements as i32,
             )?;
         } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
             self.kernels.launch_greater_equal_f64(
                 cfg,
                 &a.data,
                 &b.data,
                 &mut result.data,
                 total_elements as i32,
             )?;
         } else {
             return Err("Unsupported type for greater_equal operation".to_string());
         }
 
         Ok(result)
     }
 
     pub fn less_equal<T>(&self, a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
     where
         T: GPUNumber,
     {
         if a.shape != b.shape {
             return Err("Shape mismatch for less_equal operation".to_string());
         }
 
         let total_elements = a.shape.iter().product::<usize>();
         let mut result = CudaTensor::zeros(&self.memory_manager, a.shape.clone())?;
 
         let block_size = 256;
         let grid_size = (total_elements + block_size - 1) / block_size;
 
         let cfg = LaunchConfig {
             grid_dim: (grid_size as u32, 1, 1),
             block_dim: (block_size as u32, 1, 1),
             shared_mem_bytes: 0,
         };
 
         if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
             self.kernels.launch_less_equal_f32(
                 cfg,
                 &a.data,
                 &b.data,
                 &mut result.data,
                 total_elements as i32,
             )?;
         } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
             self.kernels.launch_less_equal_f64(
                 cfg,
                 &a.data,
                 &b.data,
                 &mut result.data,
                 total_elements as i32,
             )?;
         } else {
             return Err("Unsupported type for less_equal operation".to_string());
         }
 
         Ok(result)
     }
 
     pub fn equal<T>(&self, a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
     where
         T: GPUNumber,
     {
         if a.shape != b.shape {
             return Err("Shape mismatch for equal operation".to_string());
         }
 
         let total_elements = a.shape.iter().product::<usize>();
         let mut result = CudaTensor::zeros(&self.memory_manager, a.shape.clone())?;
 
         let block_size = 256;
         let grid_size = (total_elements + block_size - 1) / block_size;
 
         let cfg = LaunchConfig {
             grid_dim: (grid_size as u32, 1, 1),
             block_dim: (block_size as u32, 1, 1),
             shared_mem_bytes: 0,
         };
 
         if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
             self.kernels.launch_equal_f32(
                 cfg,
                 &a.data,
                 &b.data,
                 &mut result.data,
                 total_elements as i32,
             )?;
         } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
             self.kernels.launch_equal_f64(
                 cfg,
                 &a.data,
                 &b.data,
                 &mut result.data,
                 total_elements as i32,
             )?;
         } else {
             return Err("Unsupported type for equal operation".to_string());
         }
 
         Ok(result)
     }
 
     pub fn logical_not<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
     where
         T: GPUNumber,
     {
         let total_elements = input.shape.iter().product::<usize>();
         let mut result = CudaTensor::zeros(&self.memory_manager, input.shape.clone())?;
 
         let block_size = 256;
         let grid_size = (total_elements + block_size - 1) / block_size;
 
         let cfg = LaunchConfig {
             grid_dim: (grid_size as u32, 1, 1),
             block_dim: (block_size as u32, 1, 1),
             shared_mem_bytes: 0,
         };
 
         if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
             self.kernels.launch_logical_not_f32(
                 cfg,
                 &input.data,
                 &mut result.data,
                 total_elements as i32,
             )?;
         } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
             self.kernels.launch_logical_not_f64(
                 cfg,
                 &input.data,
                 &mut result.data,
                 total_elements as i32,
             )?;
         } else {
             return Err("Unsupported type for logical_not operation".to_string());
         }
 
         Ok(result)
     }
 
     pub fn in_range<T>(&self, input: &CudaTensor<T>, min_val: T, max_val: T) -> Result<CudaTensor<T>, String>
     where
         T: GPUNumber,
     {
         let total_elements = input.shape.iter().product::<usize>();
         let mut result = CudaTensor::zeros(&self.memory_manager, input.shape.clone())?;
 
         let block_size = 256;
         let grid_size = (total_elements + block_size - 1) / block_size;
 
         let cfg = LaunchConfig {
             grid_dim: (grid_size as u32, 1, 1),
             block_dim: (block_size as u32, 1, 1),
             shared_mem_bytes: 0,
         };
 
         if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
             // Cast min_val and max_val to f32 for the kernel
             let min_f32 = unsafe { std::mem::transmute::<T, f32>(min_val) };
             let max_f32 = unsafe { std::mem::transmute::<T, f32>(max_val) };
             
             self.kernels.launch_in_range_f32(
                 cfg,
                 &input.data,
                 min_f32,
                 max_f32,
                 &mut result.data,
                 total_elements as i32,
             )?;
         } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
             let min_f64 = unsafe { std::mem::transmute::<T, f64>(min_val) };
             let max_f64 = unsafe { std::mem::transmute::<T, f64>(max_val) };
             
             self.kernels.launch_in_range_f64(
                 cfg,
                 &input.data,
                 min_f64,
                 max_f64,
                 &mut result.data,
                 total_elements as i32,
             )?;
         } else {
             return Err("Unsupported type for in_range operation".to_string());
         }
 
         Ok(result)
     }
    

    /// Sign operation using CUDA kernel
    pub fn sign<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: GPUNumber,
    {
        let total_elements = input.shape.iter().product::<usize>();
        let mut result = CudaTensor::zeros(&self.memory_manager, input.shape.clone())?;

        let block_size = 256;
        let grid_size = (total_elements + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            self.kernels.launch_sign_f32(
                cfg,
                &input.data,
                &mut result.data,
                total_elements as i32,
            )?;
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            self.kernels.launch_sign_f64(
                cfg,
                &input.data,
                &mut result.data,
                total_elements as i32,
            )?;
        } else {
            return Err("Unsupported type for sign operation".to_string());
        }

        Ok(result)
    }
    
    /// Create zeros tensor with given shape (utility function)
    pub fn zeros<T>(&self, shape: &[usize]) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin + Default,
    {
        CudaTensor::zeros(self.memory, shape.to_vec())
    }

    /// Create ones tensor with given shape (utility function)
    pub fn ones<T>(&self, shape: &[usize]) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin + From<i32>,
    {
        let one = T::from(1);
        self.full(shape, one)
    }
}
