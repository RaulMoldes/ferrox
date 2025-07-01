// src/backend/cuda/ops.rs
// High-level CUDA tensor operations that mirror CPU tensor operations
// This module provides the user-facing API for neural network operations on GPU

use super::context::{CudaContextManager, CudaTensor};
use super::kernels::CudaKernels;
use crate::backend::number::CPUNumber;
use cudarc::driver::LaunchConfig;


/// Utility functions to handle tenso shape and strides calculations
/// Compute strides for unsqueeze operation
fn unsqueeze_strides(strides: &[usize], axis: usize) -> Vec<usize> {
    let mut new_strides = strides.to_vec();
    
    // Stride for new dimension: use stride of next dimension, or 1 if at end
    let new_stride = if axis < strides.len() {
        strides[axis]
    } else {
        1
    };
    
    new_strides.insert(axis, new_stride);
    new_strides
}


/// Remove all dimensions of size 1 and their corresponding strides
fn squeeze_all_dims(shape: &[usize], strides: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut new_shape = Vec::new();
    let mut new_strides = Vec::new();
    
    for (i, &dim_size) in shape.iter().enumerate() {
        if dim_size != 1 {
            new_shape.push(dim_size);
            new_strides.push(strides[i]);
        }
    }
    
    // Handle edge case: if all dimensions were size 1, keep one
    if new_shape.is_empty() {
        new_shape.push(1);
        new_strides.push(1);
    }
    
    (new_shape, new_strides)
}



// CudaOps provides a high-level interface for performing tensor operations on GPU
/// This is the main interface for performing tensor operations on GPU
/// The lifetime parameter ensures operations don't outlive the underlying CUDA resources
pub struct CudaOps<'a> {
    kernels: &'a CudaKernels,
    memory: &'a CudaContextManager,
}

impl<'a> CudaOps<'a> {
    pub fn new(kernels: &'a CudaKernels, memory: &'a CudaContextManager) -> Self {
        Self { kernels, memory }
    }

    pub fn kernels(&self) -> &CudaKernels {
        self.kernels
    }

    // ===== UTILITY METHODS =====

    /// Calculate optimal launch configuration for 1D operations
    /// Uses heuristics optimized for modern GPU architectures
    ///
    /// The CUDA configuration is set up as follows:
    /// - 1D grid with blocks of 256 threads
    /// - 32 warps per block (8 threads each) for maximum occupancy)
    ///
    /// Disclaimer: I know that hardcoding these values is not ideal. I still have not figured out how does cudarc provide access
    /// to device properties like warp size, max threads per block, etc (I do not ven know if it is feasible to do it on rust as of now).
    /// In C++ API this is done via [`cudaGetDeviceProperties`] struct.
    /// I made the decision o use this values investigating here:
    ///  https://forums.developer.nvidia.com/t/maximum-number-of-warps-and-warp-size-per-sm/234378
    /// On Pascal and later architectures, the warp size is always 32 threads. At least on this arch, the maximum number of warps per block appears to be 64 and is limited by hardware constratints.
    /// Grid size is calculated to cover all elements with proper alignment.
    /// Best practices suggest starting with 128–256 threads per block and tuning later based on profiling.
    fn get_launch_config(&self, size: usize) -> LaunchConfig {
        let block_size = 256;
        let grid_size = (size + block_size - 1) / block_size;
        LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Calculate launch configuration for 2D operations (like matrix transpose)
    /// Uses 16x16 thread blocks which are optimal for 2D memory access patterns.
    /// This value is also hardcoded based on empirical evidence and best practices.
    /// It is a  common practice to use 16x16 blocks for 2D operations on modern GPUs due to:
    /// In float32: 256 × 4B = 1 KB per tile.
    /// Two tiles (e.g., one from matrix A and one from matrix B) = 2 KB total → fits well within the 48 KB (or more) of shared memory per Streaming Multiprocessor (SM)
    /// Additionally, shared memory on most GPUs has 32 banks. A 16×16 tile maps well to these banks with minimal or no conflicts if padded carefully.
    /// This enables coalesced memory access, reducing memory transaction overhead.
    /// Specifically, during matrix multiplication, each 16×16 tile of the output matrix C can be computed by loading corresponding tiles from A and B.
    fn get_2d_launch_config(&self, rows: usize, cols: usize) -> LaunchConfig {
        let block_size = 16; // 16x16 = 256 threads per block.
        let grid_x = (cols + block_size - 1) / block_size;
        let grid_y = (rows + block_size - 1) / block_size;

        LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (block_size as u32, block_size as u32, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Create tensor filled with constant value
    /// This is a fundamental building block for scalar operations
    /// More efficient than creating dedicated scalar kernels for each operation
    pub fn full<T>(&self, shape: &[usize], value: T) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber,
    {
        let size = shape.iter().product();
        let host_data = vec![value; size];
        let gpu_data = self.memory.host_to_device(host_data)?;
        Ok(CudaTensor::new(gpu_data, shape.to_vec()))
    }

    /// Create zeros tensor - fundamental for initializing gradients and intermediate results
    pub fn zeros<T>(&self, shape: &[usize]) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + Default,
    {
        CudaTensor::zeros(self.memory, shape.to_vec())
    }

    /// Create ones tensor - useful for creating bias vectors and normalization
    pub fn ones<T>(&self, shape: &[usize]) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + From<i32>,
    {
        let one = T::from(1);
        self.full(shape, one)
    }

    /// Element-wise addition: result = a + b
    pub fn add<T>(&self, a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
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

    /// Element-wise multiplication: result = a * b
    pub fn mul<T>(&self, a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
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

    /// Element-wise division: result = a / b
    pub fn div<T>(&self, a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
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

    /// Element-wise subtraction: result = a - b
    pub fn sub<T>(&self, a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
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

    /// Element-wise power: result = a^b
    pub fn power<T>(&self, a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        if a.shape != b.shape {
            return Err("Shape mismatch for power operation".to_string());
        }

        let size = a.size();
        let mut result = CudaTensor::zeros(self.memory, a.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_power(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// Element-wise minimum: result = min(a, b)
    pub fn min_elementwise<T>(
        &self,
        a: &CudaTensor<T>,
        b: &CudaTensor<T>,
    ) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
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

    /// Element-wise maximum: result = max(a, b)
    pub fn max_elementwise<T>(
        &self,
        a: &CudaTensor<T>,
        b: &CudaTensor<T>,
    ) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
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

    /// Scalar addition: result = tensor + scalar
    pub fn add_scalar<T>(&self, a: &CudaTensor<T>, scalar: T) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let scalar_tensor = self.full(&a.shape, scalar)?;
        self.add(a, &scalar_tensor)
    }

    /// Scalar multiplication: result = tensor * scalar
    pub fn mul_scalar<T>(&self, a: &CudaTensor<T>, scalar: T) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let scalar_tensor = self.full(&a.shape, scalar)?;
        self.mul(a, &scalar_tensor)
    }

    /// Scalar division: result = tensor / scalar
    pub fn div_scalar<T>(&self, a: &CudaTensor<T>, scalar: T) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let scalar_tensor = self.full(&a.shape, scalar)?;
        self.div(a, &scalar_tensor)
    }

    /// Scalar subtraction: result = tensor - scalar
    pub fn sub_scalar<T>(&self, a: &CudaTensor<T>, scalar: T) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let scalar_tensor = self.full(&a.shape, scalar)?;
        self.sub(a, &scalar_tensor)
    }

    /// Scalar power: result = tensor^scalar
    pub fn power_scalar<T>(
        &self,
        base: &CudaTensor<T>,
        exponent: T,
    ) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let exponent_tensor = self.full(&base.shape, exponent)?;
        self.power(base, &exponent_tensor)
    }

    /// Element-wise absolute value: result = |input|
    pub fn abs<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let size = input.size();
        let mut result = CudaTensor::zeros(self.memory, input.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_abs(cfg, &input.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// Element-wise square root: result = sqrt(input)
    pub fn sqrt<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let size = input.size();
        let mut result = CudaTensor::zeros(self.memory, input.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_sqrt(cfg, &input.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// Element-wise exponential: result = exp(input)
    pub fn exp<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let size = input.size();
        let mut result = CudaTensor::zeros(self.memory, input.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_exp(cfg, &input.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// Element-wise natural logarithm: result = ln(input)
    pub fn log<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let size = input.size();
        let mut result = CudaTensor::zeros(self.memory, input.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_log(cfg, &input.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// Element-wise negation: result = -input
    pub fn negate<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let size = input.size();
        let mut result = CudaTensor::zeros(self.memory, input.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_negate(cfg, &input.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// ReLU activation: result = max(0, input)
    pub fn relu<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let size = input.size();
        let mut result = CudaTensor::zeros(self.memory, input.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_relu(cfg, &input.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// Sigmoid activation: result = 1 / (1 + exp(-input))
    pub fn sigmoid<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber,
    {
        let size = input.size();
        let mut result = CudaTensor::zeros(self.memory, input.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_activation(
            "sigmoid",
            cfg,
            &input.data,
            &mut result.data,
            size as i32,
        )?;
        Ok(result)
    }

    /// Hyperbolic tangent activation: result = tanh(input)
    pub fn tanh<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber,
    {
        let size = input.size();
        let mut result = CudaTensor::zeros(self.memory, input.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_activation(
            "hyperbolic_tanh",
            cfg,
            &input.data,
            &mut result.data,
            size as i32,
        )?;
        Ok(result)
    }

    /// Clamp values to specified range: result = clamp(input, min_val, max_val)
    pub fn clamp<T>(
        &self,
        input: &CudaTensor<T>,
        min_val: T,
        max_val: T,
    ) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
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

    /// Matrix multiplication: C = A @ B
    /// The fundamental operation of neural networks - linear transformations
    ///
    /// # Requirements:
    /// - A must be [M, K], B must be [K, N] -> Result is [M, N]
    /// - This kernel uses optimized tiled matrix multiplication for memory efficiency
    pub fn matmul<T>(&self, a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + Copy + 'static,
    {
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

        let result_shape = vec![a_shape[0], b_shape[1]];
        let result_size = result_shape.iter().product();
        let result_data = self.memory.alloc_zeros::<T>(result_size)?;
        let mut result = CudaTensor::new(result_data, result_shape);

        // Use 2D launch configuration optimized for matrix multiplication
        let cfg = self.get_2d_launch_config(a_shape[0], b_shape[1]);

        self.kernels.launch_matmul(
            cfg,
            &a.data,
            &b.data,
            &mut result.data,
            a_shape[0] as i32, // M
            b_shape[1] as i32, // N
            a_shape[1] as i32, // K
        )?;
        Ok(result)
    }

    /// 2D matrix transpose: result[j, i] = input[i, j]
    pub fn transpose_2d<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        if input.shape.len() != 2 {
            return Err("Transpose operation requires 2D tensor".to_string());
        }

        let rows = input.shape[0];
        let cols = input.shape[1];
        let output_shape = vec![cols, rows];
        let mut result = CudaTensor::zeros(self.memory, output_shape)?;

        let cfg = self.get_2d_launch_config(rows, cols);

        self.kernels.launch_transpose_2d(
            cfg,
            &input.data,
            &mut result.data,
            rows as i32,
            cols as i32,
        )?;
        Ok(result)
    }

    // ===== REDUCTION OPERATIONS =====

    /// Sum along specified axis
    pub fn sum_axis<T>(
        &self,
        input: &CudaTensor<T>,
        axis: usize,
        keep_dims: bool,
    ) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let input_shape = &input.shape;

        if axis >= input_shape.len() {
            return Err(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                input_shape.len()
            ));
        }

        let mut output_shape = input_shape.clone();
        if keep_dims {
            output_shape[axis] = 1;
        } else {
            output_shape.remove(axis);
        }

        let outer_size = input_shape[..axis].iter().product::<usize>() as i32;
        let axis_size = input_shape[axis] as i32;
        let inner_size = input_shape[axis + 1..].iter().product::<usize>() as i32;

        let mut result = CudaTensor::zeros(self.memory, output_shape)?;

        let cfg = LaunchConfig {
            grid_dim: (outer_size as u32, 1, 1),
            block_dim: (inner_size.min(1024) as u32, 1, 1),
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

    /// Maximum along specified axis
    pub fn max_along_dim<T>(
        &self,
        input: &CudaTensor<T>,
        dim: usize,
    ) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let input_shape = &input.shape;

        if dim >= input_shape.len() {
            return Err(format!(
                "Dimension {} out of bounds for tensor with {} dimensions",
                dim,
                input_shape.len()
            ));
        }

        let mut output_shape = input_shape.clone();
        output_shape.remove(dim);

        let outer_size = input_shape[..dim].iter().product::<usize>() as i32;
        let axis_size = input_shape[dim] as i32;
        let inner_size = input_shape[dim + 1..].iter().product::<usize>() as i32;

        let mut result = CudaTensor::zeros(self.memory, output_shape)?;

        let cfg = LaunchConfig {
            grid_dim: (
                ((outer_size * inner_size + 255) / 256).try_into().unwrap(),
                1,
                1,
            ),
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

    /// Sum all elements in tensor
    pub fn sum_all<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let mut current = input.clone();
        for axis in (0..input.shape.len()).rev() {
            current = self.sum_axis(&current, axis, false)?;
        }
        Ok(current)
    }

    /// Sum along multiple axes
    pub fn sum_axes<T>(
        &self,
        input: &CudaTensor<T>,
        axes: &[usize],
        keep_dims: bool,
    ) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_by(|a, b| b.cmp(a)); // Sort in descending order

        let mut current = input.clone();
        for &axis in &sorted_axes {
            current = self.sum_axis(&current, axis, keep_dims)?;
        }
        Ok(current)
    }

    /// Mean along axis: sum / count
    pub fn mean_axis<T>(
        &self,
        input: &CudaTensor<T>,
        axis: usize,
        keep_dims: bool,
    ) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let sum_result = self.sum_axis(input, axis, keep_dims)?;
        let axis_size = input.shape[axis];
        let divisor = <T as CPUNumber>::from_f64(axis_size as f64)
            .expect("Failed to convert axis size to GPU number");
        let divisor_tensor = self.full(&sum_result.shape, divisor)?;
        self.div(&sum_result, &divisor_tensor)
    }

    /// Mean of all elements
    pub fn mean_all<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let sum_result = self.sum_all(input)?;
        let total_elements = input.size();
        let divisor = <T as CPUNumber>::from_f64(total_elements as f64)
            .expect("Failed to convert total elements to GPU number");
        let divisor_tensor = self.full(&[], divisor)?; // Scalar tensor
        self.div(&sum_result, &divisor_tensor)
    }

    /// Element-wise greater-or-equal: result = (a >= b) ? 1.0 : 0.0
    pub fn greater_equal<T>(
        &self,
        a: &CudaTensor<T>,
        b: &CudaTensor<T>,
    ) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        if a.shape != b.shape {
            return Err("Shape mismatch for greater_equal operation".to_string());
        }

        let size = a.size();
        let mut result = CudaTensor::zeros(self.memory, a.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_greater_equal(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// Element-wise less-or-equal: result = (a <= b) ? 1.0 : 0.0
    pub fn less_equal<T>(
        &self,
        a: &CudaTensor<T>,
        b: &CudaTensor<T>,
    ) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        if a.shape != b.shape {
            return Err("Shape mismatch for less_equal operation".to_string());
        }

        let size = a.size();
        let mut result = CudaTensor::zeros(self.memory, a.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_less_equal(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// Element-wise equality: result = (a == b) ? 1.0 : 0.0
    pub fn equal<T>(&self, a: &CudaTensor<T>, b: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        if a.shape != b.shape {
            return Err("Shape mismatch for equal operation".to_string());
        }

        let size = a.size();
        let mut result = CudaTensor::zeros(self.memory, a.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_equal(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// Logical NOT: result = (input == 0.0) ? 1.0 : 0.0
    /// Inverts boolean tensors following IEEE 754 convention
    pub fn logical_not<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let size = input.size();
        let mut result = CudaTensor::zeros(self.memory, input.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_logical_not(cfg, &input.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// Sign function: result = sign(input) ∈ {-1, 0, 1}
    pub fn sign<T>(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let size = input.size();
        let mut result = CudaTensor::zeros(self.memory, input.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_sign(cfg, &input.data, &mut result.data, size as i32)?;
        Ok(result)
    }

    /// Range check: result = (min_val <= input <= max_val) ? 1.0 : 0.0
    pub fn in_range<T>(
        &self,
        input: &CudaTensor<T>,
        min_val: T,
        max_val: T,
    ) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let size = input.size();
        let mut result = CudaTensor::zeros(self.memory, input.shape.clone())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_in_range(
            cfg,
            &input.data,
            min_val,
            max_val,
            &mut result.data,
            size as i32,
        )?;
        Ok(result)
    }

    /// CONVOLUTIONAL OPS.
    // Add to CudaOps impl block
    pub fn conv2d<T>(
        &self,
        input: &CudaTensor<T>,
        filter: &CudaTensor<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        // Implementation for fused depthwise separable convolution
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err("Conv2D requires 4D input".to_string());
        }
        let (batch, in_channels, input_h, input_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );

        let filter_shape = filter.shape();
        if filter_shape.len() != 4 {
            return Err("Conv2D requires 4D filter".to_string());
        }
        let (out_channels, _, filter_h, filter_w) = (
            filter_shape[0],
            filter_shape[1],
            filter_shape[2],
            filter_shape[3],
        );

        let output_h = (input_h + 2 * padding.0 - filter_h) / stride.0 + 1;
        let output_w = (input_w + 2 * padding.1 - filter_w) / stride.1 + 1;

        let output_shape = vec![batch, out_channels, output_h, output_w];
        let mut result = CudaTensor::zeros(self.memory, output_shape)?;

        let cfg = LaunchConfig {
            grid_dim: (
                ((output_h * output_w + 255) / 256).try_into().unwrap(),
                batch as u32,
                out_channels as u32,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        self.kernels.launch_conv2d_forward(
            cfg,
            &input.data,
            &filter.data,
            &mut result.data,
            batch as i32,
            in_channels as i32,
            input_h as i32,
            input_w as i32,
            out_channels as i32,
            filter_h as i32,
            filter_w as i32,
            stride.0 as i32,
            stride.1 as i32,
            padding.0 as i32,
            padding.1 as i32,
        )?;

        Ok(result)
    }

    pub fn depthwise_separable_conv2d<T>(
        &self,
        input: &CudaTensor<T>,
        depthwise_filter: &CudaTensor<T>,
        pointwise_filter: &CudaTensor<T>,
    ) -> Result<CudaTensor<T>, String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err("Conv2D requires 4D input".to_string());
        }
        let (batch, in_channels, input_h, input_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );

        // Fix: use the correct variable names
        let output_shape = vec![batch, pointwise_filter.shape[0], input_h, input_w];
        let mut result = CudaTensor::zeros(self.memory, output_shape)?;
        let cfg = self.get_launch_config(batch * in_channels * input_h * input_w);

        self.kernels.launch_depthwise_separable_conv2d_fused(
            cfg,
            &input.data,
            &depthwise_filter.data,
            &pointwise_filter.data,
            &mut result.data,
            batch as i32,
            in_channels as i32,
            input_h as i32,
            input_w as i32,
        )?;

        Ok(result)
    }

    /// Zero-copy memory management
    /// This OPs allow to manipulate the view of the data without copying it
    /// It is useful for broadcasting, slicing, and reshaping tensors on GPU
    /// Zero-copy reshape - only changes shape/strides metadata
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, String> {
        let new_size: usize = new_shape.iter().product();
        let current_size = self.size();

        if new_size != current_size {
            return Err(format!(
                "Cannot reshape tensor of size {} to shape {:?} (size {})",
                current_size, new_shape, new_size
            ));
        }

        // Check if reshape is contiguous (can reuse same data layout)
        // If not contiguous, we cannot reshape without copying
        if !self.is_contiguous() {
            return Err("Cannot reshape non-contiguous tensor without copying".to_string());
        }

        Ok(Self {
            data: self.data.clone(), // Shallow clone - same GPU memory
            shape: new_shape,
            strides: compute_strides(&new_shape),
        })
    }

    /// Zero-copy broadcast - BROADCAST OPERATION (Based on: https://numpy.org/doc/2.1/reference/generated/numpy.broadcast.html)
    /// This operation allows to create a new tensor with a different shape
    /// that shares the same GPU memory as the original tensor.
    /// It uses the can_broadcast function to check if the shapes are compatible.
    /// It returns an error if the shapes cannot be broadcasted.
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Self, String> {
        if !can_broadcast(&self.shape, target_shape) {
            return Err(format!(
                "Cannot broadcast shape {:?} to {:?}",
                self.shape, target_shape
            ));
        }

        let new_strides = broadcast_strides(&self.shape, &self.strides, target_shape);

        Ok(Self {
            data: self.data.clone(), // Same GPU memory
            shape: target_shape.to_vec(),
            strides: new_strides,
        })
    }

    /// Check if tensor memory layout is contiguous
    pub fn is_contiguous(&self) -> bool {
        let expected_strides = compute_strides(&self.shape);
        self.strides == expected_strides
    }

    /// Check if shapes can be broadcasted
    /// Follows NumPy broadcasting rules:
    /// 1. Start from trailing dimensions and work backward
    /// 2. Dimensions are compatible if:
    ///    - They are equal, OR
    ///    - One of them is 1, OR  
    ///    - One of them is missing (treat as 1)
    /// 3. Missing dimensions are added as size 1 at the beginning
    /// 4. Result shape has the maximum size in each dimension
    fn can_broadcast_to(source_shape: &[usize], target_shape: &[usize]) -> bool {
        // Source cannot have more dimensions than target
        if source_shape.len() > target_shape.len() {
            return false;
        }

        let offset = target_shape.len() - source_shape.len();

        // Check each dimension from right to left
        for (i, &src_dim) in source_shape.iter().enumerate() {
            let tgt_dim = target_shape[i + offset];

            // Compatible if: equal OR source is 1
            if src_dim != 1 && src_dim != tgt_dim {
                return false;
            }
        }

        true
    }

    /// Compute broadcast strides for target shape
    /// This function calculates the strides for a new shape that is a broadcasted version of the source
    /// shape. It follows the broadcasting rules defined in `can_broadcast_to`.
    fn broadcast_strides(
        source_shape: &[usize],
        source_strides: &[usize],
        target_shape: &[usize],
    ) -> Vec<usize> {
        let mut new_strides = vec![0; target_shape.len()];
        let offset = target_shape.len() - source_shape.len();

        // Leading dimensions get stride 0 (broadcasted)
        for i in 0..offset {
            new_strides[i] = 0;
        }

        // Map existing dimensions
        for (i, (&src_dim, &src_stride)) in source_shape.iter().zip(source_strides).enumerate() {
            let tgt_idx = i + offset;
            new_strides[tgt_idx] = if src_dim == 1 { 0 } else { src_stride };
        }

        new_strides
    }



    /// Zero-copy unsqueeze - add dimension of size 1 at specified axis
    pub fn unsqueeze(&self, axis: usize) -> Result<Self, String> {
        if axis > self.shape.len() {
            return Err(format!(
                "Cannot unsqueeze at axis {} for tensor with {} dimensions", 
                axis, self.shape.len()
            ));
        }

        let mut new_shape = self.shape.clone();
        new_shape.insert(axis, 1);

        let new_strides = unsqueeze_strides(&self.strides, axis);

        Ok(Self {
            data: self.data.clone(), // Zero-copy: same GPU memory
            shape: new_shape,
            strides: new_strides,
        })
    }

    /// Zero-copy squeeze - remove dimensions of size 1
    pub fn squeeze(&self, axis: Option<usize>) -> Result<Self, String> {
        match axis {
            Some(ax) => {
                // Squeeze specific axis
                if ax >= self.shape.len() {
                    return Err(format!("Axis {} out of bounds", ax));
                }
                if self.shape[ax] != 1 {
                    return Err(format!(
                        "Cannot squeeze axis {} with size {}", 
                        ax, self.shape[ax]
                    ));
                }

                let mut new_shape = self.shape.clone();
                let mut new_strides = self.strides.clone();
                new_shape.remove(ax);
                new_strides.remove(ax);

                Ok(Self {
                    data: self.data.clone(),
                    shape: new_shape,
                    strides: new_strides,
                })
            }
            None => {
                // Squeeze all dimensions of size 1
                let (new_shape, new_strides) = squeeze_all_dims(&self.shape, &self.strides);
                
                Ok(Self {
                    data: self.data.clone(),
                    shape: new_shape,
                    strides: new_strides,
                })
            }
        }
    }
}









