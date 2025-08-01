// src/backend/cuda/ops.rs
// High-level CUDA tensor operations that mirror CPU tensor operations
// This module provides the user-facing API for neural network operations on GPU

use super::context::CudaTensor;
use super::kernels::KernelManager;
use crate::backend::cuda::context::compute_strides;
use crate::backend::manager::{alloc_cuda_slice, return_cuda_slice};
use crate::{FerroxCudaF, FerroxF};
use cudarc::driver::CudaSlice;
use cudarc::driver::LaunchConfig;
use cudarc::driver::ValidAsZeroBits;

const BLOCK_SIZE: u32 = 256;
const TILE_SIZE: u32 = 16;

// CudaOps provides a high-level interface for performing tensor operations on GPU
/// This is the main interface for performing tensor operations on GPU
/// The lifetime parameter ensures operations don't outlive the underlying CUDA resources
pub struct CudaOps<T: FerroxCudaF> {
    kernels: KernelManager,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: FerroxCudaF> CudaOps<T> {
    pub fn new(kernels: KernelManager) -> Self {
        Self {
            kernels,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn kernels(&self) -> &KernelManager {
        &self.kernels
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
        let block_size: usize = BLOCK_SIZE as usize;
        let grid_size = ((size + block_size - 1) / block_size) as u32;

        LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
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
        let block_size = TILE_SIZE as usize; // 16x16 = 256 threads per block.
        let grid_x = (cols + block_size - 1) / block_size;
        let grid_y = (rows + block_size - 1) / block_size;

        LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (block_size as u32, block_size as u32, 1),
            shared_mem_bytes: 0,
        }
    }

    fn return_tensor_to_pool(&self, allocation_id: u64, data: CudaTensor<T>) -> Result<(), String> {
        return_cuda_slice::<T>(allocation_id, data.data)
    }

    /// Helper method to use the cuda memory pool
    fn create_tensor_from_pool(&self, shape: &[usize]) -> Result<(CudaTensor<T>, u64), String> {
        let size = shape.iter().product();
        let pool_alloc = alloc_cuda_slice::<T>(size)?;

        // Create tensor from pooled allocation
        let strides = compute_strides(shape);

        Ok((
            CudaTensor {
                data: pool_alloc.data,
                shape: shape.to_vec(),
                strides,
            },
            pool_alloc.allocation_id,
        ))
    }

    fn get_slice_from_pool(&self, size: usize) -> Result<(CudaSlice<T>, u64), String> {
        let pool_alloc = alloc_cuda_slice::<T>(size)?;
        Ok((pool_alloc.data, pool_alloc.allocation_id))
    }
    /// Create tensor filled with constant value
    /// This is a fundamental building block for scalar operations
    /// More efficient than creating dedicated scalar kernels for each operation
    pub fn full(&self, shape: &[usize], value: T) -> Result<(CudaTensor<T>, u64), String> {
        let size = shape.iter().product();
        let cfg = self.get_launch_config(size);

        let (mut result, id) = self.get_slice_from_pool(size)?;

        self.kernels
            .launch_fill(cfg, &mut result, value, size.try_into().unwrap())?;

        let strides = compute_strides(shape);

        Ok((
            CudaTensor {
                data: result,
                shape: shape.to_vec(),
                strides: strides,
            },
            id,
        ))
    }

    /// Create zeros tensor - fundamental for initializing gradients and intermediate results
    pub fn zeros(&self, shape: &[usize]) -> Result<(CudaTensor<T>, u64), String> {
        self.create_tensor_from_pool(shape)
    }

    /// Create ones tensor - useful for creating bias vectors and normalization
    pub fn ones(&self, shape: &[usize]) -> Result<(CudaTensor<T>, u64), String> {
        let one = <T as FerroxF>::from_f64(1.0).unwrap();
        self.full(shape, one)
    }

    pub fn randn(&self, shape: &[usize], seed: u64) -> Result<(CudaTensor<T>, u64), String> {
        let size = shape.iter().product();

        let cfg = self.get_launch_config(size);

        let (mut result, id) = self.get_slice_from_pool(size)?;

        self.kernels
            .launch_fill_random(cfg, &mut result, size as i32, seed)?;
        let strides = compute_strides(shape);
        Ok((
            CudaTensor {
                data: result,
                shape: shape.to_vec(),
                strides: strides,
            },
            id,
        ))
    }

    /// Element-wise addition: result = a + b
    pub fn add(
        &self,
        a: &CudaTensor<T>,
        b: &CudaTensor<T>,
    ) -> Result<(CudaTensor<T>, u64), String> {
        if a.shape != b.shape {
            return Err("Shape mismatch for addition".to_string());
        }

        let size = a.size();
        let (mut result, id) = self.create_tensor_from_pool(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_add(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Element-wise multiplication: result = a * b
    pub fn mul(
        &self,
        a: &CudaTensor<T>,
        b: &CudaTensor<T>,
    ) -> Result<(CudaTensor<T>, u64), String> {
        if a.shape != b.shape {
            return Err("Shape mismatch for multiplication".to_string());
        }

        let size = a.size();
        let (mut result, id) = self.create_tensor_from_pool(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_mul(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Element-wise division: result = a / b
    pub fn div(
        &self,
        a: &CudaTensor<T>,
        b: &CudaTensor<T>,
    ) -> Result<(CudaTensor<T>, u64), String> {
        if a.shape != b.shape {
            return Err("Shape mismatch for division".to_string());
        }

        let size = a.size();
        let (mut result, id) = self.create_tensor_from_pool(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_div(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Element-wise subtraction: result = a - b
    pub fn sub(
        &self,
        a: &CudaTensor<T>,
        b: &CudaTensor<T>,
    ) -> Result<(CudaTensor<T>, u64), String> {
        if a.shape != b.shape {
            return Err("Shape mismatch for subtraction".to_string());
        }

        let size = a.size();
        let (mut result, id) = self.create_tensor_from_pool(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_sub(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Element-wise power: result = a^b
    pub fn power(
        &self,
        a: &CudaTensor<T>,
        b: &CudaTensor<T>,
    ) -> Result<(CudaTensor<T>, u64), String> {
        if a.shape != b.shape {
            return Err("Shape mismatch for power operation".to_string());
        }

        let size = a.size();
        let (mut result, id) = self.create_tensor_from_pool(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_power(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Element-wise minimum: result = min(a, b)
    pub fn min_elementwise(
        &self,

        a: &CudaTensor<T>,
        b: &CudaTensor<T>,
    ) -> Result<(CudaTensor<T>, u64), String> {
        if a.shape != b.shape {
            return Err("Shape mismatch for min operation".to_string());
        }

        let size = a.size();
        let (mut result, id) = self.create_tensor_from_pool(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_min_elementwise(
            cfg,
            &a.data,
            &b.data,
            &mut result.data,
            size as i32,
        )?;
        Ok((result, id))
    }

    /// Element-wise maximum: result = max(a, b)
    pub fn max_elementwise(
        &self,
        a: &CudaTensor<T>,
        b: &CudaTensor<T>,
    ) -> Result<(CudaTensor<T>, u64), String> {
        if a.shape != b.shape {
            return Err("Shape mismatch for max operation".to_string());
        }

        let size = a.size();
        let (mut result, id) = self.create_tensor_from_pool(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_max_elementwise(
            cfg,
            &a.data,
            &b.data,
            &mut result.data,
            size as i32,
        )?;
        Ok((result, id))
    }

    /// Scalar addition: result = tensor + scalar
    pub fn add_scalar(&self, a: &CudaTensor<T>, scalar: T) -> Result<(CudaTensor<T>, u64), String> {
        let (scalar_tensor, id) = self.full(&a.shape, scalar)?;
        let result = self.add(a, &scalar_tensor)?;
        self.return_tensor_to_pool(id, scalar_tensor)?;
        Ok(result)
    }

    /// Scalar multiplication: result = tensor * scalar
    pub fn mul_scalar(&self, a: &CudaTensor<T>, scalar: T) -> Result<(CudaTensor<T>, u64), String> {
        let (scalar_tensor, id) = self.full(&a.shape, scalar)?;
        let result = self.mul(a, &scalar_tensor)?;
        self.return_tensor_to_pool(id, scalar_tensor)?;
        Ok(result)
    }

    /// Scalar division: result = tensor / scalar
    pub fn div_scalar(&self, a: &CudaTensor<T>, scalar: T) -> Result<(CudaTensor<T>, u64), String> {
        let (scalar_tensor, id) = self.full(&a.shape, scalar)?;
        let result = self.div(a, &scalar_tensor)?;
        self.return_tensor_to_pool(id, scalar_tensor)?;
        Ok(result)
    }

    /// Scalar subtraction: result = tensor - scalar
    pub fn sub_scalar(&self, a: &CudaTensor<T>, scalar: T) -> Result<(CudaTensor<T>, u64), String> {
        let (scalar_tensor, id) = self.full(&a.shape, scalar)?;
        let result = self.sub(a, &scalar_tensor)?;
        self.return_tensor_to_pool(id, scalar_tensor)?;
        Ok(result)
    }

    /// Scalar power: result = tensor^scalar
    pub fn power_scalar(
        &self,
        base: &CudaTensor<T>,
        exponent: T,
    ) -> Result<(CudaTensor<T>, u64), String> {
        let (exponent_tensor, id) = self.full(&base.shape, exponent)?;
        let result = self.power(base, &exponent_tensor)?;
        self.return_tensor_to_pool(id, exponent_tensor)?;
        Ok(result)
    }

    /// Element-wise absolute value: result = |input|
    pub fn abs(&self, input: &CudaTensor<T>) -> Result<(CudaTensor<T>, u64), String> {
        let size = input.size();
        let (mut result, id) = self.create_tensor_from_pool(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_abs(cfg, &input.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Element-wise square root: result = sqrt(input)
    pub fn sqrt(&self, input: &CudaTensor<T>) -> Result<(CudaTensor<T>, u64), String> {
        let size = input.size();
        let (mut result, id) = self.create_tensor_from_pool(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_sqrt(cfg, &input.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Element-wise exponential: result = exp(input)
    pub fn exp(&self, input: &CudaTensor<T>) -> Result<(CudaTensor<T>, u64), String> {
        let size = input.size();
        let (mut result, id) = self.create_tensor_from_pool(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_exp(cfg, &input.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Element-wise natural logarithm: result = ln(input)
    pub fn log(&self, input: &CudaTensor<T>) -> Result<(CudaTensor<T>, u64), String> {
        let size = input.size();
        let (mut result, id) = self.create_tensor_from_pool(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_log(cfg, &input.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Element-wise negation: result = -input
    pub fn negate(&self, input: &CudaTensor<T>) -> Result<(CudaTensor<T>, u64), String> {
        let size = input.size();
        let (mut result, id) = self.create_tensor_from_pool(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_negate(cfg, &input.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// ReLU activation: result = max(0, input)
    pub fn relu(&self, input: &CudaTensor<T>) -> Result<(CudaTensor<T>, u64), String> {
        let size = input.size();
        let (mut result, id) = self.create_tensor_from_pool(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_relu(cfg, &input.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Sigmoid activation: result = 1 / (1 + exp(-input))
    pub fn sigmoid(&self, input: &CudaTensor<T>) -> Result<(CudaTensor<T>, u64), String> {
        let size = input.size();
        let (mut result, id) = self.create_tensor_from_pool(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_sigmoid(cfg, &input.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Hyperbolic tangent activation: result = tanh(input)
    pub fn tanh(&self, input: &CudaTensor<T>) -> Result<(CudaTensor<T>, u64), String> {
        let size = input.size();
        let (mut result, id) = self.create_tensor_from_pool(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_tanh(cfg, &input.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Clamp values to specified range: result = clamp(input, min_val, max_val)
    pub fn clamp(
        &self,
        input: &CudaTensor<T>,
        min_val: T,
        max_val: T,
    ) -> Result<(CudaTensor<T>, u64), String> {
        let size = input.size();
        let (mut result, id) = self.create_tensor_from_pool(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_clamp(
            cfg,
            &input.data,
            &mut result.data,
            min_val,
            max_val,
            size as i32,
        )?;
        Ok((result, id))
    }

    /// Sum reduction along specified axes or all elements
    pub fn sum_all(&self, input: &CudaTensor<T>) -> Result<(CudaTensor<T>, u64), String> {
        let size = input.size();
        let (mut result, id) = self.create_tensor_from_pool(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_sum_all(cfg, &input.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Sum reduction along specific axes
    pub fn sum_axes(
        &self,
        input: &CudaTensor<T>,
        axes: &[usize],
        keep_dims: bool,
    ) -> Result<(CudaTensor<T>, u64), String> {
        self.reduce_axes(
            input,
            axes,
            keep_dims,
            |cfg, input_data, output_data, outer_size, axis_size, inner_size| {
                self.kernels.launch_sum_axes(
                    cfg,
                    input_data,
                    output_data,
                    outer_size,
                    axis_size,
                    inner_size,
                )
            },
        )
    }

    /// Max reduction along all elements
    pub fn max_all(&self, input: &CudaTensor<T>) -> Result<(CudaTensor<T>, u64), String> {
        let size = input.size();
        let num_blocks = (size + BLOCK_SIZE as usize - 1) / BLOCK_SIZE as usize;
        let (mut result, id) = self.create_tensor_from_pool(&[num_blocks])?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_max_all(cfg, &input.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Max reduction along specific axes
    pub fn max_axes(
        &self,
        input: &CudaTensor<T>,
        axes: &[usize], // Changed from usize to &[usize]
        keep_dims: bool,
    ) -> Result<(CudaTensor<T>, u64), String> {
        self.reduce_axes(
            input,
            axes,
            keep_dims,
            |cfg, input_data, output_data, outer_size, axis_size, inner_size| {
                self.kernels.launch_max_axes(
                    cfg,
                    input_data,
                    output_data,
                    outer_size,
                    axis_size,
                    inner_size,
                )
            },
        )
    }

    /// Min reduction along all elements
    pub fn min_all(&self, input: &CudaTensor<T>) -> Result<(CudaTensor<T>, u64), String> {
        let size = input.size();
        let num_blocks = (size + BLOCK_SIZE as usize - 1) / BLOCK_SIZE as usize;
        let (mut result, id) = self.create_tensor_from_pool(&[num_blocks])?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_min_all(cfg, &input.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Min reduction along specific axes
    pub fn min_axes(
        &self,
        input: &CudaTensor<T>,
        axes: &[usize],
        keep_dims: bool,
    ) -> Result<(CudaTensor<T>, u64), String> {
        self.reduce_axes(
            input,
            axes,
            keep_dims,
            |cfg, input_data, output_data, outer_size, axis_size, inner_size| {
                self.kernels.launch_min_axes(
                    cfg,
                    input_data,
                    output_data,
                    outer_size,
                    axis_size,
                    inner_size,
                )
            },
        )
    }

    /// Mean operation (uses sum + division)
    pub fn mean_all(&self, input: &CudaTensor<T>) -> Result<(CudaTensor<T>, u64), String> {
        let (sum_result, id) = self.sum_all(input)?;
        let size_scalar = <T as FerroxF>::from_f64(1.0 / input.size() as f64)
            .ok_or("Failed to convert size to tensor type")?;
        let result = self.mul_scalar(&sum_result, size_scalar)?;
        self.return_tensor_to_pool(id, sum_result)?;
        Ok(result)
    }

    /// Mean along specific axes
    pub fn mean_axes(
        &self,
        input: &CudaTensor<T>,
        axes: &[usize],
        keep_dims: bool,
    ) -> Result<(CudaTensor<T>, u64), String> {
        let (sum_result, id) = self.sum_axes(input, axes, keep_dims)?;
        // Calculate divisor as product of all reduced dimensions
        let divisor: f64 = axes.iter().map(|&axis| input.shape[axis] as f64).product();

        let divisor_scalar = <T as crate::backend::number::FerroxF>::from_f64(1.0 / divisor)
            .ok_or("Failed to convert divisor to tensor type")?;

        let result = self.mul_scalar(&sum_result, divisor_scalar)?;
        self.return_tensor_to_pool(id, sum_result)?;
        Ok(result)
    }

    /// Matrix multiplication: C = A @ B
    /// The fundamental operation of neural networks - linear transformations
    ///
    /// # Requirements:
    /// - A must be [M, K], B must be [K, N] -> Result is [M, N]
    /// - This kernel uses optimized tiled matrix multiplication for memory efficiency
    pub fn matmul(
        &self,
        a: &CudaTensor<T>,
        b: &CudaTensor<T>,
    ) -> Result<(CudaTensor<T>, u64), String> {
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
        };

        let result_shape = [a_shape[0], b_shape[1]];

        let (mut result, id) = self.create_tensor_from_pool(&result_shape)?;

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
        Ok((result, id))
    }

    /// Element-wise greater-or-equal: result = (a >= b) ? 1.0 : 0.0
    pub fn greater_equal(
        &self,
        a: &CudaTensor<T>,
        b: &CudaTensor<T>,
    ) -> Result<(CudaTensor<T>, u64), String> {
        if a.shape != b.shape {
            return Err("Shape mismatch for greater_equal operation".to_string());
        }

        let size = a.size();
        let (mut result, id) = self.create_tensor_from_pool(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_greater_equal(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Element-wise less-or-equal: result = (a <= b) ? 1.0 : 0.0
    pub fn less_equal(
        &self,
        a: &CudaTensor<T>,
        b: &CudaTensor<T>,
    ) -> Result<(CudaTensor<T>, u64), String> {
        if a.shape != b.shape {
            return Err("Shape mismatch for less_equal operation".to_string());
        }

        let size = a.size();
        let (mut result, id) = self.create_tensor_from_pool(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_less_equal(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Element-wise greater-or-equal: result = (a >= b) ? 1.0 : 0.0
    pub fn greater(
        &self,
        a: &CudaTensor<T>,
        b: &CudaTensor<T>,
    ) -> Result<(CudaTensor<T>, u64), String> {
        if a.shape != b.shape {
            return Err("Shape mismatch for greater_equal operation".to_string());
        }

        let size = a.size();
        let (mut result, id) = self.create_tensor_from_pool(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_greater(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Element-wise less-or-equal: result = (a <= b) ? 1.0 : 0.0
    pub fn less(
        &self,
        a: &CudaTensor<T>,
        b: &CudaTensor<T>,
    ) -> Result<(CudaTensor<T>, u64), String> {
        if a.shape != b.shape {
            return Err("Shape mismatch for less_equal operation".to_string());
        }

        let size = a.size();
        let (mut result, id) = self.create_tensor_from_pool(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_less(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Element-wise equality: result = (a == b) ? 1.0 : 0.0
    pub fn equal(
        &self,
        a: &CudaTensor<T>,
        b: &CudaTensor<T>,
    ) -> Result<(CudaTensor<T>, u64), String> {
        if a.shape != b.shape {
            return Err("Shape mismatch for equal operation".to_string());
        }

        let size = a.size();
        let (mut result, id) = self.create_tensor_from_pool(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_equal(cfg, &a.data, &b.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Logical NOT: result = (input == 0.0) ? 1.0 : 0.0
    /// Inverts boolean tensors following IEEE 754 convention
    pub fn logical_not(&self, input: &CudaTensor<T>) -> Result<(CudaTensor<T>, u64), String> {
        let size = input.size();
        let (mut result, id) = self.create_tensor_from_pool(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_logical_not(cfg, &input.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Sign function: result = sign(input) ∈ {-1, 0, 1}
    pub fn sign(&self, input: &CudaTensor<T>) -> Result<(CudaTensor<T>, u64), String> {
        let size = input.size();
        let (mut result, id) = self.create_tensor_from_pool(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_sign(cfg, &input.data, &mut result.data, size as i32)?;
        Ok((result, id))
    }

    /// Range check: result = (min_val <= input <= max_val) ? 1.0 : 0.0
    pub fn in_range(
        &self,
        input: &CudaTensor<T>,
        min_val: T,
        max_val: T,
    ) -> Result<(CudaTensor<T>, u64), String> {
        let size = input.size();
        let (mut result, id) = self.create_tensor_from_pool(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_in_range(
            cfg,
            &input.data,
            min_val,
            max_val,
            &mut result.data,
            size as i32,
        )?;
        Ok((result, id))
    }

    ///  condition: result[i] = condition[i] > 0 ? true_val[i] : false_val[i]
    pub fn where_condition(
        &self,
        condition: &CudaTensor<T>,
        true_val: &CudaTensor<T>,
        false_val: &CudaTensor<T>,
    ) -> Result<(CudaTensor<T>, u64), String> {
        if condition.shape != true_val.shape || condition.shape != false_val.shape {
            return Err("Shape mismatch for  condition operation".to_string());
        }

        let size = condition.size();

        let (mut result, id) = self.create_tensor_from_pool(condition.shape())?;

        let cfg = self.get_launch_config(size);

        self.kernels.launch_where_condition(
            cfg,
            &condition.data,
            &true_val.data,
            &false_val.data,
            &mut result.data,
            size as i32,
        )?;
        Ok((result, id))
    }

    // ===== CONVOLUTION OPERATIONS =====

    /// 2D Convolution forward pass
    /// Input: [batch_size, in_channels, in_height, in_width]
    /// Filter: [out_channels, in_channels, kernel_height, kernel_width]
    /// Output: [batch_size, out_channels, out_height, out_width]
    pub fn conv2d_forward(
        &self,
        input: &CudaTensor<T>,
        filter: &CudaTensor<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<(CudaTensor<T>, u64), String> {
        // Validate input dimensions
        if input.shape.len() != 4 || filter.shape.len() != 4 {
            return Err("Conv2D requires 4D tensors [batch, channels, height, width]".to_string());
        }

        let batch_size = input.shape[0];
        let in_channels = input.shape[1];
        let in_height = input.shape[2];
        let in_width = input.shape[3];

        let out_channels = filter.shape[0];
        let filter_in_channels = filter.shape[1];
        let kernel_height = filter.shape[2];
        let kernel_width = filter.shape[3];

        // Validate channel compatibility
        if in_channels != filter_in_channels {
            return Err(format!(
                "Input channels {} don't match filter channels {}",
                in_channels, filter_in_channels
            ));
        }

        let stride_h = stride.0;
        let stride_w = stride.1;
        let pad_h = padding.0;
        let pad_w = padding.1;

        // Calculate output dimensions
        let out_height = (in_height + 2 * pad_h - kernel_height) / stride_h + 1;
        let out_width = (in_width + 2 * pad_w - kernel_width) / stride_w + 1;

        let output_shape = [batch_size, out_channels, out_height, out_width];
        let (mut result, id) = self.create_tensor_from_pool(&output_shape)?;

        // Configure launch parameters
        let tile_size = TILE_SIZE as usize;
        let grid_x = (out_width + tile_size - 1) / tile_size;
        let grid_y = (out_height + tile_size - 1) / tile_size;
        let grid_z = batch_size * out_channels;

        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, grid_z as u32),
            block_dim: (TILE_SIZE, TILE_SIZE, 1),
            shared_mem_bytes: {
                // Calculate shared memory size
                let input_tile_h = tile_size + kernel_height - 1;
                let input_tile_w = tile_size + kernel_width - 1;
                let filter_size = kernel_height * kernel_width;
                ((input_tile_h * input_tile_w + filter_size) * std::mem::size_of::<T>())
                    .try_into()
                    .unwrap()
            },
        };

        // Launch convolution kernel
        self.kernels.launch_conv2d_forward(
            cfg,
            &input.data,
            &filter.data,
            &mut result.data,
            batch_size as i32,
            in_channels as i32,
            in_height as i32,
            in_width as i32,
            out_channels as i32,
            out_height as i32,
            out_width as i32,
            kernel_height as i32,
            kernel_width as i32,
            stride_h as i32,
            stride_w as i32,
            pad_h as i32,
            pad_w as i32,
        )?;

        Ok((result, id))
    }

    /// IN - PLACE OPS (THIS DO NOT HAVE AN ASSOCIATED KERNEL SO DO NOT ATTEMPT TO SEARCH FOR IT)
    pub fn squeeze(&self, input: &mut CudaTensor<T>, axis: Option<usize>) -> Result<(), String> {
        input.squeeze(axis)
    }

    pub fn reshape(&self, input: &mut CudaTensor<T>, new_shape: Vec<usize>) -> Result<(), String> {
        input.reshape(new_shape)
    }

    pub fn transpose(
        &self,
        input: &mut CudaTensor<T>,
        axes: Option<&[usize]>,
    ) -> Result<(), String> {
        input.transpose(axes)
    }
    pub fn unsqueeze(&self, input: &mut CudaTensor<T>, axis: usize) -> Result<(), String> {
        input.unsqueeze(axis)
    }

    pub fn broadcast_to(
        &self,
        input: &mut CudaTensor<T>,
        target_shape: &[usize],
    ) -> Result<(), String> {
        input.broadcast_to(target_shape)
    }

    /// PRIVATE METHODS
    fn reduce_axes<F>(
        &self,
        input: &CudaTensor<T>,
        axes: &[usize],
        keep_dims: bool,
        kernel_launcher: F,
    ) -> Result<(CudaTensor<T>, u64), String>
    where
        F: Fn(LaunchConfig, &CudaSlice<T>, &mut CudaSlice<T>, i32, i32, i32) -> Result<(), String>,
    {
        let input_shape = &input.shape;

        // Validate all axes are within bounds
        for &axis in axes {
            if axis >= input_shape.len() {
                return Err(format!(
                    "Axis {} out of bounds for tensor with {} dimensions",
                    axis,
                    input_shape.len()
                ));
            }
        }

        // Handle single axis case
        if axes.len() == 1 {
            let axis = axes[0];
            let mut output_shape = input_shape.clone();
            if keep_dims {
                output_shape[axis] = 1;
            } else {
                output_shape.remove(axis);
            }

            let outer_size = input_shape[..axis].iter().product::<usize>() as i32;
            let axis_size = input_shape[axis] as i32;
            let inner_size = input_shape[axis + 1..].iter().product::<usize>() as i32;

            let (mut result, id) = self.create_tensor_from_pool(&output_shape)?;

            let cfg = LaunchConfig {
                grid_dim: (outer_size as u32, 1, 1),
                block_dim: (inner_size.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            kernel_launcher(
                cfg,
                &input.data,
                &mut result.data,
                outer_size,
                axis_size,
                inner_size,
            )?;
            return Ok((result, id));
        }

        // Multiple axes case
        let mut final_shape = input_shape.clone();
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_by(|a, b| b.cmp(a));

        if keep_dims {
            for &axis in axes {
                final_shape[axis] = 1;
            }
        } else {
            for &axis in &sorted_axes {
                final_shape.remove(axis);
            }
        }

        let (mut result, id) = self.create_tensor_from_pool(&final_shape)?;
        let mut current_tensor = input;
        let mut current_shape = input_shape.clone();
        let mut temp_tensors = Vec::new();

        for (step_idx, &axis) in sorted_axes.iter().enumerate() {
            let outer_size = current_shape[..axis].iter().product::<usize>() as i32;
            let axis_size = current_shape[axis] as i32;
            let inner_size = current_shape[axis + 1..].iter().product::<usize>() as i32;

            let cfg = LaunchConfig {
                grid_dim: (outer_size as u32, 1, 1),
                block_dim: (inner_size.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            if step_idx == sorted_axes.len() - 1 {
                // Final step - write directly to result
                kernel_launcher(
                    cfg,
                    &current_tensor.data,
                    &mut result.data,
                    outer_size,
                    axis_size,
                    inner_size,
                )?;
            } else {
                // Intermediate step
                let mut step_output_shape = current_shape.clone();
                step_output_shape.remove(axis);

                let (mut temp_tensor, id) = self.create_tensor_from_pool(&step_output_shape)?;

                kernel_launcher(
                    cfg,
                    &current_tensor.data,
                    &mut temp_tensor.data,
                    outer_size,
                    axis_size,
                    inner_size,
                )?;

                temp_tensors.push((temp_tensor, id));
                current_tensor = &temp_tensors[temp_tensors.len() - 1].0;
                current_shape = step_output_shape;
            }
        }

        temp_tensors
            .iter()
            .map(|(t, i)| self.return_tensor_to_pool(*i, t.clone()));

        Ok((result, id))
    }
}
