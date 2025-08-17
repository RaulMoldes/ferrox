// src/backend/cuda/ops.rs
// High-level CUDA tensor operations that mirror CPU tensor operations
// This module provides the user-facing API for neural network operations on GPU

use std::mem::ManuallyDrop;

use super::context::CudaTensor;
use super::kernels::KernelManager;
use crate::backend::cuda::context::compute_strides;
use crate::backend::manager::with_cuda_context;
use crate::backend::manager::{alloc_cuda_slice, return_cuda_slice};
use crate::{FerroxCudaF, FerroxCudaN, FerroxF, FerroxN};
use cudarc::driver::CudaSlice;
use cudarc::driver::LaunchConfig;
use cudarc::driver::ValidAsZeroBits;

const BLOCK_SIZE: u32 = 256;
const TILE_SIZE: u32 = 16;

// CudaOps provides a high-level interface for performing tensor operations on GPU
/// This is the main interface for performing tensor operations on GPU
/// The lifetime parameter ensures operations don't outlive the underlying CUDA resources
pub struct CudaOps<T: FerroxCudaN> {
    kernels: KernelManager,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: FerroxCudaN> CudaOps<T> {
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
        let grid_size = size.div_ceil(block_size) as u32;

        LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
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
        let grid_x = cols.div_ceil(block_size);
        let grid_y = rows.div_ceil(block_size);

        LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (block_size as u32, block_size as u32, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Get optimized launch configuration for reduction operations
    /// This minimizes warp stalls by ensuring proper occupancy and memory coalescing
    fn get_reduction_launch_config(
        &self,
        total_output_elements: usize,
        axis_size: usize,
    ) -> LaunchConfig {
        // Calculate optimal block size based on axis size and GPU occupancy
        let optimal_block_size = if axis_size <= 32 {
            // For small reductions, use smaller blocks to increase occupancy
            64
        } else if axis_size <= 128 {
            128
        } else if axis_size <= 512 {
            256
        } else {
            // For large reductions, use full occupancy
            512
        };

        // Grid size should match the number of output elements
        // Each block handles one output element
        let grid_size = total_output_elements.min(65535); // Max CUDA grid dimension

        LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (optimal_block_size as u32, 1, 1),
            shared_mem_bytes: 0, // We use minimal shared memory in optimized kernels
        }
    }

    /// Helper method to use the cuda memory pool
    fn get_tensor(&self, shape: &[usize]) -> Result<CudaTensor<T>, String> {
        let size = shape.iter().product();
        let pool_alloc = with_cuda_context(|ctx| ctx.alloc_zeros(size))?;

        // Create tensor from pooled allocation
        let strides = compute_strides(shape);

        Ok(CudaTensor {
            allocation: Some(pool_alloc),
            shape: shape.to_vec(),
            strides,
        })
    }

    /// Create tensor filled with constant value
    /// This is a fundamental building block for scalar operations
    /// More efficient than creating dedicated scalar kernels for each operation
    pub fn full(&self, shape: &[usize], value: T) -> Result<CudaTensor<T>, String> {
        let size = shape.iter().product();
        let cfg = self.get_launch_config(size);

        let mut tensor = self.get_tensor(shape)?;

        self.kernels
            .launch_fill(cfg, tensor.data_mut(), value, size.try_into().unwrap())?;

        Ok(tensor)
    }

    /// Create zeros tensor - fundamental for initializing gradients and intermediate results
    pub fn zeros(&self, shape: &[usize]) -> Result<CudaTensor<T>, String> {
        self.full(shape, FerroxN::zero())
    }

    /// Create ones tensor - useful for creating bias vectors and normalization
    pub fn ones(&self, shape: &[usize]) -> Result<CudaTensor<T>, String> {
        let one = <T as FerroxN>::from_f64(1.0).unwrap();
        self.full(shape, one)
    }

    pub fn randn(&self, shape: &[usize], seed: u64) -> Result<CudaTensor<T>, String> {
        let size = shape.iter().product();

        let cfg = self.get_launch_config(size);

        let mut tensor = self.get_tensor(shape)?;

        self.kernels
            .launch_fill_random(cfg, tensor.data_mut(), size as i32, seed)?;

        Ok(tensor)
    }

    /// Materialize tensor using GPU kernel - expands broadcast data efficiently
    pub fn materialize(&self, tensor: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        // Skip if already contiguous
        if !tensor.needs_materialization() {
            return Ok(tensor.clone()); // Should also be checked by the caller for efficiency
        }

        let total_elements = tensor.size();

        // Allocate new contiguous memory for materialized data
        let mut materialized_data = self.get_tensor(tensor.shape())?;

        // Prepare kernel parameters
        let shape_i32: Vec<i32> = tensor.shape.iter().map(|&x| x as i32).collect();
        let strides_i32: Vec<i32> = tensor.strides.iter().map(|&x| x as i32).collect();
        let ndim = tensor.shape.len() as i32;

        // Allocate GPU memory for shape and strides arrays
        let shape_alloc = with_cuda_context(|ctx| ctx.host_to_device(&shape_i32))?;
        let strides_alloc = with_cuda_context(|ctx| ctx.host_to_device(&strides_i32))?;

        // Launch materialization kernel
        let cfg = self.get_launch_config(total_elements);
        let result = self.kernels.launch_materialize(
            cfg,
            tensor.data(),                // Input: original small data
            materialized_data.data_mut(), // Output: expanded data
            shape_alloc.data(),           // GPU allocated shape array
            strides_alloc.data(),         // GPU allocated strides array
            ndim,
            total_elements as i32,
        );

        // Return GPU arrays to pool
        let _ = return_cuda_slice::<i32>(shape_alloc);
        let _ = return_cuda_slice::<i32>(strides_alloc);

        result?;

        Ok(materialized_data)
    }

    /// Element-wise addition: result = a + b
    pub fn add(&self, ain: &CudaTensor<T>, bin: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let a = if ain.needs_materialization() {
            &self.materialize(ain)?
        } else {
            ain
        };

        let b = if bin.needs_materialization() {
            &self.materialize(bin)?
        } else {
            bin
        };

        if a.shape != b.shape {
            return Err("Shape mismatch for addition".to_string());
        }

        let size = a.size();
        let mut result = self.get_tensor(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_add(cfg, a.data(), b.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Element-wise multiplication: result = a * b
    pub fn mul(&self, ain: &CudaTensor<T>, bin: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let a = if ain.needs_materialization() {
            &self.materialize(ain)?
        } else {
            ain
        };

        let b = if bin.needs_materialization() {
            &self.materialize(bin)?
        } else {
            bin
        };

        if a.shape != b.shape {
            return Err("Shape mismatch for multiplication".to_string());
        }

        let size = a.size();
        let mut result = self.get_tensor(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_mul(cfg, a.data(), b.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Element-wise division: result = a / b
    pub fn div(&self, ain: &CudaTensor<T>, bin: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let a = if ain.needs_materialization() {
            &self.materialize(ain)?
        } else {
            ain
        };

        let b = if bin.needs_materialization() {
            &self.materialize(bin)?
        } else {
            bin
        };
        if a.shape != b.shape {
            return Err("Shape mismatch for division".to_string());
        }

        let size = a.size();
        let mut result = self.get_tensor(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_div(cfg, a.data(), b.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Element-wise subtraction: result = a - b
    pub fn sub(&self, ain: &CudaTensor<T>, bin: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let a = if ain.needs_materialization() {
            &self.materialize(ain)?
        } else {
            ain
        };

        let b = if bin.needs_materialization() {
            &self.materialize(bin)?
        } else {
            bin
        };

        if a.shape != b.shape {
            return Err("Shape mismatch for subtraction".to_string());
        }

        let size = a.size();
        let mut result = self.get_tensor(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_sub(cfg, a.data(), b.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Element-wise power: result = a^b
    pub fn power(&self, ain: &CudaTensor<T>, bin: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let a = if ain.needs_materialization() {
            &self.materialize(ain)?
        } else {
            ain
        };

        let b = if bin.needs_materialization() {
            &self.materialize(bin)?
        } else {
            bin
        };

        if a.shape != b.shape {
            return Err("Shape mismatch for power operation".to_string());
        }

        let size = a.size();
        let mut result = self.get_tensor(a.shape())?;

        let cfg = if std::mem::size_of::<T>() == 8 {
            // f64: Use vectorization (2 elements per thread) to reduce L1TEX stalls
            LaunchConfig {
                grid_dim: ((size / 2).div_ceil(512) as u32, 1, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            }
        } else {
            // f32: Use standard configuration (1 element per thread)
            self.get_launch_config(size)
        };

        self.kernels
            .launch_power(cfg, a.data(), b.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Element-wise minimum: result = min(a, b)
    pub fn min_elementwise(
        &self,

        ain: &CudaTensor<T>,
        bin: &CudaTensor<T>,
    ) -> Result<CudaTensor<T>, String> {
        let a = if ain.needs_materialization() {
            &self.materialize(ain)?
        } else {
            ain
        };

        let b = if bin.needs_materialization() {
            &self.materialize(bin)?
        } else {
            bin
        };

        if a.shape != b.shape {
            return Err("Shape mismatch for min operation".to_string());
        }

        let size = a.size();
        let mut result = self.get_tensor(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_min_elementwise(
            cfg,
            a.data(),
            b.data(),
            result.data_mut(),
            size as i32,
        )?;
        Ok(result)
    }

    /// Element-wise maximum: result = max(a, b)
    pub fn max_elementwise(
        &self,
        ain: &CudaTensor<T>,
        bin: &CudaTensor<T>,
    ) -> Result<CudaTensor<T>, String> {
        let a = if ain.needs_materialization() {
            &self.materialize(ain)?
        } else {
            ain
        };

        let b = if bin.needs_materialization() {
            &self.materialize(bin)?
        } else {
            bin
        };

        if a.shape != b.shape {
            return Err("Shape mismatch for max operation".to_string());
        }

        let size = a.size();
        let mut result = self.get_tensor(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_max_elementwise(
            cfg,
            a.data(),
            b.data(),
            result.data_mut(),
            size as i32,
        )?;
        Ok(result)
    }

    /// Scalar addition: result = tensor + scalar
    pub fn add_scalar(&self, a: &CudaTensor<T>, scalar: T) -> Result<CudaTensor<T>, String> {
        let scalar_tensor = self.full(&a.shape, scalar)?;
        let result = self.add(a, &scalar_tensor)?;

        Ok(result)
    }

    /// Scalar multiplication: result = tensor * scalar
    pub fn mul_scalar(&self, a: &CudaTensor<T>, scalar: T) -> Result<CudaTensor<T>, String> {
        let scalar_tensor = self.full(&a.shape, scalar)?;
        let result = self.mul(a, &scalar_tensor)?;

        Ok(result)
    }

    /// Scalar division: result = tensor / scalar
    pub fn div_scalar(&self, a: &CudaTensor<T>, scalar: T) -> Result<CudaTensor<T>, String> {
        let scalar_tensor = self.full(&a.shape, scalar)?;
        let result = self.div(a, &scalar_tensor)?;

        Ok(result)
    }

    /// Scalar subtraction: result = tensor - scalar
    pub fn sub_scalar(&self, a: &CudaTensor<T>, scalar: T) -> Result<CudaTensor<T>, String> {
        let scalar_tensor = self.full(&a.shape, scalar)?;
        let result = self.sub(a, &scalar_tensor)?;

        Ok(result)
    }

    /// Scalar power: result = tensor^scalar
    pub fn power_scalar(&self, base: &CudaTensor<T>, exponent: T) -> Result<CudaTensor<T>, String> {
        let exponent_tensor = self.full(&base.shape, exponent)?;
        let result = self.power(base, &exponent_tensor)?;
        Ok(result)
    }

    pub fn reciprocal(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let size = input.size();
        let mut result = self.get_tensor(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_reciprocal(cfg, input.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Element-wise absolute value: result = |input|
    pub fn abs(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let size = input.size();
        let mut result = self.get_tensor(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_abs(cfg, input.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Element-wise square root: result = sqrt(input)
    pub fn sqrt(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let size = input.size();
        let mut result = self.get_tensor(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_sqrt(cfg, input.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Element-wise exponential: result = exp(input)
    pub fn exp(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let size = input.size();
        let mut result = self.get_tensor(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_exp(cfg, input.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Element-wise natural logarithm: result = ln(input)
    pub fn log(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let size = input.size();
        let mut result = self.get_tensor(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_log(cfg, input.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Element-wise negation: result = -input
    pub fn negate(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let size = input.size();
        let mut result = self.get_tensor(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_negate(cfg, input.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// ReLU activation: result = max(0, input)
    pub fn relu(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let size = input.size();
        let mut result = self.get_tensor(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_relu(cfg, input.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Sigmoid activation: result = 1 / (1 + exp(-input))
    pub fn sigmoid(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let size = input.size();
        let mut result = self.get_tensor(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_sigmoid(cfg, input.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    // SOFTMAX ACTIVATION.
    pub fn softmax(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let size = input.size();
        let mut result = self.get_tensor(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_softmax(cfg, input.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Batch-aware softmax along specified axis
    /// More efficient than partitioning as it processes all batches in parallel
    pub fn softmax_batched(
        &self,
        input: &CudaTensor<T>,
        axis: usize,
    ) -> Result<CudaTensor<T>, String> {
        // Validate axis
        if axis >= input.shape.len() {
            return Err(format!(
                "Softmax axis {} out of bounds for tensor with {} dimensions",
                axis,
                input.shape.len()
            ));
        }

        let mut result = self.get_tensor(input.shape())?;

        // Calculate dimensions for the kernel
        let batch_size = input.shape[..axis].iter().product::<usize>() as i32;
        let seq_length = input.shape[axis] as i32;
        let inner_size = input.shape[axis + 1..].iter().product::<usize>() as i32;
        let total_elements = input.size() as i32;

        // Each block processes one sequence (combination of batch and inner indices)
        // Total number of sequences = batch_size * inner_size
        let num_sequences = batch_size * inner_size;

        // Launch configuration: one block per sequence
        let cfg = LaunchConfig {
            grid_dim: (num_sequences as u32, 1, 1),
            block_dim: (256, 1, 1), // Optimize for sequence length up to 256*k
            shared_mem_bytes: 0,
        };

        self.kernels.launch_softmax_batched(
            cfg,
            input.data(),
            result.data_mut(),
            batch_size,
            seq_length,
            inner_size,
            total_elements,
        )?;

        Ok(result)
    }

    /// Hyperbolic tangent activation: result = tanh(input)
    pub fn tanh(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let size = input.size();
        let mut result = self.get_tensor(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_tanh(cfg, input.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Clamp values to specified range: result = clamp(input, min_val, max_val)
    pub fn clamp(
        &self,
        input: &CudaTensor<T>,
        min_val: T,
        max_val: T,
    ) -> Result<CudaTensor<T>, String> {
        let size = input.size();
        let mut result = self.get_tensor(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_clamp(
            cfg,
            input.data(),
            result.data_mut(),
            min_val,
            max_val,
            size as i32,
        )?;
        Ok(result)
    }

    /// Sum reduction along specified axes or all elements
    pub fn sum_all(&self, input_in: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let input = if input_in.needs_materialization() {
            &self.materialize(input_in)?
        } else {
            input_in
        };

        let size = input.size();
        let mut result = self.get_tensor(&[])?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_sum_all(cfg, input.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Sum reduction along specific axes
    pub fn sum_axes(
        &self,
        input_in: &CudaTensor<T>,
        axes: &[usize],
        keep_dims: bool,
    ) -> Result<CudaTensor<T>, String> {
        let input = if input_in.needs_materialization() {
            &self.materialize(input_in)?
        } else {
            input_in
        };

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
    pub fn max_all(&self, input_in: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let input = if input_in.needs_materialization() {
            &self.materialize(input_in)?
        } else {
            input_in
        };

        let size = input.size();
        let mut result = self.get_tensor(&[])?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_max_all(cfg, input.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Max reduction along specific axes
    pub fn max_axes(
        &self,
        input_in: &CudaTensor<T>,
        axes: &[usize], // Changed from usize to &[usize]
        keep_dims: bool,
    ) -> Result<CudaTensor<T>, String> {
        let input = if input_in.needs_materialization() {
            &self.materialize(input_in)?
        } else {
            input_in
        };

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
    pub fn min_all(&self, input_in: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let input = if input_in.needs_materialization() {
            &self.materialize(input_in)?
        } else {
            input_in
        };

        let size = input.size();

        let mut result = self.get_tensor(&[])?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_min_all(cfg, input.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Min reduction along specific axes
    pub fn min_axes(
        &self,
        input_in: &CudaTensor<T>,
        axes: &[usize],
        keep_dims: bool,
    ) -> Result<CudaTensor<T>, String> {
        let input = if input_in.needs_materialization() {
            &self.materialize(input_in)?
        } else {
            input_in
        };

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
    pub fn mean_all(&self, input_in: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let input = if input_in.needs_materialization() {
            &self.materialize(input_in)?
        } else {
            input_in
        };

        let sum_result = self.sum_all(input)?;
        let size_scalar = <T as FerroxN>::from_f64(1.0 / input.size() as f64)
            .ok_or("Failed to convert size to tensor type")?;
        let result = self.mul_scalar(&sum_result, size_scalar)?;

        Ok(result)
    }

    /// Mean along specific axes
    pub fn mean_axes(
        &self,
        input_in: &CudaTensor<T>,
        axes: &[usize],
        keep_dims: bool,
    ) -> Result<CudaTensor<T>, String> {
        let input = if input_in.needs_materialization() {
            &self.materialize(input_in)?
        } else {
            input_in
        };

        let sum_result = self.sum_axes(input, axes, keep_dims)?;
        // Calculate divisor as product of all reduced dimensions
        let divisor: f64 = axes.iter().map(|&axis| input.shape[axis] as f64).product();

        let divisor_scalar = <T as crate::backend::number::FerroxN>::from_f64(1.0 / divisor)
            .ok_or("Failed to convert divisor to tensor type")?;

        let result = self.mul_scalar(&sum_result, divisor_scalar)?;

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
        ain: &CudaTensor<T>,
        bin: &CudaTensor<T>,
    ) -> Result<CudaTensor<T>, String> {
        let a = if ain.needs_materialization() {
            &self.materialize(ain)?
        } else {
            ain
        };

        let b = if bin.needs_materialization() {
            &self.materialize(bin)?
        } else {
            bin
        };

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

        let mut result = self.get_tensor(&result_shape)?;

        // Use 2D launch configuration optimized for matrix multiplication
        let cfg = self.get_2d_launch_config(a_shape[0], b_shape[1]);

        self.kernels.launch_matmul(
            cfg,
            a.data(),
            b.data(),
            result.data_mut(),
            a_shape[0] as i32, // M
            b_shape[1] as i32, // N
            a_shape[1] as i32, // K
        )?;
        Ok(result)
    }

    /// Element-wise greater-or-equal: result = (a >= b) ? 1.0 : 0.0
    pub fn greater_equal(
        &self,
        ain: &CudaTensor<T>,
        bin: &CudaTensor<T>,
    ) -> Result<CudaTensor<T>, String> {
        let a = if ain.needs_materialization() {
            &self.materialize(ain)?
        } else {
            ain
        };

        let b = if bin.needs_materialization() {
            &self.materialize(bin)?
        } else {
            bin
        };

        if a.shape != b.shape {
            return Err("Shape mismatch for greater_equal operation".to_string());
        }

        let size = a.size();
        let mut result = self.get_tensor(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_greater_equal(
            cfg,
            a.data(),
            b.data(),
            result.data_mut(),
            size as i32,
        )?;
        Ok(result)
    }

    /// Element-wise less-or-equal: result = (a <= b) ? 1.0 : 0.0
    pub fn less_equal(
        &self,
        ain: &CudaTensor<T>,
        bin: &CudaTensor<T>,
    ) -> Result<CudaTensor<T>, String> {
        let a = if ain.needs_materialization() {
            &self.materialize(ain)?
        } else {
            ain
        };

        let b = if bin.needs_materialization() {
            &self.materialize(bin)?
        } else {
            bin
        };

        if a.shape != b.shape {
            return Err("Shape mismatch for less_equal operation".to_string());
        }

        let size = a.size();
        let mut result = self.get_tensor(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_less_equal(cfg, a.data(), b.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Element-wise greater-or-equal: result = (a >= b) ? 1.0 : 0.0
    pub fn greater(
        &self,
        ain: &CudaTensor<T>,
        bin: &CudaTensor<T>,
    ) -> Result<CudaTensor<T>, String> {
        let a = if ain.needs_materialization() {
            &self.materialize(ain)?
        } else {
            ain
        };

        let b = if bin.needs_materialization() {
            &self.materialize(bin)?
        } else {
            bin
        };

        if a.shape != b.shape {
            return Err("Shape mismatch for greater_equal operation".to_string());
        }

        let size = a.size();
        let mut result = self.get_tensor(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_greater(cfg, a.data(), b.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Element-wise less-or-equal: result = (a <= b) ? 1.0 : 0.0
    pub fn less(&self, ain: &CudaTensor<T>, bin: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let a = if ain.needs_materialization() {
            &self.materialize(ain)?
        } else {
            ain
        };

        let b = if bin.needs_materialization() {
            &self.materialize(bin)?
        } else {
            bin
        };

        if a.shape != b.shape {
            return Err("Shape mismatch for less_equal operation".to_string());
        }

        let size = a.size();
        let mut result = self.get_tensor(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_less(cfg, a.data(), b.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    pub fn greater_equal_scalar(
        &self,
        a: &CudaTensor<T>,
        scalar: T,
    ) -> Result<CudaTensor<T>, String> {
        let size = a.size();
        let mut result = self.get_tensor(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_greater_equal_scalar(
            cfg,
            a.data(),
            scalar,
            result.data_mut(),
            size as i32,
        )?;
        Ok(result)
    }

    /// Element-wise less-or-equal: result = (a <= b) ? 1.0 : 0.0
    pub fn less_equal_scalar(&self, a: &CudaTensor<T>, scalar: T) -> Result<CudaTensor<T>, String> {
        let size = a.size();
        let mut result = self.get_tensor(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_less_equal_scalar(
            cfg,
            a.data(),
            scalar,
            result.data_mut(),
            size as i32,
        )?;
        Ok(result)
    }

    /// Element-wise greater-or-equal: result = (a >= b) ? 1.0 : 0.0
    pub fn greater_scalar(&self, a: &CudaTensor<T>, scalar: T) -> Result<CudaTensor<T>, String> {
        let size = a.size();
        let mut result = self.get_tensor(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_greater_scalar(
            cfg,
            a.data(),
            scalar,
            result.data_mut(),
            size as i32,
        )?;
        Ok(result)
    }

    /// Element-wise less-or-equal: result = (a <= b) ? 1.0 : 0.0
    pub fn less_scalar(&self, a: &CudaTensor<T>, scalar: T) -> Result<CudaTensor<T>, String> {
        let size = a.size();
        let mut result = self.get_tensor(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_less_scalar(cfg, a.data(), scalar, result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Element-wise equality: result = (a == b) ? 1.0 : 0.0
    pub fn equal(&self, ain: &CudaTensor<T>, bin: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let a = if ain.needs_materialization() {
            &self.materialize(ain)?
        } else {
            ain
        };

        let b = if bin.needs_materialization() {
            &self.materialize(bin)?
        } else {
            bin
        };

        if a.shape != b.shape {
            return Err("Shape mismatch for equal operation".to_string());
        }

        let size = a.size();
        let mut result = self.get_tensor(a.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_equal(cfg, a.data(), b.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Logical NOT: result = (input == 0.0) ? 1.0 : 0.0
    /// Inverts boolean tensors following IEEE 754 convention
    pub fn logical_not(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let size = input.size();
        let mut result = self.get_tensor(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_logical_not(cfg, input.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Sign function: result = sign(input) ∈ {-1, 0, 1}
    pub fn sign(&self, input: &CudaTensor<T>) -> Result<CudaTensor<T>, String> {
        let size = input.size();
        let mut result = self.get_tensor(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels
            .launch_sign(cfg, input.data(), result.data_mut(), size as i32)?;
        Ok(result)
    }

    /// Range check: result = (min_val <= input <= max_val) ? 1.0 : 0.0
    pub fn in_range(
        &self,
        input: &CudaTensor<T>,
        min_val: T,
        max_val: T,
    ) -> Result<CudaTensor<T>, String> {
        let size = input.size();
        let mut result = self.get_tensor(input.shape())?;
        let cfg = self.get_launch_config(size);

        self.kernels.launch_in_range(
            cfg,
            input.data(),
            min_val,
            max_val,
            result.data_mut(),
            size as i32,
        )?;
        Ok(result)
    }

    ///  condition: result[i] = condition[i] > 0 ? true_val[i] : false_val[i]
    pub fn where_condition(
        &self,
        condition_in: &CudaTensor<T>,
        true_val_in: &CudaTensor<T>,
        false_val_in: &CudaTensor<T>,
    ) -> Result<CudaTensor<T>, String> {
        let condition = if condition_in.needs_materialization() {
            &self.materialize(condition_in)?
        } else {
            condition_in
        };

        let true_val = if true_val_in.needs_materialization() {
            &self.materialize(true_val_in)?
        } else {
            true_val_in
        };

        let false_val = if false_val_in.needs_materialization() {
            &self.materialize(false_val_in)?
        } else {
            false_val_in
        };

        if condition.shape != true_val.shape || condition.shape != false_val.shape {
            return Err("Shape mismatch for  condition operation".to_string());
        }

        let size = condition.size();

        let mut result = self.get_tensor(condition.shape())?;

        let cfg = self.get_launch_config(size);

        self.kernels.launch_where_condition(
            cfg,
            condition.data(),
            true_val.data(),
            false_val.data(),
            result.data_mut(),
            size as i32,
        )?;
        Ok(result)
    }

    // ===== CONVOLUTION OPERATIONS =====

    /// 2D Convolution forward pass
    /// Input: [batch_size, in_channels, in_height, in_width]
    /// Filter: [out_channels, in_channels, kernel_height, kernel_width]
    /// Output: [batch_size, out_channels, out_height, out_width]
    pub fn conv2d_forward(
        &self,
        input_in: &CudaTensor<T>,
        filter_in: &CudaTensor<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<CudaTensor<T>, String> {
        let input = if input_in.needs_materialization() {
            &self.materialize(input_in)?
        } else {
            input_in
        };

        let filter = if filter_in.needs_materialization() {
            &self.materialize(filter_in)?
        } else {
            filter_in
        };

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
        let mut result = self.get_tensor(&output_shape)?;

        // Configure launch parameters

        let grid_x = out_width.div_ceil(TILE_SIZE as usize);
        let grid_y = out_height.div_ceil(TILE_SIZE as usize);
        let grid_z = batch_size * out_channels;

        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, grid_z as u32),
            block_dim: (TILE_SIZE, TILE_SIZE, 1),
            shared_mem_bytes: {
                // Calculate shared memory size
                let input_tile_h = TILE_SIZE as usize + kernel_height - 1;
                let input_tile_w = TILE_SIZE as usize + kernel_width - 1;
                let filter_size = kernel_height * kernel_width;
                ((input_tile_h * input_tile_w + filter_size) * std::mem::size_of::<T>())
                    .try_into()
                    .unwrap()
            },
        };

        // Launch convolution kernel
        self.kernels.launch_conv2d_forward(
            cfg,
            input.data(),
            filter.data(),
            result.data_mut(),
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

        Ok(result)
    }

    pub fn conv1d_forward(
        &self,
        input_in: &CudaTensor<T>,
        filter_in: &CudaTensor<T>,
    ) -> Result<CudaTensor<T>, String> {
        // Materialize inputs if needed
        let input = if input_in.needs_materialization() {
            &self.materialize(input_in)?
        } else {
            input_in
        };

        let filter = if filter_in.needs_materialization() {
            &self.materialize(filter_in)?
        } else {
            filter_in
        };

        // Simple 1D convolution - expect flattened tensors
        let input_size = input.shape.iter().product::<usize>();
        let kernel_size = filter.shape.iter().product::<usize>();

        // Calculate output size
        let output_size = input_size - kernel_size + 1;
        if output_size == 0 {
            return Err("Kernel size cannot be larger than input size".to_string());
        }

        let output_shape = [output_size];
        let mut result = self.get_tensor(&output_shape)?;

        // Configure launch parameters to match benchmark
        let grid_size = output_size.div_ceil(BLOCK_SIZE as usize);
        let shared_mem_size =
            (kernel_size + BLOCK_SIZE as usize + kernel_size - 1) * std::mem::size_of::<T>();

        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: shared_mem_size as u32,
        };

        // Launch kernel
        self.kernels.launch_conv1d_forward(
            cfg,
            input.data(),
            filter.data(),
            result.data_mut(),
            input_size as i32,
            kernel_size as i32,
        )?;

        Ok(result)
    }



    pub fn cross_correlation1d(
    &self,
    input1_in: &CudaTensor<T>,
    input2_in: &CudaTensor<T>,
) -> Result<CudaTensor<T>, String> {
    // Materialize inputs if needed
    let input1 = if input1_in.needs_materialization() {
        &self.materialize(input1_in)?
    } else {
        input1_in
    };

    let input2 = if input2_in.needs_materialization() {
        &self.materialize(input2_in)?
    } else {
        input2_in
    };

    // Calculate sizes
    let input1_size = input1.shape.iter().product::<usize>();
    let input2_size = input2.shape.iter().product::<usize>();
    let kernel_size = input1_size - input2_size + 1;

    if kernel_size == 0 {
        return Err("Invalid sizes for cross-correlation".to_string());
    }

    let output_shape = [kernel_size];
    let mut result = self.get_tensor(&output_shape)?;

    // Configure launch parameters
    let grid_size = kernel_size.div_ceil(BLOCK_SIZE as usize);

    let cfg = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    // Launch kernel
    self.kernels.launch_cross_correlation1d(
        cfg,
        input1.data(),
        input2.data(),
        result.data_mut(),
        input1_size as i32,
        input2_size as i32,
        kernel_size as i32,
    )?;

    Ok(result)
}


pub fn deconv1d(
    &self,
    input_in: &CudaTensor<T>,
    filter_in: &CudaTensor<T>,
) -> Result<CudaTensor<T>, String> {
    // Materialize inputs if needed
    let grad_output = if input_in.needs_materialization() {
        &self.materialize(input_in)?
    } else {
        input_in
    };

    let filter = if filter_in.needs_materialization() {
        &self.materialize(filter_in)?
    } else {
        filter_in
    };

    // Calculate sizes
    let grad_size = grad_output.shape.iter().product::<usize>();
    let kernel_size = filter.shape.iter().product::<usize>();
    let input_size = grad_size + kernel_size - 1;

    let output_shape = [input_size];
    let mut result = self.get_tensor(&output_shape)?;

    // Configure launch parameters
    let grid_size = input_size.div_ceil(BLOCK_SIZE as usize);

    let cfg = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    // Launch deconv1d kernel
    self.kernels.launch_deconv1d(
        cfg,
        grad_output.data(),
        filter.data(),
        result.data_mut(),
        grad_size as i32,
        kernel_size as i32,
        input_size as i32,
    )?;

    Ok(result)
}

    #[allow(clippy::too_many_arguments)]
    pub fn deconv2d(
        &self,
        input_in: &CudaTensor<T>,
        filter_in: &CudaTensor<T>,
        input_shape: &[usize],
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<CudaTensor<T>, String> {
        let input = if input_in.needs_materialization() {
            &self.materialize(input_in)?
        } else {
            input_in
        };

        let filter = if filter_in.needs_materialization() {
            &self.materialize(filter_in)?
        } else {
            filter_in
        };

        // Validate dimensions
        if input.shape.len() != 4 || filter.shape.len() != 4 || input_shape.len() != 4 {
            return Err("Conv2d backward requires 4D tensors".to_string());
        }

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];

        let out_channels = input.shape[1];
        let out_height = input.shape[2];
        let out_width = input.shape[3];

        let kernel_height = filter.shape[2];
        let kernel_width = filter.shape[3];

        let stride_h = stride.0;
        let stride_w = stride.1;
        let pad_h = padding.0;
        let pad_w = padding.1;

        // Create output tensor with input shape
        let mut result = self.get_tensor(input_shape)?;

        // Configure launch parameters for input gradient
        let grid_x = in_width.div_ceil(TILE_SIZE as usize);
        let grid_y = in_height.div_ceil(TILE_SIZE as usize);
        let grid_z = batch_size * in_channels;

        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, grid_z as u32),
            block_dim: (TILE_SIZE, TILE_SIZE, 1),
            shared_mem_bytes: {
                let filter_size = kernel_height * kernel_width;
                (filter_size * std::mem::size_of::<T>()).try_into().unwrap()
            },
        };

        // Launch backward input kernel
        self.kernels.launch_deconv2d(
            cfg,
            input.data(),
            filter.data(),
            result.data_mut(),
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

        Ok(result)
    }

    /// 2D Convolution backward pass - gradient w.r.t. filter weights
    /// Used in automatic differentiation for Conv2d operator
    #[allow(clippy::too_many_arguments)]
    pub fn cross_correlation2d(
        &self,
        input_in1: &CudaTensor<T>,
        input_in2: &CudaTensor<T>,
        output_shape: &[usize],
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<CudaTensor<T>, String> {
        let input1 = if input_in1.needs_materialization() {
            &self.materialize(input_in1)?
        } else {
            input_in1
        };

        let input2 = if input_in2.needs_materialization() {
            &self.materialize(input_in2)?
        } else {
            input_in2
        };

        // Validate dimensions
        if input1.shape.len() != 4 || input2.shape.len() != 4 || output_shape.len() != 4 {
            return Err("Cross correlation requires 4D tensors".to_string());
        }

        let batch_size = input1.shape[0];
        let in_channels = input1.shape[1];
        let in_height = input1.shape[2];
        let in_width = input1.shape[3];

        let out_channels = output_shape[0];
        let out_height = input2.shape[2];
        let out_width = input2.shape[3];

        let kernel_height = output_shape[2];
        let kernel_width = output_shape[3];

        let stride_h = stride.0;
        let stride_w = stride.1;
        let pad_h = padding.0;
        let pad_w = padding.1;

        // Create output tensor with filter shape
        let mut result = self.get_tensor(output_shape)?;

        // Configure launch parameters for filter gradient
        let grid_x = kernel_width.div_ceil(16);
        let grid_y = kernel_height.div_ceil(16);
        let grid_z = in_channels * out_channels;

        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, grid_z as u32),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0, // No shared memory needed for filter gradient
        };

        // Launch backward filter kernel
        self.kernels.launch_cross_correlation2d(
            cfg,
            input1.data(),
            input2.data(),
            result.data_mut(),
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

        Ok(result)
    }

    pub fn debug_cuda_tensor(&self, tensor: &CudaTensor<T>, name: &str) -> Result<(), String> {
        with_cuda_context(|ctx| {
            let data = tensor.clone().to_vec(ctx)?;
            println!("{}: {:?}", name, data);
            Ok(())
        })?;
        Ok(())
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
    ) -> Result<CudaTensor<T>, String>
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

            let mut result = self.get_tensor(&output_shape)?;

            let total_output_elements = (outer_size * inner_size) as usize;

            let cfg = self.get_reduction_launch_config(total_output_elements, axis_size as usize);

            kernel_launcher(
                cfg,
                input.data(),
                result.data_mut(),
                outer_size,
                axis_size,
                inner_size,
            )?;
            return Ok(result);
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

        let mut result = self.get_tensor(&final_shape)?;
        let mut current_tensor = input;
        let mut current_shape = input_shape.clone();
        let mut temp_tensors = Vec::new();

        for (step_idx, &axis) in sorted_axes.iter().enumerate() {
            let outer_size = current_shape[..axis].iter().product::<usize>() as i32;
            let axis_size = current_shape[axis] as i32;
            let inner_size = current_shape[axis + 1..].iter().product::<usize>() as i32;
            let total_output_elements = (outer_size * inner_size) as usize;
            let cfg = self.get_reduction_launch_config(total_output_elements, axis_size as usize);
            if step_idx == sorted_axes.len() - 1 {
                // Final step - write directly to result
                kernel_launcher(
                    cfg,
                    current_tensor.data(),
                    result.data_mut(),
                    outer_size,
                    axis_size,
                    inner_size,
                )?;
            } else {
                // Intermediate step
                let mut step_output_shape = current_shape.clone();
                step_output_shape.remove(axis);

                let mut temp_tensor = self.get_tensor(&step_output_shape)?;

                kernel_launcher(
                    cfg,
                    current_tensor.data(),
                    temp_tensor.data_mut(),
                    outer_size,
                    axis_size,
                    inner_size,
                )?;

                temp_tensors.push(temp_tensor);
                current_tensor = &temp_tensors[temp_tensors.len() - 1];
                current_shape = step_output_shape;
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod ops_test {
    use super::*;

    use crate::backend::with_cuda_ops;

    #[test]
    fn test_bias_broadcasting() {
        // Reproduce exactamente el error que muestras: bias [0.1] -> broadcasted debería ser [0.1, 0.1, 0.1, 0.1]
        // pero está dando [0.1, -0.6, 0.0, 0.6]

        let result = with_cuda_ops(|ops: &CudaOps<f32>| {
            // Crear bias como en tu ejemplo
            let bias_data = vec![0.1f32];
            let mut bias_tensor =
                with_cuda_context(|ctx| CudaTensor::from_vec(ctx, bias_data.clone(), vec![1]))?;

            println!("Original bias shape: {:?}", bias_tensor.shape());
            println!("Original bias data: {:?}", bias_data);

            // Broadcast a [4] como en tu ejemplo
            ops.broadcast_to(&mut bias_tensor, &[4])?;

            println!("After broadcast shape: {:?}", bias_tensor.shape());
            println!("After broadcast strides: {:?}", bias_tensor.strides());
            println!(
                "Needs materialization: {}",
                bias_tensor.needs_materialization()
            );

            // Materializar
            let materialized = ops.materialize(&bias_tensor)?;
            let result_data = with_cuda_context(|ctx| materialized.to_vec(ctx))?;

            println!("Materialized data: {:?}", result_data);

            // Verificar que todos los valores son 0.1
            for (i, &value) in result_data.iter().enumerate() {
                if (value - 0.1).abs() > 1e-6 {
                    return Err(format!(
                        "Error en posición {}: esperado 0.1, obtenido {}",
                        i, value
                    ));
                }
            }

            Ok(result_data)
        });

        assert!(
            result.is_ok(),
            "Test de broadcast falló: {:?}",
            result.err()
        );
        let data = result.unwrap();
        assert_eq!(
            data,
            vec![0.1f32; 4],
            "Los datos no coinciden con el patrón esperado"
        );
    }
}
