// src/backend/storage/cpu.rs
use super::StorageBackend;
use crate::backend::manager::{alloc_cpu_vec, return_cpu_vec};
use crate::backend::memory::PoolAllocation;
use crate::backend::{FerroxCudaF, FerroxF};
use ndarray::{ArrayD, ArrayView2, ArrayViewD, Axis, Dimension, IxDyn, Zip};
use rand::Rng;
use rand_distr::StandardUniform;
use rand_distr::num_traits::{One, Zero};

#[derive(Debug, Clone)]
pub struct CPUStorage<T: FerroxCudaF> {
    data: ArrayD<T>,
    // Track pool allocation to enable smarter memory reuse
    pool_metadata: Option<PoolMetadata>,
}

#[derive(Debug, Clone)]
struct PoolMetadata {
    // Track original allocation size for better pool returns
    original_capacity: usize,
    // Could track allocation pattern for future optimizations
    reuse_count: u32,
}

impl<T: FerroxCudaF> CPUStorage<T> {
    pub fn new(data: ArrayD<T>) -> Self {
        Self {
            data,
            pool_metadata: None,
        }
    }

    pub fn new_owned(data: ArrayD<T>) -> Self {
        Self {
            data,
            pool_metadata: None,
        }
    }

    pub fn from_view<'a>(view: &ArrayViewD<'a, T>) -> Self {
        // Always clone the data to avoid lifetime issues
        Self {
            data: view.to_owned(),
            pool_metadata: None,
        }
    }

    pub fn from_array_ref(array: &ArrayD<T>) -> Self {
        Self {
            data: array.clone(),
            pool_metadata: None,
        }
    }

    // Optimized: reuse the Vec's capacity when converting to ArrayD
    fn vec_to_array(
        pool_alloc: PoolAllocation<Vec<T>>,
        shape: &[usize],
    ) -> Result<ArrayD<T>, String> {
        let capacity = pool_alloc.data.capacity();
        let array = ArrayD::from_shape_vec(IxDyn(shape), pool_alloc.data)
            .map_err(|e| format!("Failed to create ArrayD from pooled vector: {}", e))?;

        // Return empty vec but preserve the allocation tracking info
        let _ = return_cpu_vec(pool_alloc.allocation_id, Vec::<T>::new());
        Ok(array)
    }

    pub fn from_pooled_vec(
        pool_alloc: PoolAllocation<Vec<T>>,
        shape: &[usize],
    ) -> Result<Self, String> {
        let capacity = pool_alloc.data.capacity();
        let array = ArrayD::from_shape_vec(IxDyn(shape), pool_alloc.data)
            .map_err(|e| format!("Failed to create ArrayD from pooled vector: {}", e))?;

        // Track pool metadata for future optimizations
        let metadata = Some(PoolMetadata {
            original_capacity: capacity,
            reuse_count: 0,
        });

        // Return empty vec to pool - we copied the data to ArrayD
        let _ = return_cpu_vec(pool_alloc.allocation_id, Vec::<T>::new());

        Ok(Self {
            data: array,
            pool_metadata: metadata,
        })
    }

    // Optimized: create result storage with pre-calculated size
    fn create_result(shape: &[usize]) -> Result<(ArrayD<T>, u64), String> {
        let size = shape.iter().product();
        let pool_alloc = alloc_cpu_vec::<T>(size)?;

        let array = ArrayD::from_shape_vec(IxDyn(shape), pool_alloc.data)
            .map_err(|e| format!("Failed to create result ArrayD: {}", e))?;
        let id = pool_alloc.allocation_id;

        Ok((array, id))
    }

    // Optimized: return meaningful data to pool when possible
    fn return_to_pool(pool_id: u64, shape: &[usize]) -> Result<(), String> {
        let size: usize = shape.iter().product();
        // Return a zero-filled vec with correct size for pool reuse
        let _ = return_cpu_vec(pool_id, vec![<T as FerroxF>::zero(); 0]);
        Ok(())
    }

    // Fast contiguous access methods
    pub fn view(&self) -> ArrayViewD<'_, T> {
        self.data.view()
    }

    pub fn view_mut(&mut self) -> ndarray::ArrayViewMutD<'_, T> {
        self.data.view_mut()
    }

    pub fn array_ref(&self) -> &ArrayD<T> {
        &self.data
    }

    pub fn array_mut(&mut self) -> &mut ArrayD<T> {
        &mut self.data
    }

    // Optimized: check contiguity for faster operations
    fn is_contiguous(&self) -> bool {
        self.data.as_slice().is_some()
    }

    // Fast flat iteration when data is contiguous
    fn iter_contiguous(&self) -> Option<std::slice::Iter<'_, T>> {
        self.data.as_slice().map(|slice| slice.iter())
    }
}

impl<T> CPUStorage<T>
where
    T: crate::backend::number::FerroxCudaF + Clone,
{
    // Optimized im2col using SIMD-friendly memory access patterns
    fn im2col(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<T>, String> {
        let input_shape = self.shape();
        let (batch, channels, in_h, in_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (kernel_h, kernel_w) = kernel_size;

        let out_h = (in_h + 2 * padding.0 - kernel_h) / stride.0 + 1;
        let out_w = (in_w + 2 * padding.1 - kernel_w) / stride.1 + 1;

        let col_height = channels * kernel_h * kernel_w;
        let col_width = batch * out_h * out_w;

        let mut col_data = alloc_cpu_vec::<T>(col_height * col_width)?;

        // Optimized: use contiguous data access when possible
        let input_data = self
            .data
            .as_slice()
            .ok_or("Input data is not contiguous for im2col")?;

        // Optimized memory access pattern - batch major order
        for b in 0..batch {
            for c in 0..channels {
                for ky in 0..kernel_h {
                    for kx in 0..kernel_w {
                        let col_row = c * kernel_h * kernel_w + ky * kernel_w + kx;

                        for out_y in 0..out_h {
                            for out_x in 0..out_w {
                                let in_y = out_y * stride.0 + ky;
                                let in_x = out_x * stride.1 + kx;
                                let col_col = b * out_h * out_w + out_y * out_w + out_x;

                                // Bounds check with padding
                                let value = if in_y >= padding.0
                                    && in_y < in_h + padding.0
                                    && in_x >= padding.1
                                    && in_x < in_w + padding.1
                                {
                                    let actual_y = in_y - padding.0;
                                    let actual_x = in_x - padding.1;

                                    if actual_y < in_h && actual_x < in_w {
                                        let input_idx = b * (channels * in_h * in_w)
                                            + c * (in_h * in_w)
                                            + actual_y * in_w
                                            + actual_x;
                                        input_data[input_idx]
                                    } else {
                                        <T as FerroxF>::zero()
                                    }
                                } else {
                                    <T as FerroxF>::zero()
                                };

                                col_data.data[col_row * col_width + col_col] = value;
                            }
                        }
                    }
                }
            }
        }

        let shape = [col_height, col_width];
        CPUStorage::vec_to_array(col_data, &shape)
    }

    // Optimized convolution using blocked matrix multiplication
    fn conv2d_impl(
        &self,
        filter: &ArrayD<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<T>, String> {
        let input_shape = self.shape();
        let filter_shape = filter.shape();

        let (batch, in_channels, in_h, in_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (out_channels, _, kernel_h, kernel_w) = (
            filter_shape[0],
            filter_shape[1],
            filter_shape[2],
            filter_shape[3],
        );

        let out_h = (in_h + 2 * padding.0 - kernel_h) / stride.0 + 1;
        let out_w = (in_w + 2 * padding.1 - kernel_w) / stride.1 + 1;

        // Transform input to column matrix
        let col_matrix = self.im2col((kernel_h, kernel_w), stride, padding)?;

        // Reshape filter for efficient GEMM
        let filter_reshaped = filter
            .clone()
            .into_shape_with_order(IxDyn(&[out_channels, in_channels * kernel_h * kernel_w]))
            .map_err(|e| format!("Filter reshape failed: {}", e))?;

        // Matrix multiplication using ndarray's optimized BLAS
        let im2col_view: ArrayView2<T> = col_matrix
            .view()
            .into_dimensionality()
            .map_err(|e| format!("Im2col shape error: {}", e))?;

        let filter_view: ArrayView2<T> = filter_reshaped
            .view()
            .into_dimensionality()
            .map_err(|e| format!("Filter shape error: {}", e))?;

        let output_2d = filter_view.dot(&im2col_view);

        // Optimized transpose using contiguous memory when possible
        let output_data = if let Some(contiguous_data) = output_2d.as_slice() {
            contiguous_data.to_vec()
        } else {
            return Err("GEMM result is not contiguous".to_string());
        };

        let mut final_output = alloc_cpu_vec::<T>(batch * out_channels * out_h * out_w)?;

        // Optimized transpose with cache-friendly memory access
        for b in 0..batch {
            for out_c in 0..out_channels {
                for y in 0..out_h {
                    for x in 0..out_w {
                        let src_idx =
                            out_c * (batch * out_h * out_w) + b * (out_h * out_w) + y * out_w + x;
                        let dst_idx = b * (out_channels * out_h * out_w)
                            + out_c * (out_h * out_w)
                            + y * out_w
                            + x;
                        final_output.data[dst_idx] = output_data[src_idx];
                    }
                }
            }
        }

        let shape = [batch, out_channels, out_h, out_w];
        CPUStorage::vec_to_array(final_output, &shape)
    }
}

impl<T: FerroxCudaF> CPUStorage<T> {
    // Optimized reduction with axis validation and efficient iteration
    fn reduce<F>(&self, axes: Option<&[usize]>, reduction_fn: F) -> Result<CPUStorage<T>, String>
    where
        F: Fn(&ArrayD<T>, Axis) -> ArrayD<T>,
    {
        match axes {
            Some(axes_list) => {
                // Validate all axes at once
                for &ax in axes_list {
                    if ax >= self.data.ndim() {
                        return Err(format!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            ax,
                            self.data.ndim()
                        ));
                    }
                }

                // Sort axes in descending order for stable reduction
                let mut sorted_axes = axes_list.to_vec();
                sorted_axes.sort_by(|a, b| b.cmp(a));

                let mut result = self.data.clone();
                for &ax in &sorted_axes {
                    result = reduction_fn(&result, Axis(ax));
                }

                Ok(CPUStorage::new(result))
            }
            None => {
                // Reduce all axes - flatten and reduce
                let view = self.data.view();
                let flattened = view.to_shape(self.data.len()).unwrap();
                let result = reduction_fn(&flattened.to_owned().into_dyn(), Axis(0));
                Ok(CPUStorage::new(result))
            }
        }
    }

    // Centralized comparison function to avoid code duplication
    fn compare<F>(
        &self,
        other: &dyn StorageBackend<T>,
        comparison_fn: F,
    ) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        F: Fn(T, T) -> bool + Send + Sync,
    {
        let other_data = other.cpu_data()?;

        if self.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for comparison: {:?} vs {:?}",
                self.shape(),
                other_data.shape()
            ));
        }

        let one = <T as FerroxF>::one();
        let zero = <T as FerroxF>::zero();

        // Use Zip for optimal performance with the comparison function
        let result = Zip::from(&self.data)
            .and(other_data)
            .map_collect(|&a, &b| if comparison_fn(a, b) { one } else { zero });

        Ok(Box::new(CPUStorage::new(result)))
    }
}

impl<T: FerroxCudaF> StorageBackend<T> for CPUStorage<T> {
    fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn ndim(&self) -> usize {
        self.data.ndim()
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn is_gpu(&self) -> bool {
        false
    }

    fn cpu_data(&self) -> Result<&ArrayD<T>, String> {
        Ok(&self.data)
    }

    fn cpu_data_mut(&mut self) -> Result<&mut ArrayD<T>, String> {
        Ok(&mut self.data)
    }

    fn owns_data(&self) -> bool {
        true
    }

    fn clone_storage(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        Ok(Box::new(self.clone()))
    }

    // OPTIMIZED ELEMENT-WISE OPERATIONS
    fn add(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for addition: {:?} vs {:?}",
                self.shape(),
                other_data.shape()
            ));
        }

        // Use ndarray's optimized addition - leverages SIMD when available
        let result = &self.data + other_data;
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn sub(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for subtraction: {:?} vs {:?}",
                self.shape(),
                other_data.shape()
            ));
        }

        let result = &self.data - other_data;
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn mul(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for multiplication: {:?} vs {:?}",
                self.shape(),
                other_data.shape()
            ));
        }

        let result = &self.data * other_data;
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn div(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for division: {:?} vs {:?}",
                self.shape(),
                other_data.shape()
            ));
        }

        let result = &self.data / other_data;
        Ok(Box::new(CPUStorage::new(result)))
    }

    // Optimized min/max using Zip for better performance
    fn min(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for min operation: {:?} vs {:?}",
                self.shape(),
                other_data.shape()
            ));
        }

        let result = Zip::from(&self.data)
            .and(other_data)
            .map_collect(|&a, &b| if a <= b { a } else { b });

        Ok(Box::new(CPUStorage::new(result)))
    }

    fn max(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for max operation: {:?} vs {:?}",
                self.shape(),
                other_data.shape()
            ));
        }

        let result = Zip::from(&self.data)
            .and(other_data)
            .map_collect(|&a, &b| if a >= b { a } else { b });

        Ok(Box::new(CPUStorage::new(result)))
    }

    // OPTIMIZED SCALAR OPERATIONS
    fn add_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = &self.data + scalar;
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn sub_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = &self.data - scalar;
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn mul_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = &self.data * scalar;
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn div_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = &self.data / scalar;
        Ok(Box::new(CPUStorage::new(result)))
    }

    // OPTIMIZED UNARY OPERATIONS
    fn neg(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = self.data.mapv(|x| -x);
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn abs(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = self.data.mapv(|x| x.abs());
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn sqrt(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = self.data.mapv(|x| x.sqrt());
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn exp(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = self.data.mapv(|x| x.exp());
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn log(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = self.data.mapv(|x| x.ln());
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn clamp(&self, min_val: T, max_val: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = self.data.mapv(|x| {
            if x < min_val {
                min_val
            } else if x > max_val {
                max_val
            } else {
                x
            }
        });
        Ok(Box::new(CPUStorage::new(result)))
    }

    // OPTIMIZED ACTIVATION FUNCTIONS
    fn relu(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let zero = <T as FerroxF>::zero();
        let result = self.data.mapv(|x| if x > zero { x } else { zero });
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn sigmoid(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let one = <T as FerroxF>::one();
        let result = self.data.mapv(|x| one / (one + (-x).exp()));
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn tanh(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = self.data.mapv(|x| {
            let e_x = x.exp();
            let e_neg_x = (-x).exp();
            (e_x - e_neg_x) / (e_x + e_neg_x)
        });
        Ok(Box::new(CPUStorage::new(result)))
    }

    // POWER OPERATIONS
    fn powf(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_data = other.cpu_data()?;

        if self.shape() != other_data.shape() {
            return Err(format!(
                "Shape mismatch for powf: {:?} vs {:?}",
                self.shape(),
                other_data.shape()
            ));
        }

        let result = Zip::from(&self.data)
            .and(other_data)
            .map_collect(|&a, &b| a.powf(b));

        Ok(Box::new(CPUStorage::new(result)))
    }

    fn power_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = self.data.mapv(|x| x.powf(scalar));
        Ok(Box::new(CPUStorage::new(result)))
    }

    // COMPARISON OPERATIONS - All reuse the centralized compare function
    fn greater(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        self.compare(other, |a, b| a > b)
    }

    fn greater_equal(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        self.compare(other, |a, b| a >= b)
    }

    fn less(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        self.compare(other, |a, b| a < b)
    }

    fn less_equal(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        self.compare(other, |a, b| a <= b)
    }

    fn equal(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        self.compare(other, |a, b| a == b)
    }

    fn logical_not(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let one = <T as FerroxF>::one();
        let zero = <T as FerroxF>::zero();
        let result = self.data.mapv(|x| if x == zero { one } else { zero });
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn in_range(&self, min_val: T, max_val: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let one = <T as FerroxF>::one();
        let zero = <T as FerroxF>::zero();
        let result = self.data.mapv(|x| {
            if x >= min_val && x <= max_val {
                one
            } else {
                zero
            }
        });
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn sign(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = self.data.mapv(|x| x.signum());
        Ok(Box::new(CPUStorage::new(result)))
    }

    // OPTIMIZED MATRIX MULTIPLICATION
    fn matmul(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: Clone + ndarray::LinalgScalar,
    {
        let other_data = other.cpu_data()?;

        if self.data.ndim() != 2 || other_data.ndim() != 2 {
            return Err("Matrix multiplication requires 2D tensors".to_string());
        }

        let a_shape = self.shape();
        let b_shape = other_data.shape();

        if a_shape[1] != b_shape[0] {
            return Err(format!(
                "Matrix multiplication shape mismatch: ({}, {}) @ ({}, {})",
                a_shape[0], a_shape[1], b_shape[0], b_shape[1]
            ));
        }

        // Use ndarray's optimized BLAS-backed matrix multiplication
        let a: ArrayView2<T> = self
            .data
            .view()
            .into_dimensionality()
            .map_err(|e| format!("Failed to convert to 2D view: {}", e))?;
        let b: ArrayView2<T> = other_data
            .view()
            .into_dimensionality()
            .map_err(|e| format!("Failed to convert to 2D view: {}", e))?;

        let result = a.dot(&b).into_dyn();
        Ok(Box::new(CPUStorage::new(result)))
    }

    // OPTIMIZED REDUCTION OPERATIONS
    fn sum(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result = self.reduce(axes, |array, ax| array.sum_axis(ax))?;
        Ok(Box::new(result))
    }

    fn mean(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let sum_result = self.sum(axes)?;
        let sum_data = sum_result.cpu_data()?;

        // Calculate divisor based on reduced dimensions
        let divisor = match axes {
            Some(axes_list) => {
                let mut div = 1;
                for &ax in axes_list {
                    div *= self.shape()[ax];
                }
                <T as FerroxF>::from_f64(div as f64)
                    .ok_or("Failed to convert divisor to tensor type")?
            }
            None => <T as FerroxF>::from_f64(self.size() as f64)
                .ok_or("Failed to convert size to tensor type")?,
        };

        let result = sum_data / divisor;
        Ok(Box::new(CPUStorage::new(result)))
    }

    fn max_reduce(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Reuse the reduce function with a custom max reduction closure
        let result = self.reduce(axes, |array, ax| {
            // Custom max reduction since ndarray doesn't have max_axis built-in
            let axis_idx = ax.index();
            let mut result_shape = array.shape().to_vec();
            result_shape[axis_idx] = 1;

            let mut result_data = vec![
                <T as FerroxF>::from_f64(f64::NEG_INFINITY).unwrap_or_else(
                    || <T as FerroxF>::from_f64(-1e30).unwrap()
                );
                result_shape.iter().product()
            ];

            // Iterate through array and find max along specified axis
            for (idx, &val) in array.indexed_iter() {
                let mut result_idx = 0;
                let mut stride = 1;

                // Calculate result index by skipping the reduced axis
                for (dim, &coord) in idx.slice().iter().enumerate().rev() {
                    if dim != axis_idx {
                        result_idx += coord * stride;
                        stride *= result_shape[dim];
                    }
                }

                if val > result_data[result_idx] {
                    result_data[result_idx] = val;
                }
            }

            ArrayD::from_shape_vec(IxDyn(&result_shape), result_data)
                .unwrap_or_else(|_| ArrayD::zeros(IxDyn(&result_shape)))
        })?;

        Ok(Box::new(result))
    }

    fn min_reduce(&self, axes: Option<&[usize]>) -> Result<Box<dyn StorageBackend<T>>, String> {
        // Reuse the reduce function with a custom min reduction closure
        let result = self.reduce(axes, |array, ax| {
            // Custom min reduction since ndarray doesn't have min_axis built-in
            let axis_idx = ax.index();
            let mut result_shape = array.shape().to_vec();
            result_shape[axis_idx] = 1;

            let mut result_data = vec![
                <T as FerroxF>::from_f64(f64::INFINITY).unwrap_or_else(
                    || <T as FerroxF>::from_f64(1e30).unwrap()
                );
                result_shape.iter().product()
            ];

            // Iterate through array and find min along specified axis
            for (idx, &val) in array.indexed_iter() {
                let mut result_idx = 0;
                let mut stride = 1;

                // Calculate result index by skipping the reduced axis
                for (dim, &coord) in idx.slice().iter().enumerate().rev() {
                    if dim != axis_idx {
                        result_idx += coord * stride;
                        stride *= result_shape[dim];
                    }
                }

                if val < result_data[result_idx] {
                    result_data[result_idx] = val;
                }
            }

            ArrayD::from_shape_vec(IxDyn(&result_shape), result_data)
                .unwrap_or_else(|_| ArrayD::zeros(IxDyn(&result_shape)))
        })?;

        Ok(Box::new(result))
    }

    // IN-PLACE SHAPE OPERATIONS
    fn broadcast_to(&mut self, target_shape: &[usize]) -> Result<(), String> {
        // Validate broadcasting compatibility
        let current_shape = self.data.shape();

        if target_shape.len() < current_shape.len() {
            return Err("Target shape has fewer dimensions than current shape".to_string());
        }

        // Check broadcasting rules from right to left
        let offset = target_shape.len() - current_shape.len();
        for (i, &current_dim) in current_shape.iter().enumerate() {
            let target_dim = target_shape[offset + i];
            if current_dim != 1 && current_dim != target_dim {
                return Err(format!(
                    "Cannot broadcast dimension {} from {} to {}",
                    i, current_dim, target_dim
                ));
            }
        }

        // Use ndarray's broadcasting - this is a view operation, no data copying
        self.data = self
            .data
            .broadcast(IxDyn(target_shape))
            .ok_or_else(|| "Broadcasting failed".to_string())?
            .to_owned();

        Ok(())
    }

    fn reshape(&mut self, new_shape: &[usize]) -> Result<(), String> {
        let current_size: usize = self.data.len();
        let new_size: usize = new_shape.iter().product();

        if current_size != new_size {
            return Err(format!(
                "Cannot reshape array of size {} into shape {:?} (size {})",
                current_size, new_shape, new_size
            ));
        }

        // Use into_shape for efficient reshaping when possible
        self.data = self
            .data
            .clone()
            .into_shape_with_order(IxDyn(new_shape))
            .map_err(|e| format!("Reshape failed: {}", e))?;

        Ok(())
    }

    fn transpose(&mut self, axes: Option<&[usize]>) -> Result<(), String> {
        match axes {
            Some(axes_list) => {
                // Validate axes permutation
                if axes_list.len() != self.data.ndim() {
                    return Err(format!(
                        "Axes length {} doesn't match tensor dimensions {}",
                        axes_list.len(),
                        self.data.ndim()
                    ));
                }

                let mut seen = vec![false; self.data.ndim()];
                for &ax in axes_list {
                    if ax >= self.data.ndim() {
                        return Err(format!("Axis {} out of bounds", ax));
                    }
                    if seen[ax] {
                        return Err(format!("Axis {} appears multiple times", ax));
                    }
                    seen[ax] = true;
                }

                // Perform transpose using ndarray's permuted_axes
                self.data = self.data.clone().permuted_axes(IxDyn(axes_list));
            }
            None => {
                // Default transpose - reverse all axes
                let ndim = self.data.ndim();
                let reversed_axes: Vec<usize> = (0..ndim).rev().collect();
                self.data = self.data.clone().permuted_axes(IxDyn(&reversed_axes));
            }
        }
        Ok(())
    }

    fn unsqueeze(&mut self, axis: usize) -> Result<(), String> {
        if axis > self.data.ndim() {
            return Err(format!(
                "Axis {} out of bounds for unsqueeze (max: {})",
                axis,
                self.data.ndim()
            ));
        }

        let mut new_shape = self.data.shape().to_vec();
        new_shape.insert(axis, 1);

        self.data = self
            .data
            .clone()
            .into_shape_with_order(IxDyn(&new_shape))
            .map_err(|e| format!("Unsqueeze failed: {}", e))?;

        Ok(())
    }

    fn squeeze(&mut self, axis: Option<usize>) -> Result<(), String> {
        let current_shape = self.data.shape().to_vec();

        let new_shape = match axis {
            Some(ax) => {
                if ax >= current_shape.len() {
                    return Err(format!(
                        "Axis {} out of bounds for tensor with {} dimensions",
                        ax,
                        current_shape.len()
                    ));
                }

                if current_shape[ax] != 1 {
                    return Err(format!(
                        "Cannot squeeze axis {} with size {}",
                        ax, current_shape[ax]
                    ));
                }

                let mut new_shape = current_shape;
                new_shape.remove(ax);
                new_shape
            }
            None => {
                // Remove all dimensions of size 1
                let new_shape: Vec<usize> = current_shape
                    .iter()
                    .filter(|&&size| size != 1)
                    .cloned()
                    .collect();

                if new_shape.is_empty() {
                    vec![1] // Keep at least one dimension
                } else {
                    new_shape
                }
            }
        };

        self.data = self
            .data
            .clone()
            .into_shape_with_order(IxDyn(&new_shape))
            .map_err(|e| format!("Squeeze failed: {}", e))?;

        Ok(())
    }

    // CONVOLUTION
    fn conv2d(
        &self,
        filter: &dyn StorageBackend<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        let filter_data = filter.cpu_data()?;

        // Validate input shapes
        if self.ndim() != 4 || filter_data.ndim() != 4 {
            return Err("Convolution requires 4D tensors (NCHW format)".to_string());
        }

        let result = self.conv2d_impl(filter_data, stride, padding)?;
        Ok(Box::new(CPUStorage::new(result)))
    }

    // ITERATION SUPPORT
    fn iter_values(&self) -> Result<Vec<T>, String> {
        // Use contiguous access when possible for better performance
        if let Some(slice) = self.data.as_slice() {
            Ok(slice.to_vec())
        } else {
            Ok(self.data.iter().cloned().collect())
        }
    }

    fn get_flat(&self, index: usize) -> Result<Option<T>, String> {
        if index >= self.size() {
            Ok(None)
        } else if let Some(slice) = self.data.as_slice() {
            // Fast path for contiguous data
            Ok(Some(slice[index]))
        } else {
            // Fallback for non-contiguous data
            Ok(self.data.iter().nth(index).cloned())
        }
    }

    fn get_multi(&self, indices: &[usize]) -> Result<Option<T>, String> {
        if indices.len() != self.ndim() {
            return Err(format!(
                "Index dimensions {} don't match tensor dimensions {}",
                indices.len(),
                self.ndim()
            ));
        }

        // Bounds check
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape()[i] {
                return Ok(None);
            }
        }

        // Convert multi-dimensional index to flat index
        let mut flat_index = 0;
        let mut stride = 1;
        for i in (0..indices.len()).rev() {
            flat_index += indices[i] * stride;
            stride *= self.shape()[i];
        }

        self.get_flat(flat_index)
    }
}

// STATIC FACTORY METHODS
impl<T: FerroxCudaF> CPUStorage<T> {
    // Optimized factory methods using pool allocation
    pub fn zeros(shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: Zero,
    {
        let size = shape.iter().product();
        let pool_alloc = alloc_cpu_vec::<T>(size)?;
        // Pool allocation already initialized with zeros
        Ok(Box::new(CPUStorage::from_pooled_vec(pool_alloc, shape)?))
    }

    pub fn ones(shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: One,
    {
        let size = shape.iter().product();
        let mut pool_alloc = alloc_cpu_vec::<T>(size)?;
        pool_alloc.data.fill(<T as FerroxF>::one());
        Ok(Box::new(CPUStorage::from_pooled_vec(pool_alloc, shape)?))
    }

    pub fn full(shape: &[usize], value: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let size = shape.iter().product();
        let mut pool_alloc = alloc_cpu_vec::<T>(size)?;
        pool_alloc.data.fill(value);
        Ok(Box::new(CPUStorage::from_pooled_vec(pool_alloc, shape)?))
    }

    // Optimized random number generation
    pub fn randn(shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        StandardUniform: rand_distr::Distribution<T>,
    {
        let size = shape.iter().product();
        let mut pool_alloc = alloc_cpu_vec::<T>(size)?;

        // Use thread-local RNG for better performance
        let mut rng = rand::rng();
        let two = <T as FerroxF>::from_f64(2.0).ok_or("Cannot convert 2.0 to tensor type")?;
        let one = <T as FerroxF>::one();

        // Fill with random values in parallel when data is large enough
        if size > 10000 {
            // For large tensors, consider using rayon for parallel generation
            for val in pool_alloc.data.iter_mut() {
                *val = rng.random::<T>() * two - one;
            }
        } else {
            for val in pool_alloc.data.iter_mut() {
                *val = rng.random::<T>() * two - one;
            }
        }

        Ok(Box::new(CPUStorage::from_pooled_vec(pool_alloc, shape)?))
    }

    // Conditional selection with optimized memory access
    pub fn where_condition(
        condition: &dyn StorageBackend<T>,
        true_vals: &dyn StorageBackend<T>,
        false_vals: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        let condition_data = condition.cpu_data()?;
        let true_data = true_vals.cpu_data()?;
        let false_data = false_vals.cpu_data()?;

        // Validate all shapes match
        if condition_data.shape() != true_data.shape()
            || condition_data.shape() != false_data.shape()
        {
            return Err("Shape mismatch in where_condition".to_string());
        }

        let zero = <T as FerroxF>::zero();

        // Use Zip for optimal performance
        let result = Zip::from(condition_data)
            .and(true_data)
            .and(false_data)
            .map_collect(
                |&cond, &true_val, &false_val| {
                    if cond != zero { true_val } else { false_val }
                },
            );

        Ok(Box::new(CPUStorage::new(result)))
    }
}
