// src/backend/storage/cuda.rs
use super::StorageBackend;
use crate::backend::cuda::ops::CudaOps;
use crate::backend::cuda::{CudaContextManager, CudaTensor};
use crate::backend::manager::{
    return_cuda_slice, with_cuda_context, with_cuda_ops, with_cuda_pool,
};
use crate::backend::{FerroxCudaN, FerroxF, FerroxN};
#[cfg(feature = "cuda")]
use crate::FerroxCudaF;
use cudarc::driver::DeviceRepr;
use ndarray::ArrayD;

/// GPU storage
#[derive(Debug, Clone)]
pub struct CUDAStorage<T: FerroxCudaN> {
    pub cuda_data: CudaTensor<T>,
}

#[cfg(feature = "cuda")]
impl<T: DeviceRepr + FerroxCudaN> CUDAStorage<T> {
    pub fn new(cuda_data: CudaTensor<T>) -> Self {
        Self { cuda_data }
    }
}

#[cfg(feature = "cuda")]
impl<T> StorageBackend<T> for CUDAStorage<T>
where
    T: FerroxCudaN + cudarc::driver::DeviceRepr + Clone,
{
    fn shape(&self) -> &[usize] {
        self.cuda_data.shape()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + 'static> {
        self as Box<dyn std::any::Any + 'static>
    }

    fn ndim(&self) -> usize {
        self.cuda_data.shape().len()
    }

    fn size(&self) -> usize {
        self.cuda_data.shape().iter().product()
    }

    fn is_gpu(&self) -> bool {
        true
    }

    fn cpu_data(&self) -> Result<&ArrayD<T>, String> {
        Err("Data is on GPU. Use .to_cpu() to move it first".to_string())
    }

    fn cpu_data_mut(&mut self) -> Result<&mut ArrayD<T>, String> {
        Err("Data is on GPU. Use .to_cpu() to move it first".to_string())
    }

    fn owns_data(&self) -> bool {
        true
    }

    fn clone_storage(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        Ok(Box::new(self.clone()))
    }

    // ELEMENT-WISE OPERATIONS USING CUDA BACKEND

    fn add(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            // Both tensors are on GPU - use CUDA kernels directly
            let other_gpu = other
                .as_any()
                .downcast_ref::<CUDAStorage<T>>()
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for GPU addition: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda =
                with_cuda_ops(|ops: &CudaOps<T>| ops.add(&self.cuda_data, &other_gpu.cuda_data))?;

            Ok(Box::new(CUDAStorage::new(result_cuda)))
        } else {
            // Mixed GPU-CPU operation: convert CPU to GPU first
            let other_data = other.cpu_data()?;

            if self.shape() != other_data.shape() {
                return Err(format!(
                    "Shape mismatch for mixed addition: {:?} vs {:?}",
                    self.shape(),
                    other_data.shape()
                ));
            }

            // Convert CPU ArrayD to CudaTensor and perform operation
            let other_cuda = with_cuda_context(|ctx: &CudaContextManager<T>| {
                CudaTensor::from_cpu_array(ctx, other_data)
            })?;

            let result_cuda =
                with_cuda_ops(|ops: &CudaOps<T>| ops.add(&self.cuda_data, &other_cuda))?;

            Ok(Box::new(CUDAStorage::new(result_cuda)))
        }
    }

    fn sub(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .downcast_ref::<CUDAStorage<T>>()
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for GPU subtraction: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda =
                with_cuda_ops(|ops: &CudaOps<T>| ops.sub(&self.cuda_data, &other_gpu.cuda_data))?;

            Ok(Box::new(CUDAStorage::new(result_cuda)))
        } else {
            let other_data = other.cpu_data()?;

            if self.shape() != other_data.shape() {
                return Err(format!(
                    "Shape mismatch for mixed subtraction: {:?} vs {:?}",
                    self.shape(),
                    other_data.shape()
                ));
            }

            let other_cuda = with_cuda_context(|ctx: &CudaContextManager<T>| {
                CudaTensor::from_cpu_array(ctx, other_data)
            })?;
            let result_cuda =
                with_cuda_ops(|ops: &CudaOps<T>| ops.sub(&self.cuda_data, &other_cuda))?;

            Ok(Box::new(CUDAStorage::new(result_cuda)))
        }
    }

    fn mul(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .downcast_ref::<CUDAStorage<T>>()
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for GPU multiplication: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda =
                with_cuda_ops(|ops: &CudaOps<T>| ops.mul(&self.cuda_data, &other_gpu.cuda_data))?;
            Ok(Box::new(CUDAStorage::new(result_cuda)))
        } else {
            let other_data = other.cpu_data()?;

            if self.shape() != other_data.shape() {
                return Err(format!(
                    "Shape mismatch for mixed multiplication: {:?} vs {:?}",
                    self.shape(),
                    other_data.shape()
                ));
            }

            let other_cuda = with_cuda_context(|ctx: &CudaContextManager<T>| {
                CudaTensor::from_cpu_array(ctx, other_data)
            })?;
            let result_cuda =
                with_cuda_ops(|ops: &CudaOps<T>| ops.mul(&self.cuda_data, &other_cuda))?;

            Ok(Box::new(CUDAStorage::new(result_cuda)))
        }
    }

    fn div(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .downcast_ref::<CUDAStorage<T>>()
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for GPU division: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda =
                with_cuda_ops(|ops: &CudaOps<T>| ops.div(&self.cuda_data, &other_gpu.cuda_data))?;
            Ok(Box::new(CUDAStorage::new(result_cuda)))
        } else {
            let other_data = other.cpu_data()?;

            if self.shape() != other_data.shape() {
                return Err(format!(
                    "Shape mismatch for mixed division: {:?} vs {:?}",
                    self.shape(),
                    other_data.shape()
                ));
            }

            let other_cuda = with_cuda_context(|ctx: &CudaContextManager<T>| {
                CudaTensor::from_cpu_array(ctx, other_data)
            })?;
            let result_cuda =
                with_cuda_ops(|ops: &CudaOps<T>| ops.div(&self.cuda_data, &other_cuda))?;

            Ok(Box::new(CUDAStorage::new(result_cuda)))
        }
    }

    fn min(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .downcast_ref::<CUDAStorage<T>>()
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for GPU min operation: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| {
                ops.min_elementwise(&self.cuda_data, &other_gpu.cuda_data)
            })?;
            Ok(Box::new(CUDAStorage::new(result_cuda)))
        } else {
            let other_data = other.cpu_data()?;

            if self.shape() != other_data.shape() {
                return Err(format!(
                    "Shape mismatch for mixed min operation: {:?} vs {:?}",
                    self.shape(),
                    other_data.shape()
                ));
            }

            let other_cuda = with_cuda_context(|ctx: &CudaContextManager<T>| {
                CudaTensor::from_cpu_array(ctx, other_data)
            })?;
            let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| {
                ops.min_elementwise(&self.cuda_data, &other_cuda)
            })?;

            Ok(Box::new(CUDAStorage::new(result_cuda)))
        }
    }

    fn max(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .downcast_ref::<CUDAStorage<T>>()
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for GPU max operation: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| {
                ops.max_elementwise(&self.cuda_data, &other_gpu.cuda_data)
            })?;
            Ok(Box::new(CUDAStorage::new(result_cuda)))
        } else {
            let other_data = other.cpu_data()?;

            if self.shape() != other_data.shape() {
                return Err(format!(
                    "Shape mismatch for mixed max operation: {:?} vs {:?}",
                    self.shape(),
                    other_data.shape()
                ));
            }

            let other_cuda = with_cuda_context(|ctx: &CudaContextManager<T>| {
                CudaTensor::from_cpu_array(ctx, other_data)
            })?;

            let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| {
                ops.max_elementwise(&self.cuda_data, &other_cuda)
            })?;

            Ok(Box::new(CUDAStorage::new(result_cuda)))
        }
    }

    // SCALAR OPERATIONS

    fn add_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda =
            with_cuda_ops(|ops: &CudaOps<T>| ops.add_scalar(&self.cuda_data, scalar))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn sub_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda =
            with_cuda_ops(|ops: &CudaOps<T>| ops.sub_scalar(&self.cuda_data, scalar))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn mul_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda =
            with_cuda_ops(|ops: &CudaOps<T>| ops.mul_scalar(&self.cuda_data, scalar))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn div_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda =
            with_cuda_ops(|ops: &CudaOps<T>| ops.div_scalar(&self.cuda_data, scalar))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    // UNARY OPERATIONS

    fn neg(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| ops.negate(&self.cuda_data))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn abs(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| ops.abs(&self.cuda_data))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn sqrt(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| ops.sqrt(&self.cuda_data))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn clamp(&self, min_val: T, max_val: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda =
            with_cuda_ops(|ops: &CudaOps<T>| ops.clamp(&self.cuda_data, min_val, max_val))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn reciprocal(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| ops.reciprocal(&self.cuda_data))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn greater(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            // Both tensors are on GPU - use CUDA kernels
            let other_gpu = other
                .as_any()
                .downcast_ref::<CUDAStorage<T>>()
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for greater_equal: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| {
                ops.greater(&self.cuda_data, &other_gpu.cuda_data)
            })?;
            Ok(Box::new(CUDAStorage::new(result_cuda)))
        } else {
            Err("GPU-CPU mixed operations not supported for comparison operations".to_string())
        }
    }

    fn greater_equal_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda =
            with_cuda_ops(|ops: &CudaOps<T>| ops.greater_equal_scalar(&self.cuda_data, scalar))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn less_equal_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda =
            with_cuda_ops(|ops: &CudaOps<T>| ops.less_equal_scalar(&self.cuda_data, scalar))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn greater_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda =
            with_cuda_ops(|ops: &CudaOps<T>| ops.greater_scalar(&self.cuda_data, scalar))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn less_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda =
            with_cuda_ops(|ops: &CudaOps<T>| ops.less_scalar(&self.cuda_data, scalar))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn less(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .downcast_ref::<CUDAStorage<T>>()
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for less_equal: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda =
                with_cuda_ops(|ops: &CudaOps<T>| ops.less(&self.cuda_data, &other_gpu.cuda_data))?;
            Ok(Box::new(CUDAStorage::new(result_cuda)))
        } else {
            Err("GPU-CPU mixed operations not supported for comparison operations".to_string())
        }
    }

    fn greater_equal(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            // Both tensors are on GPU - use CUDA kernels
            let other_gpu = other
                .as_any()
                .downcast_ref::<CUDAStorage<T>>()
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for greater_equal: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| {
                ops.greater_equal(&self.cuda_data, &other_gpu.cuda_data)
            })?;
            Ok(Box::new(CUDAStorage::new(result_cuda)))
        } else {
            Err("GPU-CPU mixed operations not supported for comparison operations".to_string())
        }
    }

    fn less_equal(
        &self,
        other: &dyn StorageBackend<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .downcast_ref::<CUDAStorage<T>>()
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for less_equal: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| {
                ops.less_equal(&self.cuda_data, &other_gpu.cuda_data)
            })?;
            Ok(Box::new(CUDAStorage::new(result_cuda)))
        } else {
            Err("GPU-CPU mixed operations not supported for comparison operations".to_string())
        }
    }

    fn equal(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .downcast_ref::<CUDAStorage<T>>()
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for equal: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda =
                with_cuda_ops(|ops: &CudaOps<T>| ops.equal(&self.cuda_data, &other_gpu.cuda_data))?;
            Ok(Box::new(CUDAStorage::new(result_cuda)))
        } else {
            Err("GPU-CPU mixed operations not supported for comparison operations".to_string())
        }
    }

    fn logical_not(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| ops.logical_not(&self.cuda_data))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn in_range(&self, min_val: T, max_val: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda =
            with_cuda_ops(|ops: &CudaOps<T>| ops.in_range(&self.cuda_data, min_val, max_val))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn sign(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| ops.sign(&self.cuda_data))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn matmul(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .downcast_ref::<CUDAStorage<T>>()
                .ok_or("Failed to cast to GPU storage")?;

            if self.ndim() != 2 || other_gpu.ndim() != 2 {
                return Err("Matrix multiplication requires 2D tensors".to_string());
            }

            let a_shape = self.shape();
            let b_shape = other_gpu.shape();

            if a_shape[1] != b_shape[0] {
                return Err(format!(
                    "Matrix multiplication shape mismatch: ({}, {}) @ ({}, {})",
                    a_shape[0], a_shape[1], b_shape[0], b_shape[1]
                ));
            }

            let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| {
                ops.matmul(&self.cuda_data, &other_gpu.cuda_data)
            })?;
            Ok(Box::new(CUDAStorage::new(result_cuda)))
        } else {
            Err("GPU-CPU mixed operations not supported for matrix multiplication".to_string())
        }
    }

    fn sigmoid(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|cuda_ops: &CudaOps<T>| cuda_ops.sigmoid(&self.cuda_data))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn softmax(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|cuda_ops: &CudaOps<T>| cuda_ops.softmax(&self.cuda_data))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn softmax_batched(&self, axis: usize) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        T: FerroxCudaF,
    {
        let result_cuda =
            with_cuda_ops(|cuda_ops: &CudaOps<T>| cuda_ops.softmax_batched(&self.cuda_data, axis))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn relu(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|cuda_ops: &CudaOps<T>| cuda_ops.relu(&self.cuda_data))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn exp(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|cuda_ops: &CudaOps<T>| cuda_ops.exp(&self.cuda_data))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn log(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|cuda_ops: &CudaOps<T>| cuda_ops.log(&self.cuda_data))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn tanh(&self) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda = with_cuda_ops(|cuda_ops: &CudaOps<T>| cuda_ops.tanh(&self.cuda_data))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn powf(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        if other.is_gpu() {
            let other_gpu = other
                .as_any()
                .downcast_ref::<CUDAStorage<T>>()
                .ok_or("Failed to cast to GPU storage")?;

            if self.shape() != other_gpu.shape() {
                return Err(format!(
                    "Shape mismatch for powf: {:?} vs {:?}",
                    self.shape(),
                    other_gpu.shape()
                ));
            }

            let result_cuda = with_cuda_ops(|cuda_ops: &CudaOps<T>| {
                cuda_ops.power(&self.cuda_data, &other_gpu.cuda_data)
            })?;
            Ok(Box::new(CUDAStorage::new(result_cuda)))
        } else {
            Err("GPU-CPU mixed operations not supported for power operations".to_string())
        }
    }

    fn power_scalar(&self, scalar: T) -> Result<Box<dyn StorageBackend<T>>, String> {
        let result_cuda =
            with_cuda_ops(|cuda_ops: &CudaOps<T>| cuda_ops.power_scalar(&self.cuda_data, scalar))?;
        Ok(Box::new(CUDAStorage::new(result_cuda)))
    }

    fn sum(
        &self,
        axes: Option<&[usize]>,
        keep_dims: bool,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        match axes {
            Some(axes_list) => {
                if axes_list.is_empty() {
                    return Err("Empty axes list provided".to_string());
                }

                // Validate axes are within bounds
                for &ax in axes_list {
                    if ax >= self.ndim() {
                        return Err(format!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            ax,
                            self.ndim()
                        ));
                    }
                }

                // Use CUDA ops sum_axes method for multiple axes
                let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| {
                    ops.sum_axes(&self.cuda_data, axes_list, keep_dims)
                })?;
                Ok(Box::new(CUDAStorage::new(result_cuda)))
            }
            None => {
                // Sum all elements to scalar using CUDA ops
                let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| ops.sum_all(&self.cuda_data))?;
                Ok(Box::new(CUDAStorage::new(result_cuda)))
            }
        }
    }

    fn mean(
        &self,
        axes: Option<&[usize]>,
        keep_dims: bool,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        match axes {
            Some(axes_list) => {
                if axes_list.is_empty() {
                    return Err("Empty axes list provided".to_string());
                }

                // Validate axes are within bounds
                for &ax in axes_list {
                    if ax >= self.ndim() {
                        return Err(format!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            ax,
                            self.ndim()
                        ));
                    }
                }

                // For multiple axes, we compute sum then divide by product of axis sizes
                let sum_result = with_cuda_ops(|ops: &CudaOps<T>| {
                    ops.sum_axes(&self.cuda_data, axes_list, keep_dims)
                })?;

                // Calculate divisor as product of reduced dimensions
                let divisor = axes_list
                    .iter()
                    .map(|&ax| self.shape()[ax])
                    .product::<usize>() as f64;

                let divisor_scalar =
                    FerroxN::from_f64(divisor).ok_or("Failed to convert divisor to tensor type")?;

                let result_cuda =
                    with_cuda_ops(|ops: &CudaOps<T>| ops.div_scalar(&sum_result, divisor_scalar))?;

                Ok(Box::new(CUDAStorage::new(result_cuda)))
            }
            None => {
                // Mean of all elements using CUDA ops
                let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| ops.mean_all(&self.cuda_data))?;
                Ok(Box::new(CUDAStorage::new(result_cuda)))
            }
        }
    }

    fn max_reduce(
        &self,
        axes: Option<&[usize]>,
        keep_dims: bool,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        match axes {
            Some(axes_list) => {
                if axes_list.is_empty() {
                    return Err("Empty axes list provided".to_string());
                }

                // Validate axes are within bounds
                for &ax in axes_list {
                    if ax >= self.ndim() {
                        return Err(format!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            ax,
                            self.ndim()
                        ));
                    }
                }

                // Use CUDA ops max_axes method for multiple axes reduction
                let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| {
                    ops.max_axes(&self.cuda_data, axes_list, keep_dims)
                })?;
                Ok(Box::new(CUDAStorage::new(result_cuda)))
            }
            None => {
                // Max of all elements to scalar
                let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| ops.max_all(&self.cuda_data))?;
                Ok(Box::new(CUDAStorage::new(result_cuda)))
            }
        }
    }

    fn min_reduce(
        &self,
        axes: Option<&[usize]>,
        keep_dims: bool,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        match axes {
            Some(axes_list) => {
                if axes_list.is_empty() {
                    return Err("Empty axes list provided".to_string());
                }

                // Validate axes are within bounds
                for &ax in axes_list {
                    if ax >= self.ndim() {
                        return Err(format!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            ax,
                            self.ndim()
                        ));
                    }
                }

                // Use CUDA ops min_axes method for multiple axes reduction
                let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| {
                    ops.min_axes(&self.cuda_data, axes_list, keep_dims)
                })?;
                Ok(Box::new(CUDAStorage::new(result_cuda)))
            }
            None => {
                // Min of all elements to scalar
                let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| ops.min_all(&self.cuda_data))?;
                Ok(Box::new(CUDAStorage::new(result_cuda)))
            }
        }
    }

    fn broadcast_to(&mut self, target_shape: &[usize]) -> Result<(), String> {
        let broadcasted = with_cuda_ops(|ops: &CudaOps<T>| {
            ops.broadcast_to(&mut self.cuda_data, target_shape)?;
            ops.materialize(&self.cuda_data)
        })?;
        // Ensure the tensor is materialized.
        self.cuda_data = broadcasted;
        Ok(())
    }

    fn reshape(&mut self, new_shape: &[usize]) -> Result<(), String> {
        // Validate total elements remain the same
        let total_elements: usize = self.shape().iter().product();
        let new_total_elements: usize = new_shape.iter().product();

        if total_elements != new_total_elements {
            return Err(format!(
                "Cannot reshape tensor with {} elements to shape with {} elements",
                total_elements, new_total_elements
            ));
        }

        // Use CUDA ops for reshape operation
        let reshaped = with_cuda_ops(|ops: &CudaOps<T>| {
            ops.reshape(&mut self.cuda_data, new_shape.to_vec())?;
            ops.materialize(&self.cuda_data)
        })?;
        self.cuda_data = reshaped;
        Ok(())
    }

    fn transpose(&mut self, axes: Option<&[usize]>) -> Result<(), String> {
        match axes {
            Some(axes_order) => {
                // Validate axes specification
                if axes_order.len() != self.ndim() {
                    return Err(format!(
                        "Axes length {} doesn't match tensor dimensions {}",
                        axes_order.len(),
                        self.ndim()
                    ));
                }

                // Use CUDA ops for custom transpose

                let transposed = with_cuda_ops(|ops: &CudaOps<T>| {
                    ops.transpose(&mut self.cuda_data, Some(axes_order))?;
                    ops.materialize(&self.cuda_data)
                })?;
                self.cuda_data = transposed;
                Ok(())
            }
            None => {
                // Default transpose using CUDA ops
                let transposed = with_cuda_ops(|ops: &CudaOps<T>| {
                    ops.transpose(&mut self.cuda_data, None)?;
                    ops.materialize(&self.cuda_data)
                })?;
                self.cuda_data = transposed;
                Ok(())
            }
        }
    }

    fn unsqueeze(&mut self, axis: usize) -> Result<(), String> {
        // Validate axis bounds - can insert at positions 0..ndim (inclusive)
        if axis > self.ndim() {
            return Err(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                self.ndim()
            ));
        }

        let unsq = with_cuda_ops(|op: &CudaOps<T>| {
            op.unsqueeze(&mut self.cuda_data, axis)?;
            op.materialize(&self.cuda_data)
        })?;
        self.cuda_data = unsq;
        Ok(())
    }

    fn squeeze(&mut self, axis: Option<usize>) -> Result<(), String> {
        match axis {
            Some(ax) => {
                // Squeeze specific axis - validate it exists and has size 1
                if ax >= self.ndim() {
                    return Err(format!(
                        "Axis {} out of bounds for tensor with {} dimensions",
                        ax,
                        self.ndim()
                    ));
                }

                if self.shape()[ax] != 1 {
                    return Err(format!(
                        "Cannot squeeze axis {} with size {}",
                        ax,
                        self.shape()[ax]
                    ));
                }

                // Use CUDA ops for single axis squeeze
                let sq = with_cuda_ops(|ops: &CudaOps<T>| {
                    ops.squeeze(&mut self.cuda_data, Some(ax))?;
                    ops.materialize(&self.cuda_data)
                })?;
                self.cuda_data = sq;
                Ok(())
            }
            None => {
                // Remove all dimensions of size 1 using CUDA ops
                with_cuda_ops(|ops: &CudaOps<T>| ops.squeeze(&mut self.cuda_data, None))?;
                Ok(())
            }
        }
    }

    fn conv2d(
        &self,
        filter: &dyn StorageBackend<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        if filter.is_gpu() {
            // Both tensors on GPU - use CUDA kernels
            let filter_gpu = filter
                .as_any()
                .downcast_ref::<CUDAStorage<T>>()
                .ok_or("Failed to cast filter to GPU storage")?;

            let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| {
                ops.conv2d_forward(&self.cuda_data, &filter_gpu.cuda_data, stride, padding)
            })?;
            Ok(Box::new(CUDAStorage::new(result_cuda)))
        } else {
            // Mixed GPU/CPU - fall back to CPU computation
            Err(
                "Mixed GPU/CPU convolution not supported - move both tensors to same device"
                    .to_string(),
            )
        }
    }

    fn deconv2d(
        &self,
        filter: &dyn StorageBackend<T>,
        output_shape: &[usize],
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        let filter_shape = filter.shape();
        // Validate input dimensions for conv2d
        if output_shape.len() != 4 || filter_shape.len() != 4 || self.shape().len() != 4 {
            return Err(
                "Deconv2D requires 4D tensors [batch, channels, height, width]".to_string(),
            );
        }

        if filter.is_gpu() {
            // Both tensors on GPU - use CUDA kernels
            let filter_gpu = filter
                .as_any()
                .downcast_ref::<CUDAStorage<T>>()
                .ok_or("Failed to cast filter to GPU storage")?;

            let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| {
                ops.deconv2d(
                    &self.cuda_data,
                    &filter_gpu.cuda_data,
                    output_shape,
                    stride,
                    padding,
                )
            })?;

            Ok(Box::new(CUDAStorage::new(result_cuda)))
        } else {
            // Mixed GPU/CPU - fall back to CPU computation
            Err(
                "Mixed GPU/CPU deconvolution not supported - move both tensors to same device"
                    .to_string(),
            )
        }
    }

    fn cross_correlation2d(
        &self,
        other: &dyn StorageBackend<T>,
        output_shape: &[usize],
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_shape = other.shape();
        if self.shape().len() != 4 || other_shape.len() != 4 {
            return Err(
                "Cross correlation requires 4D tensors [batch, channels, height, width]"
                    .to_string(),
            );
        }

        if other.is_gpu() {
            // Both tensors on GPU - use CUDA kernels
            let filter_gpu = other
                .as_any()
                .downcast_ref::<CUDAStorage<T>>()
                .ok_or("Failed to cast filter to GPU storage")?;

            let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| {
                ops.cross_correlation2d(
                    &self.cuda_data,
                    &filter_gpu.cuda_data,
                    output_shape,
                    stride,
                    padding,
                )
            })?;

            Ok(Box::new(CUDAStorage::new(result_cuda)))
        } else {
            // Mixed GPU/CPU - fall back to CPU computation
            Err(
                "Mixed GPU/CPU deconvolution not supported - move both tensors to same device"
                    .to_string(),
            )
        }
    }

    // Note that this ops require moving the data to the CPU
    fn iter_values(&self) -> Result<Vec<T>, String> {
        Err("Cannot iterate over a tensor in cuda. Move to cpu first.".to_string())
    }

    fn get_flat(&self, _index: usize) -> Result<Option<T>, String> {
        Err("Cannot index over a tensor in cuda. Move to cpu first.".to_string())
    }

    fn get_multi(&self, _indices: &[usize]) -> Result<Option<T>, String> {
        Err("Cannot index over a tensor in cuda. Move to cpu first.".to_string())
    }

    fn conv1d(&self, filter: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let filter_shape = filter.shape();
        // Validate input dimensions for conv2d
        if filter_shape.len() != 1 || self.shape().len() != 1 {
            return Err("Conv1D requires flattened tensors ".to_string());
        }

        if filter.is_gpu() {
            // Both tensors on GPU - use CUDA kernels
            let filter_gpu = filter
                .as_any()
                .downcast_ref::<CUDAStorage<T>>()
                .ok_or("Failed to cast filter to GPU storage")?;

            let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| {
                ops.conv1d_forward(&self.cuda_data, &filter_gpu.cuda_data)
            })?;

            Ok(Box::new(CUDAStorage::new(result_cuda)))
        } else {
            // Mixed GPU/CPU - fall back to CPU computation
            Err(
                "Mixed GPU/CPU 1D convolution not supported - move both tensors to same device"
                    .to_string(),
            )
        }
    }


     fn cross_correlation1d(&self, other: &dyn StorageBackend<T>) -> Result<Box<dyn StorageBackend<T>>, String> {
        let other_shape = other.shape();
        // Validate input dimensions for conv2d
        if other_shape.len() != 1 || self.shape().len() != 1 {
            return Err("Cross correlation 1D requires flattened tensors ".to_string());
        }

        if other.is_gpu() {
            // Both tensors on GPU - use CUDA kernels
            let other_gpu = other
                .as_any()
                .downcast_ref::<CUDAStorage<T>>()
                .ok_or("Failed to cast other to GPU storage")?;

            let result_cuda = with_cuda_ops(|ops: &CudaOps<T>| {
                ops.conv1d_forward(&self.cuda_data, &other_gpu.cuda_data)
            })?;

            Ok(Box::new(CUDAStorage::new(result_cuda)))
        } else {
            // Mixed GPU/CPU - fall back to CPU computation
            Err(
                "Mixed GPU/CPU 1D convolution not supported - move both tensors to same device"
                    .to_string(),
            )
        }
    }


    /*fn execute_custom_op<R>(&self, op: Box<dyn CustomOperation<T, R>>) -> Result<R, String> {
        // Executes custom operation through CUDA ops interface
        with_cuda_ops(|cuda_ops: &CudaOps<T>| op.execute_cuda(cuda_ops, &self.cuda_data))
    }*/
}

#[cfg(feature = "cuda")]
impl<T: cudarc::driver::DeviceRepr + FerroxCudaN> CUDAStorage<T> {
    pub fn zeros(shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
        T: cudarc::driver::ValidAsZeroBits,
    {
        // Get default CUDA backend from manager

        // Create GPU tensor with zeros directly
        let cuda_tensor = with_cuda_ops(|ops: &CudaOps<T>| ops.zeros(shape))?;
        Ok(Box::new(CUDAStorage::new(cuda_tensor)))
    }

    pub fn ones(shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
    {
        // Create GPU tensor with ones directly
        let cuda_tensor = with_cuda_ops(|ops: &CudaOps<T>| ops.ones(shape))?;
        Ok(Box::new(CUDAStorage::new(cuda_tensor)))
    }

    pub fn full(shape: &[usize], value: T) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
    {
        // Create GPU tensor filled with value
        let cuda_tensor = with_cuda_ops(|ops: &CudaOps<T>| ops.full(shape, value))?;
        Ok(Box::new(CUDAStorage::new(cuda_tensor)))
    }

    pub fn randn(shape: &[usize]) -> Result<Box<dyn StorageBackend<T>>, String>
    where
        Self: Sized,
    {
        let cuda_tensor = with_cuda_ops(|ops: &CudaOps<T>| ops.randn(shape, 1000))?;
        Ok(Box::new(CUDAStorage::new(cuda_tensor)))
    }

    pub fn where_condition(
        condition: &CUDAStorage<T>,
        true_vals: &CUDAStorage<T>,
        false_vals: &CUDAStorage<T>,
    ) -> Result<Box<dyn StorageBackend<T>>, String> {
        if condition.is_gpu() && true_vals.is_gpu() && false_vals.is_gpu() {
            let cuda_tensor = with_cuda_ops(|ops: &CudaOps<T>| {
                ops.where_condition(
                    &condition.cuda_data,
                    &true_vals.cuda_data,
                    &false_vals.cuda_data,
                )
            })?;

            Ok(Box::new(CUDAStorage::new(cuda_tensor)))
        } else {
            Err("Not all provided tensors are on GPU storage".to_string())
        }
    }
}
