// src/backend/manager.rs

use crate::backend::number::FerroxCudaF;
use crate::backend::storage::{CPUStorage, StorageBackend};
use crate::backend::Device;
use ndarray::ArrayD;
use std::sync::{Arc, OnceLock};

#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaContextManager;
#[cfg(feature = "cuda")]
use crate::backend::storage::CUDAStorage;
#[cfg(feature = "cuda")]
use crate::backend::cuda::ops::CudaOps;

pub struct BackendManager<T: FerroxCudaF> {
    #[cfg(feature = "cuda")]
    cuda_backend: Option<CudaContextManager<T>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: FerroxCudaF> Default for BackendManager<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FerroxCudaF> BackendManager<T> {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "cuda")]
            cuda_backend: None,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn init() -> Self {
        #[cfg(not(feature = "cuda"))]
        let manager = Self::new();

        #[cfg(feature = "cuda")]
        let mut manager = Self::new();

        #[cfg(feature = "cuda")]
        {
            if let Ok(cuda_backend) = CudaContextManager::from_device_id(0) {
                manager.cuda_backend = Some(cuda_backend);
                println!("CUDA backend initialized successfully");
            } else {
                println!("CUDA backend not available, using CPU only");
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            println!("CUDA feature not enabled, using CPU only");
        }

        manager
    }

    pub fn has_cuda(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.cuda_backend.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_backend(&self) -> Option<&CudaContextManager<T>> {
        self.cuda_backend.as_ref()
    }

    pub fn best_device(&self) -> Device {
        if self.has_cuda() {
            #[cfg(feature = "cuda")]
            {
                Device::CUDA(0)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Device::CPU
            }
        } else {
            Device::CPU
        }
    }

    /// Validate device is available and return it
    pub fn validate_device(&self, device: Device) -> Result<Device, String> {
        match device {
            Device::CPU => Ok(device),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                if self.has_cuda() {
                    Ok(device)
                } else {
                    Err("CUDA device requested but CUDA not available".to_string())
                }
            }
        }
    }

    /// Create storage that matches the device - prevents CPU device + CUDA storage
    pub fn create_storage(
        &self,
        shape: &[usize],
        device: Device,
    ) -> Result<(Device, Box<dyn StorageBackend<T>>), String>
    where
        T: Clone + rand_distr::num_traits::One,
    {
        let validated_device = self.validate_device(device)?;

        let storage: Box<dyn StorageBackend<T>> = match validated_device {
            Device::CPU => {
                CPUStorage::<T>::zeros(shape)?
            }
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                CUDAStorage::<T>::zeros(shape)?
            }
        };

        Ok((validated_device, storage))
    }

    /// Create storage on best available device
    pub fn create_storage_auto(&self, shape: &[usize]) -> Result<(Device, Box<dyn StorageBackend<T>>), String>
    where
        T: Clone + rand_distr::num_traits::One,
    {
        let device = self.best_device();
        self.create_storage(shape, device)
    }

    /// Create storage with ones
    pub fn create_ones_storage(
        &self,
        shape: &[usize],
        device: Device,
    ) -> Result<(Device, Box<dyn StorageBackend<T>>), String>
    where
        T: Clone + rand_distr::num_traits::One,
    {
        let validated_device = self.validate_device(device)?;

        let storage: Box<dyn StorageBackend<T>> = match validated_device {
            Device::CPU => {
                CPUStorage::<T>::ones(shape)?
            }
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                CUDAStorage::<T>::ones(shape)?
            }
        };

        Ok((validated_device, storage))
    }

    /// Create storage with specific value
    pub fn create_full_storage(
        &self,
        shape: &[usize],
        device: Device,
        value: T,
    ) -> Result<(Device, Box<dyn StorageBackend<T>>), String>
    where
        T: Clone + rand_distr::num_traits::One,
    {
        let validated_device = self.validate_device(device)?;

        let storage: Box<dyn StorageBackend<T>> = match validated_device {
            Device::CPU => {
                CPUStorage::<T>::full(shape, value)?
            }
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                CUDAStorage::<T>::full(shape, value)?
            }
        };

        Ok((validated_device, storage))
    }

    /// Create storage with random normal distribution
    pub fn create_randn_storage(
        &self,
        shape: &[usize],
        device: Device,
    ) -> Result<(Device, Box<dyn StorageBackend<T>>), String>
    where
        T: Clone + rand_distr::num_traits::One,
        rand::distr::StandardUniform: rand_distr::Distribution<T>,
    {
        let validated_device = self.validate_device(device)?;

        let storage: Box<dyn StorageBackend<T>> = match validated_device {
            Device::CPU => {
                CPUStorage::<T>::randn(shape)?
            }
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                CUDAStorage::<T>::randn(shape)?
            }
        };

        Ok((validated_device, storage))
    }

    /// Create storage from existing CPU data
    pub fn create_storage_from_data(
        &self,
        data: &ArrayD<T>,
        device: Device,
    ) -> Result<(Device, Box<dyn StorageBackend<T>>), String>
    where
        T: Clone + rand_distr::num_traits::One,
    {
        let validated_device = self.validate_device(device)?;

        let storage: Box<dyn StorageBackend<T>> = match validated_device {
            Device::CPU => {
                Box::new(CPUStorage::<T>::new(data.clone()))
            }
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                let cuda_tensor = with_cuda_context(|ctx: &CudaContextManager<T>| {
                    crate::backend::cuda::CudaTensor::from_cpu_array(ctx, data)
                })?;
                Box::new(CUDAStorage::<T>::new(cuda_tensor))
            }
        };

        Ok((validated_device, storage))
    }

    /// Create storage from data on best available device
    pub fn create_storage_from_data_auto(
        &self,
        data: &ArrayD<T>,
    ) -> Result<(Device, Box<dyn StorageBackend<T>>), String>
    where
        T: Clone + rand_distr::num_traits::One,
    {
        let device = self.best_device();
        self.create_storage_from_data(data, device)
    }

    /// Transfer storage to different device - manager handles the complexity
    pub fn move_storage(
        &self,
        storage: Box<dyn StorageBackend<T>>,
        target_device: Device,
    ) -> Result<(Device, Box<dyn StorageBackend<T>>), String>
    where
        T: Clone,
    {
        let validated_target = self.validate_device(target_device)?;

        // Check if already on correct device type
        let needs_transfer = match validated_target {
            Device::CPU => storage.is_gpu(),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => !storage.is_gpu(),
        };

        if !needs_transfer {
            return Ok((validated_target, storage));
        }

        // Need to transfer between device types
        match validated_target {
            Device::CPU => {
                // GPU -> CPU: get GPU data and create CPU storage
                #[cfg(feature = "cuda")]
                {
                    if let Some(gpu_storage) = storage
                        .as_any()
                        .and_then(|any| any.downcast_ref::<CUDAStorage<T>>())
                    {
                        let host_data = with_cuda_context(|ctx: &CudaContextManager<T>| {
                            gpu_storage.cuda_data.to_vec(ctx)
                        })?;

                        let cpu_array = ArrayD::from_shape_vec(
                            ndarray::IxDyn(gpu_storage.cuda_data.shape()),
                            host_data,
                        )
                        .map_err(|e| format!("Failed to create CPU array: {}", e))?;

                        let cpu_storage: Box<dyn StorageBackend<T>> = Box::new(CPUStorage::<T>::new(cpu_array));
                        return Ok((validated_target, cpu_storage));
                    }
                }
                Err("Cannot transfer GPU storage to CPU without CUDA feature".to_string())
            }
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                // CPU -> GPU: get CPU data and create GPU storage from it
                let cpu_data = storage.cpu_data()?;

                let cuda_tensor = with_cuda_context(|ctx: &CudaContextManager<T>| {
                    crate::backend::cuda::CudaTensor::from_cpu_array(ctx, cpu_data)
                })?;

                let cuda_storage: Box<dyn StorageBackend<T>> = Box::new(CUDAStorage::<T>::new(cuda_tensor));
                Ok((validated_target, cuda_storage))
            }
        }
    }
}

// Global instances
static BACKEND_F32: OnceLock<BackendManager<f32>> = OnceLock::new();
static BACKEND_F64: OnceLock<BackendManager<f64>> = OnceLock::new();

pub fn get_f32_backend() -> &'static BackendManager<f32> {
    BACKEND_F32.get_or_init(|| BackendManager::<f32>::init())
}

pub fn get_f64_backend() -> &'static BackendManager<f64> {
    BACKEND_F64.get_or_init(|| BackendManager::<f64>::init())
}

pub fn has_f32_cuda() -> bool {
    get_f32_backend().has_cuda()
}

pub fn best_f32_device() -> Device {
    get_f32_backend().best_device()
}

pub fn has_f64_cuda() -> bool {
    get_f64_backend().has_cuda()
}

pub fn best_f64_device() -> Device {
    get_f64_backend().best_device()
}

pub fn get_backend<T: FerroxCudaF>() -> &'static BackendManager<T> {
    match std::any::TypeId::of::<T>() {
        id if id == std::any::TypeId::of::<f32>() => {
            unsafe { std::mem::transmute(get_f32_backend()) }
        }
        id if id == std::any::TypeId::of::<f64>() => {
            unsafe { std::mem::transmute(get_f64_backend()) }
        }
        _ => panic!("Unsupported float type for backend"),
    }
}

pub fn has_cuda<T: FerroxCudaF>() -> bool {
    match std::any::TypeId::of::<T>() {
        id if id == std::any::TypeId::of::<f32>() => has_f32_cuda(),
        id if id == std::any::TypeId::of::<f64>() => has_f64_cuda(),
        _ => false,
    }
}

pub fn best_device<T: FerroxCudaF>() -> Device {
    match std::any::TypeId::of::<T>() {
        id if id == std::any::TypeId::of::<f32>() => best_f32_device(),
        id if id == std::any::TypeId::of::<f64>() => best_f64_device(),
        _ => Device::CPU,
    }
}

#[cfg(feature = "cuda")]
pub fn with_cuda_context<F, R, T>(f: F) -> Result<R, String>
where
    T: FerroxCudaF,
    F: FnOnce(&CudaContextManager<T>) -> Result<R, String>,
{
    let backend: &'static BackendManager<T> = get_backend::<T>();
    let context_manager: &CudaContextManager<T> =
        backend.cuda_backend().ok_or("CUDA backend not available")?;

    f(&context_manager)
}

#[cfg(feature = "cuda")]
pub fn with_cuda_ops<F, R, T>(f: F) -> Result<R, String>
where
    T: FerroxCudaF,
    F: FnOnce(&CudaOps<T>) -> Result<R, String>,
{
    let backend: &'static BackendManager<T> = get_backend::<T>();
    let context_manager: &CudaContextManager<T> =
        backend.cuda_backend().ok_or("CUDA backend not available")?;

    let ops: Arc<CudaOps<T>> = context_manager.ops();
    f(&ops)
}
