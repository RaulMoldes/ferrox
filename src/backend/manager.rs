// src/backend/manager.rs
use crate::backend::Device;
#[cfg(feature = "cuda")]
use crate::backend::memory::CudaMemoryPool;
#[allow(unused_imports)]
use crate::backend::memory::{MemoryPool, PoolAllocation};
use crate::backend::number::FerroxCudaF;
use crate::backend::storage::{CPUStorage, StorageBackend};
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;
use ndarray::ArrayD;
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaContextManager;
#[cfg(feature = "cuda")]
use crate::backend::cuda::ops::CudaOps;
#[cfg(feature = "cuda")]
use crate::backend::storage::CUDAStorage;

pub struct BackendManager<T: FerroxCudaF> {
    #[cfg(feature = "cuda")]
    cuda_backend: Option<CudaContextManager<T>>,
    #[cfg(feature = "cuda")]
    cuda_pool: Option<std::sync::Mutex<CudaMemoryPool<T>>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: FerroxCudaF> Default for BackendManager<T> {
    fn default() -> Self {
        Self::init()
    }
}

impl<T: FerroxCudaF> BackendManager<T> {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "cuda")]
            cuda_backend: None,
            #[cfg(feature = "cuda")]
            cuda_pool: None,
            _phantom: std::marker::PhantomData,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_pool(&self) -> Option<&std::sync::Mutex<CudaMemoryPool<T>>> {
        self.cuda_pool.as_ref()
    }

    pub fn init() -> Self {
        #[cfg(not(feature = "cuda"))]
        {
            println!("CUDA feature not enabled, using CPU only");
            Self::new()
        }

        #[cfg(feature = "cuda")]
        {
            // Create base manager with CPU pool
            let mut manager = Self::new();

            // Try to initialize CUDA backend and pool
            if let Ok(cuda_backend) = CudaContextManager::<T>::new() {
                let stream = match cuda_backend.stream_manager().get_stream("memset") {
                    Some(memset_stream) => memset_stream,
                    None => cuda_backend.stream_manager().default_stream(),
                };

                let cuda_pool = CudaMemoryPool::new(stream.clone());
                manager.cuda_backend = Some(cuda_backend);
                manager.cuda_pool = Some(Mutex::new(cuda_pool));
                println!("CUDA backend and pool initialized successfully");
            } else {
                println!("CUDA backend not available, using CPU only");
            }

            manager
        }
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
            Device::CPU => CPUStorage::<T>::zeros(shape)?,
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => CUDAStorage::<T>::zeros(shape)?,
        };

        Ok((validated_device, storage))
    }

    /// Create storage on best available device
    pub fn create_storage_auto(
        &self,
        shape: &[usize],
    ) -> Result<(Device, Box<dyn StorageBackend<T>>), String>
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
            Device::CPU => CPUStorage::<T>::ones(shape)?,
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => CUDAStorage::<T>::ones(shape)?,
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
            Device::CPU => CPUStorage::<T>::full(shape, value)?,
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => CUDAStorage::<T>::full(shape, value)?,
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
            Device::CPU => CPUStorage::<T>::randn(shape)?,
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => CUDAStorage::<T>::randn(shape)?,
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
            Device::CPU => Box::new(CPUStorage::<T>::new(data.clone())),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                let (cuda_tensor, id) = with_cuda_context(|ctx: &CudaContextManager<T>| {
                    crate::backend::cuda::CudaTensor::from_cpu_array(ctx, data)
                })?;
                Box::new(CUDAStorage::<T>::new(cuda_tensor, Some(id)))
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

                        let cpu_storage: Box<dyn StorageBackend<T>> =
                            Box::new(CPUStorage::<T>::new(cpu_array));
                        return Ok((validated_target, cpu_storage));
                    }
                }
                Err("Cannot transfer GPU storage to CPU without CUDA feature".to_string())
            }
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                // CPU -> GPU: get CPU data and create GPU storage from it
                let cpu_data = storage.cpu_data()?;

                let (cuda_tensor, id) = with_cuda_context(|ctx: &CudaContextManager<T>| {
                    crate::backend::cuda::CudaTensor::from_cpu_array(ctx, cpu_data)
                })?;

                let cuda_storage: Box<dyn StorageBackend<T>> =
                    Box::new(CUDAStorage::<T>::new(cuda_tensor, Some(id)));
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
        id if id == std::any::TypeId::of::<f32>() => unsafe {
            std::mem::transmute::<&BackendManager<f32>, &BackendManager<T>>(get_f32_backend())
        },
        id if id == std::any::TypeId::of::<f64>() => unsafe {
            std::mem::transmute::<&BackendManager<f64>, &BackendManager<T>>(get_f64_backend())
        },
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

#[cfg(feature = "cuda")]
pub fn with_cuda_pool<F, R, T>(f: F) -> Result<R, String>
where
    T: FerroxCudaF,
    F: FnOnce(&mut CudaMemoryPool<T>) -> Result<R, String>,
{
    let backend: &'static BackendManager<T> = get_backend::<T>();
    let pool_mutex = backend.cuda_pool().ok_or("CUDA pool not available")?;
    let mut pool = pool_mutex
        .lock()
        .map_err(|e| format!("Failed to lock CUDA pool: {}", e))?;

    f(&mut pool)
}

// Pool-aware allocation wrappers that pass context manager
#[cfg(feature = "cuda")]
pub fn alloc_cuda_slice<T>(size: usize) -> Result<PoolAllocation<CudaSlice<T>>, String>
where
    T: FerroxCudaF,
{
    with_cuda_pool(|pool: &mut CudaMemoryPool<T>| pool.allocate(size))
}

// Main function for CUDA ops and context to return memory to pool
#[cfg(feature = "cuda")]
pub fn return_cuda_slice<T: FerroxCudaF>(
    allocation_id: u64,
    slice: CudaSlice<T>,
) -> Result<(), String> {
    with_cuda_pool(|pool: &mut CudaMemoryPool<T>| pool.return_to_pool(allocation_id, slice))
}

#[cfg(test)]
mod backend_manager_tests {
    use super::*;
    use crate::backend::Device;

    #[test]
    fn test_backend_manager_initialization() {


        // Test best device selection returns valid device
        let f32_device = best_f32_device();
        let f64_device = best_f64_device();

        match f32_device {
            Device::CPU => assert!(true), // CPU always available
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => assert!(has_f32_cuda()), // CUDA only if available
        }

        match f64_device {
            Device::CPU => assert!(true),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => assert!(has_f64_cuda()),
        }
    }
}
