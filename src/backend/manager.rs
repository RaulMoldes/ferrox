use crate::backend::memory::PoolConfig;
// src/backend/manager.rs
#[cfg(feature = "cuda")]
use crate::backend::memory::CudaMemoryPool;
#[allow(unused_imports)]
use crate::backend::memory::{MemoryPool, PoolAllocation};
use crate::backend::number::{FerroxCudaN, FerroxN};
use crate::backend::storage::{CPUStorage, StorageBackend};
use crate::backend::Device;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;
use ndarray::ArrayD;
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "cuda")]
use crate::backend::cuda::ops::CudaOps;
#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaContextManager;
#[cfg(feature = "cuda")]
use crate::backend::storage::CUDAStorage;

pub struct BackendManager<T: FerroxCudaN> {
    #[cfg(feature = "cuda")]
    cuda_backend: Option<CudaContextManager<T>>,
    #[cfg(feature = "cuda")]
    cuda_pool: Option<std::sync::Mutex<CudaMemoryPool<T>>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: FerroxCudaN> Default for BackendManager<T> {
    fn default() -> Self {
        Self::init(None)
    }
}

impl<T: FerroxCudaN> BackendManager<T> {
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

    #[allow(unused_variables)]
    pub fn init(pool_config: Option<PoolConfig>) -> Self {
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

                let cuda_pool = CudaMemoryPool::new(stream.clone(), pool_config);
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

    pub fn move_storage(
        &self,
        storage: Box<dyn StorageBackend<T>>,
        target_device: Device,
    ) -> Result<(Device, Box<dyn StorageBackend<T>>), String>
    where
        T: Clone,
    {
        let validated_target = self.validate_device(target_device)?;

        let needs_transfer = match validated_target {
            Device::CPU => storage.is_gpu(),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => !storage.is_gpu(),
        };

        if !needs_transfer {
            return Ok((validated_target, storage));
        }

        match validated_target {
            Device::CPU => {
                // GPU -> CPU
                #[cfg(feature = "cuda")]
                {
                    // downcast by value: consumes storage and returns Box<CUDAStorage<T>>
                    let gpu_storage_box = match storage.into_any().downcast::<CUDAStorage<T>>() {
                        Ok(b) => b,
                        Err(_) => return Err(
                            "The storage is not CUDAstorage, cannot be transferred from GPU->CPU"
                                .to_string(),
                        ),
                    };

                    // Move the content inside the box
                    let gpu_storage: CUDAStorage<T> = *gpu_storage_box;

                    // IMPORTANT: take ownership of the shape before moving out of cuda_data
                    let shape_vec: Vec<usize> = gpu_storage.cuda_data.shape().to_vec();

                    // Move `cuda_data`inside the closure
                    let host_data = with_cuda_context(move |ctx: &CudaContextManager<T>| {
                        // `to_vec` consumes the tensor using move.
                        gpu_storage.cuda_data.to_vec(ctx)
                    })?;

                    // Build  ArrayD
                    let cpu_array = ArrayD::from_shape_vec(ndarray::IxDyn(&shape_vec), host_data)
                        .map_err(|e| format!("Failed to create CPU array: {}", e))?;

                    let cpu_storage: Box<dyn StorageBackend<T>> =
                        Box::new(CPUStorage::<T>::new(cpu_array));
                    Ok((validated_target, cpu_storage))
                }

                #[cfg(not(feature = "cuda"))]
                {
                    return Err(
                        "CUDA feature no habilitada; no es posible transferir desde GPU"
                            .to_string(),
                    );
                }
            }

            #[cfg(feature = "cuda")]
            Device::CUDA(_device_idx) => {
                // CPU -> GPU
                let cpu_data = storage.cpu_data()?;

                let cuda_tensor = with_cuda_context(|ctx: &CudaContextManager<T>| {
                    crate::backend::cuda::CudaTensor::from_cpu_array(ctx, cpu_data)
                })?;

                let cuda_storage: Box<dyn StorageBackend<T>> =
                    Box::new(CUDAStorage::<T>::new(cuda_tensor));
                Ok((validated_target, cuda_storage))
            }
        }
    }
}

// Global instances
static BACKEND_F32: OnceLock<BackendManager<f32>> = OnceLock::new();
static BACKEND_F64: OnceLock<BackendManager<f64>> = OnceLock::new();

#[allow(clippy::redundant_closure)]
pub fn get_f32_backend() -> &'static BackendManager<f32> {
    BACKEND_F32.get_or_init(|| BackendManager::<f32>::init(None))
}
#[allow(clippy::redundant_closure)]
pub fn get_f64_backend() -> &'static BackendManager<f64> {
    BACKEND_F64.get_or_init(|| BackendManager::<f64>::init(None))
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

pub fn get_backend<T: FerroxCudaN>() -> &'static BackendManager<T> {
    match std::any::TypeId::of::<T>() {
        id if id == std::any::TypeId::of::<f32>() => unsafe {
            std::mem::transmute::<&BackendManager<f32>, &BackendManager<T>>(get_f32_backend())
        },
        id if id == std::any::TypeId::of::<f64>() => unsafe {
            std::mem::transmute::<&BackendManager<f64>, &BackendManager<T>>(get_f64_backend())
        },
        id if id == std::any::TypeId::of::<i64>() => unsafe {
            std::mem::transmute::<&BackendManager<f64>, &BackendManager<T>>(get_f64_backend())
        },
        id if id == std::any::TypeId::of::<i32>() => unsafe {
            std::mem::transmute::<&BackendManager<f32>, &BackendManager<T>>(get_f32_backend())
        },
        _ => panic!("Unsupported type for backend"),
    }
}

pub fn has_cuda<T: FerroxCudaN>() -> bool {
    match std::any::TypeId::of::<T>() {
        id if id == std::any::TypeId::of::<f32>() => has_f32_cuda(),
        id if id == std::any::TypeId::of::<f64>() => has_f64_cuda(),
        _ => false,
    }
}

pub fn best_device<T: FerroxCudaN>() -> Device {
    match std::any::TypeId::of::<T>() {
        id if id == std::any::TypeId::of::<f32>() => best_f32_device(),
        id if id == std::any::TypeId::of::<f64>() => best_f64_device(),
        _ => Device::CPU,
    }
}

#[cfg(feature = "cuda")]
pub fn with_cuda_context<F, R, T>(f: F) -> Result<R, String>
where
    T: FerroxCudaN,
    F: FnOnce(&CudaContextManager<T>) -> Result<R, String>,
{
    let backend: &'static BackendManager<T> = get_backend::<T>();
    let context_manager: &CudaContextManager<T> =
        backend.cuda_backend().ok_or("CUDA backend not available")?;

    f(context_manager)
}

#[cfg(feature = "cuda")]
pub fn with_cuda_ops<F, R, T>(f: F) -> Result<R, String>
where
    T: FerroxCudaN,
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
    T: FerroxCudaN,
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
    T: FerroxCudaN,
{
    with_cuda_pool(|pool: &mut CudaMemoryPool<T>| pool.allocate(size))
}

// Main function for CUDA ops and context to return memory to pool
#[cfg(feature = "cuda")]
pub fn return_cuda_slice<T: FerroxCudaN>(
    alloc: PoolAllocation<CudaSlice<T>>,
) -> Result<(), String> {
    with_cuda_pool(|pool: &mut CudaMemoryPool<T>| pool.deallocate(alloc))
}

#[cfg(feature = "cuda")]
pub fn cleanup_pool<T: FerroxCudaN>() -> Result<(), String> {
    with_cuda_pool(|pool: &mut CudaMemoryPool<T>| pool.cleanup())
}

#[cfg(feature = "cuda")]
pub fn show_memory_stats<T: FerroxCudaN>() -> Result<(), String> {
    with_cuda_pool(|pool: &mut CudaMemoryPool<T>| {
        pool.print_stats();
        Ok(())
    })
}

#[cfg(test)]
mod backend_manager_tests {
    use super::*;
    use crate::backend::storage::StorageBackend;
    use crate::backend::{get_backend, Device, FerroxCudaN};
    use ndarray::{Array1, ArrayD};

    // Test basic backend initialization for both f32 and f64
    #[test]
    #[allow(clippy::cmp_null, clippy::ptr_eq)]
    fn test_backend_initialization() {
        let f32_backend = get_backend::<f32>();
        let f64_backend = get_backend::<f64>();

        // Both backends should be available

        assert!(f32_backend as *const _ != std::ptr::null());

        assert!(f64_backend as *const _ != std::ptr::null());

        println!("F32 backend has CUDA: {}", f32_backend.has_cuda());
        println!("F64 backend has CUDA: {}", f64_backend.has_cuda());
    }

    // Test that when CUDA feature is enabled, default device is CUDA
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_feature_default_backend() {
        let f32_backend = get_backend::<f32>();
        let f64_backend = get_backend::<f64>();

        // With CUDA feature enabled, backends should prefer CUDA if available
        let f32_best = f32_backend.best_device();
        let f64_best = f64_backend.best_device();

        if f32_backend.has_cuda() {
            assert!(
                matches!(f32_best, Device::CUDA(_)),
                "F32 backend should default to CUDA when available, got: {:?}",
                f32_best
            );
        }

        if f64_backend.has_cuda() {
            assert!(
                matches!(f64_best, Device::CUDA(_)),
                "F64 backend should default to CUDA when available, got: {:?}",
                f64_best
            );
        }

        println!("F32 best device: {:?}", f32_best);
        println!("F64 best device: {:?}", f64_best);
    }

    // Test that without CUDA feature, default device is CPU
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_no_cuda_feature_default_backend() {
        let f32_backend = get_backend::<f32>();
        let f64_backend = get_backend::<f64>();

        // Without CUDA feature, should always be CPU
        assert!(!f32_backend.has_cuda(), "F32 backend should not have CUDA");
        assert!(!f64_backend.has_cuda(), "F64 backend should not have CUDA");

        let f32_best = f32_backend.best_device();
        let f64_best = f64_backend.best_device();

        assert!(
            matches!(f32_best, Device::CPU),
            "F32 backend should default to CPU without CUDA feature"
        );
        assert!(
            matches!(f64_best, Device::CPU),
            "F64 backend should default to CPU without CUDA feature"
        );
    }

    // Test CPU to GPU data movement with proper synchronization
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cpu_to_gpu_movement() {
        let backend = get_backend::<f32>();

        // Skip test if CUDA not available
        if !backend.has_cuda() {
            println!("Skipping CPU->GPU test: CUDA not available");
            return;
        }

        // Synchronize CUDA context before test to ensure clean state
        if let Some(cuda_backend) = backend.cuda_backend() {
            cuda_backend
                .synchronize()
                .expect("Failed to synchronize CUDA before test");
        }

        // Create test data on CPU
        let test_data = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let array_data = test_data.into_dyn();

        // Create CPU storage first
        let (cpu_device, cpu_storage) = backend
            .create_storage_from_data(&array_data, Device::CPU)
            .expect("Failed to create CPU storage");

        assert!(matches!(cpu_device, Device::CPU));
        assert!(!cpu_storage.is_gpu(), "CPU storage should not be on GPU");

        // Move to GPU
        let (gpu_device, gpu_storage) = backend
            .move_storage(cpu_storage, Device::CUDA(0))
            .expect("Failed to move storage from CPU to GPU");

        assert!(
            matches!(gpu_device, Device::CUDA(_)),
            "Should be on GPU device"
        );
        assert!(gpu_storage.is_gpu(), "GPU storage should be on GPU");

        // Synchronize after GPU operations to ensure completion
        if let Some(cuda_backend) = backend.cuda_backend() {
            cuda_backend
                .synchronize()
                .expect("Failed to synchronize CUDA after GPU operations");
        }

        println!("Successfully moved data from CPU to GPU");
        println!("Original shape: {:?}", array_data.shape());
        println!("GPU storage shape: {:?}", gpu_storage.shape());
        assert_eq!(
            array_data.shape(),
            gpu_storage.shape(),
            "Shapes should match after movement"
        );
    }

    // Test GPU to CPU data movement with proper synchronization
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gpu_to_cpu_movement() {
        let backend = get_backend::<f32>();

        // Skip test if CUDA not available
        if !backend.has_cuda() {
            println!("Skipping GPU->CPU test: CUDA not available");
            return;
        }

        // Synchronize CUDA context before test
        if let Some(cuda_backend) = backend.cuda_backend() {
            cuda_backend
                .synchronize()
                .expect("Failed to synchronize CUDA before test");
        }

        // Create test data and move to GPU first
        let test_data = Array1::from_vec(vec![10.0f32, 20.0, 30.0, 40.0]);
        let array_data = test_data.into_dyn();

        let (_, gpu_storage) = backend
            .create_storage_from_data(&array_data, Device::CUDA(0))
            .expect("Failed to create GPU storage");

        assert!(gpu_storage.is_gpu(), "Storage should be on GPU");

        // Synchronize after GPU creation
        if let Some(cuda_backend) = backend.cuda_backend() {
            cuda_backend
                .synchronize()
                .expect("Failed to synchronize after GPU creation");
        }

        // Move back to CPU
        let (cpu_device, cpu_storage) = backend
            .move_storage(gpu_storage, Device::CPU)
            .expect("Failed to move storage from GPU to CPU");

        assert!(matches!(cpu_device, Device::CPU));
        assert!(!cpu_storage.is_gpu(), "CPU storage should not be on GPU");

        // Synchronize after CPU transfer
        if let Some(cuda_backend) = backend.cuda_backend() {
            cuda_backend
                .synchronize()
                .expect("Failed to synchronize after CPU transfer");
        }

        // Verify data integrity
        let cpu_data = cpu_storage.cpu_data().expect("Failed to get CPU data");

        assert_eq!(cpu_data.shape(), array_data.shape(), "Shapes should match");

        // Check data values are preserved
        for (original, retrieved) in array_data.iter().zip(cpu_data.iter()) {
            assert!(
                (original - retrieved).abs() < 1e-6,
                "Data values should be preserved: {} != {}",
                original,
                retrieved
            );
        }

        println!("Successfully moved data from GPU back to CPU with data integrity");
    }

    // Test round-trip CPU -> GPU -> CPU movement with synchronization
    #[cfg(feature = "cuda")]
    #[test]
    fn test_roundtrip_cpu_gpu_cpu_movement() {
        let backend = get_backend::<f32>();

        if !backend.has_cuda() {
            println!("Skipping round-trip test: CUDA not available");
            return;
        }

        // Synchronize before starting
        if let Some(cuda_backend) = backend.cuda_backend() {
            cuda_backend
                .synchronize()
                .expect("Failed to synchronize CUDA before test");
        }

        // Original test data
        let original_data =
            Array1::from_vec(vec![1.5f32, -2.7, std::f32::consts::PI, 0.0, -100.25]);
        let original_array = original_data.clone().into_dyn();
        let original_shape = original_array.shape();
        println!("Original shape {:?}", original_shape);
        // CPU -> GPU -> CPU round trip
        let (_, cpu_storage1) = backend
            .create_storage_from_data(&original_array, Device::CPU)
            .expect("Failed to create initial CPU storage");

        let (_, gpu_storage) = backend
            .move_storage(cpu_storage1, Device::CUDA(0))
            .expect("Failed to move CPU->GPU");

        // Synchronize after CPU->GPU transfer
        if let Some(cuda_backend) = backend.cuda_backend() {
            cuda_backend
                .synchronize()
                .expect("Failed to synchronize after CPU->GPU");
        }

        let (_, cpu_storage2) = backend
            .move_storage(gpu_storage, Device::CPU)
            .expect("Failed to move GPU->CPU");

        // Final synchronization
        if let Some(cuda_backend) = backend.cuda_backend() {
            cuda_backend
                .synchronize()
                .expect("Failed to synchronize after GPU->CPU");
        }

        // Verify final data matches original
        let final_data = cpu_storage2
            .cpu_data()
            .expect("Failed to get final CPU data");

        assert_eq!(
            final_data.shape(),
            original_array.shape(),
            "Final shape should match original"
        );

        for (original, final_val) in original_array.iter().zip(final_data.iter()) {
            assert!(
                (original - final_val).abs() < 1e-6,
                "Round-trip should preserve data: {} != {}",
                original,
                final_val
            );
        }

        println!("Round-trip CPU->GPU->CPU movement successful with data integrity");
    }

    // Test device validation
    #[test]
    fn test_device_validation() {
        let backend = get_backend::<f32>();

        // CPU should always be valid
        let cpu_result = backend.validate_device(Device::CPU);
        assert!(cpu_result.is_ok(), "CPU device should always be valid");

        #[cfg(feature = "cuda")]
        {
            let cuda_result = backend.validate_device(Device::CUDA(0));
            if backend.has_cuda() {
                assert!(
                    cuda_result.is_ok(),
                    "CUDA device should be valid when available"
                );
            } else {
                assert!(
                    cuda_result.is_err(),
                    "CUDA device should be invalid when not available"
                );
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Without CUDA feature, CUDA devices should be rejected at compile time
            // This test just ensures CPU works
            assert!(matches!(backend.best_device(), Device::CPU));
        }
    }

    // Test storage creation on different devices
    #[test]
    fn test_storage_creation_different_devices() {
        let backend = get_backend::<f32>();
        let test_shape = &[2, 3];

        // CPU storage should always work
        let cpu_result = backend.create_storage(test_shape, Device::CPU);
        assert!(cpu_result.is_ok(), "CPU storage creation should work");

        let (cpu_device, cpu_storage) = cpu_result.unwrap();
        assert!(matches!(cpu_device, Device::CPU));
        assert_eq!(cpu_storage.shape(), test_shape);

        #[cfg(feature = "cuda")]
        {
            if backend.has_cuda() {
                let gpu_result = backend.create_storage(test_shape, Device::CUDA(0));
                assert!(
                    gpu_result.is_ok(),
                    "GPU storage creation should work when CUDA available"
                );

                let (gpu_device, gpu_storage) = gpu_result.unwrap();
                assert!(matches!(gpu_device, Device::CUDA(_)));
                assert_eq!(gpu_storage.shape(), test_shape);
                assert!(gpu_storage.is_gpu(), "GPU storage should report as GPU");
            }
        }
    }

    // Test auto device selection
    #[test]
    fn test_auto_device_selection() {
        let backend = get_backend::<f32>();
        let test_shape = &[3, 3];

        let auto_result = backend.create_storage_auto(test_shape);
        assert!(auto_result.is_ok(), "Auto storage creation should work");

        let (selected_device, storage) = auto_result.unwrap();

        #[cfg(feature = "cuda")]
        {
            if backend.has_cuda() {
                assert!(
                    matches!(selected_device, Device::CUDA(_)),
                    "Auto selection should prefer CUDA when available"
                );
                assert!(
                    storage.is_gpu(),
                    "Auto-selected storage should be on GPU when CUDA available"
                );
            } else {
                assert!(
                    matches!(selected_device, Device::CPU),
                    "Auto selection should use CPU when CUDA unavailable"
                );
                assert!(
                    !storage.is_gpu(),
                    "Auto-selected storage should be on CPU when CUDA unavailable"
                );
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            assert!(
                matches!(selected_device, Device::CPU),
                "Auto selection should always use CPU without CUDA feature"
            );
            assert!(
                !storage.is_gpu(),
                "Storage should be on CPU without CUDA feature"
            );
        }

        println!("Auto-selected device: {:?}", selected_device);
        println!("Storage is on GPU: {}", storage.is_gpu());
    }

    // Test backend manager is consistent across calls
    #[test]
    fn test_backend_consistency() {
        let backend1 = get_backend::<f32>();
        let backend2 = get_backend::<f32>();

        // Should be same instance (static singleton)
        assert_eq!(
            backend1 as *const _, backend2 as *const _,
            "Backend manager should be singleton"
        );

        // Should have consistent CUDA availability
        assert_eq!(
            backend1.has_cuda(),
            backend2.has_cuda(),
            "CUDA availability should be consistent"
        );

        assert_eq!(
            format!("{:?}", backend1.best_device()),
            format!("{:?}", backend2.best_device()),
            "Best device should be consistent"
        );
    }

    // Test large data movement to verify memory management with synchronization
    #[cfg(feature = "cuda")]
    #[test]
    fn test_large_data_movement() {
        let backend = get_backend::<f32>();

        if !backend.has_cuda() {
            println!("Skipping large data test: CUDA not available");
            return;
        }

        // Synchronize before test
        if let Some(cuda_backend) = backend.cuda_backend() {
            cuda_backend
                .synchronize()
                .expect("Failed to synchronize before large data test");
        }

        // Create larger test data (1MB of f32s)
        let large_size = 262144; // 1MB / 4 bytes
        let large_data: Vec<f32> = (0..large_size).map(|i| i as f32 * 0.1).collect();
        let large_array = ArrayD::from_shape_vec(vec![large_size], large_data.clone())
            .expect("Failed to create large array");

        // Test CPU -> GPU movement with large data
        let (_, cpu_storage) = backend
            .create_storage_from_data(&large_array, Device::CPU)
            .expect("Failed to create large CPU storage");

        let (_, gpu_storage) = backend
            .move_storage(cpu_storage, Device::CUDA(0))
            .expect("Failed to move large data to GPU");

        // Synchronize after GPU transfer
        if let Some(cuda_backend) = backend.cuda_backend() {
            cuda_backend
                .synchronize()
                .expect("Failed to synchronize after large GPU transfer");
        }

        let (_, final_cpu_storage) = backend
            .move_storage(gpu_storage, Device::CPU)
            .expect("Failed to move large data back to CPU");

        // Final synchronization
        if let Some(cuda_backend) = backend.cuda_backend() {
            cuda_backend
                .synchronize()
                .expect("Failed to synchronize after large CPU transfer");
        }

        // Verify data integrity on a sample of values
        let final_data = final_cpu_storage
            .cpu_data()
            .expect("Failed to get final large CPU data");

        // Check first, middle, and last values
        let indices = [0, large_size / 2, large_size - 1];
        for &i in &indices {
            let expected = large_data[i];
            let actual = final_data[[i]];
            assert!(
                (expected - actual).abs() < 1e-5,
                "Large data value mismatch at index {}: {} != {}",
                i,
                expected,
                actual
            );
        }

        println!(
            "Successfully moved large data ({} elements) with integrity",
            large_size
        );
    }

    // Test multiple dtype support
    #[test]
    fn test_multiple_dtypes() {
        let f32_backend = get_backend::<f32>();
        let f64_backend = get_backend::<f64>();

        // Both should work with their respective types
        let f32_shape = &[2, 2];
        let f64_shape = &[3, 3];

        let f32_result = f32_backend.create_storage_auto(f32_shape);
        let f64_result = f64_backend.create_storage_auto(f64_shape);

        assert!(f32_result.is_ok(), "F32 backend should work");
        assert!(f64_result.is_ok(), "F64 backend should work");

        let (f32_device, f32_storage) = f32_result.unwrap();
        let (f64_device, f64_storage) = f64_result.unwrap();

        assert_eq!(f32_storage.shape(), f32_shape);
        assert_eq!(f64_storage.shape(), f64_shape);

        println!("F32 backend device: {:?}", f32_device);
        println!("F64 backend device: {:?}", f64_device);

        // Both should have same CUDA availability
        assert_eq!(
            f32_backend.has_cuda(),
            f64_backend.has_cuda(),
            "Both dtypes should have same CUDA availability"
        );
    }
}
