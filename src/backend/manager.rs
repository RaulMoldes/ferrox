// src/backend/manager.rs
// Backend manager that handles both CPU and CUDA backends.
// Simple approach with separate F32 and F64 entry points.

use crate::backend::Device;
#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaContextManager;
use crate::GPUFloat;
use std::sync::OnceLock;
use crate::backend::cuda::ops::CudaOps;
use std::sync::Arc;
/// Simple backend manager that coordinates CPU and CUDA operations
pub struct BackendManager<T: GPUFloat> {
    #[cfg(feature = "cuda")]
    cuda_backend: Option<CudaContextManager<T>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: GPUFloat> Default for BackendManager<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: GPUFloat> BackendManager<T> {
    /// Create new backend manager
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "cuda")]
            cuda_backend: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Initialize with CUDA if available
    pub fn init() -> Self {
        #[cfg(not(feature = "cuda"))]
        let manager = Self::new();

        #[cfg(feature = "cuda")]
        let mut manager = Self::new();

        #[cfg(feature = "cuda")]
        {
            // Try to initialize CUDA backend, but don't fail if unavailable
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

    /// Check if CUDA is available
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

    /// Get CUDA backend if available
    #[cfg(feature = "cuda")]
    pub fn cuda_backend(&self) -> Option<&CudaContextManager<T>> {
        self.cuda_backend.as_ref()
    }

    /// Get the best device for operations
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
}

// Global backend manager instances
static BACKEND_F32: OnceLock<BackendManager<f32>> = OnceLock::new();
static BACKEND_F64: OnceLock<BackendManager<f64>> = OnceLock::new();

// ===== F32 FUNCTIONS =====

/// Get the global F32 backend manager
pub fn get_f32_backend() -> &'static BackendManager<f32> {
    BACKEND_F32.get_or_init(|| BackendManager::<f32>::init())
}

/// Check if CUDA is available for F32
pub fn has_f32_cuda() -> bool {
    get_f32_backend().has_cuda()
}

/// Get the best device for F32
pub fn best_f32_device() -> Device {
    get_f32_backend().best_device()
}

// ===== F64 FUNCTIONS =====

/// Get the global F64 backend manager
pub fn get_f64_backend() -> &'static BackendManager<f64> {
    BACKEND_F64.get_or_init(|| BackendManager::<f64>::init())
}

/// Check if CUDA is available for F64
pub fn has_f64_cuda() -> bool {
    get_f64_backend().has_cuda()
}

/// Get the best device for F64
pub fn best_f64_device() -> Device {
    get_f64_backend().best_device()
}


// ===== GENERIC DISPATCH (WITHOUT TRAIT) =====

/// Get the global backend manager for type T
pub fn get_backend<T: GPUFloat>() -> &'static BackendManager<T> {
    match std::any::TypeId::of::<T>() {
        id if id == std::any::TypeId::of::<f32>() => {
            // SAFETY: We've verified T is f32
            unsafe { std::mem::transmute(get_f32_backend()) }
        }
        id if id == std::any::TypeId::of::<f64>() => {
            // SAFETY: We've verified T is f64
            unsafe { std::mem::transmute(get_f64_backend()) }
        }
        _ => panic!("Unsupported float type for backend"),
    }
}

/// Check if CUDA is available for type T
pub fn has_cuda<T: GPUFloat>() -> bool {
    match std::any::TypeId::of::<T>() {
        id if id == std::any::TypeId::of::<f32>() => has_f32_cuda(),
        id if id == std::any::TypeId::of::<f64>() => has_f64_cuda(),
        _ => false,
    }
}

/// Get the best available device for type T
pub fn best_device<T: GPUFloat>() -> Device {
    match std::any::TypeId::of::<T>() {
        id if id == std::any::TypeId::of::<f32>() => best_f32_device(),
        id if id == std::any::TypeId::of::<f64>() => best_f64_device(),
        _ => Device::CPU,
    }
}



#[cfg(feature = "cuda")]
pub fn with_cuda_context<F, R, T>(f: F) -> Result<R, String>
where
    T: GPUFloat,
    F: FnOnce(&CudaContextManager<T>) -> Result<R, String>,
{
    let backend: &'static BackendManager<T> = get_backend::<T>();
    let context_manager: &CudaContextManager<T> = backend.cuda_backend().ok_or("CUDA backend not available")?;

    f(&context_manager)
}

#[cfg(feature = "cuda")]
pub fn with_cuda_ops<F, R, T>(f: F) -> Result<R, String>
where
    T: GPUFloat,
    F: FnOnce(&CudaOps<T>) -> Result<R, String>,
{
    let backend: &'static BackendManager<T> = get_backend::<T>();
    let context_manager: &CudaContextManager<T> = backend.cuda_backend().ok_or("CUDA backend not available")?;

    let ops: Arc<CudaOps<T>> = context_manager.ops();
    f(&ops)
}
