// src/backend/cuda/context.rs
use super::kernels::{CudaKernels, load_all_kernels};
use super::memory::{CudaMemoryManager, CudaTensor};
use super::ops::CudaOps;
use cudarc::driver::CudaContext;
use std::sync::Arc;
use std::default::Default;

/// Main CUDA backend that manages context and kernels
pub struct CudaBackend {
    context: Arc<CudaContext>, // CudaContext::new() returns this type directly, // Default stream for memory operations
    kernels: CudaKernels,
    context_id: usize, // Unique identifier for the CUDA context
    memory_manager: CudaMemoryManager,
}

impl CudaBackend {
    /// Creates a new CUDA backend for the specified context
    pub fn new(context_id: usize) -> Result<Self, String> {
        // CudaContext::new() returns Arc<CudaContext> already, not CudaContext
        let context = CudaContext::new(context_id)
            .map_err(|e| format!("Failed to initialize CUDA context {}: {}", context_id, e))?;

        // context is already Arc<CudaContext>, so we can clone it directly
        let mut kernels = CudaKernels::new(context.clone());
        let memory_manager = CudaMemoryManager::new(context.clone())?;
        // Load all kernels during initialization
        load_all_kernels(&mut kernels)?;

        Ok(Self {
            context,
            kernels,
            context_id,
            memory_manager,
        })
    }

    pub fn default_stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.context.default_stream()
    }

    /// Returns reference to the CUDA context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    pub fn id(&self) -> usize {
        self.context_id
    }

    /// Returns reference to the kernel manager
    pub fn kernels(&self) -> &CudaKernels {
        &self.kernels
    }

    /// Returns mutable reference to the kernel manager
    pub fn kernels_mut(&mut self) -> &mut CudaKernels {
        &mut self.kernels
    }

    /// Synchronizes the context (waits for all operations to complete)
    pub fn synchronize(&self) -> Result<(), String> {
        self.context
            .synchronize()
            .map_err(|e| format!("CUDA synchronization failed: {}", e))
    }

    /// Returns context name for debugging
    pub fn name(&self) -> String {
        format!("CUDA context {}", self.context_id)
    }

    /// Creates a CUDA tensor from CPU data
    /// This method takes host data and copies it to GPU memory
    pub fn create_tensor_from_cpu<T>(
        &self,
        data: Vec<T>,
        shape: Vec<usize>,
    ) -> Result<CudaTensor<T>, String>
    where
        T: cudarc::driver::DeviceRepr
            + Clone
            + cudarc::driver::ValidAsZeroBits
            + std::marker::Unpin + Default,
    {
        // Validate that data size matches shape
        let size = shape.iter().product::<usize>();
        if data.len() != size {
            return Err(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                size
            ));
        }

        // Create memory manager for this context
        let memory_manager = self.memory_manager();

        // Transfer data from host to context
        let cuda_data = memory_manager.host_to_device(data)?;

        // Create and return the CUDA tensor
        Ok(CudaTensor::new(cuda_data, shape))
    }

    /// Returns reference to memory manager
    /// This method provides access to the memory manager for external use
    pub fn memory_manager(&self) -> &CudaMemoryManager {
        &self.memory_manager
    }

    /// Returns reference to operations interface
    /// This method provides access to CUDA operations for tensor computations
    pub fn ops(&self) -> CudaOps<'_> {
        let memory = self.memory_manager();
        CudaOps::new(&self.kernels, memory)
    }
}

impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBackend")
            .field("context_id", &self.context_id)
            .field("context_name", &self.name())
            .finish()
    }
}
