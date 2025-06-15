// src/backend/cuda/kernels.rs
use cudarc::driver::{CudaDevice, CudaFunction, LaunchConfig};
use std::collections::HashMap;
use std::sync::Arc;

/// Embedded PTX kernels
pub const ADD_PTX: &[u8] = include_bytes!("../../../kernels/add.ptx");
pub const MATMUL_PTX: &[u8] = include_bytes!("../../../kernels/matmul.ptx");
pub const RELU_PTX: &[u8] = include_bytes!("../../../kernels/relu.ptx");

/// CUDA kernel manager that handles loading and executing kernels
pub struct CudaKernels {
    device: Arc<CudaDevice>,
    functions: HashMap<String, CudaFunction>,
}

impl CudaKernels {
    /// Creates a new CUDA kernels manager for the specified device
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            functions: HashMap::new(),
        }
    }

    /// Loads a single kernel from PTX bytes
    pub fn load_kernel(&mut self, name: &str, ptx_bytes: &[u8]) -> Result<(), String> {
        let ptx_str = std::str::from_utf8(ptx_bytes)
            .map_err(|e| format!("Invalid PTX UTF-8: {}", e))?;

        match name {
            "add" => {
                self.device.load_ptx(ptx_str.into(), "add_module", &["add"])
                    .map_err(|e| format!("Failed to load add kernel: {}", e))?;
                let func = self.device.get_func("add_module", "add")
                    .ok_or_else(|| "Failed to get add function".to_string())?;
                self.functions.insert("add".to_string(), func);
            },
            "matmul" => {
                self.device.load_ptx(ptx_str.into(), "matmul_module", &["matmul"])
                    .map_err(|e| format!("Failed to load matmul kernel: {}", e))?;
                let func = self.device.get_func("matmul_module", "matmul")
                    .ok_or_else(|| "Failed to get matmul function".to_string())?;
                self.functions.insert("matmul".to_string(), func);
            },
            "relu" => {
                self.device.load_ptx(ptx_str.into(), "relu_module", &["relu"])
                    .map_err(|e| format!("Failed to load relu kernel: {}", e))?;
                let func = self.device.get_func("relu_module", "relu")
                    .ok_or_else(|| "Failed to get relu function".to_string())?;
                self.functions.insert("relu".to_string(), func);
            },
            _ => return Err(format!("Unknown kernel name: {}", name)),
        }

        Ok(())
    }

    /// Gets a cloned kernel function by name for launching
    /// This is the preferred method since launch() consumes the function
    pub fn get_function_cloned(&self, name: &str) -> Option<CudaFunction> {
        self.functions.get(name).cloned()
    }

    /// Gets a reference to a loaded kernel function by name
    /// Use this only for inspection, not for launching
    pub fn get_function(&self, name: &str) -> Option<&CudaFunction> {
        self.functions.get(name)
    }

    /// Convenience method to launch a kernel by name handling the cloning internally
    /// More effient than calling `launch_kernel` with a cloned function
    pub fn launch_kernel<Params>(&self, name: &str, cfg: LaunchConfig, params: Params) -> Result<(), String> 
    where
        Params: cudarc::driver::LaunchParam,
    {
        let kernel = self.get_function_cloned(name)
            .ok_or_else(|| format!("Kernel '{}' not found", name))?;
        
        unsafe {
            kernel.launch(cfg, params)
                .map_err(|e| format!("Failed to launch kernel '{}': {}", name, e))
        }
    }

    /// Returns reference to the CUDA device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Lists all loaded kernel names
    pub fn loaded_kernels(&self) -> Vec<&String> {
        self.functions.keys().collect()
    }
}

/// Loads all available kernels into the kernel manager
pub fn load_all_kernels(kernels: &mut CudaKernels) -> Result<(), String> {
    let kernel_list = [
        ("add", ADD_PTX),
        ("matmul", MATMUL_PTX),
        ("relu", RELU_PTX),
    ];

    for (name, ptx_bytes) in kernel_list.iter() {
        kernels.load_kernel(name, ptx_bytes)?;
    }

    println!("✓ Loaded {} CUDA kernels successfully", kernel_list.len());
    Ok(())
}