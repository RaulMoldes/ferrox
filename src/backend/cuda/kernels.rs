// src/backend/cuda/kernels.rs
use cudarc::driver::{CudaDevice, CudaFunction, LaunchConfig, LaunchAsync};
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
                self.device.load_ptx(ptx_str.into(), "add_module", &["elementwise_add"])
                    .map_err(|e| format!("Failed to load add kernel: {}", e))?;
                let func = self.device.get_func("add_module", "elementwise_add")
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
                self.device.load_ptx(ptx_str.into(), "relu_module", &["relu_forward"])
                    .map_err(|e| format!("Failed to load relu kernel: {}", e))?;
                let func = self.device.get_func("relu_module", "relu_forward")
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

    /// Gets a reference to a loaded kernel function by name (Not cloned)
    /// Should only be used if you don't need to launch immediately
    pub fn get_function(&self, name: &str) -> Option<&CudaFunction> {
        self.functions.get(name)
    }

    /// Convenience method to launch add kernel
    pub fn launch_add(&self, cfg: LaunchConfig, a: &cudarc::driver::CudaSlice<f32>, b: &cudarc::driver::CudaSlice<f32>, c: &mut cudarc::driver::CudaSlice<f32>, size: i32) -> Result<(), String> {
        let kernel = self.get_function_cloned("add")
            .ok_or_else(|| "Add kernel not found".to_string())?;
        
        unsafe {
            kernel.launch(cfg, (a, b, c, size))
                .map_err(|e| format!("Failed to launch add kernel: {}", e))
        }
    }

    /// Convenience method to launch relu kernel
    pub fn launch_relu(&self, cfg: LaunchConfig, input: &cudarc::driver::CudaSlice<f32>, output: &mut cudarc::driver::CudaSlice<f32>, size: i32) -> Result<(), String> {
        let kernel = self.get_function_cloned("relu")
            .ok_or_else(|| "ReLU kernel not found".to_string())?;
        
        unsafe {
            kernel.launch(cfg, (input, output, size))
                .map_err(|e| format!("Failed to launch relu kernel: {}", e))
        }
    }

    /// Convenience method to launch matmul kernel
    pub fn launch_matmul(&self, cfg: LaunchConfig, a: &cudarc::driver::CudaSlice<f32>, b: &cudarc::driver::CudaSlice<f32>, c: &mut cudarc::driver::CudaSlice<f32>, m: i32, n: i32, k: i32) -> Result<(), String> {
        let kernel = self.get_function_cloned("matmul")
            .ok_or_else(|| "MatMul kernel not found".to_string())?;
        
        unsafe {
            kernel.launch(cfg, (a, b, c, m, n, k))
                .map_err(|e| format!("Failed to launch matmul kernel: {}", e))
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