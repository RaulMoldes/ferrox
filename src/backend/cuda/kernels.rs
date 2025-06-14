// src/backend/cuda/kernels.rs
use cudarc::driver::{CudaDevice, CudaFunction};
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
          _ => return Err(format!("Unknown kernel: {}", name)),
      }
      
      Ok(())
  }
    /// Gets a loaded kernel function by name
    pub fn get_function(&self, name: &str) -> Option<&CudaFunction> {
        self.functions.get(name)
    }

    /// Returns reference to the CUDA device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
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