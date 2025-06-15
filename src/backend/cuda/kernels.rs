// src/backend/cuda/kernels.rs
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};
use std::collections::HashMap;
use std::sync::Arc;

/// Embedded PTX kernels
pub const ADD_PTX: &[u8] = include_bytes!("../../../kernels/add.ptx");
pub const MATMUL_PTX: &[u8] = include_bytes!("../../../kernels/matmul.ptx");
pub const RELU_PTX: &[u8] = include_bytes!("../../../kernels/relu.ptx");
pub const MUL_PTX: &[u8] = include_bytes!("../../../kernels/mul.ptx");
pub const DIV_PTX: &[u8] = include_bytes!("../../../kernels/div.ptx");
pub const EXP_PTX: &[u8] = include_bytes!("../../../kernels/exp.ptx");
//pub const LOG_PTX: &[u8] = include_bytes!("../../../kernels/log.ptx");
pub const SIGMOID_PTX: &[u8] = include_bytes!("../../../kernels/sigmoid.ptx");
//pub const TANH_PTX: &[u8] = include_bytes!("../../../kernels/tanh.ptx");

/// CUDA kernel manager that handles loading and executing kernels
pub struct CudaKernels {
    device: Arc<CudaDevice>,
    // Maintains a Stack of loaded kernel functions.
    // This allows us to inmmediately launch kernels without needing to
    // retrieve them from the device each time.
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
        let ptx_str =
            std::str::from_utf8(ptx_bytes).map_err(|e| format!("Invalid PTX UTF-8: {}", e))?;

        match name {
            "add" => {
                self.device
                    .load_ptx(ptx_str.into(), "add_module", &["elementwise_add"])
                    .map_err(|e| format!("Failed to load add kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("add_module", "elementwise_add")
                    .ok_or_else(|| "Failed to get add function".to_string())?;
                self.functions.insert("add".to_string(), func);
            }
            "matmul" => {
                self.device
                    .load_ptx(ptx_str.into(), "matmul_module", &["matmul"])
                    .map_err(|e| format!("Failed to load matmul kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("matmul_module", "matmul")
                    .ok_or_else(|| "Failed to get matmul function".to_string())?;
                self.functions.insert("matmul".to_string(), func);
            }
            "relu" => {
                self.device
                    .load_ptx(ptx_str.into(), "relu_module", &["relu_forward"])
                    .map_err(|e| format!("Failed to load relu kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("relu_module", "relu_forward")
                    .ok_or_else(|| "Failed to get relu function".to_string())?;
                self.functions.insert("relu".to_string(), func);
            }
            "mul" => {
                self.device
                    .load_ptx(ptx_str.into(), "mul_module", &["elementwise_mul"])
                    .map_err(|e| format!("Failed to load mul kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("mul_module", "elementwise_mul")
                    .ok_or_else(|| "Failed to get mul function".to_string())?;
                self.functions.insert("mul".to_string(), func);
            }
            "exp" => {
                self.device
                    .load_ptx(ptx_str.into(), "exp_module", &["exp_forward"])
                    .map_err(|e| format!("Failed to load exp kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("exp_module", "exp_forward")
                    .ok_or_else(|| "Failed to get exp function".to_string())?;
                self.functions.insert("exp".to_string(), func);
            }
            "sigmoid" => {
                self.device
                    .load_ptx(ptx_str.into(), "sigmoid_module", &["sigmoid_forward"])
                    .map_err(|e| format!("Failed to load sigmoid kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("sigmoid_module", "sigmoid_forward")
                    .ok_or_else(|| "Failed to get sigmoid function".to_string())?;
                self.functions.insert("sigmoid".to_string(), func);
            }
            "div" => {
                self.device
                    .load_ptx(ptx_str.into(), "div_module", &["elementwise_div"])
                    .map_err(|e| format!("Failed to load div kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("div_module", "elementwise_div")
                    .ok_or_else(|| "Failed to get div function".to_string())?;
                self.functions.insert("div".to_string(), func);
            }
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
    pub fn launch_add<T>(
        &self,
        cfg: LaunchConfig,
        a: &cudarc::driver::CudaSlice<T>,
        b: &cudarc::driver::CudaSlice<T>,
        c: &mut cudarc::driver::CudaSlice<T>,
        size: i32,
    ) -> Result<(), String> 
    where 
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        let kernel = self
            .get_function_cloned("add")
            .ok_or_else(|| "Add kernel not found".to_string())?;

        unsafe {
            kernel
                .launch(cfg, (a, b, c, size))
                .map_err(|e| format!("Failed to launch add kernel: {}", e))
        }
    }

    /// Convenience method to launch element-wise division kernel
    pub fn launch_div<T>(
        &self,
        cfg: LaunchConfig,
        a: &cudarc::driver::CudaSlice<T>,
        b: &cudarc::driver::CudaSlice<T>,
        c: &mut cudarc::driver::CudaSlice<T>,
        size: i32,
    ) -> Result<(), String> 
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        let kernel = self
            .get_function_cloned("div")
            .ok_or_else(|| "Division kernel not found".to_string())?;

        unsafe {
            kernel
                .launch(cfg, (a, b, c, size))
                .map_err(|e| format!("Failed to launch div kernel: {}", e))
        }
    }

    /// Launch element-wise multiplication kernel
    pub fn launch_mul<T>(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        size: i32,
    ) -> Result<(), String> 
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        let kernel = self
            .get_function_cloned("mul")
            .ok_or_else(|| "Mul kernel not found".to_string())?;

        unsafe {
            kernel
                .launch(cfg, (a, b, c, size))
                .map_err(|e| format!("Failed to launch mul kernel: {}", e))
        }
    }

    /// Convenience method to launch relu kernel
    pub fn launch_relu<T>(
        &self,
        cfg: LaunchConfig,
        input: &cudarc::driver::CudaSlice<T>,
        output: &mut cudarc::driver::CudaSlice<T>,
        size: i32,
    ) -> Result<(), String> 
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        let kernel = self
            .get_function_cloned("relu")
            .ok_or_else(|| "ReLU kernel not found".to_string())?;

        unsafe {
            kernel
                .launch(cfg, (input, output, size))
                .map_err(|e| format!("Failed to launch relu kernel: {}", e))
        }
    }

    /// Launch activation function kernels (exp, log, sigmoid, tanh)
    pub fn launch_activation<T>(
        &self,
        kernel_name: &str,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String> 
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
        {
        let kernel = self
            .get_function_cloned(kernel_name)
            .ok_or_else(|| format!("{} kernel not found", kernel_name))?;

        unsafe {
            kernel
                .launch(cfg, (input, output, size))
                .map_err(|e| format!("Failed to launch {} kernel: {}", kernel_name, e))
        }
    }

    /// Convenience method to launch matmul kernel
    pub fn launch_matmul<T>(
        &self,
        cfg: LaunchConfig,
        a: &cudarc::driver::CudaSlice<T>,
        b: &cudarc::driver::CudaSlice<T>,
        c: &mut cudarc::driver::CudaSlice<T>,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), String> 
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        let kernel = self
            .get_function_cloned("matmul")
            .ok_or_else(|| "MatMul kernel not found".to_string())?;

        unsafe {
            kernel
                .launch(cfg, (a, b, c, m, n, k))
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
        ("mul", MUL_PTX),
        ("div", DIV_PTX),
        ("matmul", MATMUL_PTX),
        ("relu", RELU_PTX),
        ("exp", EXP_PTX),
        // ("log", LOG_PTX),
        ("sigmoid", SIGMOID_PTX),
        //  ("tanh", TANH_PTX),
    ];

    for (name, ptx_bytes) in kernel_list.iter() {
        kernels.load_kernel(name, ptx_bytes)?;
    }

    println!("✓ Loaded {} CUDA kernels successfully", kernel_list.len());
    Ok(())
}
