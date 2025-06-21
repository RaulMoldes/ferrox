// src/backend/cuda/kernels.rs
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use std::collections::HashMap;
use std::sync::Arc;
use std::env::Args;

/// Embedded PTX kernels
pub const ELEMENTWISE_PTX: &[u8] = include_bytes!("../../../kernels/elementwise.ptx");
pub const ACTIVATIONS_PTX: &[u8] = include_bytes!("../../../kernels/activations.ptx");
pub const MATMUL_PTX: &[u8] = include_bytes!("../../../kernels/matmul.ptx");
pub const REDUCES_PTX: &[u8] = include_bytes!("../../../kernels/reduce.ptx");
pub const TRANSPOSE_PTX: &[u8] = include_bytes!("../../../kernels/transpose.ptx");
pub const COMPARISON_PTX: &[u8] = include_bytes!("../../../kernels/comparison.ptx");


/// Kernel configuration for automatic loading
struct KernelConfig {
    name: &'static str,
    ptx: &'static [u8],
    module: &'static str,
    functions: &'static [&'static str],
}

/// All kernel configurations
const KERNEL_CONFIGS: &[KernelConfig] = &[
    KernelConfig { name: "elementwise", ptx: ELEMENTWISE_PTX, module: "elementwise_module", functions: &["elementwise_add","elementwise_sqrt", "elemenwise_abs", "elementwise_mul","elementwise_div","elementwise_sub","elementwise_pow", "elementwise_min", "elementwise_max", "elementwise_exp", "elementwise_log", "elementwise_negate"] },
    
    KernelConfig { name: "matmul", ptx: MATMUL_PTX, module: "matmul_module", functions: &["matmul"] },
    KernelConfig { name: "activations", ptx: ACTIVATIONS_PTX, module: "activations_module", functions: &["relu", "sigmoid", "hyperbolic_tanh"] },


    KernelConfig { name: "sum_axis", ptx: REDUCES_PTX, module: "reduces_module", functions: &["sum_axis", "max_along_dim"] },
    KernelConfig { name: "transpose", ptx: TRANSPOSE_PTX, module: "transpose_module", functions: &["transpose_2d"] },


    KernelConfig { name: "comparison", ptx: COMPARISON_PTX, module: "comparison_module", 
        functions: &["greater_equal", "greater_equal_f64", "less_equal", "less_equal_f64", 
                    "equal", "equal_f64", "logical_not", "logical_not_f64", 
                    "in_range", "in_range_f64", "sign", "sign_f64","clamp", "clamp_f64"] },
];


// ------------------------------------------------------------------------------------------------------//
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

    /// Load kernel using its configuration.
    /// This makes it elegant to load kernels from PTX bytecode
    /// and automatically handles function names based on the configuration.
    pub fn load_kernel(&mut self, name: &str, ptx_bytes: &[u8]) -> Result<(), String> {
        let config = KERNEL_CONFIGS.iter()
            .find(|k| k.name == name)
            .ok_or_else(|| format!("Unknown kernel: {}", name))?;

        let ptx_str = std::str::from_utf8(ptx_bytes)
            .map_err(|e| format!("Invalid PTX UTF-8: {}", e))?;

        self.device
            .load_ptx(ptx_str.into(), config.module, config.functions)
            .map_err(|e| format!("Failed to load {} kernel: {}", name, e))?;

        // Store all functions from this module
        for &func_name in config.functions {
            if let Some(func) = self.device.get_func(config.module, func_name) {
                self.functions.insert(name.to_string(), func);
            }
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

    fn launch_kernel<T, P>(&self, kernel_name: &str, cfg: LaunchConfig, params: P) -> Result<(), String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
         P: cudarc::driver::LaunchAsync<Args>,
    {
        let kernel = self.get_function_cloned(kernel_name)
            .ok_or_else(|| format!("{} kernel not found", kernel_name))?;

        unsafe {
            kernel.launch(cfg, params)
                .map_err(|e| format!("Failed to launch {} kernel: {}", kernel_name, e))
        }
    }

    // Simplified launch methods using the generic launcher
    pub fn launch_add<T>(&self, cfg: LaunchConfig, a: &CudaSlice<T>, b: &CudaSlice<T>, 
                         c: &mut CudaSlice<T>, size: i32) -> Result<(), String>
    where T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        self.launch_kernel::<T, _>("add", cfg, (a, b, c, size))
    }

    pub fn launch_mul<T>(&self, cfg: LaunchConfig, a: &CudaSlice<T>, b: &CudaSlice<T>, 
                         c: &mut CudaSlice<T>, size: i32) -> Result<(), String>
    where T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        self.launch_kernel::<T, _>("mul", cfg, (a, b, c, size))
    }

    pub fn launch_div<T>(&self, cfg: LaunchConfig, a: &CudaSlice<T>, b: &CudaSlice<T>, 
                         c: &mut CudaSlice<T>, size: i32) -> Result<(), String>
    where T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        self.launch_kernel::<T, _>("div", cfg, (a, b, c, size))
    }

    pub fn launch_sub<T>(&self, cfg: LaunchConfig, a: &CudaSlice<T>, b: &CudaSlice<T>, 
                         c: &mut CudaSlice<T>, size: i32) -> Result<(), String>
    where T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        self.launch_kernel::<T, _>("sub", cfg, (a, b, c, size))
    }

    pub fn launch_power<T>(&self, cfg: LaunchConfig, a: &CudaSlice<T>, b: &CudaSlice<T>, 
                           c: &mut CudaSlice<T>, size: i32) -> Result<(), String>
    where T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        self.launch_kernel::<T, _>("power", cfg, (a, b, c, size))
    }

    pub fn launch_relu<T>(&self, cfg: LaunchConfig, input: &CudaSlice<T>, 
                          output: &mut CudaSlice<T>, size: i32) -> Result<(), String>
    where T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        self.launch_kernel::<T, _>("relu", cfg, (input, output, size))
    }

    /// Unified activation launcher
    pub fn launch_activation<T>(&self, kernel_name: &str, cfg: LaunchConfig, 
                               input: &CudaSlice<T>, output: &mut CudaSlice<T>, size: i32) -> Result<(), String>
    where T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        self.launch_kernel::<T, _>(kernel_name, cfg, (input, output, size))
    }

    pub fn launch_matmul<T>(&self, cfg: LaunchConfig, a: &CudaSlice<T>, b: &CudaSlice<T>, 
                           c: &mut CudaSlice<T>, m: i32, n: i32, k: i32) -> Result<(), String>
    where T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        self.launch_kernel::<T, _>("matmul", cfg, (a, b, c, m, n, k))
    }

    pub fn launch_clamp<T>(&self, cfg: LaunchConfig, input: &CudaSlice<T>, output: &mut CudaSlice<T>,
                          min_val: T, max_val: T, size: i32) -> Result<(), String>
    where T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        self.launch_kernel::<T, _>("clamp", cfg, (input, output, min_val, max_val, size))
    }

    pub fn launch_sum_axis<T>(&self, cfg: LaunchConfig, input: &CudaSlice<T>, output: &mut CudaSlice<T>,
                             outer_size: i32, axis_size: i32, inner_size: i32) -> Result<(), String>
    where T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        self.launch_kernel::<T, _>("sum_axis", cfg, (input, output, outer_size, axis_size, inner_size))
    }

    pub fn launch_negate<T>(&self, cfg: LaunchConfig, input: &CudaSlice<T>, 
                           output: &mut CudaSlice<T>, size: i32) -> Result<(), String>
    where T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        self.launch_kernel::<T, _>("negate", cfg, (input, output, size))
    }

    pub fn launch_transpose_2d<T>(&self, cfg: LaunchConfig, input: &CudaSlice<T>, 
                                 output: &mut CudaSlice<T>, rows: i32, cols: i32) -> Result<(), String>
    where T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        self.launch_kernel::<T, _>("transpose", cfg, (input, output, rows, cols))
    }

    pub fn launch_min_elementwise<T>(&self, cfg: LaunchConfig, a: &CudaSlice<T>, b: &CudaSlice<T>,
                                    c: &mut CudaSlice<T>, size: i32) -> Result<(), String>
    where T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        self.launch_kernel::<T, _>("min", cfg, (a, b, c, size))
    }

    pub fn launch_max_elementwise<T>(&self, cfg: LaunchConfig, a: &CudaSlice<T>, b: &CudaSlice<T>,
                                    c: &mut CudaSlice<T>, size: i32) -> Result<(), String>
    where T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        self.launch_kernel::<T, _>("max", cfg, (a, b, c, size))
    }

    pub fn launch_abs<T>(&self, cfg: LaunchConfig, input: &CudaSlice<T>, 
                        output: &mut CudaSlice<T>, size: i32) -> Result<(), String>
    where T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        self.launch_kernel::<T, _>("abs", cfg, (input, output, size))
    }

    pub fn launch_sqrt<T>(&self, cfg: LaunchConfig, input: &CudaSlice<T>, 
                         output: &mut CudaSlice<T>, size: i32) -> Result<(), String>
    where T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        self.launch_kernel::<T, _>("sqrt", cfg, (input, output, size))
    }

    pub fn launch_max_along_dim<T>(&self, cfg: LaunchConfig, input: &CudaSlice<T>, output: &mut CudaSlice<T>,
                                  outer_size: i32, axis_size: i32, inner_size: i32) -> Result<(), String>
    where T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + std::marker::Unpin,
    {
        self.launch_kernel::<T, _>("max_along_dim", cfg, (input, output, outer_size, axis_size, inner_size))
    }

    // Comparison operations
    pub fn launch_greater_equal_f32(&self, cfg: LaunchConfig, a: &CudaSlice<f32>, b: &CudaSlice<f32>,
                                   result: &mut CudaSlice<f32>, size: i32) -> Result<(), String> {
        self.launch_kernel::<f32, _>("greater_equal", cfg, (a, b, result, size))
    }

    pub fn launch_greater_equal_f64(&self, cfg: LaunchConfig, a: &CudaSlice<f64>, b: &CudaSlice<f64>,
                                   result: &mut CudaSlice<f64>, size: i32) -> Result<(), String> {
        self.launch_kernel::<f64, _>("greater_equal_f64", cfg, (a, b, result, size))
    }

    pub fn launch_less_equal_f32(&self, cfg: LaunchConfig, a: &CudaSlice<f32>, b: &CudaSlice<f32>,
                                result: &mut CudaSlice<f32>, size: i32) -> Result<(), String> {
        self.launch_kernel::<f32, _>("less_equal", cfg, (a, b, result, size))
    }

    pub fn launch_less_equal_f64(&self, cfg: LaunchConfig, a: &CudaSlice<f64>, b: &CudaSlice<f64>,
                                result: &mut CudaSlice<f64>, size: i32) -> Result<(), String> {
        self.launch_kernel::<f64, _>("less_equal_f64", cfg, (a, b, result, size))
    }

    pub fn launch_equal_f32(&self, cfg: LaunchConfig, a: &CudaSlice<f32>, b: &CudaSlice<f32>,
                           result: &mut CudaSlice<f32>, size: i32) -> Result<(), String> {
        self.launch_kernel::<f32, _>("equal", cfg, (a, b, result, size))
    }

    pub fn launch_equal_f64(&self, cfg: LaunchConfig, a: &CudaSlice<f64>, b: &CudaSlice<f64>,
                           result: &mut CudaSlice<f64>, size: i32) -> Result<(), String> {
        self.launch_kernel::<f64, _>("equal_f64", cfg, (a, b, result, size))
    }

    // Unary comparison operations
    pub fn launch_logical_not_f32(&self, cfg: LaunchConfig, input: &CudaSlice<f32>,
                                 result: &mut CudaSlice<f32>, size: i32) -> Result<(), String> {
        self.launch_kernel::<f32, _>("logical_not", cfg, (input, result, size))
    }

    pub fn launch_logical_not_f64(&self, cfg: LaunchConfig, input: &CudaSlice<f64>,
                                 result: &mut CudaSlice<f64>, size: i32) -> Result<(), String> {
        self.launch_kernel::<f64, _>("logical_not_f64", cfg, (input, result, size))
    }

    pub fn launch_in_range_f32(&self, cfg: LaunchConfig, input: &CudaSlice<f32>, min_val: f32, max_val: f32,
                              result: &mut CudaSlice<f32>, size: i32) -> Result<(), String> {
        self.launch_kernel::<f32, _>("in_range", cfg, (input, min_val, max_val, result, size))
    }

    pub fn launch_in_range_f64(&self, cfg: LaunchConfig, input: &CudaSlice<f64>, min_val: f64, max_val: f64,
                              result: &mut CudaSlice<f64>, size: i32) -> Result<(), String> {
        self.launch_kernel::<f64, _>("in_range_f64", cfg, (input, min_val, max_val, result, size))
    }

    pub fn launch_sign_f32(&self, cfg: LaunchConfig, input: &CudaSlice<f32>,
                          result: &mut CudaSlice<f32>, size: i32) -> Result<(), String> {
        self.launch_kernel::<f32, _>("sign", cfg, (input, result, size))
    }

    pub fn launch_sign_f64(&self, cfg: LaunchConfig, input: &CudaSlice<f64>,
                          result: &mut CudaSlice<f64>, size: i32) -> Result<(), String> {
        self.launch_kernel::<f64, _>("sign_f64", cfg, (input, result, size))
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

/// ------------------------------------------------------------------------------------------------------//
/// Loads all predefined CUDA kernels into the provided CudaKernels instance
pub fn load_all_kernels(kernels: &mut CudaKernels) -> Result<(), String> {
    for config in KERNEL_CONFIGS {
        kernels.load_kernel(config.name, config.ptx)?;
    }
    
    println!("✓ Loaded {} CUDA kernels successfully", KERNEL_CONFIGS.len());
    Ok(())
}