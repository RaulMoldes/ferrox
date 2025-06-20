// src/backend/cuda/kernels.rs
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use std::collections::HashMap;
use std::sync::Arc;

/// Embedded PTX kernels
pub const ADD_PTX: &[u8] = include_bytes!("../../../kernels/add.ptx");
pub const MATMUL_PTX: &[u8] = include_bytes!("../../../kernels/matmul.ptx");
pub const RELU_PTX: &[u8] = include_bytes!("../../../kernels/relu.ptx");
pub const MUL_PTX: &[u8] = include_bytes!("../../../kernels/mul.ptx");
pub const DIV_PTX: &[u8] = include_bytes!("../../../kernels/div.ptx");
pub const EXP_PTX: &[u8] = include_bytes!("../../../kernels/exp.ptx");
pub const LOG_PTX: &[u8] = include_bytes!("../../../kernels/log.ptx");
pub const SIGMOID_PTX: &[u8] = include_bytes!("../../../kernels/sigmoid.ptx");
pub const TANH_PTX: &[u8] = include_bytes!("../../../kernels/tanh.ptx");
pub const POWER_PTX: &[u8] = include_bytes!("../../../kernels/powf.ptx");
pub const SUB_PTX: &[u8] = include_bytes!("../../../kernels/sub.ptx");
pub const CLAMP_PTX: &[u8] = include_bytes!("../../../kernels/clamp.ptx");
pub const MAX_PTX: &[u8] = include_bytes!("../../../kernels/max.ptx");
pub const SUM_AXIS_PTX: &[u8] = include_bytes!("../../../kernels/sum_axis.ptx");
pub const NEGATE_PTX: &[u8] = include_bytes!("../../../kernels/negate.ptx");
pub const TRANSPOSE_PTX: &[u8] = include_bytes!("../../../kernels/transpose.ptx");
pub const MIN_PTX: &[u8] = include_bytes!("../../../kernels/min.ptx");
pub const COMPARISON_PTX: &[u8] = include_bytes!("../../../kernels/comparison.ptx");
pub const SIGN_PTX: &[u8] = include_bytes!("../../../kernels/sign.ptx");

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
            "log" => {
                self.device
                    .load_ptx(ptx_str.into(), "log_module", &["log_forward"])
                    .map_err(|e| format!("Failed to load log kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("log_module", "log_forward")
                    .ok_or_else(|| "Failed to get log function".to_string())?;
                self.functions.insert("log".to_string(), func);
            }
            "tanh" => {
                self.device
                    .load_ptx(ptx_str.into(), "tanh_module", &["tanh_forward"])
                    .map_err(|e| format!("Failed to load tanh kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("tanh_module", "tanh_forward")
                    .ok_or_else(|| "Failed to get tanh function".to_string())?;
                self.functions.insert("tanh".to_string(), func);
            }
            "power" => {
                self.device
                    .load_ptx(ptx_str.into(), "power_module", &["power_forward"])
                    .map_err(|e| format!("Failed to load power kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("power_module", "power_forward")
                    .ok_or_else(|| "Failed to get power function".to_string())?;
                self.functions.insert("power".to_string(), func);
            }
            "sub" => {
                self.device
                    .load_ptx(ptx_str.into(), "sub_module", &["elementwise_sub"])
                    .map_err(|e| format!("Failed to load sub kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("sub_module", "elementwise_sub")
                    .ok_or_else(|| "Failed to get sub function".to_string())?;
                self.functions.insert("sub".to_string(), func);
            }
            "clamp" => {
                self.device
                    .load_ptx(ptx_str.into(), "clamp_module", &["clamp_kernel"])
                    .map_err(|e| format!("Failed to load clamp kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("clamp_module", "clamp_kernel")
                    .ok_or_else(|| "Failed to get clamp function".to_string())?;
                self.functions.insert("clamp".to_string(), func);
            }
            "sum_axis" => {
                self.device
                    .load_ptx(ptx_str.into(), "sum_axis_module", &["sum_axis_kernel"])
                    .map_err(|e| format!("Failed to load sum_axis kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("sum_axis_module", "sum_axis_kernel")
                    .ok_or_else(|| "Failed to get sum_axis function".to_string())?;
                self.functions.insert("sum_axis".to_string(), func);
            }
            "negate" => {
                self.device
                    .load_ptx(ptx_str.into(), "negate_module", &["negate_kernel"])
                    .map_err(|e| format!("Failed to load negate kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("negate_module", "negate_kernel")
                    .ok_or_else(|| "Failed to get negate function".to_string())?;
                self.functions.insert("negate".to_string(), func);
            }
            "transpose" => {
                self.device
                    .load_ptx(ptx_str.into(), "transpose_module", &["transpose_2d"])
                    .map_err(|e| format!("Failed to load transpose kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("transpose_module", "transpose_2d")
                    .ok_or_else(|| "Failed to get transpose function".to_string())?;
                self.functions.insert("transpose".to_string(), func);
            }
            "min" => {
                self.device
                    .load_ptx(ptx_str.into(), "min_module", &["element_min"])
                    .map_err(|e| format!("Failed to load min kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("min_module", "element_min")
                    .ok_or_else(|| "Failed to get min function".to_string())?;
                self.functions.insert("min".to_string(), func);
            }
            "max" => {
                self.device
                    .load_ptx(ptx_str.into(), "max_module", &["element_max"])
                    .map_err(|e| format!("Failed to load max kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("max_module", "element_max")
                    .ok_or_else(|| "Failed to get max function".to_string())?;
                self.functions.insert("max".to_string(), func);
            }
            "abs" => {
                self.device
                    .load_ptx(ptx_str.into(), "abs_module", &["element_abs"])
                    .map_err(|e| format!("Failed to load abs kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("abs_module", "element_abs")
                    .ok_or_else(|| "Failed to get abs function".to_string())?;
                self.functions.insert("abs".to_string(), func);
            }
            "sqrt" => {
                // sqrt can use the existing activation kernel loading pattern
                self.device
                    .load_ptx(ptx_str.into(), "sqrt_module", &["element_sqrt"])
                    .map_err(|e| format!("Failed to load sqrt kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("sqrt_module", "element_sqrt")
                    .ok_or_else(|| "Failed to get sqrt function".to_string())?;
                self.functions.insert("sqrt".to_string(), func);
            }
            "max_along_dim" => {
                self.device
                    .load_ptx(ptx_str.into(), "reduction_module", &["max_reduce_along_dim"])
                    .map_err(|e| format!("Failed to load max_along_dim kernel: {}", e))?;
                let func = self
                    .device
                    .get_func("reduction_module", "max_reduce_along_dim")
                    .ok_or_else(|| "Failed to get max_along_dim function".to_string())?;
                self.functions.insert("max_along_dim".to_string(), func);
            }           
            // Comparison kernels
            "greater_equal" => {
                self.device
                    .load_ptx(ptx_str.into(), "comparison_module", &["greater_equal_kernel", "greater_equal_kernel_f64"])
                    .map_err(|e| format!("Failed to load greater_equal kernel: {}", e))?;
                
                let func_f32 = self
                    .device
                    .get_func("comparison_module", "greater_equal_kernel")
                    .ok_or_else(|| "Failed to get greater_equal function".to_string())?;
                self.functions.insert("greater_equal".to_string(), func_f32);
                
                let func_f64 = self
                    .device
                    .get_func("comparison_module", "greater_equal_kernel_f64")
                    .ok_or_else(|| "Failed to get greater_equal_f64 function".to_string())?;
                self.functions.insert("greater_equal_f64".to_string(), func_f64);
            }
            
            "less_equal" => {
                self.device
                    .load_ptx(ptx_str.into(), "comparison_module", &["less_equal_kernel", "less_equal_kernel_f64"])
                    .map_err(|e| format!("Failed to load less_equal kernel: {}", e))?;
                
                let func_f32 = self
                    .device
                    .get_func("comparison_module", "less_equal_kernel")
                    .ok_or_else(|| "Failed to get less_equal function".to_string())?;
                self.functions.insert("less_equal".to_string(), func_f32);
                
                let func_f64 = self
                    .device
                    .get_func("comparison_module", "less_equal_kernel_f64")
                    .ok_or_else(|| "Failed to get less_equal_f64 function".to_string())?;
                self.functions.insert("less_equal_f64".to_string(), func_f64);
            }
            
            "equal" => {
                self.device
                    .load_ptx(ptx_str.into(), "comparison_module", &["equal_kernel", "equal_kernel_f64"])
                    .map_err(|e| format!("Failed to load equal kernel: {}", e))?;
                
                let func_f32 = self
                    .device
                    .get_func("comparison_module", "equal_kernel")
                    .ok_or_else(|| "Failed to get equal function".to_string())?;
                self.functions.insert("equal".to_string(), func_f32);
                
                let func_f64 = self
                    .device
                    .get_func("comparison_module", "equal_kernel_f64")
                    .ok_or_else(|| "Failed to get equal_f64 function".to_string())?;
                self.functions.insert("equal_f64".to_string(), func_f64);
            }
            "logical_not" => {
                self.device
                    .load_ptx(ptx_str.into(), "comparison_module", &["logical_not_kernel", "logical_not_kernel_f64"])
                    .map_err(|e| format!("Failed to load logical_not kernel: {}", e))?;
                
                let func_f32 = self
                    .device
                    .get_func("comparison_module", "logical_not_kernel")
                    .ok_or_else(|| "Failed to get logical_not function".to_string())?;
                self.functions.insert("logical_not".to_string(), func_f32);
                
                let func_f64 = self
                    .device
                    .get_func("comparison_module", "logical_not_kernel_f64")
                    .ok_or_else(|| "Failed to get logical_not_f64 function".to_string())?;
                self.functions.insert("logical_not_f64".to_string(), func_f64);
            }
            
            "in_range" => {
                self.device
                    .load_ptx(ptx_str.into(), "comparison_module", &["in_range_kernel", "in_range_kernel_f64"])
                    .map_err(|e| format!("Failed to load in_range kernel: {}", e))?;
                
                let func_f32 = self
                    .device
                    .get_func("comparison_module", "in_range_kernel")
                    .ok_or_else(|| "Failed to get in_range function".to_string())?;
                self.functions.insert("in_range".to_string(), func_f32);
                
                let func_f64 = self
                    .device
                    .get_func("comparison_module", "in_range_kernel_f64")
                    .ok_or_else(|| "Failed to get in_range_f64 function".to_string())?;
                self.functions.insert("in_range_f64".to_string(), func_f64);
            }
            "sign" => {
                self.device
                    .load_ptx(ptx_str.into(), "sign_module", &["sign_kernel", "sign_kernel_f64"])
                    .map_err(|e| format!("Failed to load sign kernel: {}", e))?;
                
                let func_f32 = self
                    .device
                    .get_func("sign_module", "sign_kernel")
                    .ok_or_else(|| "Failed to get sign function".to_string())?;
                self.functions.insert("sign".to_string(), func_f32);
                
                let func_f64 = self
                    .device
                    .get_func("sign_module", "sign_kernel_f64")
                    .ok_or_else(|| "Failed to get sign_f64 function".to_string())?;
                self.functions.insert("sign_f64".to_string(), func_f64);
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
        T: cudarc::driver::DeviceRepr
            + Clone
            + cudarc::driver::ValidAsZeroBits
            + std::marker::Unpin,
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
        T: cudarc::driver::DeviceRepr
            + Clone
            + cudarc::driver::ValidAsZeroBits
            + std::marker::Unpin,
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
        a: &CudaSlice<T>,
        b: &CudaSlice<T>,
        c: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: cudarc::driver::DeviceRepr
            + Clone
            + cudarc::driver::ValidAsZeroBits
            + std::marker::Unpin,
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
        T: cudarc::driver::DeviceRepr
            + Clone
            + cudarc::driver::ValidAsZeroBits
            + std::marker::Unpin,
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
        T: cudarc::driver::DeviceRepr
            + Clone
            + cudarc::driver::ValidAsZeroBits
            + std::marker::Unpin,
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
        T: cudarc::driver::DeviceRepr
            + Clone
            + cudarc::driver::ValidAsZeroBits
            + std::marker::Unpin,
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

    pub fn launch_power<T>(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<T>,
        b: &CudaSlice<T>,
        c: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: cudarc::driver::DeviceRepr
            + Clone
            + cudarc::driver::ValidAsZeroBits
            + std::marker::Unpin,
    {
        let kernel = self
            .get_function_cloned("power")
            .ok_or_else(|| "Power kernel not found".to_string())?;

        unsafe {
            kernel
                .launch(cfg, (a, b, c, size))
                .map_err(|e| format!("Failed to launch power kernel: {}", e))
        }
    }

    /// Launch element-wise subtraction kernel
    pub fn launch_sub<T>(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<T>,
        b: &CudaSlice<T>,
        c: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let kernel = self
            .get_function_cloned("sub")
            .ok_or_else(|| "Sub kernel not found".to_string())?;

        unsafe {
            kernel
                .launch(cfg, (a, b, c, size))
                .map_err(|e| format!("Failed to launch sub kernel: {}", e))
        }
    }

    /// Launch clamp kernel (clamps values between min and max)
    pub fn launch_clamp<T>(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        min_val: T,
        max_val: T,
        size: i32,
    ) -> Result<(), String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let kernel = self
            .get_function_cloned("clamp")
            .ok_or_else(|| "Clamp kernel not found".to_string())?;

        unsafe {
            kernel
                .launch(cfg, (input, output, min_val, max_val, size))
                .map_err(|e| format!("Failed to launch clamp kernel: {}", e))
        }
    }

    /// Launch sum along axis kernel
    pub fn launch_sum_axis<T>(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        outer_size: i32,
        axis_size: i32,
        inner_size: i32,
    ) -> Result<(), String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let kernel = self
            .get_function_cloned("sum_axis")
            .ok_or_else(|| "Sum axis kernel not found".to_string())?;

        unsafe {
            kernel
                .launch(cfg, (input, output, outer_size, axis_size, inner_size))
                .map_err(|e| format!("Failed to launch sum_axis kernel: {}", e))
        }
    }

    /// Launch negate kernel (unary minus)
    pub fn launch_negate<T>(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let kernel = self
            .get_function_cloned("negate")
            .ok_or_else(|| "Negate kernel not found".to_string())?;

        unsafe {
            kernel
                .launch(cfg, (input, output, size))
                .map_err(|e| format!("Failed to launch negate kernel: {}", e))
        }
    }

    /// Launch 2D transpose kernel
    pub fn launch_transpose_2d<T>(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        rows: i32,
        cols: i32,
    ) -> Result<(), String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let kernel = self
            .get_function_cloned("transpose")
            .ok_or_else(|| "Transpose kernel not found".to_string())?;

        unsafe {
            kernel
                .launch(cfg, (input, output, rows, cols))
                .map_err(|e| format!("Failed to launch transpose kernel: {}", e))
        }
    }

    pub fn launch_min_elementwise<T>(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<T>,
        b: &CudaSlice<T>,
        c: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let kernel = self
            .get_function_cloned("min")
            .ok_or_else(|| "Min kernel not found".to_string())?;

        unsafe {
            kernel
                .launch(cfg, (a, b, c, size))
                .map_err(|e| format!("Failed to launch min kernel: {}", e))
        }
    }

    pub fn launch_max_elementwise<T>(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<T>,
        b: &CudaSlice<T>,
        c: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let kernel = self
            .get_function_cloned("max")
            .ok_or_else(|| "Max kernel not found".to_string())?;

        unsafe {
            kernel
                .launch(cfg, (a, b, c, size))
                .map_err(|e| format!("Failed to launch max kernel: {}", e))
        }
    }

    pub fn launch_abs<T>(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let kernel = self
            .get_function_cloned("abs")
            .ok_or_else(|| "Abs kernel not found".to_string())?;

        unsafe {
            kernel
                .launch(cfg, (input, output, size))
                .map_err(|e| format!("Failed to launch abs kernel: {}", e))
        }
    }

    pub fn launch_sqrt<T>(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let kernel = self
            .get_function_cloned("sqrt")
            .ok_or_else(|| "Sqrt kernel not found".to_string())?;

        unsafe {
            kernel
                .launch(cfg, (input, output, size))
                .map_err(|e| format!("Failed to launch sqrt kernel: {}", e))
        }
    }

    pub fn launch_max_along_dim<T>(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        outer_size: i32,
        axis_size: i32,
        inner_size: i32,
    ) -> Result<(), String>
    where
        T: cudarc::driver::DeviceRepr + Clone + cudarc::driver::ValidAsZeroBits + Unpin,
    {
        let kernel = self
            .get_function_cloned("max_along_dim")
            .ok_or_else(|| "Max along dim kernel not found".to_string())?;
    
        unsafe {
            // Usar la versión simplificada que coincide con el kernel
            kernel
                .launch(cfg, (input, output, outer_size, axis_size, inner_size))
                .map_err(|e| format!("Failed to launch max_along_dim kernel: {}", e))
        }
    }


    // Launch methods for comparison operations
    pub fn launch_greater_equal_f32(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        result: &mut CudaSlice<f32>,
        size: i32,
    ) -> Result<(), String> {
        let func = self.functions.get("greater_equal")
            .ok_or("Greater equal kernel not loaded")?;

        unsafe {
            func.launch_async(cfg, (a, b, result, size))
                .map_err(|e| format!("Failed to launch greater_equal kernel: {}", e))?;
        }
        Ok(())
    }

    pub fn launch_greater_equal_f64(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<f64>,
        b: &CudaSlice<f64>,
        result: &mut CudaSlice<f64>,
        size: i32,
    ) -> Result<(), String> {
        let func = self.functions.get("greater_equal_f64")
            .ok_or("Greater equal f64 kernel not loaded")?;

        unsafe {
            func.launch_async(cfg, (a, b, result, size))
                .map_err(|e| format!("Failed to launch greater_equal_f64 kernel: {}", e))?;
        }
        Ok(())
    }

    pub fn launch_less_equal_f32(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        result: &mut CudaSlice<f32>,
        size: i32,
    ) -> Result<(), String> {
        let func = self.functions.get("less_equal")
            .ok_or("Less equal kernel not loaded")?;

        unsafe {
            func.launch_async(cfg, (a, b, result, size))
                .map_err(|e| format!("Failed to launch less_equal kernel: {}", e))?;
        }
        Ok(())
    }

    pub fn launch_less_equal_f64(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<f64>,
        b: &CudaSlice<f64>,
        result: &mut CudaSlice<f64>,
        size: i32,
    ) -> Result<(), String> {
        let func = self.functions.get("less_equal_f64")
            .ok_or("Less equal f64 kernel not loaded")?;

        unsafe {
            func.launch_async(cfg, (a, b, result, size))
                .map_err(|e| format!("Failed to launch less_equal_f64 kernel: {}", e))?;
        }
        Ok(())
    }

    pub fn launch_equal_f32(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        result: &mut CudaSlice<f32>,
        size: i32,
    ) -> Result<(), String> {
        let func = self.functions.get("equal")
            .ok_or("Equal kernel not loaded")?;

        unsafe {
            func.launch_async(cfg, (a, b, result, size))
                .map_err(|e| format!("Failed to launch equal kernel: {}", e))?;
        }
        Ok(())
    }

    pub fn launch_equal_f64(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<f64>,
        b: &CudaSlice<f64>,
        result: &mut CudaSlice<f64>,
        size: i32,
    ) -> Result<(), String> {
        let func = self.functions.get("equal_f64")
            .ok_or("Equal f64 kernel not loaded")?;

        unsafe {
            func.launch_async(cfg, (a, b, result, size))
                .map_err(|e| format!("Failed to launch equal_f64 kernel: {}", e))?;
        }
        Ok(())
    }

    pub fn launch_logical_not_f32(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<f32>,
        result: &mut CudaSlice<f32>,
        size: i32,
    ) -> Result<(), String> {
        let func = self.functions.get("logical_not")
            .ok_or("Logical not kernel not loaded")?;

        unsafe {
            func.launch_async(cfg, (input, result, size))
                .map_err(|e| format!("Failed to launch logical_not kernel: {}", e))?;
        }
        Ok(())
    }

    pub fn launch_logical_not_f64(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<f64>,
        result: &mut CudaSlice<f64>,
        size: i32,
    ) -> Result<(), String> {
        let func = self.functions.get("logical_not_f64")
            .ok_or("Logical not f64 kernel not loaded")?;

        unsafe {
            func.launch_async(cfg, (input, result, size))
                .map_err(|e| format!("Failed to launch logical_not_f64 kernel: {}", e))?;
        }
        Ok(())
    }

    pub fn launch_in_range_f32(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<f32>,
        min_val: f32,
        max_val: f32,
        result: &mut CudaSlice<f32>,
        size: i32,
    ) -> Result<(), String> {
        let func = self.functions.get("in_range")
            .ok_or("In range kernel not loaded")?;

        unsafe {
            func.launch_async(cfg, (input, min_val, max_val, result, size))
                .map_err(|e| format!("Failed to launch in_range kernel: {}", e))?;
        }
        Ok(())
    }

    pub fn launch_in_range_f64(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<f64>,
        min_val: f64,
        max_val: f64,
        result: &mut CudaSlice<f64>,
        size: i32,
    ) -> Result<(), String> {
        let func = self.functions.get("in_range_f64")
            .ok_or("In range f64 kernel not loaded")?;

        unsafe {
            func.launch_async(cfg, (input, min_val, max_val, result, size))
                .map_err(|e| format!("Failed to launch in_range_f64 kernel: {}", e))?;
        }
        Ok(())
    }

    // SIGN LAUNCHERS
    /// Launches the sign kernel for f32 inputs
    pub fn launch_sign_f32(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<f32>,
        result: &mut CudaSlice<f32>,
        size: i32,
    ) -> Result<(), String> {
        let func = self.functions.get("sign")
            .ok_or("Sign kernel not loaded")?;

        unsafe {
            func.launch_async(cfg, (input, result, size))
                .map_err(|e| format!("Failed to launch sign kernel: {}", e))?;
        }
        Ok(())
    }

    pub fn launch_sign_f64(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<f64>,
        result: &mut CudaSlice<f64>,
        size: i32,
    ) -> Result<(), String> {
        let func = self.functions.get("sign_f64")
            .ok_or("Sign f64 kernel not loaded")?;

        unsafe {
            func.launch_async(cfg, (input, result, size))
                .map_err(|e| format!("Failed to launch sign_f64 kernel: {}", e))?;
        }
        Ok(())
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
        ("log", LOG_PTX),
        ("sigmoid", SIGMOID_PTX),
        ("tanh", TANH_PTX),
        ("power", POWER_PTX),
        ("sub", SUB_PTX),
        ("clamp", CLAMP_PTX),
        ("sum_axis", SUM_AXIS_PTX),
        ("negate", NEGATE_PTX),
        ("transpose", TRANSPOSE_PTX),
        ("comparison", COMPARISON_PTX),
        ("sign", SIGN_PTX),
        ("min", MIN_PTX),
        ("max", MAX_PTX),
        ("abs", ABS_PTX),
        ("sqrt", SQRT_PTX),
        ("max_along_dim", MAX_ALONG_DIM_PTX),
    ];

    for (name, ptx_bytes) in kernel_list.iter() {
        kernels.load_kernel(name, ptx_bytes)?;
    }

    println!("✓ Loaded {} CUDA kernels successfully", kernel_list.len());
    Ok(())
}
