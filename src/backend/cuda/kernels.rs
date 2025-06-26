// src/backend/cuda/kernels.rs
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Embedded PTX kernels
pub const ELEMENTWISE_PTX: &[u8] = include_bytes!("../../../kernels/elementwise.ptx");
pub const ACTIVATIONS_PTX: &[u8] = include_bytes!("../../../kernels/activations.ptx");
pub const MATMUL_PTX: &[u8] = include_bytes!("../../../kernels/matmul.ptx");
pub const REDUCES_PTX: &[u8] = include_bytes!("../../../kernels/reduce.ptx");
pub const TRANSPOSE_PTX: &[u8] = include_bytes!("../../../kernels/transpose.ptx");
pub const COMPARISON_PTX: &[u8] = include_bytes!("../../../kernels/comparison.ptx");

// Generic kernel launch macro
macro_rules! launch_kernel {
    ($self:expr, $kernel_name:expr, $cfg:expr, $params:expr) => {{
        let kernel = $self
            .get_function_cloned($kernel_name)
            .ok_or_else(|| format!("{} kernel not found", $kernel_name))?;

        unsafe {
            kernel
                .launch($cfg, $params)
                .map_err(|e| format!("Failed to launch {} kernel: {}", $kernel_name, e))
        }
    }};
}

/// Kernel configuration for automatic loading
struct KernelConfig {
    name: &'static str,
    ptx: &'static [u8],
    module: &'static str,
    functions: &'static [&'static str],
}

/// Fixed kernel configurations with corrected names
const KERNEL_CONFIGS: &[KernelConfig] = &[
    KernelConfig {
        name: "elementwise",
        ptx: ELEMENTWISE_PTX,
        module: "elementwise_module",
        functions: &[
            // f32 versions
            "elementwise_add",
            "elementwise_sqrt",
            "elementwise_abs",
            "elementwise_mul",
            "elementwise_div",
            "elementwise_sub",
            "elementwise_pow",
            "elementwise_min",
            "elementwise_max",
            "elementwise_exp",
            "elementwise_log",
            "elementwise_negate",
            // f64 versions
            "elementwise_add_f64",
            "elementwise_sqrt_f64",
            "elementwise_abs_f64",
            "elementwise_mul_f64",
            "elementwise_div_f64",
            "elementwise_sub_f64",
            "elementwise_pow_f64",
            "elementwise_min_f64",
            "elementwise_max_f64",
            "elementwise_exp_f64",
            "elementwise_log_f64",
            "elementwise_negate_f64",
        ],
    },
    KernelConfig {
        name: "matmul",
        ptx: MATMUL_PTX,
        module: "matmul_module",
        functions: &["matmul", "matmul_f64"],
    },
    KernelConfig {
        name: "activations",
        ptx: ACTIVATIONS_PTX,
        module: "activations_module",
        functions: &[
            "relu",
            "sigmoid",
            "hyperbolic_tangent",
            "relu_f64",
            "sigmoid_f64",
            "hyperbolic_tangent_f64",
        ],
    },
    KernelConfig {
        name: "reduces",
        ptx: REDUCES_PTX,
        module: "reduces_module",
        functions: &[
            "sum_axis",
            "max_along_dim",
            "sum_axis_f64",
            "max_along_dim_f64",
        ],
    },
    KernelConfig {
        name: "transpose",
        ptx: TRANSPOSE_PTX,
        module: "transpose_module",
        functions: &["transpose_2d", "transpose_2d_f64"],
    },
    KernelConfig {
        name: "comparison",
        ptx: COMPARISON_PTX,
        module: "comparison_module",
        functions: &[
            "greater_equal",
            "greater_equal_f64",
            "less_equal",
            "less_equal_f64",
            "equal",
            "equal_f64",
            "logical_not",
            "logical_not_f64",
            "in_range",
            "in_range_f64",
            "sign",
            "sign_f64",
            "clamp",
            "clamp_f64",
        ],
    },
];

/// CUDA kernel manager
pub struct CudaKernels {
    device: Arc<CudaContext>,
    functions: HashMap<String, CudaFunction>,
}

impl CudaKernels {
    pub fn new(device: Arc<CudaContext>) -> Self {
        Self {
            device,
            functions: HashMap::new(),
        }
    }

    pub fn load_kernel(&mut self, name: &str, ptx_bytes: &[u8]) -> Result<(), String> {
        let config = KERNEL_CONFIGS
            .iter()
            .find(|k| k.name == name)
            .ok_or_else(|| format!("Unknown kernel: {}", name))?;

        let ptx_str =
            std::str::from_utf8(ptx_bytes).map_err(|e| format!("Invalid PTX UTF-8: {}", e))?;

        let module = self.device
            .load_module(ptx_str.into())
            .map_err(|e| format!("Failed to load {} kernel: {}", name, e))?;

        // Store the function on the stack
        for &func_name in config.functions {
            if let Some(func) = module.load_func(func_name) {
                self.functions.insert(func_name.to_string(), func);
            }
        }

        Ok(())
    }

    pub fn get_function_cloned(&self, name: &str) -> Option<CudaFunction> {
        self.functions.get(name).cloned()
    }

    pub fn get_function(&self, name: &str) -> Option<&CudaFunction> {
        self.functions.get(name)
    }

    // ===== HELPER METHODS =====

    /// Determines the correct CUDA kernel name based on the Rust type at compile time
    /// This is the core of my type-safe kernel dispatch system
    /// We use Rust's TypeId system to check the concrete type T at compile time
    /// - For f64 types: appends "_f64" suffix to match our CUDA kernel naming convention
    /// - For all other types (primarily f32): uses the base name directly
    /// This allows us to:
    /// - Automatically select the correct precision kernel based on T with no runtime overhead (type checking is done at compile time)
    /// - Type safety - impossible to accidentally call wrong precision kernel
    /// - Consistent naming convention across all kernels allows for maintainability and readability.
    fn get_kernel_name<T>(&self, base_name: &str) -> String
    where
        T: 'static,
    {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            format!("{}_f64", base_name)
        } else {
            base_name.to_string()
        }
    }

    /// Generic launcher for binary elementwise operations (a ○ b = result)
    /// This handles all operations where two tensors of equal shape produce one result tensor
    /// The 'static bound ensures T's type information is available at compile time
    /// for our kernel name resolution system
    fn launch_binary_elementwise<T>(
        &self,
        kernel_base: &str,
        cfg: LaunchConfig,
        a: &CudaSlice<T>,
        b: &CudaSlice<T>,
        result: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let kernel_name = self.get_kernel_name::<T>(kernel_base);
        launch_kernel!(self, &kernel_name, cfg, (a, b, result, size))
    }

    /// Generic launcher for unary elementwise operations (f(input) = output)
    /// Essential for activation functions and mathematical transformations
    fn launch_unary_elementwise<T>(
        &self,
        kernel_base: &str,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let kernel_name = self.get_kernel_name::<T>(kernel_base);
        launch_kernel!(self, &kernel_name, cfg, (input, output, size))
    }

    /// Generic launcher for reduction operations along tensor dimensions
    /// These are critical for implementing neural network operations efficiently
    fn launch_reduction<T>(
        &self,
        kernel_base: &str,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        outer_size: i32,
        axis_size: i32,
        inner_size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let kernel_name = self.get_kernel_name::<T>(kernel_base);
        launch_kernel!(
            self,
            &kernel_name,
            cfg,
            (input, output, outer_size, axis_size, inner_size)
        )
    }

    // ===== PUBLIC API - BINARY ELEMENTWISE OPERATIONS =====
    // These operations form the backbone of neural network computation
    // All functions automatically dispatch to the correct precision kernel (f32/f64)

    /// Element-wise tensor addition: result[i] = a[i] + b[i]
    pub fn launch_add<T>(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<T>,
        b: &CudaSlice<T>,
        c: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_binary_elementwise("elementwise_add", cfg, a, b, c, size)
    }

    /// Element-wise tensor multiplication: result[i] = a[i] * b[i]
    pub fn launch_mul<T>(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<T>,
        b: &CudaSlice<T>,
        c: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_binary_elementwise("elementwise_mul", cfg, a, b, c, size)
    }

    /// Element-wise tensor division: result[i] = a[i] / b[i]
    pub fn launch_div<T>(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<T>,
        b: &CudaSlice<T>,
        c: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_binary_elementwise("elementwise_div", cfg, a, b, c, size)
    }

    /// Element-wise tensor subtraction: result[i] = a[i] - b[i]
    pub fn launch_sub<T>(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<T>,
        b: &CudaSlice<T>,
        c: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_binary_elementwise("elementwise_sub", cfg, a, b, c, size)
    }

    /// Element-wise power operation: result[i] = a[i] ^ b[i]
    pub fn launch_power<T>(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<T>,
        b: &CudaSlice<T>,
        c: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_binary_elementwise("elementwise_pow", cfg, a, b, c, size)
    }

    /// Element-wise minimum: result[i] = min(a[i], b[i])
    pub fn launch_min_elementwise<T>(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<T>,
        b: &CudaSlice<T>,
        c: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_binary_elementwise("elementwise_min", cfg, a, b, c, size)
    }

    /// Element-wise maximum: result[i] = max(a[i], b[i])  
    pub fn launch_max_elementwise<T>(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<T>,
        b: &CudaSlice<T>,
        c: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_binary_elementwise("elementwise_max", cfg, a, b, c, size)
    }

    // Unary elementwise operations
    pub fn launch_abs<T>(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_unary_elementwise("elementwise_abs", cfg, input, output, size)
    }

    pub fn launch_sqrt<T>(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_unary_elementwise("elementwise_sqrt", cfg, input, output, size)
    }

    pub fn launch_exp<T>(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_unary_elementwise("elementwise_exp", cfg, input, output, size)
    }

    pub fn launch_log<T>(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_unary_elementwise("elementwise_log", cfg, input, output, size)
    }

    pub fn launch_negate<T>(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_unary_elementwise("elementwise_negate", cfg, input, output, size)
    }

    pub fn launch_relu<T>(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_unary_elementwise("relu", cfg, input, output, size)
    }

    // Reduction operations
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
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_reduction(
            "sum_axis", cfg, input, output, outer_size, axis_size, inner_size,
        )
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
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_reduction(
            "max_along_dim",
            cfg,
            input,
            output,
            outer_size,
            axis_size,
            inner_size,
        )
    }

    // Special cases that need custom parameters
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
        T: crate::backend::number::GPUNumber + 'static,
    {
        let kernel_name = self.get_kernel_name::<T>("clamp");
        launch_kernel!(
            self,
            &kernel_name,
            cfg,
            (input, output, min_val, max_val, size)
        )
    }

    pub fn launch_matmul<T>(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<T>,
        b: &CudaSlice<T>,
        c: &mut CudaSlice<T>,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let kernel_name = self.get_kernel_name::<T>("matmul");
        launch_kernel!(self, &kernel_name, cfg, (a, b, c, m, n, k))
    }

    pub fn launch_transpose_2d<T>(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        rows: i32,
        cols: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let kernel_name = self.get_kernel_name::<T>("transpose_2d");
        launch_kernel!(self, &kernel_name, cfg, (input, output, rows, cols))
    }

    // ===== COMPARISON OPERATIONS =====

    /// Generic binary comparison launcher - automatically dispatches to f32/f64 kernels
    /// This is the core pattern for all element-wise comparison operations in our ML framework
    /// The kernel selection happens at compile time based on the tensor's data type
    fn launch_binary_comparison<T>(
        &self,
        kernel_base: &str,
        cfg: LaunchConfig,
        a: &CudaSlice<T>,
        b: &CudaSlice<T>,
        result: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let kernel_name = self.get_kernel_name::<T>(kernel_base);
        launch_kernel!(self, &kernel_name, cfg, (a, b, result, size))
    }

    /// Generic unary comparison launcher for operations like logical_not, sign
    /// Essential for activation function derivatives and boolean tensor operations
    fn launch_unary_comparison<T>(
        &self,
        kernel_base: &str,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        result: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let kernel_name = self.get_kernel_name::<T>(kernel_base);
        launch_kernel!(self, &kernel_name, cfg, (input, result, size))
    }

    /// Element-wise greater-than-or-equal comparison: result[i] = (a[i] >= b[i]) ? 1.0 : 0.0
    /// Critical for gradient clipping and ReLU derivative computation
    /// Returns 1.0 for true, 0.0 for false - standard convention for boolean tensors in ML
    pub fn launch_greater_equal<T>(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<T>,
        b: &CudaSlice<T>,
        result: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_binary_comparison("greater_equal", cfg, a, b, result, size)
    }

    /// Element-wise less-than-or-equal comparison: result[i] = (a[i] <= b[i]) ? 1.0 : 0.0
    /// Used in loss function computations and threshold-based operations
    pub fn launch_less_equal<T>(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<T>,
        b: &CudaSlice<T>,
        result: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_binary_comparison("less_equal", cfg, a, b, result, size)
    }

    /// Element-wise equality comparison: result[i] = (a[i] == b[i]) ? 1.0 : 0.0
    /// Essential for mask generation and accuracy computation in classification tasks
    /// Note: For floating point, this is exact equality - use with caution or implement tolerance-based version
    pub fn launch_equal<T>(
        &self,
        cfg: LaunchConfig,
        a: &CudaSlice<T>,
        b: &CudaSlice<T>,
        result: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_binary_comparison("equal", cfg, a, b, result, size)
    }

    /// Logical NOT operation: result[i] = (input[i] == 0.0) ? 1.0 : 0.0
    /// Inverts boolean tensors - critical for implementing complex boolean logic in loss functions
    /// Follows IEEE 754 convention: 0.0 is false, any non-zero value is true
    pub fn launch_logical_not<T>(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        result: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_unary_comparison("logical_not", cfg, input, result, size)
    }

    /// Sign function: result[i] = sign(input[i]) where sign(x) = {-1 if x<0, 0 if x=0, 1 if x>0}
    /// Critical for implementing certain activation functions and gradient sign-based optimizers (like SignSGD)
    /// This is mathematically different from logical operations - preserves the sign information
    pub fn launch_sign<T>(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        result: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        self.launch_unary_comparison("sign", cfg, input, result, size)
    }

    /// Range check operation: result[i] = (min_val <= input[i] <= max_val) ? 1.0 : 0.0
    /// Essential for gradient clipping bounds checking and implementing custom activation functions
    /// This is a compound operation that's more efficient than separate comparisons
    /// Template parameters allow this to work with both f32 and f64 automatically
    pub fn launch_in_range<T>(
        &self,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        min_val: T,
        max_val: T,
        result: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber + 'static,
    {
        let kernel_name = self.get_kernel_name::<T>("in_range");
        launch_kernel!(
            self,
            &kernel_name,
            cfg,
            (input, min_val, max_val, result, size)
        )
    }

    // Keep existing unified activation launcher
    pub fn launch_activation<T>(
        &self,
        kernel_name: &str,
        cfg: LaunchConfig,
        input: &CudaSlice<T>,
        output: &mut CudaSlice<T>,
        size: i32,
    ) -> Result<(), String>
    where
        T: crate::backend::number::GPUNumber,
    {
        launch_kernel!(self, kernel_name, cfg, (input, output, size))
    }

    pub fn device(&self) -> &Arc<CudaContext> {
        &self.device
    }

    pub fn loaded_kernels(&self) -> Vec<&String> {
        self.functions.keys().collect()
    }
}

/// Load all predefined CUDA kernels
pub fn load_all_kernels(kernels: &mut CudaKernels) -> Result<(), String> {
    for config in KERNEL_CONFIGS {
        kernels.load_kernel(config.name, config.ptx)?;
    }

    println!("Loaded {} CUDA kernels successfully", KERNEL_CONFIGS.len());
    Ok(())
}
