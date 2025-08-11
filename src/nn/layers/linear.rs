// src/nn/layers/linear.rs
// Linear layer implementation using the computational graph engine
// This is the fundamental building block for feedforward neural networks

use crate::backend::number::FerroxCudaF;
use crate::backend::{Device, Tensor};
use crate::graph::{AutoFerroxEngine, NodeId};
use crate::nn::parameter::Parameter;
use crate::nn::Module;
use crate::ops::Transpose;
use crate::ops::{Add, BroadcastTo, MatMul};

/// Linear transformation layer: y = x * W^T + b
/// This implements the basic feedforward layer found in most neural networks
/// Weight matrix is stored as [out_features, in_features] to match PyTorch convention
#[derive(Debug)]
pub struct Linear<T>
where
    T: FerroxCudaF,
{
    /// Weight matrix [out_features, in_features] - transposed for efficiency
    pub weight: Parameter<T>,
    /// Optional bias vector [out_features]
    pub bias: Option<Parameter<T>>,
    /// Number of input features
    pub in_features: usize,
    /// Number of output features
    pub out_features: usize,
    /// Training mode flag
    training: bool,
    parameter_maps: std::cell::RefCell<Option<std::collections::HashMap<String, NodeId>>>,
}

impl<T> Linear<T>
where
    T: FerroxCudaF,
{
    /// Create a new linear layer with specified input and output features
    /// Initializes weights with Xavier uniform distribution and bias with zeros
    pub fn new(in_features: usize, out_features: usize, bias: bool, device: Device) -> Self {
        Self::new_with_device(in_features, out_features, bias, device)
    }

    /// Create a new linear layer with specified device
    pub fn new_with_device(
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: Device,
    ) -> Self {
        // Weight matrix shape: [out_features, in_features] for efficient computation
        // This allows input @ weight.T + bias without needing transpose in forward pass
        let mut weight =
            Parameter::xavier_uniform_with_device(&[out_features, in_features], device);

        weight.set_name("weight".to_string());

        // Initialize bias to zeros if requested
        let bias_param = if bias {
            let mut b = Parameter::zeros_with_device(&[out_features], device);
            b.set_name("bias".to_string());
            Some(b)
        } else {
            None
        };

        Self {
            weight,
            bias: bias_param,
            in_features,
            out_features,
            training: true,
            parameter_maps: std::cell::RefCell::new(None),
        }
    }

    /// Create linear layer with custom weight and bias tensors
    pub fn from_tensors(
        weight: Tensor<T>,
        bias: Option<Tensor<T>>,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self, String> {
        // Validate weight shape
        if weight.shape() != [out_features, in_features] {
            return Err(format!(
                "Weight shape {:?} doesn't match expected [out_features={}, in_features={}]",
                weight.shape(),
                out_features,
                in_features
            ));
        }

        // Validate bias shape if provided
        if let Some(ref bias_tensor) = bias {
            if bias_tensor.shape() != [out_features] {
                return Err(format!(
                    "Bias shape {:?} doesn't match expected [out_features={}]",
                    bias_tensor.shape(),
                    out_features
                ));
            }
        }

        let mut weight_param = Parameter::new(weight);
        weight_param.set_name("weight".to_string());
        let mut bias_param = bias.map(|b| Parameter::new(b));
        if let Some(ref mut bias) = bias_param {
            bias.set_name("bias".to_string());
        }
        Ok(Self {
            weight: weight_param,
            bias: bias_param,
            in_features,
            out_features,
            training: true,
             parameter_maps: std::cell::RefCell::new(None),
        })
    }

    /// Get weight tensor reference
    pub fn weight_tensor(&self) -> &Tensor<T> {
        &self.weight.data
    }

    /// Get bias tensor reference (if exists)
    pub fn bias_tensor(&self) -> Option<&Tensor<T>> {
        self.bias.as_ref().map(|b| &b.data)
    }

    /// Check if layer has bias
    pub fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

       // Helper method to get parameter node IDs (cached from create_parameters_in_graph)
    fn get_parameter_node(&self, param_name: &str) -> Result<NodeId, String> {
        let nodes_ref = self.parameter_maps.borrow();
        match nodes_ref.as_ref() {
            Some(node_map) => {
                node_map.get(param_name)
                    .copied()
                    .ok_or_else(|| format!("Parameter '{}' not found in node mapping", param_name))
            }
            None => {
                Err("Parameters not yet created in graph. Call create_parameters_in_graph() first.".to_string())
            }
        }
    }
}

impl<T> Module<T> for Linear<T>
where
    T: FerroxCudaF + rand_distr::num_traits::FromPrimitive,
{


    fn create_parameters_in_graph(
        &self,
        engine: &mut AutoFerroxEngine<T>,
    ) -> std::collections::HashMap<String, NodeId> {
        let mut param_map = std::collections::HashMap::new();
        println!("Initializing parameters in graph!");
        // Create weight node
        let weight_node = engine.create_variable(self.weight.data.clone(), self.weight.requires_grad);
        param_map.insert("weight".to_string(), weight_node);

        // Create bias node if present
        if let Some(ref bias_param) = self.bias {
            let bias_node = engine.create_variable(bias_param.data.clone(), bias_param.requires_grad);
            param_map.insert("bias".to_string(), bias_node);
        }

        // Store the node mappings for use in forward()
        *self.parameter_maps.borrow_mut() = Some(param_map.clone());

        param_map
    }

    /// Forward pass: y = x @ W^T + b
    /// Input shape: [batch_size, in_features] or [in_features]
    /// Output shape: [batch_size, out_features] or [out_features]
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        // Get input tensor to validate dimensions
        let input_tensor = graph
            .get_tensor(input)
            .ok_or("Input tensor not found in graph")?;
;    // Validate input dimensions
        let input_shape = input_tensor.shape();
        let expected_last_dim = self.in_features;

        // Support both 1D [in_features] and 2D [batch_size, in_features] inputs
        match input_shape.len() {
            1 => {
                if input_shape[0] != expected_last_dim {
                    return Err(format!(
                        "Input shape {:?} doesn't match expected last dimension {}",
                        input_shape, expected_last_dim
                    ));
                }
            }
            2 => {
                if input_shape[1] != expected_last_dim {
                    return Err(format!(
                        "Input shape {:?} doesn't match expected last dimension {}",
                        input_shape, expected_last_dim
                    ));
                }
            }
            _ => {
                return Err(format!(
                    "Linear layer only supports 1D or 2D inputs, got shape {:?}",
                    input_shape
                ));
            }
        }

        // Create weight node in computational graph

        let weight_node = self.get_parameter_node("weight")?;

        // Apply linear transformation: input @ weight^T
        // Since our weight is stored as [out_features, in_features], we need to transpose it
        // Use Transpose operation through the computational graph for proper gradient flow
        let transpose_op = Box::new(Transpose::new());
        let weight_t_node = graph
            .apply_operation(transpose_op, vec![weight_node])
            .map_err(|e| format!("Weight transpose failed: {}", e))?;

        // Perform matrix multiplication: output = input @ weight^T
        let matmul_op = Box::new(MatMul);
        let linear_result = graph
            .apply_operation(matmul_op, vec![input, weight_t_node])
            .map_err(|e| format!("Linear transformation failed: {}", e))?;

        // Add bias if present
        if let Some(ref bias_param) = self.bias {

            // Extract shape information first to avoid borrow checker issues
            let result_shape = {
                let linear_result_tensor = graph
                    .get_tensor(linear_result)
                    .ok_or("Linear result tensor not found in graph")?;
                linear_result_tensor.shape().to_vec() // Copy shape to owned Vec
            };


            let bias_shape = bias_param.data.shape();

            // Create bias node in computational graph
            let bias_node = self.get_parameter_node("bias")?;



            // Check if we need to broadcast bias to match result shape
            if bias_shape != result_shape.as_slice() {
                // Broadcast bias to match the result shape
                // bias shape [out_features] -> [batch_size, out_features] (or [out_features])
                let broadcast_op = Box::new(BroadcastTo::new(result_shape));
                let broadcasted_bias = graph
                    .apply_operation(broadcast_op, vec![bias_node])
                    .map_err(|e| format!("Bias broadcasting failed: {}", e))?;

                // Now perform element-wise addition with matching shapes
                let add_op = Box::new(Add::new());
                let final_result = graph
                    .apply_operation(add_op, vec![linear_result, broadcasted_bias])
                    .map_err(|e| format!("Bias addition failed: {}", e))?;

                Ok(final_result)
            } else {
                // Shapes already match, direct addition
                let add_op = Box::new(Add::new());
                let final_result = graph
                    .apply_operation(add_op, vec![linear_result, bias_node])
                    .map_err(|e| format!("Bias addition failed: {}", e))?;



                Ok(final_result)
            }
        } else {
            Ok(linear_result)
        }
    }

    /// Collect all parameters (weight and bias)
    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    /// Collect mutable parameter references
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }

    /// Training mode getter
    fn training(&self) -> bool {
        self.training
    }

    /// Set training mode
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}



// Tests unitarios y de integración para validar consistencia exacta entre CPU y GPU
// en la capa Linear - forward pass, backward pass y actualización de parámetros
#[cfg(all(test, feature = "cuda"))]
mod linear_layer_tests {
    use crate::backend::{Device, Tensor};
    use crate::graph::AutoFerroxEngine;
    use crate::nn::layers::Linear;
    use crate::nn::losses::MSELoss;
    use crate::nn::optim::SGD;
    use crate::nn::Module;
    use crate::nn::losses::{Loss, ReductionType};
    use crate::nn::optim::Optimizer;

    const TOLERANCE: f32 = 1e-6; // Precision tolerance for CPU-GPU comparison

    /// Helper to create identical test data on both devices
    fn create_test_input(device: Device) -> Tensor<f32> {
        // Fixed seed data for reproducible tests
        let data = vec![
            1.0, -0.5, 2.0,
            0.0, 1.5, -1.0,
            -2.0, 0.5, 1.0,
            3.0, -1.5, 0.0
        ];
        Tensor::from_vec_with_device(data, &[4, 3], device)
            .expect("Failed to create test input")
    }

    /// Helper to create identical target data on both devices
    fn create_test_target(device: Device) -> Tensor<f32> {
        let data = vec![1.0, 0.0, -1.0, 2.0];
        Tensor::from_vec_with_device(data, &[4, 1], device)
            .expect("Failed to create test target")
    }

    /// Helper to create identical Linear layer on both devices with same weights
    fn create_identical_linear_layers() -> (Linear<f32>, Linear<f32>) {
        // Create weight and bias data with fixed values for consistency
        let weight_data = vec![
            0.1, -0.2, 0.3,  // First output neuron weights
        ];
        let bias_data = vec![0.1]; // Single output bias

        let weight_cpu = Tensor::from_vec_with_device(weight_data.clone(), &[1, 3], Device::CPU)
            .expect("Failed to create CPU weight");
        let bias_cpu = Tensor::from_vec_with_device(bias_data.clone(), &[1], Device::CPU)
            .expect("Failed to create CPU bias");

        let weight_gpu = Tensor::from_vec_with_device(weight_data, &[1, 3], Device::CUDA(0))
            .expect("Failed to create GPU weight");
        let bias_gpu = Tensor::from_vec_with_device(bias_data, &[1], Device::CUDA(0))
            .expect("Failed to create GPU bias");

        let linear_cpu = Linear::from_tensors(weight_cpu, Some(bias_cpu), 3, 1)
            .expect("Failed to create CPU linear layer");
        let linear_gpu = Linear::from_tensors(weight_gpu, Some(bias_gpu), 3, 1)
            .expect("Failed to create GPU linear layer");

        (linear_cpu, linear_gpu)
    }

    #[test]
    fn test_linear_forward_cpu_gpu_consistency() {
        // Skip if CUDA not available
        if !crate::backend::manager::get_backend::<f32>().has_cuda() {
            println!("Skipping GPU test: CUDA not available");
            return;
        }

        let (linear_cpu, linear_gpu) = create_identical_linear_layers();

        let input_cpu = create_test_input(Device::CPU);
        let input_gpu = create_test_input(Device::CUDA(0));

        let mut engine_cpu = AutoFerroxEngine::new();
        let mut engine_gpu = AutoFerroxEngine::new();

        // Forward pass on CPU
        println!("Debugging forward pass on cpu");
        let input_node_cpu = engine_cpu.create_variable(input_cpu, false);

        let output_node_cpu = linear_cpu.forward(&mut engine_cpu, input_node_cpu)
            .expect("CPU forward pass failed");
        let output_cpu = engine_cpu.get_tensor(output_node_cpu)
            .expect("CPU output not found");

        // Forward pass on GPU
        println!("Debugging forward pass on gpu");
        let input_node_gpu = engine_gpu.create_variable(input_gpu, false);
        let output_node_gpu = linear_gpu.forward(&mut engine_gpu, input_node_gpu)
            .expect("GPU forward pass failed");
        let output_gpu = engine_gpu.get_tensor(output_node_gpu)
            .expect("GPU output not found");

        // Convert GPU output to CPU for comparison
        let output_gpu_on_cpu = output_gpu.clone().to_device(Device::CPU)
            .expect("Failed to move GPU output to CPU");

        // Synchronize GPU operations before comparison
        #[cfg(feature = "cuda")]
        {
            use crate::backend::manager::get_backend;
            let backend = get_backend::<f32>();
            if let Some(cuda_backend) = backend.cuda_backend() {
                cuda_backend.synchronize().expect("GPU sync failed");
            }
        }

        // Verify shapes match
        assert_eq!(output_cpu.shape(), output_gpu_on_cpu.shape(),
            "Output shapes don't match: CPU {:?} vs GPU {:?}",
            output_cpu.shape(), output_gpu_on_cpu.shape());

        // Verify values match exactly within tolerance
        let cpu_data = output_cpu.as_slice().expect("Failed to get CPU data");
        let gpu_data = output_gpu_on_cpu.as_slice().expect("Failed to get GPU data");

        assert_eq!(cpu_data.len(), gpu_data.len(), "Data length mismatch");

        for (i, (&cpu_val, &gpu_val)) in cpu_data.iter().zip(gpu_data.iter()).enumerate() {
            let diff = (cpu_val - gpu_val).abs();
            assert!(diff < TOLERANCE,
                "Forward pass value mismatch at index {}: CPU = {}, GPU = {}, diff = {}",
                i, cpu_val, gpu_val, diff);
        }

        println!("✓ Linear forward pass CPU-GPU consistency test passed");
    }



    #[test]
    fn test_linear_backward_cpu_gpu_consistency() {
        if !crate::backend::manager::get_backend::<f32>().has_cuda() {
            println!("Skipping GPU test: CUDA not available");
            return;
        }

        let (linear_cpu, linear_gpu) = create_identical_linear_layers();
        println!("===DEBUGGING CPU===");
        let input_cpu = create_test_input(Device::CPU);
        let input_gpu = create_test_input(Device::CUDA(0));
        let target_cpu = create_test_target(Device::CPU);
        let target_gpu = create_test_target(Device::CUDA(0));

        let mut engine_cpu = AutoFerroxEngine::new();
        let mut engine_gpu = AutoFerroxEngine::new();

        let loss_fn = MSELoss::new(ReductionType::Mean);

        // CPU backward pass
        let input_node_cpu = engine_cpu.create_variable(input_cpu, true);
        let target_node_cpu = engine_cpu.create_variable(target_cpu, false);

        // Create parameter nodes for gradient tracking
        let param_map_cpu = linear_cpu.create_parameters_in_graph(&mut engine_cpu);

        let output_node_cpu = linear_cpu.forward(&mut engine_cpu, input_node_cpu)
            .expect("CPU forward failed");
        let loss_node_cpu = loss_fn.forward(&mut engine_cpu, output_node_cpu, target_node_cpu)
            .expect("CPU loss computation failed");

        let loss_tensor = engine_cpu.get_tensor(loss_node_cpu).unwrap();
        println!("Loss cpu {:?}", loss_tensor.clone().into_data().unwrap().as_slice().unwrap());

        engine_cpu.backward(loss_node_cpu).expect("CPU backward failed");
        println!("===DEBUGGING GPU===");
        // GPU backward pass
        let input_node_gpu = engine_gpu.create_variable(input_gpu, true);
        let target_node_gpu = engine_gpu.create_variable(target_gpu, false);

        let param_map_gpu = linear_gpu.create_parameters_in_graph(&mut engine_gpu);

        let output_node_gpu = linear_gpu.forward(&mut engine_gpu, input_node_gpu)
            .expect("GPU forward failed");
        let loss_node_gpu = loss_fn.forward(&mut engine_gpu, output_node_gpu, target_node_gpu)
            .expect("GPU loss computation failed");

        let loss_tensor = engine_gpu.get_tensor(loss_node_gpu).unwrap();
        println!("Loss gpu {:?}", loss_tensor.clone().into_data().unwrap().as_slice().unwrap());

        engine_gpu.backward(loss_node_gpu).expect("GPU backward failed");

        // Synchronize GPU before comparison
        #[cfg(feature = "cuda")]
        {
            use crate::backend::manager::get_backend;
            let backend = get_backend::<f32>();
            if let Some(cuda_backend) = backend.cuda_backend() {
                cuda_backend.synchronize().expect("GPU sync failed");
            }
        }

        // Compare gradients for weight and bias
        for (param_name, &cpu_node) in &param_map_cpu {
            let gpu_node = param_map_gpu.get(param_name)
                .expect(&format!("GPU parameter {} not found", param_name));

            let cpu_grad = engine_cpu.get_gradient(cpu_node)
                .expect(&format!("CPU gradient for {} not found", param_name));
            let gpu_grad = engine_gpu.get_gradient(*gpu_node)
                .expect(&format!("GPU gradient for {} not found", param_name));

            // Move GPU gradient to CPU for comparison
            let gpu_grad_on_cpu = gpu_grad.clone().to_device(Device::CPU)
                .expect("Failed to move GPU gradient to CPU");

            assert_eq!(cpu_grad.shape(), gpu_grad_on_cpu.shape(),
                "Gradient shapes don't match for {}: CPU {:?} vs GPU {:?}",
                param_name, cpu_grad.shape(), gpu_grad_on_cpu.shape());

            let cpu_grad_data = cpu_grad.as_slice().expect("Failed to get CPU gradient data");
            let gpu_grad_data = gpu_grad_on_cpu.as_slice().expect("Failed to get GPU gradient data");

            for (i, (&cpu_val, &gpu_val)) in cpu_grad_data.iter().zip(gpu_grad_data.iter()).enumerate() {
                let diff = (cpu_val - gpu_val).abs();
                assert!(diff < TOLERANCE,
                    "Gradient mismatch for {} at index {}: CPU = {}, GPU = {}, diff = {}",
                    param_name, i, cpu_val, gpu_val, diff);
            }
        }

        println!("✓ Linear backward pass CPU-GPU consistency test passed");
    }

    #[test]
    fn test_linear_parameter_updates_cpu_gpu_consistency() {
        if !crate::backend::manager::get_backend::<f32>().has_cuda() {
            println!("Skipping GPU test: CUDA not available");
            return;
        }

        let (linear_cpu, linear_gpu) = create_identical_linear_layers();

        let mut engine_cpu = AutoFerroxEngine::new();
        let mut engine_gpu = AutoFerroxEngine::new();

        // Create identical optimizers
        let mut optimizer_cpu = SGD::with_defaults(0.1);
        let mut optimizer_gpu = SGD::with_defaults(0.1);

        // Register parameters with optimizers
        let param_map_cpu = linear_cpu.create_parameters_in_graph(&mut engine_cpu);
        let param_map_gpu = linear_gpu.create_parameters_in_graph(&mut engine_gpu);

        for &node_id in param_map_cpu.values() {
            optimizer_cpu.add_param(0, node_id);
        }
        for &node_id in param_map_gpu.values() {
            optimizer_gpu.add_param(0, node_id);
        }

        // Simulate identical gradients
        for (param_name, &cpu_node) in &param_map_cpu {
            let gpu_node = param_map_gpu[param_name];

            // Create identical gradient tensors
            let grad_data = if param_name == "weight" {
                vec![0.01, -0.02, 0.03] // 3 gradient values for weight
            } else {
                vec![0.05] // 1 gradient value for bias
            };

            let grad_shape = if param_name == "weight" { vec![1, 3] } else { vec![1] };

            let grad_cpu = Tensor::from_vec_with_device(grad_data.clone(), &grad_shape, Device::CPU)
                .expect("Failed to create CPU gradient");
            let grad_gpu = Tensor::from_vec_with_device(grad_data, &grad_shape, Device::CUDA(0))
                .expect("Failed to create GPU gradient");

            engine_cpu.set_gradient(cpu_node, grad_cpu);
            engine_gpu.set_gradient(gpu_node, grad_gpu);
        }

        // Store original parameter values for comparison
        let mut original_cpu_params = std::collections::HashMap::new();
        let mut original_gpu_params = std::collections::HashMap::new();

        for (param_name, &cpu_node) in &param_map_cpu {
            let gpu_node = param_map_gpu[param_name];

            let cpu_param = engine_cpu.get_tensor_data(cpu_node).unwrap().clone();
            let gpu_param = engine_gpu.get_tensor_data(gpu_node).unwrap().clone();

            original_cpu_params.insert(param_name, cpu_param);
            original_gpu_params.insert(param_name, gpu_param);
        }

        // Apply optimizer step
        optimizer_cpu.step(&mut engine_cpu).expect("CPU optimizer step failed");
        optimizer_gpu.step(&mut engine_gpu).expect("GPU optimizer step failed");

        // Synchronize GPU
        #[cfg(feature = "cuda")]
        {
            use crate::backend::manager::get_backend;
            let backend = get_backend::<f32>();
            if let Some(cuda_backend) = backend.cuda_backend() {
                cuda_backend.synchronize().expect("GPU sync failed");
            }
        }

        // Compare parameter updates
        for (param_name, &cpu_node) in &param_map_cpu {
            let gpu_node = param_map_gpu[param_name];

            let updated_cpu_param = engine_cpu.get_tensor_data(cpu_node).unwrap();
            let updated_gpu_param = engine_gpu.get_tensor_data(gpu_node).unwrap();

            // Move GPU parameter to CPU for comparison
            let updated_gpu_param_on_cpu = updated_gpu_param.clone().to_device(Device::CPU)
                .expect("Failed to move GPU parameter to CPU");

            assert_eq!(updated_cpu_param.shape(), updated_gpu_param_on_cpu.shape(),
                "Updated parameter shapes don't match for {}", param_name);

            let cpu_data = updated_cpu_param.as_slice().expect("Failed to get CPU parameter data");
            let gpu_data = updated_gpu_param_on_cpu.as_slice().expect("Failed to get GPU parameter data");

            for (i, (&cpu_val, &gpu_val)) in cpu_data.iter().zip(gpu_data.iter()).enumerate() {
                let diff = (cpu_val - gpu_val).abs();
                assert!(diff < TOLERANCE,
                    "Parameter update mismatch for {} at index {}: CPU = {}, GPU = {}, diff = {}",
                    param_name, i, cpu_val, gpu_val, diff);
            }

            // Verify that parameters actually changed
            let original_cpu = &original_cpu_params[param_name];
            let original_cpu_data = original_cpu.as_slice().expect("Failed to get original CPU data");

            let params_changed = cpu_data.iter().zip(original_cpu_data.iter())
                .any(|(&new_val, &old_val)| (new_val - old_val).abs() > TOLERANCE);

            assert!(params_changed, "Parameters for {} did not change after optimization step", param_name);
        }

        println!("✓ Linear parameter updates CPU-GPU consistency test passed");
    }
}
