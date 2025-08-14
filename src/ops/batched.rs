use crate::backend::{FerroxCudaF, Tensor};
use crate::ops::Operator;

#[derive(Debug, Clone)]
pub struct Softmax {
    axis: Option<usize>, // None means apply to entire tensor (current behavior)
}

impl<T> Operator<T> for Softmax
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 1 {
            return Err("Softmax operation requires exactly 1 input".to_string());
        }

        let input_tensor = inputs[0];

        match self.axis {
            None => {
                // Fallback to existing whole-tensor softmax implementation
                input_tensor.softmax()
            }
            Some(axis) => {
                // Use optimized batch-aware kernel instead of partitioning
                if axis >= input_tensor.ndim() {
                    return Err(format!(
                        "Softmax axis {} out of bounds for tensor with {} dimensions",
                        axis,
                        input_tensor.ndim()
                    ));
                }

                // Call the batch-aware softmax method
                input_tensor.softmax_batched(axis)
            }
        }
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        _inputs: &mut [&Tensor<T>],
        outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if outputs.shape() != grad_output.shape() {
            return Err("Softmax gradient: shape mismatch".to_string());
        }

        match self.axis {
            None => {
                // Use existing gradient computation for entire tensor
                self.compute_global_gradient(grad_output, outputs)
            }
            Some(axis) => {
                // Use optimized batch-aware gradient computation
                self.compute_batch_gradient(grad_output, outputs, axis)
            }
        }
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }

    fn num_inputs(&self) -> usize {
        1
    }
}

impl Softmax {
    pub fn new(axis: Option<usize>) -> Self {
        Self { axis }
    }

    // Helper method for global gradient computation
    fn compute_global_gradient<T>(
        &self,
        grad_output: Tensor<T>,
        outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String>
    where
        T: FerroxCudaF,
    {
        let element_wise_prod = grad_output.mul(outputs)?;
        let mut sum_prod = element_wise_prod.sum(None, false)?;
        sum_prod.broadcast_to(outputs.shape())?;
        let grad_diff = grad_output.sub(&sum_prod)?;
        let grad_input = outputs.mul(&grad_diff)?;
        Ok(vec![grad_input])
    }

    /// Optimized gradient computation for batch softmax
    /// Uses the mathematical property: ∇softmax = softmax * (∇L - Σ(softmax * ∇L))
    /// where the sum is computed along the softmax axis
    fn compute_batch_gradient<T>(
        &self,
        grad_output: Tensor<T>,
        outputs: &Tensor<T>,
        axis: usize,
    ) -> Result<Vec<Tensor<T>>, String>
    where
        T: FerroxCudaF,
    {
        // Element-wise product: softmax * grad_output
        let element_wise_prod = grad_output.mul(outputs)?;

        // Sum along the softmax axis, keeping dimensions
        let mut sum_prod = element_wise_prod.sum(Some(&[axis]), true)?;

        // Broadcast sum back to original shape for subtraction
        sum_prod.broadcast_to(outputs.shape())?;

        // Compute: grad_output - broadcasted_sum
        let grad_diff = grad_output.sub(&sum_prod)?;

        // Final gradient: softmax * (grad_output - sum)
        let grad_input = outputs.mul(&grad_diff)?;

        Ok(vec![grad_input])
    }
}

/// 2D Convolution operation for batched inputs
/// Input shape: [batch_size, in_channels, height, width]
/// Filter shape: [out_channels, in_channels, kernel_height, kernel_width]
/// Output shape: [batch_size, out_channels, out_height, out_width]
#[derive(Debug, Clone)]
pub struct Conv2d {
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl Conv2d {
    pub fn new(stride: (usize, usize), padding: (usize, usize)) -> Self {
        Self { stride, padding }
    }
}

impl<T> Operator<T> for Conv2d
where
    T: FerroxCudaF,
{
    fn compute(&self, inputs: &mut [&Tensor<T>]) -> Result<Tensor<T>, String> {
        if inputs.len() != 2 {
            return Err("Conv2d operation requires exactly 2 inputs (input, filter)".to_string());
        }

        let input = inputs[0];
        let filter = inputs[1];

        // Validate input dimensions for batched convolution
        if input.ndim() != 4 {
            return Err(
                "Conv2d input must be 4D [batch_size, in_channels, height, width]".to_string(),
            );
        }

        if filter.ndim() != 4 {
            return Err(
                "Conv2d filter must be 4D [out_channels, in_channels, kernel_height, kernel_width]"
                    .to_string(),
            );
        }

        // Validate channel compatibility
        let input_channels = input.shape()[1];
        let filter_channels = filter.shape()[1];
        if input_channels != filter_channels {
            return Err(format!(
                "Input channels {} don't match filter channels {}",
                input_channels, filter_channels
            ));
        }

        // Perform batched convolution using existing backend
        input.conv2d(filter, self.stride, self.padding)
    }

    fn gradient(
        &self,
        grad_output: Tensor<T>,
        inputs: &mut [&Tensor<T>],
        _outputs: &Tensor<T>,
    ) -> Result<Vec<Tensor<T>>, String> {
        if inputs.len() != 2 {
            return Err("Conv2d operation requires exactly 2 inputs".to_string());
        }

        let input = inputs[0]; // [batch, in_channels, in_h, in_w]
        let filter = inputs[1]; // [out_channels, in_channels, kernel_h, kernel_w]

        // grad_output: [batch, out_channels, out_h, out_w]
        let input_shape = input.shape();
        // 1. Gradient w.r.t. input (data gradient) using deconvolution
        let grad_input = grad_output.deconv2d(filter, input_shape, self.stride, self.padding)?;

        // 2. Gradient w.r.t. filter (weight gradient) using cross-correlation
        let grad_filter =
            input.cross_correlation(&grad_output, filter.shape(), self.stride, self.padding)?;

        Ok(vec![grad_input, grad_filter])
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn clone_op(&self) -> Box<dyn Operator<T>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod conv2d_tests {
    use crate::backend::manager::best_f32_device;
    use crate::backend::Tensor;
    use crate::ops::batched::Conv2d;
    use crate::ops::Operator;
    use crate::FerroxN;
    #[test]
    fn test_conv2d_forward() {
        println!("Testing Conv2D forward pass...");

        // Test configuration matching PyTorch
        let batch_size = 1;
        let in_channels = 3;
        let in_height = 5;
        let in_width = 5;
        let out_channels = 4;
        let kernel_height = 3;
        let kernel_width = 3;
        let stride = (1, 1);
        let padding = (1, 1);

        let device = best_f32_device();

        // Input tensor [1, 3, 5, 5] - exact PyTorch input values
        let input_data = vec![
            // Channel 0
            0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650,
            0.700, 0.750, 0.800, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500,
            0.550, // Channel 1
            0.600, 0.650, 0.700, 0.750, 0.800, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400,
            0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.100, 0.150, 0.200, 0.250,
            0.300, // Channel 2
            0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.100, 0.150,
            0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750,
            0.800,
        ];

        let input = Tensor::from_vec_with_device(
            input_data,
            &[batch_size, in_channels, in_height, in_width],
            device,
        )
        .expect("Failed to create input tensor");

        // Filter tensor [4, 3, 3, 3] - exact PyTorch filter values
        let filter_data = vec![
            // Filter 0
            0.010, 0.030, 0.050, 0.070, 0.090, 0.110, 0.130, 0.150, 0.170, 0.190, 0.210, 0.230,
            0.250, 0.270, 0.290, 0.310, 0.330, 0.350, 0.370, 0.390, 0.010, 0.030, 0.050, 0.070,
            0.090, 0.110, 0.130, // Filter 1
            0.150, 0.170, 0.190, 0.210, 0.230, 0.250, 0.270, 0.290, 0.310, 0.330, 0.350, 0.370,
            0.390, 0.010, 0.030, 0.050, 0.070, 0.090, 0.110, 0.130, 0.150, 0.170, 0.190, 0.210,
            0.230, 0.250, 0.270, // Filter 2
            0.290, 0.310, 0.330, 0.350, 0.370, 0.390, 0.010, 0.030, 0.050, 0.070, 0.090, 0.110,
            0.130, 0.150, 0.170, 0.190, 0.210, 0.230, 0.250, 0.270, 0.290, 0.310, 0.330, 0.350,
            0.370, 0.390, 0.010, // Filter 3
            0.030, 0.050, 0.070, 0.090, 0.110, 0.130, 0.150, 0.170, 0.190, 0.210, 0.230, 0.250,
            0.270, 0.290, 0.310, 0.330, 0.350, 0.370, 0.390, 0.010, 0.030, 0.050, 0.070, 0.090,
            0.110, 0.130, 0.150,
        ];

        let filter = Tensor::from_vec_with_device(
            filter_data,
            &[out_channels, in_channels, kernel_height, kernel_width],
            device,
        )
        .expect("Failed to create filter tensor");

        // Perform convolution
        let conv_op = Conv2d::new(stride, padding);
        let mut inputs = [&input, &filter];
        let result = conv_op
            .compute(&mut inputs)
            .expect("Failed to compute convolution");

        println!("Output shape: {:?}", result.shape());

        // Get result data
        let result_data = result.to_vec().expect("Failed to get result data");

        // PyTorch reference output
        let pytorch_result = vec![
            // Channel 0
            0.7780, 1.1820, 1.3320, 1.4820, 0.9700, 1.1345, 1.7940, 2.0185, 2.2430, 1.6215, 1.2345,
            1.9940, 2.2185, 2.4430, 1.8215, 1.0345, 1.5940, 1.8185, 2.0430, 1.4215, 0.6480, 1.0500,
            1.1860, 1.3220, 1.0120, // Channel 1
            0.8080, 1.4700, 1.6460, 1.8220, 1.3120, 1.5395, 2.3860, 2.6595, 2.9330, 1.9845, 1.1395,
            1.8860, 2.1595, 2.4330, 1.6845, 1.3395, 2.2860, 2.5595, 2.8330, 1.9845, 1.0380, 1.6180,
            1.8000, 1.9820, 1.3540, // Channel 2
            0.8780, 1.4780, 1.6800, 1.8820, 1.4740, 1.4245, 2.2180, 2.5205, 2.8230, 1.9675, 1.7245,
            2.7180, 3.0205, 3.3230, 2.3675, 1.4245, 2.3180, 2.6205, 2.9230, 2.1675, 1.1680, 1.8060,
            2.0340, 2.2620, 1.5160, // Channel 3
            0.8680, 1.3260, 1.4940, 1.6620, 1.0960, 1.1295, 1.8500, 2.0815, 2.3130, 1.5905, 1.1295,
            1.9500, 2.1815, 2.4130, 1.6905, 1.1295, 1.7500, 1.9815, 2.2130, 1.4905, 0.5980, 1.0340,
            1.1680, 1.3020, 0.9180,
        ];

        // Compare results
        let tolerance = 1e-3;
        let mut max_diff = 0.0f32;
        let mut mismatches = 0;

        println!("Comparing results (showing first 10 and any mismatches):");
        for (i, (my_val, pytorch_val)) in result_data.iter().zip(pytorch_result.iter()).enumerate()
        {
            let diff = (my_val - pytorch_val).abs();
            max_diff = max_diff.max(diff);

            if i < 10 || diff > tolerance {
                println!(
                    "  [{:3}] My: {:.4}, PyTorch: {:.4}, Diff: {:.6}",
                    i, my_val, pytorch_val, diff
                );
                if diff > tolerance {
                    mismatches += 1;
                }
            }
        }

        println!("Results:");
        println!("  Maximum difference: {:.6}", max_diff);
        println!("  Tolerance: {:.6}", tolerance);
        println!("  Mismatches: {}/{}", mismatches, result_data.len());

        assert!(
            max_diff < tolerance,
            "Forward pass validation failed: max diff {:.6} > tolerance {:.6}",
            max_diff,
            tolerance
        );

        println!("✓ Forward pass validation PASSED");
    }

    #[test]
    fn test_conv2d_gradients_pytorch_reference() {
        println!("Testing Conv2D gradients against PyTorch reference...");

        // Same configuration as forward test
        let batch_size = 1;
        let in_channels = 3;
        let in_height = 5;
        let in_width = 5;
        let out_channels = 4;
        let kernel_height = 3;
        let kernel_width = 3;
        let stride = (1, 1);
        let padding = (1, 1);

        let device = best_f32_device();

        // Same input as forward test
        let input_data = vec![
            0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650,
            0.700, 0.750, 0.800, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500,
            0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350,
            0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.100, 0.150, 0.200,
            0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800,
            0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650,
            0.700, 0.750, 0.800,
        ];

        let input = Tensor::from_vec_with_device(
            input_data,
            &[batch_size, in_channels, in_height, in_width],
            device,
        )
        .expect("Failed to create input tensor");

        // Same filter as forward test
        let filter_data = vec![
            0.010, 0.030, 0.050, 0.070, 0.090, 0.110, 0.130, 0.150, 0.170, 0.190, 0.210, 0.230,
            0.250, 0.270, 0.290, 0.310, 0.330, 0.350, 0.370, 0.390, 0.010, 0.030, 0.050, 0.070,
            0.090, 0.110, 0.130, 0.150, 0.170, 0.190, 0.210, 0.230, 0.250, 0.270, 0.290, 0.310,
            0.330, 0.350, 0.370, 0.390, 0.010, 0.030, 0.050, 0.070, 0.090, 0.110, 0.130, 0.150,
            0.170, 0.190, 0.210, 0.230, 0.250, 0.270, 0.290, 0.310, 0.330, 0.350, 0.370, 0.390,
            0.010, 0.030, 0.050, 0.070, 0.090, 0.110, 0.130, 0.150, 0.170, 0.190, 0.210, 0.230,
            0.250, 0.270, 0.290, 0.310, 0.330, 0.350, 0.370, 0.390, 0.010, 0.030, 0.050, 0.070,
            0.090, 0.110, 0.130, 0.150, 0.170, 0.190, 0.210, 0.230, 0.250, 0.270, 0.290, 0.310,
            0.330, 0.350, 0.370, 0.390, 0.010, 0.030, 0.050, 0.070, 0.090, 0.110, 0.130, 0.150,
        ];

        let filter = Tensor::from_vec_with_device(
            filter_data,
            &[out_channels, in_channels, kernel_height, kernel_width],
            device,
        )
        .expect("Failed to create filter tensor");

        // Gradient output - exact PyTorch grad_output values
        let grad_output_data = vec![
            0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000, 0.100, 0.200,
            0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000, 0.100, 0.200, 0.300, 0.400,
            0.500, 0.600, 0.700, 0.800, 0.900, 1.000, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600,
            0.700, 0.800, 0.900, 1.000, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800,
            0.900, 1.000, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000,
            0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000, 0.100, 0.200,
            0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000, 0.100, 0.200, 0.300, 0.400,
            0.500, 0.600, 0.700, 0.800, 0.900, 1.000, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600,
            0.700, 0.800, 0.900, 1.000,
        ];

        let out_height = (in_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
        let out_width = (in_width + 2 * padding.1 - kernel_width) / stride.1 + 1;

        let grad_output = Tensor::from_vec_with_device(
            grad_output_data,
            &[batch_size, out_channels, out_height, out_width],
            device,
        )
        .expect("Failed to create grad_output tensor");

        // Compute forward pass and gradients
        let conv_op = Conv2d::new(stride, padding);
        let mut inputs = [&input, &filter];
        let outputs = conv_op
            .compute(&mut inputs)
            .expect("Failed to compute forward pass");

        let gradients = conv_op
            .gradient(grad_output, &mut inputs, &outputs)
            .expect("Failed to compute gradients");

        assert_eq!(
            gradients.len(),
            2,
            "Should have gradients for input and filter"
        );

        let grad_input = &gradients[0];
        let grad_filter = &gradients[1];

        println!("Gradient shapes:");
        println!("  Input gradient: {:?}", grad_input.shape());
        println!("  Filter gradient: {:?}", grad_filter.shape());

        // Get gradient data
        let grad_input_data = grad_input.to_vec().expect("Failed to get grad_input");
        let grad_filter_data = grad_filter.to_vec().expect("Failed to get grad_filter");

        // PyTorch reference gradients (from your output)
        let pytorch_grad_input = vec![
            1.0160, 1.8040, 2.2120, 2.6200, 2.0080, 1.6320, 2.8620, 3.4620, 4.0620, 3.0960, 1.3520,
            2.4420, 3.0420, 3.6420, 2.8160, 1.6320, 2.8620, 3.4620, 4.0620, 3.0960, 0.8800, 1.6120,
            2.0440, 2.4760, 1.9200, 1.2880, 2.1480, 2.6680, 3.1880, 2.1440, 2.1600, 3.7380, 4.5460,
            5.3540, 3.8800, 2.0800, 3.5180, 4.3260, 5.1340, 3.6000, 2.1600, 3.7380, 4.5460, 5.3540,
            3.8800, 1.5520, 2.5560, 3.1000, 3.6440, 2.4560, 1.3600, 2.0920, 2.5240, 2.9560, 1.9600,
            1.7680, 2.9340, 3.5900, 4.2460, 2.8240, 2.0880, 3.1140, 3.7700, 4.4260, 2.9440, 1.7680,
            2.9340, 3.5900, 4.2460, 2.8240, 1.1440, 1.7800, 2.1960, 2.6120, 1.8720,
        ];

        let pytorch_grad_filter = vec![
            3.8300, 4.7750, 3.6500, 3.9500, 4.8750, 3.6500, 4.6800, 5.7750, 4.4000, 5.0300, 6.1500,
            4.6500, 4.8000, 5.8750, 4.4000, 3.3300, 4.1500, 3.1500, 3.6800, 4.5250, 3.4000, 5.6500,
            6.8750, 5.1500, 4.5300, 5.5250, 4.1500, 3.3300, 4.1500, 3.1500, 5.3000, 6.6250, 5.1000,
            3.6800, 4.5250, 3.4000, 4.5300, 5.5250, 4.1500, 5.6500, 7.0000, 5.3500, 3.8300, 4.7750,
            3.6500, 4.6800, 5.7750, 4.4000, 6.0000, 7.3750, 5.6000, 5.0300, 6.1500, 4.6500, 3.8300,
            4.7750, 3.6500, 3.9500, 4.8750, 3.6500, 4.6800, 5.7750, 4.4000, 5.0300, 6.1500, 4.6500,
            4.8000, 5.8750, 4.4000, 3.3300, 4.1500, 3.1500, 3.6800, 4.5250, 3.4000, 5.6500, 6.8750,
            5.1500, 4.5300, 5.5250, 4.1500, 3.3300, 4.1500, 3.1500, 5.3000, 6.6250, 5.1000, 3.6800,
            4.5250, 3.4000, 4.5300, 5.5250, 4.1500, 5.6500, 7.0000, 5.3500, 3.8300, 4.7750, 3.6500,
            4.6800, 5.7750, 4.4000, 6.0000, 7.3750, 5.6000, 5.0300, 6.1500, 4.6500,
        ];

        let tolerance = 1e-3;

        // Check input gradient
        println!("Checking input gradient (first 10 values):");
        let mut input_mismatches = 0;
        for (i, (my_val, pytorch_val)) in grad_input_data
            .iter()
            .zip(pytorch_grad_input.iter())
            .enumerate()
        {
            let diff = (my_val - pytorch_val).abs();
            if i < 10 || diff > tolerance {
                println!(
                    "  [{:2}] My: {:.4}, PyTorch: {:.4}, Diff: {:.6}",
                    i, my_val, pytorch_val, diff
                );
                if diff > tolerance {
                    input_mismatches += 1;
                }
            }
        }

        // Check filter gradient
        println!("Checking filter gradient (first 10 values):");
        let mut filter_mismatches = 0;
        for (i, (my_val, pytorch_val)) in grad_filter_data
            .iter()
            .zip(pytorch_grad_filter.iter())
            .enumerate()
        {
            let diff = (my_val - pytorch_val).abs();
            if diff > tolerance {
                println!(
                    "  [{:2}] My: {:.4}, PyTorch: {:.4}, Diff: {:.6}",
                    i, my_val, pytorch_val, diff
                );
                if diff > tolerance {
                    filter_mismatches += 1;
                }
            }
        }

        println!("Gradient validation results:");
        println!(
            "  Input grad mismatches: {}/{}",
            input_mismatches,
            pytorch_grad_input.len()
        );
        println!(
            "  Filter grad mismatches: {}/{}",
            filter_mismatches,
            pytorch_grad_filter.len()
        );
        assert_eq!(filter_mismatches, 0, "Filter gradient validation failed");
        assert_eq!(input_mismatches, 0, "Input gradient validation failed");
        assert_eq!(filter_mismatches, 0, "Filter gradient validation failed");

        println!("✓ Gradient validation PASSED");
    }

    #[test]
    fn test_conv2d_shape_calculations() {
        println!("Testing Conv2D shape calculations...");

        let device = best_f32_device();

        let test_cases = vec![
            // (in_h, in_w, k_h, k_w, s_h, s_w, p_h, p_w, expected_out_h, expected_out_w)
            (5, 5, 3, 3, 1, 1, 1, 1, 5, 5), // Same padding
            (5, 5, 3, 3, 1, 1, 0, 0, 3, 3), // Valid padding
            (6, 6, 3, 3, 2, 2, 1, 1, 3, 3), // Stride 2
            (4, 4, 2, 2, 1, 1, 0, 0, 3, 3), // 2x2 kernel
        ];

        for (in_h, in_w, k_h, k_w, s_h, s_w, p_h, p_w, exp_h, exp_w) in test_cases {
            let batch_size = 1;
            let in_channels = 2;
            let out_channels = 3;

            // Create test tensors
            let input_size = batch_size * in_channels * in_h * in_w;
            let input_data: Vec<f32> = (0..input_size).map(|i| i as f32 * 0.1).collect();
            let input = Tensor::from_vec_with_device(
                input_data,
                &[batch_size, in_channels, in_h, in_w],
                device,
            )
            .expect("Failed to create input");

            let filter_size = out_channels * in_channels * k_h * k_w;
            let filter_data: Vec<f32> = (0..filter_size).map(|i| i as f32 * 0.05 + 0.1).collect();
            let filter = Tensor::from_vec_with_device(
                filter_data,
                &[out_channels, in_channels, k_h, k_w],
                device,
            )
            .expect("Failed to create filter");

            // Test convolution
            let conv_op = Conv2d::new((s_h, s_w), (p_h, p_w));
            let mut inputs = [&input, &filter];
            let result = conv_op
                .compute(&mut inputs)
                .expect("Failed to compute convolution");

            // Verify output shape
            let expected_shape = [batch_size, out_channels, exp_h, exp_w];
            assert_eq!(
                result.shape(),
                &expected_shape,
                "Shape mismatch for case {}x{} kernel {}x{} stride {}x{} pad {}x{}",
                in_h,
                in_w,
                k_h,
                k_w,
                s_h,
                s_w,
                p_h,
                p_w
            );

            println!(
                "  ✓ {}x{} -> {}x{} (k{}x{}, s{}x{}, p{}x{})",
                in_h, in_w, exp_h, exp_w, k_h, k_w, s_h, s_w, p_h, p_w
            );
        }

        println!("✓ Shape calculation tests PASSED");
    }
}
