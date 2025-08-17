// Refactored convolutions module - split into logical standalone functions
// I've separated the pure computational functions from the CPUStorage methods

use crate::backend::storage::{CPUStorage, StorageBackend};
use crate::{FerroxCudaF, FerroxCudaN, FerroxN};
use ndarray::{Array1, Array2, ArrayD, ArrayView1, Ix1, IxDyn};

/// Function to convert image patches to column matrix (im2col)
/// This transforms 4D convolution into efficient 2D matrix multiplication
fn im2col<T>(
    input_data: &[T],
    input_shape: &[usize], // [batch, channels, height, width]
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<ArrayD<T>, String>
where
    T: crate::backend::number::FerroxCudaN + Clone,
{
    let (batch, channels, in_h, in_w) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let (kernel_h, kernel_w) = kernel_size;

    let out_h = (in_h + 2 * padding.0 - kernel_h) / stride.0 + 1;
    let out_w = (in_w + 2 * padding.1 - kernel_w) / stride.1 + 1;

    let col_height = channels * kernel_h * kernel_w;
    let col_width = batch * out_h * out_w;

    let mut col_data = vec![FerroxN::zero(); col_height * col_width];

    // Extract patches and arrange them as columns for matrix multiplication
    for b in 0..batch {
        for c in 0..channels {
            for ky in 0..kernel_h {
                for kx in 0..kernel_w {
                    let col_row = c * kernel_h * kernel_w + ky * kernel_w + kx;

                    for out_y in 0..out_h {
                        for out_x in 0..out_w {
                            let in_y = out_y * stride.0 + ky;
                            let in_x = out_x * stride.1 + kx;
                            let col_col = b * out_h * out_w + out_y * out_w + out_x;

                            // Handle padding by checking bounds
                            if in_y >= padding.0
                                && in_y < in_h + padding.0
                                && in_x >= padding.1
                                && in_x < in_w + padding.1
                            {
                                let actual_y = in_y - padding.0;
                                let actual_x = in_x - padding.1;

                                if actual_y < in_h && actual_x < in_w {
                                    let input_idx = b * (channels * in_h * in_w)
                                        + c * (in_h * in_w)
                                        + actual_y * in_w
                                        + actual_x;
                                    col_data[col_row * col_width + col_col] = input_data[input_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    ArrayD::from_shape_vec(IxDyn(&[col_height, col_width]), col_data)
        .map_err(|e| format!("Failed to create im2col matrix: {e}"))
}

/// Function to convert column matrix back to image (col2im)
/// Used in deconvolution to reconstruct image from patches
fn col2im<T>(
    col_matrix: ArrayD<T>,
    output_shape: &[usize], // [batch, channels, height, width]
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<ArrayD<T>, String>
where
    T: crate::backend::number::FerroxCudaN + Clone,
{
    let (batch_size, channels, out_height, out_width) = (
        output_shape[0],
        output_shape[1],
        output_shape[2],
        output_shape[3],
    );
    let (kernel_h, kernel_w) = kernel_size;
    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;

    // Calculate sliding window dimensions
    let padded_h = out_height + 2 * pad_h;
    let padded_w = out_width + 2 * pad_w;
    let col_h = (padded_h - kernel_h) / stride_h + 1;
    let col_w = (padded_w - kernel_w) / stride_w + 1;

    // Initialize padded output image
    let mut padded_output = ArrayD::zeros(IxDyn(&[batch_size, channels, padded_h, padded_w]));

    let col_flat = col_matrix.as_slice().ok_or("Col matrix not contiguous")?;

    // Map column matrix back to image
    for b in 0..batch_size {
        for c in 0..channels {
            for ky in 0..kernel_h {
                for kx in 0..kernel_w {
                    for col_y in 0..col_h {
                        for col_x in 0..col_w {
                            // Calculate position in padded image
                            let img_y = col_y * stride_h + ky;
                            let img_x = col_x * stride_w + kx;

                            // Index in col_matrix
                            let col_idx = b * (channels * kernel_h * kernel_w * col_h * col_w)
                                + c * (kernel_h * kernel_w * col_h * col_w)
                                + ky * (kernel_w * col_h * col_w)
                                + kx * (col_h * col_w)
                                + col_y * col_w
                                + col_x;

                            if col_idx < col_flat.len() {
                                // Accumulate in image (important: ACCUMULATE, don't overwrite)
                                padded_output[[b, c, img_y, img_x]] += col_flat[col_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // Extract central part (remove padding)
    let mut result = ArrayD::zeros(IxDyn(output_shape));
    for b in 0..batch_size {
        for c in 0..channels {
            for h in 0..out_height {
                for w in 0..out_width {
                    result[[b, c, h, w]] = padded_output[[b, c, h + pad_h, w + pad_w]];
                }
            }
        }
    }

    Ok(result)
}

/// Transpose output from [out_channels, batch * out_h * out_w] to [batch, out_channels, out_h, out_w]
fn transpose_conv_output<T>(
    output_data: &[T],
    batch: usize,
    out_channels: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<T>
where
    T: FerroxCudaN + Clone,
{
    let mut final_output = vec![FerroxN::zero(); batch * out_channels * out_h * out_w];

    for out_c in 0..out_channels {
        for b in 0..batch {
            for y in 0..out_h {
                for x in 0..out_w {
                    let src_idx =
                        out_c * (batch * out_h * out_w) + b * (out_h * out_w) + y * out_w + x;
                    let dst_idx = b * (out_channels * out_h * out_w)
                        + out_c * (out_h * out_w)
                        + y * out_w
                        + x;
                    final_output[dst_idx] = output_data[src_idx];
                }
            }
        }
    }

    final_output
}

/// Reshape array to 2D safely, handling memory layout issues
fn safe_reshape_to_2d<T>(array: &ArrayD<T>, new_shape: [usize; 2]) -> Result<Array2<T>, String>
where
    T: FerroxCudaN + Clone,
{
    // Convert to owned data first to ensure contiguous layout
    let data: Vec<T> = array.iter().cloned().collect();
    Array2::from_shape_vec(new_shape, data).map_err(|e| format!("Failed to reshape to 2D: {e}"))
}

impl<T> CPUStorage<T>
where
    T: FerroxCudaN + Clone,
{
    /// Im2col wrapper that uses the storage's data
    fn im2col(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<T>, String> {
        let input_data = self
            .cpu_data()?
            .as_slice()
            .ok_or("Input data is not contiguous")?;
        let input_shape = self.shape();

        im2col(input_data, input_shape, kernel_size, stride, padding)
    }

    /// Standard 2D convolution implementation using im2col + GEMM
    pub fn conv2d_impl(
        &self,
        filter: &ArrayD<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<T>, String> {
        let input_shape = self.shape();
        let filter_shape = filter.shape();

        let (batch, in_channels, in_h, in_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (out_channels, _, kernel_h, kernel_w) = (
            filter_shape[0],
            filter_shape[1],
            filter_shape[2],
            filter_shape[3],
        );

        let out_h = (in_h + 2 * padding.0 - kernel_h) / stride.0 + 1;
        let out_w = (in_w + 2 * padding.1 - kernel_w) / stride.1 + 1;

        // Transform input to column matrix
        let col_matrix = self.im2col((kernel_h, kernel_w), stride, padding)?;

        // Reshape filter for matrix multiplication
        let filter_2d =
            safe_reshape_to_2d(filter, [out_channels, in_channels * kernel_h * kernel_w])?;

        let col_2d = safe_reshape_to_2d(
            &col_matrix,
            [in_channels * kernel_h * kernel_w, batch * out_h * out_w],
        )?;

        // Perform convolution as matrix multiplication
        let output_2d = filter_2d.dot(&col_2d);

        // Get output data and transpose
        let output_data = output_2d.as_slice().ok_or("Output is not contiguous")?;

        let final_output = transpose_conv_output(output_data, batch, out_channels, out_h, out_w);

        ArrayD::from_shape_vec(IxDyn(&[batch, out_channels, out_h, out_w]), final_output)
            .map_err(|e| format!("Failed to create output tensor: {e}"))
    }

    /// Deconvolution implementation (gradient w.r.t. input)
    pub fn deconv2d_impl(
        &self,
        filter: &ArrayD<T>,
        input_shape: &[usize],
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<T>, String> {
        let grad_output = self.cpu_data()?;
        let grad_shape = grad_output.shape();
        let filter_shape = filter.shape();

        let (batch_size, out_channels, grad_h, grad_w) =
            (grad_shape[0], grad_shape[1], grad_shape[2], grad_shape[3]);
        let (_, in_channels, kernel_h, kernel_w) = (
            filter_shape[0],
            filter_shape[1],
            filter_shape[2],
            filter_shape[3],
        );

        // Reshape tensors for matrix multiplication
        let filter_2d =
            safe_reshape_to_2d(filter, [out_channels, in_channels * kernel_h * kernel_w])?;

        let grad_2d =
            safe_reshape_to_2d(grad_output, [out_channels, batch_size * grad_h * grad_w])?;

        // Matrix multiplication: filter.T @ grad_output
        let col_2d = filter_2d.t().dot(&grad_2d);

        // Convert back to column matrix format and apply col2im
        let col_matrix = col_2d.into_dyn();
        col2im(
            col_matrix,
            input_shape,
            (kernel_h, kernel_w),
            stride,
            padding,
        )
    }

    /// Cross-correlation implementation (gradient w.r.t. filter)
    pub fn cross_correlation2d_impl(
        &self,
        grad_output: &ArrayD<T>,
        output_shape: &[usize],
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<T>, String> {
        // Get input as column matrix
        let col_matrix = self.im2col((output_shape[2], output_shape[3]), stride, padding)?;

        let grad_shape = grad_output.shape();

        // Reshape grad_output for matrix multiplication
        let grad_2d = safe_reshape_to_2d(
            grad_output,
            [grad_shape[1], grad_shape[0] * grad_shape[2] * grad_shape[3]],
        )?;

        let col_2d =
            safe_reshape_to_2d(&col_matrix, [col_matrix.shape()[0], col_matrix.shape()[1]])?;

        // Matrix multiplication: grad_output @ col_matrix.T
        let result_2d = grad_2d.dot(&col_2d.t());

        // Safely convert back to filter shape
        let result_data: Vec<T> = result_2d.iter().cloned().collect();
        ArrayD::from_shape_vec(IxDyn(output_shape), result_data)
            .map_err(|e| format!("Filter reshape failed: {e}"))
    }

    /// Performs 1D convolution on flattened arrays
    /// This matches the CUDA kernel behavior for simple 1D convolution
    pub fn conv1d_impl(&self, filter_data: &ArrayD<T>) -> Result<ArrayD<T>, String> {
        let arrayd = self.cpu_data()?;
        let input: ArrayView1<T> = arrayd
            .view()
            .into_dimensionality::<Ix1>()
            .map_err(|_| "Input array is not 1D".to_string())?;

        let filter: ArrayView1<T> = filter_data
            .view()
            .into_dimensionality::<Ix1>()
            .map_err(|_| "Filter array is not 1D".to_string())?;
        let input_size = input.len();
        let kernel_size = filter.len();

        // Calculate output size
        let output_size = input_size.saturating_sub(kernel_size).saturating_add(1);
        if output_size == 0_usize {
            return Err("Kernel size cannot be larger than input size".to_string());
        }

        let mut output = Array1::zeros(output_size);

        // Perform 1D convolution
        for i in 0..output_size {
            let mut sum = FerroxN::zero();
            for j in 0..kernel_size {
                sum += input[i + j] * filter[j];
            }
            output[i] = sum;
        }

        Ok(output.into_dyn())
    }



    /// Performs 1D cross-correlation for gradient computation
    /// Used to compute filter gradients in conv1d backward pass
    /// input: original input tensor, grad_output: gradient from next layer
    /// Returns: gradient w.r.t. filter
    pub fn cross_correlation1d(
        &self,
        input2_data: &ArrayD<T>,
    ) -> Result<ArrayD<T>, String> {

         let arrayd = self.cpu_data()?;
        let input1: ArrayView1<T> = arrayd
            .view()
            .into_dimensionality::<Ix1>()
            .map_err(|_| "Input 1 array is not 1D".to_string())?;

         let input2: ArrayView1<T> = input2_data
            .view()
            .into_dimensionality::<Ix1>()
            .map_err(|_| "Input 2 array is not 1D".to_string())?;

        let input1_size = input1.len();
        let input2_size = input2.len();

        // Calculate kernel size from input and output sizes
        let kernel_size = input1_size.saturating_sub(input2_size).saturating_add(1);
        if kernel_size == 0 {
            return Err("Invalid sizes for cross-correlation".to_string());
        }

        let mut output = Array1::zeros(kernel_size);

        // Cross-correlation: for each position in filter, compute dot product
        for k in 0..kernel_size {
            let mut sum = FerroxN::zero();
            for i in 0..input2_size {
                sum += input1[i + k] * input2[i];
            }
            output[k] = sum;
        }

        Ok(output.into_dyn())
    }


    /// Performs 1D deconvolution for gradient computation
    /// Used to compute input gradients in conv1d backward pass
    /// grad_output: gradient from next layer, filter: convolution filter
    /// Returns: gradient w.r.t. input
    pub fn deconv1d_impl(&self, filter_data: &ArrayD<T>) -> Result<ArrayD<T>, String> {
        let arrayd = self.cpu_data()?;
        let grad_output: ArrayView1<T> = arrayd
            .view()
            .into_dimensionality::<Ix1>()
            .map_err(|_| "Grad output array is not 1D".to_string())?;

        let filter: ArrayView1<T> = filter_data
            .view()
            .into_dimensionality::<Ix1>()
            .map_err(|_| "Filter array is not 1D".to_string())?;

        let grad_size = grad_output.len();
        let kernel_size = filter.len();
        let input_size = grad_size + kernel_size - 1;

        let mut grad_input = Array1::zeros(input_size);

        // Deconvolution: for each position in grad_input, accumulate from relevant grad_output positions
        for i in 0..input_size {
            let mut sum = FerroxN::zero();

            // For each position in grad_output that affects grad_input[i]
            for j in 0..grad_size {
                let k = i.saturating_sub(j); // Filter index (will be flipped)
                if k == 0_usize && k < kernel_size {
                    // Use flipped filter: filter[kernel_size - 1 - k]
                    let filter_idx = kernel_size - 1 - k;
                    sum += grad_output[j] * filter[filter_idx];
                }
            }

            grad_input[i] = sum;
        }

        Ok(grad_input.into_dyn())
    }
}
