use crate::backend::storage::{CPUStorage, StorageBackend};
use crate::{FerroxCudaF, FerroxCudaN, FerroxN};
use ndarray::{Array1, Array2, ArrayD, ArrayView1, Ix1, IxDyn};

/// Trait that defines pooling operations
/// This allows me to abstract max and average pooling with the same generic function
trait PoolingOp<T: FerroxCudaN> {
    /// Initial value for the accumulator (e.g., min_value for max, zero for avg)
    fn init_value() -> T;

    /// Accumulate a value into the accumulator
    /// For max pooling: takes maximum
    /// For avg pooling: adds to sum and increments count
    fn accumulate(accumulator: &mut T, value: T, valid_count: &mut i32, is_valid: bool);

    /// Finalize the result from accumulator
    /// For max pooling: returns accumulator directly
    /// For avg pooling: divides sum by valid count
    fn finalize(accumulator: T, valid_count: i32) -> T;
}

/// Max pooling operation implementation
struct MaxPoolOp;

impl<T: FerroxCudaN> PoolingOp<T> for MaxPoolOp {
    fn init_value() -> T {
        T::min_value()
    }

    fn accumulate(accumulator: &mut T, value: T, _valid_count: &mut i32, is_valid: bool) {
        if is_valid && value > *accumulator {
            *accumulator = value;
        }
    }

    fn finalize(accumulator: T, _valid_count: i32) -> T {
        accumulator
    }
}

/// Average pooling operation implementation
struct AvgPoolOp;

impl<T: FerroxCudaN> PoolingOp<T> for AvgPoolOp {
    fn init_value() -> T {
        FerroxN::zero()
    }

    fn accumulate(accumulator: &mut T, value: T, valid_count: &mut i32, is_valid: bool) {
        if is_valid {
            *accumulator += value;
            *valid_count += 1;
        }
    }

    fn finalize(accumulator: T, valid_count: i32) -> T {
        if valid_count > 0 {
            let count_t = FerroxN::from_i32(valid_count).unwrap_or(FerroxN::one());
            accumulator / count_t
        } else {
            FerroxN::zero()
        }
    }
}


/// Trait that defines unpooling operations for gradient computation
/// This allows me to abstract max and average unpooling with the same generic function
trait UnpoolingOp<T: FerroxCudaN> {
    /// Initial value for gradient accumulator
    fn init_value() -> T;

    /// Accumulate gradient for a specific input position
    /// For max unpooling: only accumulate if this position was selected
    /// For avg unpooling: accumulate scaled gradient (1/kernel_area)
    fn accumulate_gradient(
        accumulator: &mut T,
        grad_output_val: T,
        original_input_val: T,
        pooled_output_val: T,
        kernel_area: f32,
    );

    /// Finalize the accumulated gradient
    fn finalize_gradient(accumulator: T) -> T;
}

/// Max unpooling operation - gradients only flow to positions that were selected as maximum
struct MaxUnpoolOp;

impl<T: FerroxCudaN> UnpoolingOp<T> for MaxUnpoolOp {
    fn init_value() -> T {
        FerroxN::zero()
    }

    fn accumulate_gradient(
        accumulator: &mut T,
        grad_output_val: T,
        original_input_val: T,
        pooled_output_val: T,
        _kernel_area: f32,
    ) {
        // Only accumulate gradient if this position produced the max value
        // Use small epsilon for floating point comparison
        let epsilon = FerroxN::from_f32(1e-8).unwrap_or(FerroxN::zero());
        let diff = if original_input_val > pooled_output_val {
            original_input_val - pooled_output_val
        } else {
            pooled_output_val - original_input_val
        };

        if diff < epsilon {
            *accumulator += grad_output_val;
        }
    }

    fn finalize_gradient(accumulator: T) -> T {
        accumulator
    }
}

/// Average unpooling operation - gradients are distributed uniformly to all contributing positions
struct AvgUnpoolOp;

impl<T: FerroxCudaN> UnpoolingOp<T> for AvgUnpoolOp {
    fn init_value() -> T {
        FerroxN::zero()
    }

    fn accumulate_gradient(
        accumulator: &mut T,
        grad_output_val: T,
        _original_input_val: T,
        _pooled_output_val: T,
        kernel_area: f32,
    ) {
        // Distribute gradient uniformly - each input position gets 1/kernel_area of the output gradient
        let scale = FerroxN::from_f32(1.0 / kernel_area).unwrap_or(FerroxN::one());
        *accumulator += grad_output_val * scale;
    }

    fn finalize_gradient(accumulator: T) -> T {
        accumulator
    }
}


impl<T> CPUStorage<T>
where
    T: FerroxCudaN,
{
    /// Generic 2D pooling implementation
    /// Used by both max and average pooling operations
    fn pool2d_impl<Op: PoolingOp<T>>(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<ArrayD<T>, String> {
        let input_shape = self.shape();
        if input_shape.len() != 4 {
            return Err("Input must be 4D tensor [N, C, H, W]".to_string());
        }

        let (n, c, h, w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );

        // Calculate output dimensions
        let h_out = (h + 2 * padding - kernel_size) / stride + 1;
        let w_out = (w + 2 * padding - kernel_size) / stride + 1;

        let input_data = self.cpu_data()?;
        let input_slice = input_data.as_slice().ok_or("Input data not contiguous")?;

        let mut output_data = vec![FerroxN::zero(); n * c * h_out * w_out];

        // Iterate over output positions
        for batch in 0..n {
            for channel in 0..c {
                for out_h in 0..h_out {
                    for out_w in 0..w_out {
                        let h_start = (out_h * stride) as i32 - padding as i32;
                        let w_start = (out_w * stride) as i32 - padding as i32;

                        let mut accumulator = Op::init_value();
                        let mut valid_count = 0;

                        // Pool over kernel window
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let h_pos = h_start + kh as i32;
                                let w_pos = w_start + kw as i32;

                                // Check bounds
                                let is_valid = h_pos >= 0 && h_pos < h as i32
                                              && w_pos >= 0 && w_pos < w as i32;

                                if is_valid {
                                    let input_idx = batch * (c * h * w)
                                        + channel * (h * w)
                                        + (h_pos as usize) * w
                                        + (w_pos as usize);

                                    let val = input_slice[input_idx];
                                    Op::accumulate(&mut accumulator, val, &mut valid_count, true);
                                }
                            }
                        }

                        let output_idx = batch * (c * h_out * w_out)
                            + channel * (h_out * w_out)
                            + out_h * w_out
                            + out_w;

                        output_data[output_idx] = Op::finalize(accumulator, valid_count);
                    }
                }
            }
        }

        ArrayD::from_shape_vec(IxDyn(&[n, c, h_out, w_out]), output_data)
            .map_err(|e| format!("Failed to create output tensor: {e}"))
    }

    /// Generic 1D pooling implementation
    /// Used by both max and average pooling operations
    fn pool1d_impl<Op: PoolingOp<T>>(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<ArrayD<T>, String> {
        let input_shape = self.shape();
        if input_shape.len() != 3 {
            return Err("Input must be 3D tensor [N, C, L]".to_string());
        }

        let (n, c, l) = (input_shape[0], input_shape[1], input_shape[2]);

        // Calculate output dimensions
        let l_out = (l + 2 * padding - kernel_size) / stride + 1;

        let input_data = self.cpu_data()?;
        let input_slice = input_data.as_slice().ok_or("Input data not contiguous")?;

        let mut output_data = vec![FerroxN::zero(); n * c * l_out];

        // Iterate over output positions
        for batch in 0..n {
            for channel in 0..c {
                for out_l in 0..l_out {
                    let l_start = (out_l * stride) as i32 - padding as i32;

                    let mut accumulator = Op::init_value();
                    let mut valid_count = 0;

                    // Pool over kernel window
                    for kl in 0..kernel_size {
                        let l_pos = l_start + kl as i32;

                        // Check bounds
                        let is_valid = l_pos >= 0 && l_pos < l as i32;

                        if is_valid {
                            let input_idx = batch * (c * l) + channel * l + (l_pos as usize);
                            let val = input_slice[input_idx];
                            Op::accumulate(&mut accumulator, val, &mut valid_count, true);
                        }
                    }

                    let output_idx = batch * (c * l_out) + channel * l_out + out_l;
                    output_data[output_idx] = Op::finalize(accumulator, valid_count);
                }
            }
        }

        ArrayD::from_shape_vec(IxDyn(&[n, c, l_out]), output_data)
            .map_err(|e| format!("Failed to create output tensor: {e}"))
    }

    // Specialized pooling functions - these are just thin wrappers around the generic implementation

    /// CPU implementation of 2D max pooling
    /// Matches the CUDA kernel behavior exactly
    pub fn maxpool2d_impl(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<ArrayD<T>, String> {
        self.pool2d_impl::<MaxPoolOp>(kernel_size, stride, padding)
    }

    /// CPU implementation of 2D average pooling
    /// Matches the CUDA kernel behavior exactly
    pub fn avgpool2d_impl(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<ArrayD<T>, String> {
        self.pool2d_impl::<AvgPoolOp>(kernel_size, stride, padding)
    }

    /// CPU implementation of 1D max pooling
    /// Matches the CUDA kernel behavior exactly
    pub fn maxpool1d_impl(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<ArrayD<T>, String> {
        self.pool1d_impl::<MaxPoolOp>(kernel_size, stride, padding)
    }

    /// CPU implementation of 1D average pooling
    /// Matches the CUDA kernel behavior exactly
    pub fn avgpool1d_impl(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<ArrayD<T>, String> {
        self.pool1d_impl::<AvgPoolOp>(kernel_size, stride, padding)
    }
}


impl<T> CPUStorage<T>
where
    T: FerroxCudaN,
{
    /// Generic 2D unpooling implementation
    /// Used by both max and average unpooling gradient operations
    fn unpool2d_impl<Op: UnpoolingOp<T>>(
        &self,
        grad_output: &ArrayD<T>,
        original_input: &ArrayD<T>,
        pooled_output: &ArrayD<T>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<ArrayD<T>, String> {
        let input_shape = original_input.shape();
        let output_shape = grad_output.shape();

        if input_shape.len() != 4 || output_shape.len() != 4 {
            return Err("Unpool2D requires 4D tensors [N, C, H, W]".to_string());
        }

        let (n, c, h, w) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
        let (_, _, h_out, w_out) = (output_shape[0], output_shape[1], output_shape[2], output_shape[3]);

        let grad_output_slice = grad_output.as_slice().ok_or("Grad output not contiguous")?;
        let original_input_slice = original_input.as_slice().ok_or("Original input not contiguous")?;
        let pooled_output_slice = pooled_output.as_slice().ok_or("Pooled output not contiguous")?;

        let mut grad_input_data = vec![FerroxN::zero(); n * c * h * w];
        let kernel_area = (kernel_size * kernel_size) as f32;

        // Iterate over each input position
        for batch in 0..n {
            for channel in 0..c {
                for in_h in 0..h {
                    for in_w in 0..w {
                        let input_idx = batch * (c * h * w) + channel * (h * w) + in_h * w + in_w;
                        let original_val = original_input_slice[input_idx];

                        let mut grad_accumulator = Op::init_value();

                        // Find all output positions that this input position could have contributed to
                        for out_h in 0..h_out {
                            for out_w in 0..w_out {
                                let h_start = (out_h * stride) as i32 - padding as i32;
                                let w_start = (out_w * stride) as i32 - padding as i32;
                                let h_end = h_start + kernel_size as i32;
                                let w_end = w_start + kernel_size as i32;

                                // Check if current input position is within this output's window
                                let in_h_i32 = in_h as i32;
                                let in_w_i32 = in_w as i32;

                                if in_h_i32 >= h_start && in_h_i32 < h_end &&
                                   in_w_i32 >= w_start && in_w_i32 < w_end {

                                    let output_idx = batch * (c * h_out * w_out) +
                                                   channel * (h_out * w_out) +
                                                   out_h * w_out + out_w;

                                    let grad_output_val = grad_output_slice[output_idx];
                                    let pooled_val = pooled_output_slice[output_idx];

                                    Op::accumulate_gradient(
                                        &mut grad_accumulator,
                                        grad_output_val,
                                        original_val,
                                        pooled_val,
                                        kernel_area,
                                    );
                                }
                            }
                        }

                        grad_input_data[input_idx] = Op::finalize_gradient(grad_accumulator);
                    }
                }
            }
        }

        ArrayD::from_shape_vec(IxDyn(&[n, c, h, w]), grad_input_data)
            .map_err(|e| format!("Failed to create gradient tensor: {e}"))
    }

    /// Generic 1D unpooling implementation
    /// Used by both max and average unpooling gradient operations
    fn unpool1d_impl<Op: UnpoolingOp<T>>(
        &self,
        grad_output: &ArrayD<T>,
        original_input: &ArrayD<T>,
        pooled_output: &ArrayD<T>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<ArrayD<T>, String> {
        let input_shape = original_input.shape();
        let output_shape = grad_output.shape();

        if input_shape.len() != 3 || output_shape.len() != 3 {
            return Err("Unpool1D requires 3D tensors [N, C, L]".to_string());
        }

        let (n, c, l) = (input_shape[0], input_shape[1], input_shape[2]);
        let (_, _, l_out) = (output_shape[0], output_shape[1], output_shape[2]);

        let grad_output_slice = grad_output.as_slice().ok_or("Grad output not contiguous")?;
        let original_input_slice = original_input.as_slice().ok_or("Original input not contiguous")?;
        let pooled_output_slice = pooled_output.as_slice().ok_or("Pooled output not contiguous")?;

        let mut grad_input_data = vec![FerroxN::zero(); n * c * l];
        let kernel_area = kernel_size as f32;

        // Iterate over each input position
        for batch in 0..n {
            for channel in 0..c {
                for in_l in 0..l {
                    let input_idx = batch * (c * l) + channel * l + in_l;
                    let original_val = original_input_slice[input_idx];

                    let mut grad_accumulator = Op::init_value();

                    // Find all output positions that this input position could have contributed to
                    for out_l in 0..l_out {
                        let l_start = (out_l * stride) as i32 - padding as i32;
                        let l_end = l_start + kernel_size as i32;

                        // Check if current input position is within this output's window
                        let in_l_i32 = in_l as i32;

                        if in_l_i32 >= l_start && in_l_i32 < l_end {
                            let output_idx = batch * (c * l_out) + channel * l_out + out_l;

                            let grad_output_val = grad_output_slice[output_idx];
                            let pooled_val = pooled_output_slice[output_idx];

                            Op::accumulate_gradient(
                                &mut grad_accumulator,
                                grad_output_val,
                                original_val,
                                pooled_val,
                                kernel_area,
                            );
                        }
                    }

                    grad_input_data[input_idx] = Op::finalize_gradient(grad_accumulator);
                }
            }
        }

        ArrayD::from_shape_vec(IxDyn(&[n, c, l]), grad_input_data)
            .map_err(|e| format!("Failed to create gradient tensor: {e}"))
    }

    // Specialized unpooling functions - thin wrappers around the generic implementation

    /// CPU implementation of 2D max unpooling for gradients
    /// Returns gradients only to positions that were selected as maximum
    pub fn max_unpool2d_impl(
        &self,
        grad_output: &ArrayD<T>,
        original_input: &ArrayD<T>,
        pooled_output: &ArrayD<T>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<ArrayD<T>, String> {
        self.unpool2d_impl::<MaxUnpoolOp>(
            grad_output, original_input, pooled_output,
            kernel_size, stride, padding
        )
    }

    /// CPU implementation of 2D average unpooling for gradients
    /// Distributes gradients uniformly to all contributing positions
    pub fn avg_unpool2d_impl(
        &self,
        grad_output: &ArrayD<T>,
        original_input: &ArrayD<T>,
        pooled_output: &ArrayD<T>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<ArrayD<T>, String> {
        self.unpool2d_impl::<AvgUnpoolOp>(
            grad_output, original_input, pooled_output,
            kernel_size, stride, padding
        )
    }

    /// CPU implementation of 1D max unpooling for gradients
    /// Returns gradients only to positions that were selected as maximum
    pub fn max_unpool1d_impl(
        &self,
        grad_output: &ArrayD<T>,
        original_input: &ArrayD<T>,
        pooled_output: &ArrayD<T>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<ArrayD<T>, String> {
        self.unpool1d_impl::<MaxUnpoolOp>(
            grad_output, original_input, pooled_output,
            kernel_size, stride, padding
        )
    }

    /// CPU implementation of 1D average unpooling for gradients
    /// Distributes gradients uniformly to all contributing positions
    pub fn avg_unpool1d_impl(
        &self,
        grad_output: &ArrayD<T>,
        original_input: &ArrayD<T>,
        pooled_output: &ArrayD<T>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<ArrayD<T>, String> {
        self.unpool1d_impl::<AvgUnpoolOp>(
            grad_output, original_input, pooled_output,
            kernel_size, stride, padding
        )
    }
}
