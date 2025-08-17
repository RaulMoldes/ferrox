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
