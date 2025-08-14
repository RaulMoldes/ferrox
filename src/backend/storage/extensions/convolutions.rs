use crate::backend::storage::{CPUStorage, StorageBackend};
use ndarray::{ArrayD, IxDyn, Array2};
use crate::{FerroxCudaF, FerroxN};

impl<T> CPUStorage<T>
where
    T: crate::backend::number::FerroxCudaN + Clone,
{
    /// Convert image patches to column matrix (im2col) - reused from original impl
    /// This transforms 4D convolution into efficient 2D matrix multiplication
    fn im2col(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<T>, String> {
        let input_shape = self.cpu_data()?.shape();
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

        // Use effective data access for logical views
        let input_data = if let Some(data) = self.cpu_data()?.as_slice() {
            data
        } else {
            return Err("Input data is empty or not contiguous!".to_string());
        };

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
                                        col_data[col_row * col_width + col_col] =
                                            input_data[input_idx];
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

    /// Standard 2D convolution implementation using im2col + GEMM
    pub fn conv2d_impl(
        &self,
        filter: &ArrayD<T>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<T>, String> {
        let input_shape = self.shape();
        let output_shape = filter.shape();

        let (batch, in_channels, in_h, in_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (out_channels, _, kernel_h, kernel_w) = (
            output_shape[0],
            output_shape[1],
            output_shape[2],
            output_shape[3],
        );

        let out_h = (in_h + 2 * padding.0 - kernel_h) / stride.0 + 1;
        let out_w = (in_w + 2 * padding.1 - kernel_w) / stride.1 + 1;

        // Transform input to column matrix for efficient GEMM
        let col_matrix = self.im2col((kernel_h, kernel_w), stride, padding)?;

        // Reshape filter for matrix multiplication
        let filter_reshaped = filter
            .clone()
            .into_shape_with_order(IxDyn(&[out_channels, in_channels * kernel_h * kernel_w]))
            .map_err(|e| format!("Filter reshape failed: {e}"))?;

        // Perform convolution as matrix multiplication: filter @ col_matrix
        let im2col_view: ndarray::ArrayView2<T> = col_matrix
            .view()
            .into_dimensionality()
            .map_err(|e| format!("Shape error: {e}"))?;

        let filter_view: ndarray::ArrayView2<T> = filter_reshaped
            .view()
            .into_dimensionality()
            .map_err(|e| format!("Shape error: {e}"))?;

        let output_2d = filter_view.dot(&im2col_view);

        // Transpose result from [out_channels, batch * out_h * out_w] to [batch, out_channels, out_h, out_w]
        let output_data: Vec<T> = if let Some(out_slice) = output_2d.as_slice() {
            out_slice.to_vec()
        } else {
            return Err("Failed to get contiguous output data".to_string());
        };
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

        ArrayD::from_shape_vec(IxDyn(&[batch, out_channels, out_h, out_w]), final_output)
            .map_err(|e| format!("Failed to create output tensor: {e}"))
    }


    /// col2im - Convierte column matrix de vuelta a imagen
    /// Usado en deconvolución para reconstruir la imagen desde patches
    fn col2im(
        input: ArrayD<T>,
        output_shape: &[usize], // [batch, channels, height, width]
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<T>, String> {
        let (batch_size, channels, out_height, out_width) = (
            output_shape[0], output_shape[1], output_shape[2], output_shape[3]
        );
        let (kernel_h, kernel_w) = kernel_size;
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;

        // Calcular dimensiones de la "ventana deslizante"
        let padded_h = out_height + 2 * pad_h;
        let padded_w = out_width + 2 * pad_w;
        let col_h = (padded_h - kernel_h) / stride_h + 1;
        let col_w = (padded_w - kernel_w) / stride_w + 1;

        // Inicializar imagen de salida (con padding)
        let mut padded_output = ArrayD::zeros(IxDyn(&[batch_size, channels, padded_h, padded_w]));

        // Recorrer la column matrix y mapear de vuelta a imagen
        for b in 0..batch_size {
            for c in 0..channels {
                for ky in 0..kernel_h {
                    for kx in 0..kernel_w {
                        for col_y in 0..col_h {
                            for col_x in 0..col_w {
                                // Calcular posición en la imagen padded
                                let img_y = col_y * stride_h + ky;
                                let img_x = col_x * stride_w + kx;

                                // Índice en col_matrix
                                let col_idx = b * (channels * kernel_h * kernel_w * col_h * col_w) +
                                            c * (kernel_h * kernel_w * col_h * col_w) +
                                            ky * (kernel_w * col_h * col_w) +
                                            kx * (col_h * col_w) +
                                            col_y * col_w +
                                            col_x;

                                // Obtener valor de col_matrix (aplanar manualmente)
                                let col_flat = input.as_slice().ok_or("Col matrix not contiguous")?;
                                if col_idx < col_flat.len() {
                                    // Acumular en la imagen (importante: ACUMULAR, no sobrescribir)
                                    padded_output[[b, c, img_y, img_x]] += col_flat[col_idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Extraer la parte central (quitar padding)
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

    pub fn deconv2d_impl(
    &self, // self es grad_output
    filter: &ArrayD<T>, // Solo necesitas filter, no input
    input_shape: &[usize],
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<ArrayD<T>, String> {
    let grad_output = self; // self es grad_output
    let grad_shape = grad_output.shape();
    let filter_shape = filter.shape();

    let (batch_size, out_channels, grad_h, grad_w) = (
        grad_shape[0], grad_shape[1], grad_shape[2], grad_shape[3]
    );
    let (_, in_channels, kernel_h, kernel_w) = (
        filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]
    );

    // 1. Reshape filter: [out_c, in_c, kh, kw] -> [out_c, in_c * kh * kw]
    let filter_matrix = filter.clone()
        .into_shape_with_order(IxDyn(&[
            out_channels,
            in_channels * kernel_h * kernel_w
        ]))
        .map_err(|e| format!("Filter reshape failed: {e}"))?;

    // 2. Reshape grad_output: [batch, out_c, grad_h, grad_w] -> [out_c, batch * grad_h * grad_w]
    let grad_matrix = grad_output.cpu_data()?.clone() // NO uses cpu_data() aquí
        .into_shape_with_order(IxDyn(&[
            out_channels,
            batch_size * grad_h * grad_w
        ]))
        .map_err(|e| format!("Grad_output reshape failed: {e}"))?;

    // 3. Usar ndarray's dot para eficiencia (en lugar de loops manuales)
    let filter_2d: Array2<T> = filter_matrix
        .into_dimensionality()
        .map_err(|e| format!("Filter dimensionality error: {e}"))?;

    let grad_2d: Array2<T> = grad_matrix
        .into_dimensionality()
        .map_err(|e| format!("Grad dimensionality error: {e}"))?;

    // filter.T @ grad_output = [in_c * kh * kw, batch * grad_h * grad_w]
    let col_2d = filter_2d.t().dot(&grad_2d);

    // 4. col2im necesita la column matrix en el formato correcto
    let col_matrix = col_2d.into_dyn();

    // 5. Aplicar col2im
    Self::col2im(
        col_matrix, // Pasa referencia, no ownership
        input_shape,
        (kernel_h, kernel_w),
        stride,
        padding,
    )
}

    pub fn cross_correlation_impl(
        &self,
        input2: &ArrayD<T>,
        output_shape: &[usize],
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<T>, String>
    {

        let col_matrix = self.im2col((output_shape[2], output_shape[3]), stride, padding)?;


        let other_shape = input2.shape();
        let corr_matrix = input2.clone().into_shape_with_order(IxDyn(&[
                other_shape[1], // out_channels
                other_shape[0] * other_shape[2] * other_shape[3] // batch * h * w
            ]))
            .map_err(|e| format!("Grad reshape failed: {e}"))?;

        // Use BLAS dot product for efficient matrix multiplication
        let grad_2d: Array2<T> = corr_matrix
            .into_dimensionality()
            .map_err(|e| format!("Dimensionality error: {e}"))?;

        let col_2d: Array2<T> = col_matrix
            .into_dimensionality()
            .map_err(|e| format!("Dimensionality error: {e}"))?;

        // BLAS multiplication: grad_2d @ col_2d.t()
        let result_2d = grad_2d.dot(&col_2d.t());

        // Reshape back to filter shape
        let grad_filter = result_2d
            .into_dyn()
            .into_shape_with_order(IxDyn(output_shape))
            .map_err(|e| format!("Filter reshape failed: {e}"))?;

        Ok(grad_filter)
    }



}
