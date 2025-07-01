#[cfg(test)]
mod tests {
    use crate::Tensor;
    use crate::backend::GPUNumber;
    use crate::backend::cpu;
    use ndarray::Array4;

    fn assert_tensors_close<T>(a: &Tensor<T>, b: &Tensor<T>, tolerance: T)
    where
        T: GPUNumber + Clone + PartialOrd + std::fmt::Debug,
    {
        assert_eq!(a.shape(), b.shape(), "Shapes don't match");

        let a_data = a.data.as_slice().unwrap();
        let b_data = b.data.as_slice().unwrap();

        for (i, (&val_a, &val_b)) in a_data.iter().zip(b_data.iter()).enumerate() {
            let diff = if val_a > val_b {
                val_a - val_b
            } else {
                val_b - val_a
            };
            assert!(
                diff < tolerance,
                "Values differ at index {}: {:?} vs {:?} (diff: {:?})",
                i,
                val_a,
                val_b,
                diff
            );
        }
    }

    #[test]
    fn test_tensor_creation_with_device() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.len(), 4);
        assert_eq!(tensor.device(), &cpu());

        let zeros = Tensor::<f64>::zeros_with_device(&[3, 3], cpu());
        assert_eq!(zeros.shape(), &[3, 3]);
    }

    #[test]
    fn test_tensor_matmul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();

        let result = a.matmul(&b).unwrap();
        let expected = Tensor::from_vec(vec![22.0, 28.0, 49.0, 64.0], &[2, 2]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_tensor_activations() {
        let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();

        // Test ReLU
        let relu_result = input.relu();
        let expected_relu = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        assert_eq!(relu_result, expected_relu);

        // Test Sigmoid (should be between 0 and 1)
        let sigmoid_result = input.sigmoid();
        for &val in sigmoid_result.data().iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }

        // Test Exp
        let exp_result = input.exp();
        assert!(exp_result.data().iter().all(|&x| x > 0.0));

        // Test Negate
        let neg_result = input.negate();
        let expected_neg = Tensor::from_vec(vec![2.0, 1.0, 0.0, -1.0, -2.0], &[5]).unwrap();
        assert_eq!(neg_result, expected_neg);
    }

    #[test]
    fn test_tensor_scalar_operations() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let add_scalar = tensor.add_scalar(5.0);
        let expected_add = Tensor::from_vec(vec![6.0, 7.0, 8.0, 9.0], &[2, 2]).unwrap();
        assert_eq!(add_scalar, expected_add);

        let mul_scalar = tensor.mul_scalar(2.0);
        let expected_mul = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[2, 2]).unwrap();
        assert_eq!(mul_scalar, expected_mul);

        let div_scalar = tensor.div_scalar(2.0);
        let expected_div = Tensor::from_vec(vec![0.5, 1.0, 1.5, 2.0], &[2, 2]).unwrap();
        assert_eq!(div_scalar, expected_div);
    }

    #[test]
    fn test_tensor_transpose_comprehensive() {
        // Test 2D transpose
        let tensor_2d = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Default transpose (should swap axes)
        let transposed_default = tensor_2d.transpose(None).unwrap();
        assert_eq!(transposed_default.shape(), &[3, 2]);

        // Explicit axes transpose
        let transposed_explicit = tensor_2d.transpose(Some(&[1, 0])).unwrap();
        assert_eq!(transposed_explicit.shape(), &[3, 2]);
        assert_eq!(transposed_default.data(), transposed_explicit.data());

        // Test 3D transpose
        let tensor_3d = Tensor::from_vec((0..24).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();

        // Default transpose (reverse all axes: [2,3,4] -> [4,3,2])
        let transposed_3d_default = tensor_3d.transpose(None).unwrap();
        assert_eq!(transposed_3d_default.shape(), &[4, 3, 2]);

        // Custom permutation: [2,3,4] -> [4,2,3] (axes [2,0,1])
        let transposed_3d_custom = tensor_3d.transpose(Some(&[2, 0, 1])).unwrap();
        assert_eq!(transposed_3d_custom.shape(), &[4, 2, 3]);

        // Test 1D and 0D tensors (should be unchanged)
        let tensor_1d = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let transposed_1d = tensor_1d.transpose(None).unwrap();
        assert_eq!(transposed_1d.shape(), &[3]);
        assert_eq!(transposed_1d.data(), tensor_1d.data());

        let tensor_0d = Tensor::from_vec(vec![42.0], &[]).unwrap();
        let transposed_0d = tensor_0d.transpose(None).unwrap();
        assert_eq!(transposed_0d.shape(), &[]);
        assert_eq!(transposed_0d.data(), tensor_0d.data());
    }

    #[test]
    fn test_transpose_error_cases() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Invalid axes length
        assert!(tensor.transpose(Some(&[0])).is_err());
        assert!(tensor.transpose(Some(&[0, 1, 2])).is_err());

        // Invalid axes values
        assert!(tensor.transpose(Some(&[0, 2])).is_err()); // 2 is out of bounds
        assert!(tensor.transpose(Some(&[0, 0])).is_err()); // duplicate axis
        assert!(tensor.transpose(Some(&[1, 1])).is_err()); // duplicate axis
    }

    #[test]
    fn test_tensor_broadcasting() {
        let a = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let target_shape = &[2, 3];

        let broadcasted = a.broadcast_to(target_shape).unwrap();
        let expected = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[2, 3]).unwrap();
        assert_eq!(broadcasted, expected);
    }

    #[test]
    fn test_tensor_squeeze_unsqueeze() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();

        // Test squeeze
        let squeezed = tensor.squeeze(Some(0)).unwrap();
        assert_eq!(squeezed.shape(), &[3]);

        // Test unsqueeze
        let unsqueezed = squeezed.unsqueeze(1);
        assert_eq!(unsqueezed.shape(), &[3, 1]);
    }

    #[test]
    fn test_tensor_error_handling() {
        // Test shape mismatch in addition
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        assert!(a.add(&b).is_err());

        // Test invalid matrix multiplication
        let c = Tensor::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
        let d = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();

        assert!(c.matmul(&d).is_err());
    }

    #[test]
    fn test_tensor_reductions() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Test sum
        let sum_all = tensor.sum(None);
        assert_eq!(sum_all.data().iter().next().unwrap(), &21.0);

        // Test mean
        let mean_all = tensor.mean(None);
        assert_eq!(mean_all.data().iter().next().unwrap(), &3.5);

        // Test sum along axis
        let sum_axis0 = tensor.sum(Some(0));
        assert_eq!(sum_axis0.shape(), &[3]);

        let sum_axis1 = tensor.sum(Some(1));
        assert_eq!(sum_axis1.shape(), &[2]);
    }

    #[test]
    fn test_conv2d_basic() {
        // Simple 2x2 conv on 3x3 input
        let input_data = Array4::from_shape_vec(
            (1, 1, 3, 3),
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let input = Tensor::new(input_data.into_dyn());

        // 2x2 filter
        let filter_data =
            Array4::from_shape_vec((1, 1, 2, 2), vec![1.0f32, 0.0, 0.0, 1.0]).unwrap();
        let filter = Tensor::new(filter_data.into_dyn());

        let result = input.conv2d(&filter, (1, 1), (0, 0)).unwrap();

        // Expected: 1*1 + 2*0 + 4*0 + 5*1 = 6
        //           2*1 + 3*0 + 5*0 + 6*1 = 8
        //           4*1 + 5*0 + 7*0 + 8*1 = 12
        //           5*1 + 6*0 + 8*0 + 9*1 = 14
        let expected_data =
            Array4::from_shape_vec((1, 1, 2, 2), vec![6.0f32, 8.0, 12.0, 14.0]).unwrap();
        let expected = Tensor::new(expected_data.into_dyn());

        assert_tensors_close(&result, &expected, 1e-6f32);
    }

    #[test]
    fn test_conv2d_with_padding() {
        // 2x2 input
        let input_data = Array4::from_shape_vec((1, 1, 2, 2), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let input = Tensor::new(input_data.into_dyn());

        // 2x2 filter
        let filter_data =
            Array4::from_shape_vec((1, 1, 2, 2), vec![1.0f32, 1.0, 1.0, 1.0]).unwrap();
        let filter = Tensor::new(filter_data.into_dyn());

        // With padding (1,1), output should be 3x3
        let result = input.conv2d(&filter, (1, 1), (1, 1)).unwrap();
        assert_eq!(result.shape(), &[1, 1, 3, 3]);

        // Corner should be just the input value (1.0)
        // Center should be sum of all inputs (10.0)
        let result_data = result.data.as_slice().unwrap();
        assert!((result_data[0] - 1.0).abs() < 1e-6); // Top-left corner
        assert!((result_data[4] - 10.0).abs() < 1e-6); // Center
    }

    #[test]
    fn test_conv2d_stride() {
        // 4x4 input
        let input_data = Array4::from_shape_vec(
            (1, 1, 4, 4),
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .unwrap();
        let input = Tensor::new(input_data.into_dyn());

        // 2x2 filter
        let filter_data =
            Array4::from_shape_vec((1, 1, 2, 2), vec![1.0f32, 0.0, 0.0, 1.0]).unwrap();
        let filter = Tensor::new(filter_data.into_dyn());

        // Stride 2 should give 2x2 output from 4x4 input with 2x2 kernel
        let result = input.conv2d(&filter, (2, 2), (0, 0)).unwrap();
        assert_eq!(result.shape(), &[1, 1, 2, 2]);

        // Should sample at positions (0,0), (0,2), (2,0), (2,2)
        // Values: 1+6=7, 3+8=11, 9+14=23, 11+16=27
        let expected_data =
            Array4::from_shape_vec((1, 1, 2, 2), vec![7.0f32, 11.0, 23.0, 27.0]).unwrap();
        let expected = Tensor::new(expected_data.into_dyn());

        assert_tensors_close(&result, &expected, 1e-6f32);
    }

    #[test]
    fn test_conv2d_multiple_channels() {
        // 2 input channels, 2x2 spatial
        let input_data = Array4::from_shape_vec(
            (1, 2, 2, 2),
            vec![
                1.0f32, 2.0, 3.0, 4.0, // Channel 0
                5.0, 6.0, 7.0, 8.0,
            ], // Channel 1
        )
        .unwrap();
        let input = Tensor::new(input_data.into_dyn());

        // 1 output channel, 2 input channels, 2x2 kernel
        let filter_data = Array4::from_shape_vec(
            (1, 2, 2, 2),
            vec![
                1.0f32, 0.0, 0.0, 1.0, // For input channel 0
                0.5, 0.5, 0.5, 0.5,
            ], // For input channel 1
        )
        .unwrap();
        let filter = Tensor::new(filter_data.into_dyn());

        let result = input.conv2d(&filter, (1, 1), (0, 0)).unwrap();
        assert_eq!(result.shape(), &[1, 1, 1, 1]);

        // Expected: (1*1 + 4*1) + (5*0.5 + 6*0.5 + 7*0.5 + 8*0.5) = 5 + 13 = 18
        let result_val = result.data.as_slice().unwrap()[0];
        assert!((result_val - 18.0).abs() < 1e-6);
    }

    #[test]
    fn test_depthwise_conv2d_basic() {
        // 2 channels, 3x3 spatial
        let input_data = Array4::from_shape_vec(
            (1, 2, 3, 3),
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // Channel 0
                10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            ], // Channel 1
        )
        .unwrap();
        let input = Tensor::new(input_data.into_dyn());

        // 2 filters, one per channel, 2x2 each
        let filter_data = Array4::from_shape_vec(
            (2, 1, 2, 2),
            vec![
                1.0f32, 0.0, 0.0, 1.0, // Filter for channel 0
                0.5, 0.5, 0.5, 0.5,
            ], // Filter for channel 1
        )
        .unwrap();
        let filter = Tensor::new(filter_data.into_dyn());

        let result = input.depthwise_conv2d(&filter, (1, 1), (0, 0)).unwrap();
        assert_eq!(result.shape(), &[1, 2, 2, 2]); // Same number of channels as input

        let result_data = result.data.as_slice().unwrap();

        // Channel 0: 1*1 + 5*1 = 6, 2*1 + 6*1 = 8, etc.
        assert!((result_data[0] - 6.0).abs() < 1e-6); // (0,0) in channel 0
        assert!((result_data[1] - 8.0).abs() < 1e-6); // (0,1) in channel 0

        // Channel 1: (10+11+13+14)*0.5 = 24
        assert!((result_data[4] - 24.0).abs() < 1e-6); // (0,0) in channel 1
    }

    #[test]
    fn test_depthwise_conv2d_channel_independence() {
        // Test that channels are processed independently
        let input_data = Array4::from_shape_vec(
            (1, 2, 2, 2),
            vec![
                1.0f32, 0.0, 0.0, 0.0, // Channel 0: only top-left has value
                0.0, 0.0, 0.0, 2.0,
            ], // Channel 1: only bottom-right has value
        )
        .unwrap();
        let input = Tensor::new(input_data.into_dyn());

        // Identity-like filters
        let filter_data = Array4::from_shape_vec(
            (2, 1, 1, 1),
            vec![1.0f32, 1.0f32], // 1x1 filters for both channels
        )
        .unwrap();
        let filter = Tensor::new(filter_data.into_dyn());

        let result = input.depthwise_conv2d(&filter, (1, 1), (0, 0)).unwrap();
        let result_data = result.data.as_slice().unwrap();

        // Channel 0: input[0,0,0,0] = 1.0 should be at output[0,0,0,0] (index 0)
        assert!((result_data[0] - 1.0).abs() < 1e-6);

        // Channel 1: input[0,1,1,1] = 2.0 should be at output[0,1,1,1] (index 7)
        assert!((result_data[7] - 2.0).abs() < 1e-6);

        // Cross-contamination check: other positions should be 0
        assert!((result_data[1] - 0.0).abs() < 1e-6);
        assert!((result_data[4] - 0.0).abs() < 1e-6); // Channel 1, position [0,0]
    }

    #[test]
    fn test_depthwise_separable_conv2d() {
        // 2 channels, 3x3 input
        let input_data = Array4::from_shape_vec(
            (1, 2, 3, 3),
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // Channel 0
                9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
            ], // Channel 1
        )
        .unwrap();
        let input = Tensor::new(input_data.into_dyn());

        // Depthwise filters: 2 channels, 2x2 kernels
        let depthwise_filter_data = Array4::from_shape_vec(
            (2, 1, 2, 2),
            vec![
                1.0f32, 0.0, 0.0, 1.0, // Channel 0 filter
                1.0, 1.0, 1.0, 1.0,
            ], // Channel 1 filter
        )
        .unwrap();
        let depthwise_filter = Tensor::new(depthwise_filter_data.into_dyn());

        // Pointwise filters: 3 output channels from 2 input channels, 1x1 kernels
        let pointwise_filter_data = Array4::from_shape_vec(
            (3, 2, 1, 1),
            vec![
                1.0f32, 0.5, // Output channel 0: 1.0*ch0 + 0.5*ch1
                0.0, 1.0, // Output channel 1: 0.0*ch0 + 1.0*ch1
                1.0, 1.0,
            ], // Output channel 2: 1.0*ch0 + 1.0*ch1
        )
        .unwrap();
        let pointwise_filter = Tensor::new(pointwise_filter_data.into_dyn());

        let result = input
            .depthwise_separable_conv2d(&depthwise_filter, &pointwise_filter, (1, 1), (0, 0))
            .unwrap();

        // Should have 3 output channels, 2x2 spatial
        assert_eq!(result.shape(), &[1, 3, 2, 2]);

        // Just verify it runs without error and produces reasonable output
        let result_data = result.data.as_slice().unwrap();
        assert!(result_data.iter().all(|&x| x.is_finite()));
    }
}
