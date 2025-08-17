use ferrox::backend::{Device, FerroxCudaF, Tensor};
use ferrox::dataset::TensorDataset;
use ferrox::FerroxN;
use rand_distr::{Distribution, Normal};

/// Generate synthetic regression dataset
/// Creates simple linear relationship: y = 0.3*x1 + 0.1*x2 + noise
#[allow(dead_code)]
pub fn generate_regression_data<T>(
    num_samples: usize,
    input_size: usize,
    device: Device,
) -> Result<TensorDataset<T>, String>
where
    T: FerroxCudaF,
{
    let mut rng = rand::rng();
    let normal = Normal::new(0.0, 0.5).unwrap();

    let mut input_data = Vec::with_capacity(num_samples * input_size);
    let mut target_data = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let mut features = Vec::new();

        // Generate input features
        for _ in 0..input_size {
            let value = normal.sample(&mut rng) as f64;
            features.push(value);
            input_data.push(<T as FerroxN>::from_f64(value).ok_or("Failed to convert input data")?);
        }

        // Simple linear relationship with minimal noise
        let x1 = features[0];
        let x2 = if input_size > 1 { features[1] } else { 0.0 };
        let noise = normal.sample(&mut rng) * 0.05;
        let target = 0.3 * x1 + 0.1 * x2 + noise;

        target_data.push(<T as FerroxN>::from_f64(target).ok_or("Failed to convert target data")?);
    }

    let inputs = Tensor::from_vec_with_device(input_data, &[num_samples, input_size], device)?;
    let targets = Tensor::from_vec_with_device(target_data, &[num_samples, 1], device)?;

    TensorDataset::from_tensor(inputs, targets)
}

/// Generate synthetic binary classification dataset
#[allow(dead_code)]
pub fn generate_binary_classification_data<T>(
    num_samples: usize,
    input_size: usize,
    device: Device,
) -> Result<TensorDataset<T>, String>
where
    T: FerroxCudaF,
{
    let mut rng = rand::rng();
    let normal = Normal::new(0.0, 0.8).unwrap();

    let mut input_data = Vec::with_capacity(num_samples * input_size);
    let mut target_data = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let mut features = Vec::new();

        // Generate input features
        for _ in 0..input_size {
            let value = normal.sample(&mut rng) as f64;
            features.push(value);
            input_data.push(<T as FerroxN>::from_f64(value).ok_or("Failed to convert input data")?);
        }

        // Create linear decision boundary: y = sign(w1*x1 + w2*x2 + bias)
        let decision_value = 0.7 * features[0] + 0.5 * features.get(1).unwrap_or(&0.0) - 0.1;
        let label = if decision_value > 0.0 { 1.0 } else { 0.0 };

        target_data.push(<T as FerroxN>::from_f64(label).ok_or("Failed to convert target data")?);
    }

    let inputs = Tensor::from_vec_with_device(input_data, &[num_samples, input_size], device)?;
    let targets = Tensor::from_vec_with_device(target_data, &[num_samples, 1], device)?;

    TensorDataset::from_tensor(inputs, targets)
}

/// Generate synthetic multiclass classification dataset
/// Creates clear patterns with proper class separation
#[allow(dead_code)]
pub fn generate_multiclass_classification_data<T>(
    num_samples: usize,
    input_size: usize,
    num_classes: usize,
    device: Device,
) -> Result<TensorDataset<T>, String>
where
    T: FerroxCudaF,
{
    let mut input_data = Vec::with_capacity(num_samples * input_size);
    let mut target_data = Vec::with_capacity(num_samples * num_classes);

    for i in 0..num_samples {
        let class = i % num_classes;

        // Create clear patterns for each class
        for j in 0..input_size {
            let value = match (class, j) {
                // Class 0: first feature = +20, others = -10
                (0, 0) => 20.0,
                (0, _) => -10.0,

                // Class 1: second feature = +20, others = -10
                (1, 1) => 20.0,
                (1, _) => -10.0,

                // Class 2: third feature = +20, others = -10
                (2, 2) => 20.0,
                (2, _) => -10.0,

                // For additional classes, use modulo pattern
                (c, f) if c == f => 20.0,
                _ => -10.0,
            };

            input_data.push(<T as FerroxN>::from_f64(value).ok_or("Input conversion failed")?);
        }

        // One-hot targets
        for c in 0..num_classes {
            target_data.push(
                <T as FerroxN>::from_f64(if c == class { 1.0 } else { 0.0 })
                    .ok_or("Target conversion failed")?,
            );
        }
    }

    let inputs = Tensor::from_vec_with_device(input_data, &[num_samples, input_size], device)?;
    let targets = Tensor::from_vec_with_device(target_data, &[num_samples, num_classes], device)?;

    TensorDataset::from_tensor(inputs, targets)
}

/// Generate synthetic image data for classification
/// Creates simple patterns: class 0 = horizontal stripes, class 1 = vertical stripes, class 2 = diagonal
#[allow(dead_code)]
pub fn generate_synthetic_image_data<T>(
    num_samples: usize,
    num_classes: usize,
    image_size: usize,
    channels: usize,
    device: Device,
) -> Result<TensorDataset<T>, String>
where
    T: FerroxCudaF,
{
    let mut input_data = Vec::with_capacity(num_samples * channels * image_size * image_size);
    let mut target_data = Vec::with_capacity(num_samples * num_classes);

    for i in 0..num_samples {
        let class = i % num_classes;

        // Generate image for each channel
        for _c in 0..channels {
            // Create VERY simple, high-contrast patterns
            for y in 0..image_size {
                for x in 0..image_size {
                    let pixel_value = match class {
                        // Class 0: Top half bright, bottom half dark
                        0 => {
                            if y < image_size / 2 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        // Class 1: Left half bright, right half dark
                        1 => {
                            if x < image_size / 2 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        // Class 2: Center bright, edges dark
                        2 => {
                            let center_x = image_size / 2;
                            let center_y = image_size / 2;
                            let quarter_size = image_size / 4;

                            if (x as i32 - center_x as i32).abs() < quarter_size as i32
                                && (y as i32 - center_y as i32).abs() < quarter_size as i32
                            {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        // Additional classes: simple diagonal split
                        _ => {
                            if x + y < image_size {
                                1.0
                            } else {
                                0.0
                            }
                        }
                    };

                    input_data.push(
                        <T as FerroxN>::from_f64(pixel_value)
                            .ok_or("Failed to convert pixel value")?,
                    );
                }
            }
        }

        // Create one-hot target
        for c in 0..num_classes {
            target_data.push(
                <T as FerroxN>::from_f64(if c == class { 1.0 } else { 0.0 })
                    .ok_or("Failed to convert target value")?,
            );
        }
    }

    let inputs = Tensor::from_vec_with_device(
        input_data,
        &[num_samples, channels, image_size, image_size],
        device,
    )?;
    let targets = Tensor::from_vec_with_device(target_data, &[num_samples, num_classes], device)?;

    TensorDataset::from_tensor(inputs, targets)
}
