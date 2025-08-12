// src/data/dataset.rs
use crate::backend::{Device, FerroxCudaF, Tensor};





pub trait Dataset<T>
where
    T: FerroxCudaF,
{
    /// Get a single sample by index
    fn get_item(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>), String>;

    /// Total number of samples in the dataset
    fn len(&self) -> usize;

    /// Check if dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    
}

/// Basic tensor dataset for supervised learning
#[derive(Debug, Clone)]
pub struct TensorDataset<T>
where
    T: FerroxCudaF,
{
    /// Input tensors - shape: [num_samples, *input_dims]
    inputs: Tensor<T>,
    /// Target tensors - shape: [num_samples, *target_dims]
    targets: Tensor<T>,
    /// Number of samples in dataset
    num_samples: usize,
}

impl<T> TensorDataset<T>
where
    T: FerroxCudaF + Clone,
{
    /// Create new tensor dataset from input and target tensors
    pub fn new(inputs: Tensor<T>, targets: Tensor<T>) -> Result<Self, String> {
        if inputs.shape()[0] != targets.shape()[0] {
            return Err(format!(
                "Input batch size {} doesn't match target batch size {}",
                inputs.shape()[0], targets.shape()[0]
            ));
        }

        let device = inputs.device();
        let num_samples = inputs.shape()[0];

        let targets = if targets.device() != device {
            targets.to_device(device)?
        } else {
            targets
        };

        Ok(Self {
            inputs,
            targets,
            num_samples,
        })
    }

    /// Extract slice data for sample at given index from tensor
    fn extract_slice(&self, tensor: &Tensor<T>, index: usize) -> Result<&[T], String> {
        let elements_per_sample: usize = tensor.shape()[1..].iter().product();
        let start = index * elements_per_sample;
        tensor.slice_range(start, elements_per_sample)
    }

    /// Create tensor from slice data with sample shape
    fn create_sample(&self, slice_data: &[T], sample_shape: &[usize]) -> Result<Tensor<T>, String> {
        Tensor::from_vec_with_device(slice_data.to_vec(), sample_shape, self.device())
    }

    /// Get device where dataset tensors are stored
    pub fn device(&self) -> Device {
        self.inputs.device()
    }

    /// Move entire dataset to different device
    pub fn to_device(self, target_device: Device) -> Result<Self, String> {
        if self.inputs.device() == target_device {
            return Ok(self);
        }

        let inputs = self.inputs.to_device(target_device)?;
        let targets = self.targets.to_device(target_device)?;

        Ok(Self {
            inputs,
            targets,
            num_samples: self.num_samples,
        })
    }

    /// Get input tensor shape excluding batch dimension
    pub fn input_shape(&self) -> &[usize] {
        &self.inputs.shape()[1..]
    }

    /// Get target tensor shape excluding batch dimension
    pub fn target_shape(&self) -> &[usize] {
        &self.targets.shape()[1..]
    }
}

impl<T> Dataset<T> for TensorDataset<T>
where
    T: FerroxCudaF + Clone,
{
    /// Extract single sample using tensor slicing operations
    fn get_item(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>), String> {
        if index >= self.num_samples {
            return Err(format!(
                "Index {} out of bounds for dataset with {} samples",
                index, self.num_samples
            ));
        }

        let input_slice = self.extract_slice(&self.inputs, index)?;
        let target_slice = self.extract_slice(&self.targets, index)?;

        let input_sample = self.create_sample(input_slice, self.input_shape())?;
        let target_sample = self.create_sample(target_slice, self.target_shape())?;

        Ok((input_sample, target_sample))
    }

    fn len(&self) -> usize {
        self.num_samples
    }
}
