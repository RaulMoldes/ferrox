// src/data/dataset.rs
use crate::backend::{FerroxCudaF, Tensor};

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
                inputs.shape()[0],
                targets.shape()[0]
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

    // Create dataset from single tensor - for unsupervised learning
    pub fn from_tensor(inputs: Tensor<T>, targets: Tensor<T>) -> Result<Self, String> {
        if inputs.shape()[0] != targets.shape()[0] {
            return Err(format!(
                "Input batch size {} doesn't match target batch size {}",
                inputs.shape()[0],
                targets.shape()[0]
            ));
        }

        let device = inputs.device();
        let num_samples = inputs.shape()[0];

        // Ensure targets are on same device as inputs
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

    // Create batched dataset using tensor's into_batches method
    pub fn into_batches(
        self,
        batch_size: usize,
        drop_last: bool,
    ) -> Result<BatchedDataset<T>, String> {
        if batch_size == 0 {
            return Err("Batch size must be greater than 0".to_string());
        }

        // Use tensor's into_batches method to create input batches
        let input_batches = self.inputs.into_batches(batch_size, drop_last)?;
        let target_batches = self.targets.into_batches(batch_size, drop_last)?;

        Ok(BatchedDataset::new(input_batches, target_batches))
    }
}

impl<T> Dataset<T> for TensorDataset<T>
where
    T: FerroxCudaF + Clone,
{
    /// Extract single sample using tensor slicing operations
    fn get_item(&self, _index: usize) -> Result<(Tensor<T>, Tensor<T>), String> {
        Ok((self.inputs.clone(), self.targets.clone()))
    }

    fn len(&self) -> usize {
        self.num_samples
    }
}

#[derive(Debug)]
pub struct BatchedDataset<T>
where
    T: FerroxCudaF,
{
    input_batches: Vec<Tensor<T>>,
    target_batches: Vec<Tensor<T>>,
    num_batches: usize,
}

impl<T> BatchedDataset<T>
where
    T: FerroxCudaF + Clone,
{
    // Create from pre-computed batch vectors
    fn new(input_batches: Vec<Tensor<T>>, target_batches: Vec<Tensor<T>>) -> Self {
        let num_batches = input_batches.len();
        Self {
            input_batches,
            target_batches,
            num_batches,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Tensor<T>, &Tensor<T>)> {
        self.input_batches.iter().zip(self.target_batches.iter())
    }

    // Check if empty
    pub fn is_empty(&self) -> bool {
        self.num_batches == 0
    }
}

impl<T> Dataset<T> for BatchedDataset<T>
where
    T: FerroxCudaF + Clone,
{
    /// Extract single sample using tensor slicing operations
    fn get_item(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>), String> {
        if index < self.len() {
            let target = self.target_batches[index].clone();
            let input = self.input_batches[index].clone();
            Ok((input, target))
        } else {
            Err("Out of bounds error. Index is too big".to_string())
        }
    }

    // Get total number of batches
    fn len(&self) -> usize {
        self.num_batches
    }
}

impl<T> IntoIterator for BatchedDataset<T>
where
    T: FerroxCudaF + Clone,
{
    type Item = (Tensor<T>, Tensor<T>);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        // zip de los vectores para emparejar input y target en tuplas
        self.input_batches
            .into_iter()
            .zip(self.target_batches)
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<'a, T> IntoIterator for &'a BatchedDataset<T>
where
    T: FerroxCudaF + Clone,
{
    type Item = (&'a Tensor<T>, &'a Tensor<T>);
    type IntoIter =
        std::iter::Zip<std::slice::Iter<'a, Tensor<T>>, std::slice::Iter<'a, Tensor<T>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.input_batches.iter().zip(self.target_batches.iter())
    }
}
