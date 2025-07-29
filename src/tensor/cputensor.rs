use crate::backend::number::{CPUNumber, GPUFloat};
use crate::backend::{Device, default_device};
use crate::tensor::storage::{CPUOwnedStorage, StorageBackend};
use ndarray::{Array, ArrayD, Axis, IxDyn};
use std::ops::{Index, IndexMut};

// Tensor wrapper to handle dynamic arrays more elegantly
#[derive(Debug)]
pub struct CPUTensor<T>
where
    T: GPUFloat,
{
    // This `data` field is the main data storage of the tensor on CPU.
    pub data: ArrayD<T>, // As I documented in the device module, this will be changed toa generic type <T>
    // This way I will be able to use different data types in the future.
    // For now, we will keep it as f64 for simplicity.
    pub device: Device,
    storage: Option<Box<dyn StorageBackend<T>>>,
}

impl<T> Clone for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    fn clone(&self) -> Self {
        let storage = if let Some(storage_ref) = &self.storage {
            // Try to downcast to CPUOwnedStorage if possible
            if let Some(any_ref) = storage_ref.as_any() {
                if let Some(cpu_storage) = any_ref.downcast_ref::<CPUOwnedStorage<T>>() {
                    Some(Box::new(cpu_storage.clone()) as Box<dyn StorageBackend<T>>)
                } else if let Ok(storage) = storage_ref.clone_storage() {
                    Some(storage)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        Self {
            data: self.data.clone(),
            device: self.device.clone(),
            storage,
        }
    }
}

#[cfg(feature = "cuda")]
impl<T> CPUTensor<T>
where
    T: GPUFloat,
{
    /// Helper to eliminate repeated backend access pattern.
    /// This removes the need to repeatedly call `get_backend()` in every method.
    /// Allows us to execute a closure with the CUDA backend.
    fn with_cuda_backend<F, R>(&self, f: F) -> Result<R, String>
    where
        F: FnOnce(&crate::backend::cuda::CudaBackend) -> Result<R, String>,
    {
        let backend = get_backend();
        let cuda_backend = backend.cuda_backend().ok_or("CUDA backend not available")?;
        f(cuda_backend)
    }
}

// Main implementation block with basic operations
impl<T> CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    // Basically a constructor for the Tensor struct.
    // It takes an ArrayD<T> and returns a Tensor with storage backend.
    // The device is set to the default device if not provided.
    // This is similar to how PyTorch and TensorFlow work, where the device is set to the default device if not specified.
    // We could also take advantage of Rust's default trait to set the device to the default device if not provided.
    // In the course that I am following (dlsyscourse.org), the Tensor is actually inheriting from a graph node.
    // In rust we do not have inheritance, so we will just use composition.
    // Additionally, I decided to reverse the composition hierarchy as opposite to the course,
    // so the tensor is used to represent data, and is the main layer of abstraction over the device.
    // The graph node is a separate struct (check src/graph/node.rs), which indeed has a data property, that in ferrrox will be a tensor.
    // In the course, the graph node was the main layer of abstraction over the device, and the tensor inherits from it.
    pub fn new(data: ArrayD<T>) -> Self {
        // Create storage backend from the data
        let storage = crate::tensor::storage::CPUOwnedStorage::new(data.clone());

        Self {
            data, // Keep for now - TODO: Remove this field in final migration step
            device: crate::backend::default_device(),
            storage: Some(Box::new(storage)),
        }
    }

    // Internal method to create tensor with specific storage backend
    // This will become the primary constructor once data field is removed
    fn new_with_storage<S>(storage: S) -> Result<Self, String>
    where
        S: crate::tensor::storage::StorageBackend<T> + 'static,
    {
        // Extract data from storage for backward compatibility
        let data = match storage.cpu_data() {
            Ok(cpu_data) => cpu_data.clone(),
            Err(_) => {
                // For GPU storage, create empty CPU data as placeholder
                ndarray::ArrayD::zeros(ndarray::IxDyn(&[]))
            }
        };

        Ok(Self {
            data, // TODO: Remove this field in final migration step
            device: crate::backend::default_device(),
            storage: Some(Box::new(storage)),
        })
    }

    // Creates a new tensor with the given data and device.
    // This is useful when you want to create a tensor with a specific device, for example, when you want to use a GPU.
    // In the future, we could also add a method to create a tensor with a specific data type, but for now we will keep it simple.
    // The device is set to the default device if not provided.
    // This is similar to how PyTorch and TensorFlow work, where the device is set to the default device if not specified.
    // Ideally, we should not be bound to ndarray backend here because it defaults to CPU, but it is okay for now as I prefer to focus more on the automatic differentiation engine thing.
    pub fn new_with_device(data: ArrayD<T>, device: crate::backend::Device) -> Self {
        let storage = crate::tensor::storage::CPUOwnedStorage::new(data.clone());
        Self {
            data, // TODO: Remove this field in final migration step
            device,
            storage: Some(Box::new(storage)),
        }
    }

    // Random numbers
    // Generates a tensor with random numbers from a normal distribution using the storage backend approach
    pub fn randn(shape: &[usize]) -> Self {
        let device = crate::backend::default_device();

        let data_f64 = device.randn(shape);
        let data =
            data_f64.mapv(|x| <T as crate::backend::number::CPUNumber>::from_f64(x).unwrap());
        let storage = crate::tensor::storage::CPUOwnedStorage::new(data.clone());
        Self {
            data, // TODO: Remove this field in final migration step
            device,
            storage: Some(Box::new(storage)),
        }
    }

    // Random numbers with specific device
    pub fn randn_with_device(shape: &[usize], device: crate::backend::Device) -> Self {
        // Generates a tensor with random numbers from a normal distribution.
        let data_f64 = device.randn(shape);
        let data =
            data_f64.mapv(|x| <T as crate::backend::number::CPUNumber>::from_f64(x).unwrap());
        let storage = crate::tensor::storage::CPUOwnedStorage::new(data.clone());
        Self {
            data, // TODO: Remove this field in final migration step
            device,
            storage: Some(Box::new(storage)),
        }
    }

    // Random integer numbers
    // Generates a tensor with random integer numbers using the storage backend approach
    pub fn randint(shape: &[usize]) -> Self {
        let device = crate::backend::default_device();
        let data_i64 = device.randint(shape);
        let data =
            data_i64.mapv(|x| <T as crate::backend::number::CPUNumber>::from_i64(x).unwrap());
        let storage = crate::tensor::storage::CPUOwnedStorage::new(data.clone());
        Self {
            data, // TODO: Remove this field in final migration step
            device,
            storage: Some(Box::new(storage)),
        }
    }

    // Random integer numbers with specific device
    pub fn randint_with_device(shape: &[usize], device: crate::backend::Device) -> Self {
        // Generates a tensor with random integer numbers.
        let data_i64 = device.randint(shape);
        let data =
            data_i64.mapv(|x| <T as crate::backend::number::CPUNumber>::from_i64(x).unwrap());
        let storage = crate::tensor::storage::CPUOwnedStorage::new(data.clone());
        Self {
            data, // TODO: Remove this field in final migration step
            device,
            storage: Some(Box::new(storage)),
        }
    }

    // Creates a tensor from a Rust vector. Again we are bound to ndarray backend here, but it is okay for now.
    // This function takes a vector of T and a shape, and returns a tensor with the given shape using storage backend.
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self, String> {
        let total_elements: usize = shape.iter().product();
        if data.len() != total_elements {
            return Err(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                total_elements
            ));
        }

        match ndarray::Array::from_shape_vec(ndarray::IxDyn(shape), data) {
            Ok(array) => {
                // Create storage backend for the array
                let storage = crate::tensor::storage::CPUOwnedStorage::new(array.clone());
                Ok(Self {
                    data: array, // TODO: Remove this field in final migration step
                    device: crate::backend::default_device(),
                    storage: Some(Box::new(storage)),
                })
            }
            Err(e) => Err(format!("Failed to create tensor: {e}")),
        }
    }

    // Creates a tensor from a Rust vector with specific device
    pub fn from_vec_with_device(
        data: Vec<T>,
        shape: &[usize],
        device: crate::backend::Device,
    ) -> Result<Self, String> {
        let total_elements: usize = shape.iter().product();
        if data.len() != total_elements {
            return Err(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                total_elements
            ));
        }

        match ndarray::Array::from_shape_vec(ndarray::IxDyn(shape), data) {
            Ok(array) => {
                // Create storage backend for the array
                let storage = crate::tensor::storage::CPUOwnedStorage::new(array.clone());
                Ok(Self {
                    data: array, // TODO: Remove this field in final migration step
                    device,
                    storage: Some(Box::new(storage)),
                })
            }
            Err(e) => Err(format!("Failed to create tensor: {e}")),
        }
    }

    // Create tensor from existing storage backend (internal use)
    // This is the most direct way to create tensors and will become primary after migration
    pub(crate) fn from_storage_backend(
        storage: Box<dyn crate::tensor::storage::StorageBackend<T>>,
        device: crate::backend::Device,
    ) -> Result<Self, String> {
        let data = match storage.cpu_data() {
            Ok(cpu_data) => cpu_data.clone(),
            Err(_) => ndarray::ArrayD::zeros(ndarray::IxDyn(&[])),
        };

        Ok(Self {
            data,
            device,
            storage: Some(storage),
        })
    }

    // I decided not to implement the empty() function as it is useless in practice.
    // The empty function in numpy creates an uninitialized array, which is unsafe in Rust.
    // Instead, we will use the zeros() function to create a tensor with zeroes.
    // If you want to use uninitialized arrays, you can use `Array::uninit` but it is unsafe.

    // Some utility functions to get information about the tensor.
    // These functions are similar to the ones in PyTorch and TensorFlow, and they return the shape, number of dimensions, length, data, and device of the tensor.
    pub fn shape(&self) -> &[usize] {
        self.storage
            .as_ref()
            .expect("Tensor must have storage backend")
            .shape()
    }

    // Get the tensor dimensions from the underlying storage backend
    pub fn ndim(&self) -> usize {
        self.storage
            .as_ref()
            .expect("Tensor must have storage backend")
            .ndim()
    }

    // Get the total tensor size from the underlying storage backend
    pub fn size(&self) -> usize {
        self.storage
            .as_ref()
            .expect("Tensor must have storage backend")
            .size()
    }

    /// Check if tensor is on CUDA
    pub fn is_cuda(&self) -> bool {
        if !self.device.is_cuda() {
            return false;
        }

        self.storage.as_ref().map(|s| s.is_gpu()).unwrap_or(false)
    }

    // Get CPU data reference
    /// This is be the main way to access tensor data
    pub fn cpu_data(&self) -> Result<&ArrayD<T>, String> {
        self.storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?
            .cpu_data()
    }

    /// Get mutable CPU data
    pub fn cpu_data_mut(&mut self) -> Result<&mut ArrayD<T>, String> {
        self.storage
            .as_mut()
            .ok_or("Tensor has no storage backend")?
            .cpu_data_mut()
    }

    /// Check if tensor owns its data
    pub fn owns_data(&self) -> bool {
        self.storage
            .as_ref()
            .map(|s| s.owns_data())
            .unwrap_or(false)
    }

    /// Deprecated: Legacy data access for backward compatibility
    /// This will be removed once migration is complete
    #[deprecated(note = "Use cpu_data() or get_cpu_data() instead")]
    pub fn data(&self) -> &ArrayD<T> {
        // Try to get from storage first, fallback to direct field access
        if let Ok(data) = self.cpu_data() {
            data
        } else {
            &self.data // This will be removed eventually
        }
    }

    /// Safe data access with GPU->CPU sync when needed
    pub fn get_cpu_data(&self) -> Result<std::borrow::Cow<ArrayD<T>>, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        // Try to get CPU data directly
        match storage.cpu_data() {
            Ok(data) => Ok(std::borrow::Cow::Borrowed(data)),
            Err(_) if storage.is_gpu() => {
                // GPU storage needs sync - create temporary CPU data
                #[cfg(feature = "cuda")]
                {
                    if let Some(gpu_storage) = storage.as_any().and_then(|any| {
                        any.downcast_ref::<crate::tensor::storage::GPUOwnedStorage<T>>()
                    }) {
                        use crate::backend::manager::get_backend;
                        let backend = get_backend();
                        let cuda_backend =
                            backend.cuda_backend().ok_or("CUDA backend not available")?;

                        let host_data = gpu_storage.cuda_data.to_vec(cuda_backend)?;
                        let cpu_array = ndarray::ArrayD::from_shape_vec(
                            ndarray::IxDyn(gpu_storage.cuda_data.shape()),
                            host_data,
                        )
                        .map_err(|e| format!("Failed to create CPU array: {}", e))?;

                        return Ok(std::borrow::Cow::Owned(cpu_array));
                    }
                }
                Err("Cannot sync GPU data without CUDA feature".to_string())
            }
            Err(e) => Err(e),
        }
    }

    /// Get number of elements using storage backend
    pub fn len(&self) -> usize {
        self.storage.as_ref().map(|s| s.size()).unwrap_or(0)
    }

    /// Check if tensor is empty using storage backend
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get device reference
    pub fn device(&self) -> &crate::backend::Device {
        &self.device
    }

    /// Extract data from tensor, consuming it
    /// This method handles both CPU and GPU tensors by syncing to CPU first
    pub fn into_data(self) -> Result<ArrayD<T>, String> {
        let storage = self.storage.ok_or("Tensor has no storage backend")?;

        // Try to get CPU data directly
        match storage.cpu_data() {
            Ok(data) => Ok(data.clone()),
            Err(_) if storage.is_gpu() => {
                // GPU storage - need to sync to CPU first
                #[cfg(feature = "cuda")]
                {
                    if let Some(gpu_storage) = storage.as_any().and_then(|any| {
                        any.downcast_ref::<crate::tensor::storage::GPUOwnedStorage<T>>()
                    }) {
                        use crate::backend::manager::get_backend;
                        let backend = get_backend();
                        let cuda_backend =
                            backend.cuda_backend().ok_or("CUDA backend not available")?;

                        let host_data = gpu_storage.cuda_data.to_vec(cuda_backend)?;
                        let cpu_array = ndarray::ArrayD::from_shape_vec(
                            ndarray::IxDyn(gpu_storage.cuda_data.shape()),
                            host_data,
                        )
                        .map_err(|e| format!("Failed to create CPU array: {}", e))?;

                        return Ok(cpu_array);
                    }
                }
                Err("Cannot extract GPU data without CUDA feature".to_string())
            }
            Err(e) => Err(format!("Failed to extract data: {}", e)),
        }
    }
}

impl<T> CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    // Element-wise operations.
    // These are operations that are applied to each element of the tensor.
    // They are easily parallelizable and can be implemented using ndarray's mapv method.
    // The mapv method applies a function to each element of the array and returns a new array with the results.
    pub fn add(&self, other: &Self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;
        let other_storage = other
            .storage
            .as_ref()
            .ok_or("Other tensor has no storage backend")?;

        // Use the ElementwiseOps trait - works for both CPU and future CUDA implementations
        let result_storage = storage.add(other_storage.as_ref())?;

        // Create new tensor with result storage, preserving device context
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Element-wise subtraction using storage trait
    pub fn sub(&self, other: &Self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;
        let other_storage = other
            .storage
            .as_ref()
            .ok_or("Other tensor has no storage backend")?;

        let result_storage = storage.sub(other_storage.as_ref())?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Element-wise multiplication using storage trait
    pub fn mul(&self, other: &Self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;
        let other_storage = other
            .storage
            .as_ref()
            .ok_or("Other tensor has no storage backend")?;

        let result_storage = storage.mul(other_storage.as_ref())?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Element-wise division using storage trait
    pub fn div(&self, other: &Self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;
        let other_storage = other
            .storage
            .as_ref()
            .ok_or("Other tensor has no storage backend")?;

        let result_storage = storage.div(other_storage.as_ref())?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Element-wise minimum using storage trait
    pub fn min(&self, other: &Self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;
        let other_storage = other
            .storage
            .as_ref()
            .ok_or("Other tensor has no storage backend")?;

        let result_storage = storage.min(other_storage.as_ref())?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Element-wise maximum using storage trait
    pub fn max(&self, other: &Self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;
        let other_storage = other
            .storage
            .as_ref()
            .ok_or("Other tensor has no storage backend")?;

        let result_storage = storage.max(other_storage.as_ref())?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Scalar addition - more efficient than broadcasting
    pub fn add_scalar(&self, scalar: T) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.add_scalar(scalar)?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: T) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.mul_scalar(scalar)?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Scalar substraction
    pub fn sub_scalar(&self, scalar: T) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.sub_scalar(scalar)?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Scalar division
    pub fn div_scalar(&self, scalar: T) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.div_scalar(scalar)?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Unary negation - replaces your negate() method
    pub fn neg(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.neg()?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Element-wise absolute value using storage trait
    pub fn abs(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.abs()?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Element-wise clamp using storage trait
    pub fn clamp(&self, min_val: T, max_val: T) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.clamp(min_val, max_val)?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Element-wise square root
    /// Returns a new tensor with square root values
    /// Validates that all values are non-negative
    pub fn sqrt(&self) -> Result<Self, String> {
        // Check for negative values first
        let has_negative = self.data.iter().any(|&x| x < <T as CPUNumber>::zero());
        if has_negative {
            return Err("Cannot compute square root of negative values".to_string());
        }

        let result_data: Vec<T> = self.data.iter().map(|&x| x.sqrt()).collect();

        let result_array = ndarray::Array::from_shape_vec(self.data.raw_dim(), result_data)
            .map_err(|e| format!("Failed to create result tensor: {e}",))?;

        Ok(Self::new_with_device(result_array, self.device.clone()))
    }
}

impl<T> CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    /// Element-wise greater than or equal comparison using storage trait
    pub fn greater_equal(&self, other: &Self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;
        let other_storage = other
            .storage
            .as_ref()
            .ok_or("Other tensor has no storage backend")?;

        let result_storage = storage.greater_equal(other_storage.as_ref())?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Element-wise less than or equal comparison using storage trait
    pub fn less_equal(&self, other: &Self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;
        let other_storage = other
            .storage
            .as_ref()
            .ok_or("Other tensor has no storage backend")?;

        let result_storage = storage.less_equal(other_storage.as_ref())?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Element-wise equality comparison using storage trait
    pub fn equal(&self, other: &Self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;
        let other_storage = other
            .storage
            .as_ref()
            .ok_or("Other tensor has no storage backend")?;

        let result_storage = storage.equal(other_storage.as_ref())?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Logical NOT operation using storage trait
    pub fn logical_not(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.logical_not()?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Range check operation using storage trait
    pub fn in_range(&self, min_val: T, max_val: T) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.in_range(min_val, max_val)?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Sign function using storage trait
    pub fn sign(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.sign()?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    // Detach operation - creates a new tensor that shares data but detaches from graph
    // Need to check if this is the right way to do it.
    // In Pytorch i think the detach operation sets the requires_grad flag to false, but we don't have that concept at the tensor level.
    // We can just return a new tensor with the same data and device, but without any gradient tracking.
    pub fn detach(&self) -> CPUTensor<T> {
        CPUTensor::new_with_device(self.data.clone(), self.device.clone())
    }

    /// Returns an iterator over elements in row-major order
    pub fn iter(&self) -> ndarray::iter::Iter<'_, T, ndarray::IxDyn> {
        self.data.iter()
    }

    /// Returns a mutable iterator over elements in row-major order
    pub fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, T, ndarray::IxDyn> {
        self.data.iter_mut()
    }

    /// Returns an iterator over elements with their indices
    pub fn indexed_iter(&self) -> ndarray::iter::IndexedIter<'_, T, ndarray::IxDyn> {
        self.data.indexed_iter()
    }

    /// Returns a mutable iterator over elements with their indices
    pub fn indexed_iter_mut(&mut self) -> ndarray::iter::IndexedIterMut<'_, T, ndarray::IxDyn> {
        self.data.indexed_iter_mut()
    }

    /// Collect all elements into a Vec in row-major order.
    /// Works for both CPU and CUDA tensors.
    pub fn to_vec(&self) -> Result<Vec<T>, String> {
        Ok(self.data.iter().copied().collect())
    }

    /// Conditional selection: where condition is true, use true_vals, else false_vals
    pub fn where_condition(
        condition: &CPUTensor<T>,
        true_vals: &CPUTensor<T>,
        false_vals: &CPUTensor<T>,
    ) -> Result<CPUTensor<T>, String> {
        let condition_vec = condition.to_vec()?;
        let true_vec = true_vals.to_vec()?;
        let false_vec = false_vals.to_vec()?;

        let result_vec: Vec<T> = condition_vec
            .iter()
            .zip(true_vec.iter())
            .zip(false_vec.iter())
            .map(|((&cond, &true_val), &false_val)| {
                if cond > <T as CPUNumber>::zero() {
                    true_val
                } else {
                    false_val
                }
            })
            .collect();

        CPUTensor::from_vec(result_vec, condition.shape())
    }
}

impl<T> CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    /// SLICING SUPPORT
    ///
    /// Get immutable slice view of tensor data using storage backend
    /// Zero-cost access to underlying memory for efficient operations
    pub fn as_slice(&self) -> Result<&[T], String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        // Only works for CPU storage
        if storage.is_gpu() {
            return Err("Cannot get slice from GPU tensor. Use .to_cpu() first".to_string());
        }

        let cpu_data = storage.cpu_data()?;
        cpu_data
            .as_slice()
            .ok_or("ArrayD is not contiguous - cannot convert to slice".to_string())
    }

    /// Get mutable slice view of tensor data using storage backend
    /// Enables in-place operations without reallocations
    /// Only works for owned CPU storage
    pub fn as_slice_mut(&mut self) -> Result<&mut [T], String> {
        let storage = self
            .storage
            .as_mut()
            .ok_or("Tensor has no storage backend")?;

        // Only works for CPU storage
        if storage.is_gpu() {
            return Err(
                "Cannot get mutable slice from GPU tensor. Use .to_cpu() first".to_string(),
            );
        }

        // Check if storage owns its data
        if !storage.owns_data() {
            return Err("Cannot get mutable slice from borrowed storage".to_string());
        }

        let cpu_data = storage.cpu_data_mut()?;
        cpu_data
            .as_slice_mut()
            .ok_or("ArrayD is not contiguous - cannot convert to mutable slice".to_string())
    }

    /// Get slice of specific length starting from offset using storage
    /// Useful for accessing sub-tensors or specific regions
    pub fn slice_range(&self, start: usize, len: usize) -> Result<&[T], String> {
        let full_slice = self.as_slice()?;
        if start + len > full_slice.len() {
            return Err(format!(
                "Slice range [{}, {}) exceeds tensor size {}",
                start,
                start + len,
                full_slice.len()
            ));
        }
        Ok(&full_slice[start..start + len])
    }

    /// Get mutable slice of specific length starting from offset using storage
    pub fn slice_range_mut(&mut self, start: usize, len: usize) -> Result<&mut [T], String> {
        let total_len = self.len(); // Uses storage.size()
        if start + len > total_len {
            return Err(format!(
                "Slice range [{}, {}) exceeds tensor size {}",
                start,
                start + len,
                total_len
            ));
        }
        let full_slice = self.as_slice_mut()?;
        Ok(&mut full_slice[start..start + len])
    }

    /// Create tensor from existing slice (copies data) using storage backend
    /// Useful for creating tensors from memory pool slices
    pub fn from_slice(slice: &[T], shape: &[usize]) -> Result<Self, String> {
        let expected_len: usize = shape.iter().product();
        if slice.len() != expected_len {
            return Err(format!(
                "Slice length {} doesn't match shape {:?} (expected {})",
                slice.len(),
                shape,
                expected_len
            ));
        }

        let array = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), slice.to_vec())
            .map_err(|e| format!("Failed to create tensor from slice: {}", e))?;

        // Create using storage backend
        let storage = crate::tensor::storage::CPUOwnedStorage::new(array.clone());
        Ok(Self {
            data: array, // TODO: Remove this field in final migration step
            device: crate::backend::default_device(),
            storage: Some(Box::new(storage)),
        })
    }

    /// Create tensor from existing slice with specific device
    pub fn from_slice_with_device(
        slice: &[T],
        shape: &[usize],
        device: crate::backend::Device,
    ) -> Result<Self, String> {
        let expected_len: usize = shape.iter().product();
        if slice.len() != expected_len {
            return Err(format!(
                "Slice length {} doesn't match shape {:?} (expected {})",
                slice.len(),
                shape,
                expected_len
            ));
        }

        let array = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), slice.to_vec())
            .map_err(|e| format!("Failed to create tensor from slice: {}", e))?;

        // Create using storage backend
        let storage = crate::tensor::storage::CPUOwnedStorage::new(array.clone());
        Ok(Self {
            data: array, // TODO: Remove this field in final migration step
            device,
            storage: Some(Box::new(storage)),
        })
    }

    /// Check if tensor data is contiguous (for slice operations)
    pub fn is_contiguous(&self) -> bool {
        if let Ok(data) = self.cpu_data() {
            data.as_slice().is_some()
        } else {
            // GPU tensors are always considered contiguous
            true
        }
    }

    /// Clone the tensor into owned storage (converts borrowed to owned)
    pub fn to_owned(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let cpu_data = if storage.is_gpu() {
            // For GPU storage, sync to CPU first
            self.get_cpu_data()?.into_owned()
        } else {
            storage.cpu_data()?.clone()
        };

        let owned_storage = crate::tensor::storage::CPUOwnedStorage::new(cpu_data.clone());
        Ok(Self {
            data: cpu_data, // TODO: Remove this field in final migration step
            device: self.device.clone(),
            storage: Some(Box::new(owned_storage)),
        })
    }

    /// Create a flattened view of the tensor as 1D
    /// Returns a new tensor with same data but 1D shape
    pub fn flatten(&self) -> Result<Self, String> {
        let cpu_data = self.get_cpu_data()?.into_owned();
        let total_elements = cpu_data.len();

        let flattened = cpu_data
            .into_shape(ndarray::IxDyn(&[total_elements]))
            .map_err(|e| format!("Failed to flatten tensor: {}", e))?;

        Ok(Self::new_with_device(flattened, self.device.clone()))
    }
}

// Implementation for floating-point operations
impl<T> CPUTensor<T>
where
    T: GPUFloat,
{
    /// Matrix multiplication using storage trait
    pub fn matmul(&self, other: &Self) -> Result<Self, String>
    where
        T: Clone + ndarray::LinalgScalar,
    {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;
        let other_storage = other
            .storage
            .as_ref()
            .ok_or("Other tensor has no storage backend")?;

        let result_storage = storage.matmul(other_storage.as_ref())?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Sigmoid activation function using storage trait
    pub fn sigmoid(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.sigmoid()?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// ReLU activation function using storage trait
    pub fn relu(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.relu()?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Exponential function using storage trait
    pub fn exp(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.exp()?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Natural logarithm using storage trait
    pub fn log(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.log()?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Hyperbolic tangent using storage trait
    pub fn tanh(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.tanh()?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Element-wise power using storage trait
    pub fn powf(&self, other: &Self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;
        let other_storage = other
            .storage
            .as_ref()
            .ok_or("Other tensor has no storage backend")?;

        let result_storage = storage.powf(other_storage.as_ref())?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Scalar power using storage trait
    pub fn power_scalar(&self, scalar: T) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.power_scalar(scalar)?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }
}

// Implementation for reduction operations and tensor manipulations
impl<T> CPUTensor<T>
where
    T: GPUFloat + Clone + rand_distr::num_traits::Zero + rand_distr::num_traits::FromPrimitive,
{
    // Reduction operations.
    // Sum reduction along multiple axes using storage backend
    /// This replaces the old sum_axes method to use the storage backend
    pub fn sum(&self, axes: Option<&[usize]>) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.sum(axes)?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Mean reduction along multiple axes using storage backend
    pub fn mean(&self, axes: Option<&[usize]>) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.mean(axes)?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Maximum values reduction along multiple axes using storage backend
    pub fn max_reduce(&self, axes: Option<&[usize]>) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.max_reduce(axes)?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }

    /// Minimum values reduction along multiple axes using storage backend
    pub fn min_reduce(&self, axes: Option<&[usize]>) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.min_reduce(axes)?;
        Self::from_storage_backend(result_storage, self.device.clone())
    }
}

impl<T> CPUTensor<T>
where
    T: GPUFloat + Clone + rand_distr::num_traits::Zero + rand_distr::num_traits::FromPrimitive,
{
    /// Broadcasting for gradient computation and tensor operations
    /// Uses storage backend for consistent behavior across CPU/GPU
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<CPUTensor<T>, String> {
        if let Some(storage) = &self.storage {
            let new_storage = storage.broadcast_to(target_shape)?;
            Ok(CPUTensor {
                data: new_storage.cpu_data()?.clone(),
                device: self.device.clone(),
                storage: Some(new_storage),
            })
        } else {
            // Create storage backend and use it - no fallback needed
            let storage = Box::new(CPUOwnedStorage::new(self.data.clone()));
            let new_storage = storage.broadcast_to(target_shape)?;
            Ok(CPUTensor {
                data: new_storage.cpu_data()?.clone(),
                device: self.device.clone(),
                storage: Some(new_storage),
            })
        }
    }

    /// Reshape operation - change tensor shape while preserving total elements
    /// Validates element count consistency before reshaping
    pub fn reshape(&self, new_shape: &[usize]) -> Result<CPUTensor<T>, String> {
        if let Some(storage) = &self.storage {
            let new_storage = storage.reshape(new_shape)?;
            Ok(CPUTensor {
                data: new_storage.cpu_data()?.clone(),
                device: self.device.clone(),
                storage: Some(new_storage),
            })
        } else {
            let storage = Box::new(CPUOwnedStorage::new(self.data.clone()));
            let new_storage = storage.reshape(new_shape)?;
            Ok(CPUTensor {
                data: new_storage.cpu_data()?.clone(),
                device: self.device.clone(),
                storage: Some(new_storage),
            })
        }
    }

    /// Transpose operation - permute tensor axes
    /// If axes is None, performs default transpose (reverse all axes)
    /// If axes provided, must be valid permutation of 0..ndim
    pub fn transpose(&self, axes: Option<&[usize]>) -> Result<CPUTensor<T>, String> {
        if let Some(storage) = &self.storage {
            let new_storage = storage.transpose(axes)?;
            Ok(CPUTensor {
                data: new_storage.cpu_data()?.clone(),
                device: self.device.clone(),
                storage: Some(new_storage),
            })
        } else {
            let storage = Box::new(CPUOwnedStorage::new(self.data.clone()));
            let new_storage = storage.transpose(axes)?;
            Ok(CPUTensor {
                data: new_storage.cpu_data()?.clone(),
                device: self.device.clone(),
                storage: Some(new_storage),
            })
        }
    }

    /// Add dimension of size 1 at specified axis
    /// Similar to tf.expand_dims - axis can be 0..ndim (inclusive)
    pub fn unsqueeze(&self, axis: usize) -> Result<CPUTensor<T>, String> {
        if let Some(storage) = &self.storage {
            let new_storage = storage.unsqueeze(axis)?;
            Ok(CPUTensor {
                data: new_storage.cpu_data()?.clone(),
                device: self.device.clone(),
                storage: Some(new_storage),
            })
        } else {
            let storage = Box::new(CPUOwnedStorage::new(self.data.clone()));
            let new_storage = storage.unsqueeze(axis)?;
            Ok(CPUTensor {
                data: new_storage.cpu_data()?.clone(),
                device: self.device.clone(),
                storage: Some(new_storage),
            })
        }
    }

    /// Remove dimensions of size 1 from tensor
    /// If axis is Some(ax), removes only specified axis if it has size 1
    /// If axis is None, removes all dimensions with size 1
    pub fn squeeze(&self, axis: Option<usize>) -> Result<CPUTensor<T>, String> {
        if let Some(storage) = &self.storage {
            let new_storage = storage.squeeze(axis)?;
            Ok(CPUTensor {
                data: new_storage.cpu_data()?.clone(),
                device: self.device.clone(),
                storage: Some(new_storage),
            })
        } else {
            let storage = Box::new(CPUOwnedStorage::new(self.data.clone()));
            let new_storage = storage.squeeze(axis)?;
            Ok(CPUTensor {
                data: new_storage.cpu_data()?.clone(),
                device: self.device.clone(),
                storage: Some(new_storage),
            })
        }
    }

    /// TensorFlow-style expand_dims - alias for unsqueeze
    /// Adds dimension of size 1 at specified axis
    /// Implemented through unsqueeze to avoid code duplication
    pub fn expand_dims(&self, axis: usize) -> Result<CPUTensor<T>, String> {
        self.unsqueeze(axis)
    }
}

//// -------------------------------------------------------------------
/// CONVOLUTION OPERATIONS
/// --------------------------------------------------------------------
/// I decided to separate this on a different block to aid for readability.
///-----------------------------------------------------------------------
impl<T> CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    /// Convert image patches to column matrix (im2col)
    /// I am going to try to explain why these is needed.
    /// Basically, in order to perform convolution, we need to convert the input image patches into a column matrix.
    /// If we did not do this, we would have to perform 4 nested loops to perform the convolution operation,
    /// which would be very inefficient (O(n^4) complexity).
    /// The im2col operation converts the input image patches into a column matrix, which allows
    /// us to perform the convolution operation using matrix multiplication.
    /// The output shape of the im2col operation is [channels * kernel_h * kernel_w, batch * out_h * out_w],
    /// where:
    /// - channels: number of input channels
    /// - kernel_h: height of the convolution kernel
    /// - kernel_w: width of the convolution kernel
    /// - batch: number of input images in the batch
    /// - out_h: height of the output feature map
    /// - out_w: width of the output feature map
    /// An example here would be:
    /// Given an input tensor with shape [1, 3, 5, 5] (1 batch, 3 channels, 5 height, 5 width),
    /// and a kernel size of (2, 2) with stride (1, 1) and padding (0, 0),
    /// the output shape of the im2col operation would be [
    /// [3 * 2 * 2, 1 * 4 * 4] = [12, 16].
    /// This means we have 12 rows (one for each channel and kernel element)
    /// and 16 columns (one for each position in the output feature map).
    ///
    /// If you dive deeper into the impl, you will see that we also have a nested loop
    /// that iterates over the kernel height and width, and for each kernel element,
    /// we iterate over the output feature map positions.
    ///
    /// You could be tempted to think that it is the same as performing a convolution operation, in a nested loop
    /// as we are not getting away from it completely, but the key idea is that the outer part of the loop is MUCH SMALLER than if we had to iterate over
    /// the input image height and width, as an image can have thousands of pixels, while the kernel size is usually small (3x3, 5x5, etc.).
    fn im2col(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<T>, String> {
        let input_shape = self.shape();
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

        let mut col_data = vec![<T as CPUNumber>::zero(); col_height * col_width];
        let input_data = self.data.as_slice().unwrap();

        for b in 0..batch {
            for c in 0..channels {
                // Note that here we are iterating over the kernel height and width,
                // and for each kernel element we iterate over the output feature map.
                // This is the key part of the im2col operation.
                for ky in 0..kernel_h {
                    for kx in 0..kernel_w {
                        let col_row = c * kernel_h * kernel_w + ky * kernel_w + kx;

                        for out_y in 0..out_h {
                            for out_x in 0..out_w {
                                // Calculate the input coordinates based on the output coordinates,
                                // kernel size, stride, and padding.
                                let in_y = out_y * stride.0 + ky;
                                let in_x = out_x * stride.1 + kx;

                                let col_col = b * out_h * out_w + out_y * out_w + out_x;

                                // Check if the input coordinates are within the padded input dimensions
                                // and if they are, we can safely access the input data.
                                // If they are not, we skip this position, preventing out-of-bounds access.
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
            .map_err(|e| format!("Failed to create im2col matrix: {}", e))
    }

    /// 2D Convolution
    pub fn conv2d(
        &self,
        filter: &Self,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self, String> {
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

        // im2col: [in_channels * kernel_h * kernel_w, batch * out_h * out_w]
        // Compute im2col.
        // Aka from:
        //
        // [[[[
        // 1,  2,  3,  4,
        // 5,  6,  7,  8,
        // 9, 10, 11, 12,
        // 13, 14, 15, 16
        //]]]]
        //
        // to:
        //  [[ 1,  2,  3,  6],
        //   [ 5,  6,  7, 10],
        //   [ 9, 10, 11, 14],
        //   [ 2,  3,  4,  7],
        //   [ 6,  7,  8, 11],
        //   [10, 11, 12, 15],
        //   [ 3,  4,  0,  8],   <-- if padding or stride changes, more truncation/zeros
        //   [ 7,  8,  0, 12],
        //   [11, 12,  0, 16]]
        //
        // This way, we extract patches of the input tensor and flatten them into columns,
        // which allows us to perform the convolution as a matrix multiplication.
        let col_matrix = self.im2col((kernel_h, kernel_w), stride, padding)?;

        // Reshape filter: [out_channels, in_channels * kernel_h * kernel_w]
        let filter_reshaped = filter
            .data
            .clone()
            .into_shape_with_order(IxDyn(&[out_channels, in_channels * kernel_h * kernel_w]))
            .map_err(|e| format!("Filter reshape failed: {}", e))?;

        // Compute the output using matrix multiplication
        // We need to convert the data to 2D views. Nte that this does not actually create a copy
        // of the data, it just creates a view of the data with the specified shape.
        let im2col_view: ndarray::ArrayView2<T> = col_matrix.view().into_dimensionality().unwrap();
        let filter_view: ndarray::ArrayView2<T> =
            filter_reshaped.view().into_dimensionality().unwrap();
        // GEMM: filter_reshaped @ col_matrix = [out_channels, batch * out_h * out_w]
        let output_2d = filter_view.dot(&im2col_view);

        // Then we reshape back to the original shape.
        // Reshape to [batch, out_channels, out_h, out_w]
        let output_data: Vec<T> = output_2d.as_slice().unwrap().to_vec();
        let mut final_output = vec![<T as CPUNumber>::zero(); batch * out_channels * out_h * out_w];

        // Transpose from [out_channels, batch * out_h * out_w] to [batch, out_channels, out_h, out_w]
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

        let output_array =
            ArrayD::from_shape_vec(IxDyn(&[batch, out_channels, out_h, out_w]), final_output)
                .map_err(|e| format!("Failed to create output tensor: {}", e))?;

        Ok(Self::new(output_array))
    }

    /// Depthwise separable convolution
    /// The key idea of this operation is to first apply a depthwise convolution,
    /// which applies a separate filter to each input channel independently,
    /// and then apply a pointwise convolution, which mixes the output channels.
    /// The output shape is [batch, out_channels, out_h, out_w] where
    /// out_channels is the number of filters in the pointwise convolution.
    /// I think it is very inefficient on CPU but as I also have the GPU , I decided to add it so that
    /// I can keep it as a fallback when gpu is not available.
    pub fn depthwise_separable_conv2d(
        &self,
        depthwise_filter: &Self,
        pointwise_filter: &Self,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self, String> {
        // Depthwise convolution first
        // For each input channel, we apply the corresponding depthwise filter.
        // The depthwise filter shape is [in_channels, 1, kernel_h, kernel_w]
        // The output shape will be [batch, in_channels, out_h, out_w]
        // where out_h and out_w are computed based on the input shape, stride, and
        // padding.
        // The pointwise filter shape is [out_channels, in_channels, 1, 1]
        // The output shape will be [batch, out_channels, out_h, out_w]
        // where out_channels is the number of filters in the pointwise convolution.
        let depthwise_result = self.depthwise_conv2d(depthwise_filter, stride, padding)?;
        depthwise_result.conv2d(pointwise_filter, (1, 1), (0, 0))
    }

    /// Depthwise convolution (efficient channel-wise implementation)
    /// Depthwise convolution applies a separate filter to each input channel independently.
    /// This is different from standard convolution where filters mix across channels.
    /// The output shape is [batch, channels, out_h, out_w] where each channel is convolved with its own filter.
    /// I will provide a link to this explanation
    /// which really helped me to understand how depthwise separable
    /// convolutions work: https://medium.com/data-science/understanding-depthwise-separable-convolutions-and-the-efficiency-of-mobilenets-6de3d6b62503
    pub fn depthwise_conv2d(
        &self,
        filter: &Self,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self, String> {
        let input_shape = self.shape();
        let filter_shape = filter.shape();

        let (batch, channels, in_h, in_w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (filter_channels, _, kernel_h, kernel_w) = (
            filter_shape[0],
            filter_shape[1],
            filter_shape[2],
            filter_shape[3],
        );

        if channels != filter_channels {
            return Err("Channel count mismatch in depthwise conv".to_string());
        }

        let out_h = (in_h + 2 * padding.0 - kernel_h) / stride.0 + 1;
        let out_w = (in_w + 2 * padding.1 - kernel_w) / stride.1 + 1;

        let mut output_data = vec![<T as CPUNumber>::zero(); batch * channels * out_h * out_w];
        let input_data = self.data.as_slice().unwrap();
        let filter_data = filter.data.as_slice().unwrap();

        // Each channel processed independently - no cross-channel mixing
        for b in 0..batch {
            for c in 0..channels {
                for out_y in 0..out_h {
                    for out_x in 0..out_w {
                        let mut sum = <T as CPUNumber>::zero();

                        // Convolve single channel with its corresponding filter
                        for ky in 0..kernel_h {
                            for kx in 0..kernel_w {
                                let in_y = out_y * stride.0 + ky;
                                let in_x = out_x * stride.1 + kx;

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
                                        let filter_idx =
                                            c * (kernel_h * kernel_w) + ky * kernel_w + kx;

                                        sum = sum + input_data[input_idx] * filter_data[filter_idx];
                                    }
                                }
                            }
                        }

                        let output_idx = b * (channels * out_h * out_w)
                            + c * (out_h * out_w)
                            + out_y * out_w
                            + out_x;
                        output_data[output_idx] = sum;
                    }
                }
            }
        }

        let output_array =
            ArrayD::from_shape_vec(IxDyn(&[batch, channels, out_h, out_w]), output_data)
                .map_err(|e| format!("Failed to create output tensor: {}", e))?;

        Ok(Self::new(output_array))
    }
}
// Implementation for tensor creation with Zero trait
impl<T> CPUTensor<T>
where
    T: GPUFloat + Clone + rand_distr::num_traits::Zero,
{
    // Initialization functions for creating tensors with specific shapes.
    // They all have a `_with_device` variant that allows specifying the device.
    // Zeroes
    pub fn zeros(shape: &[usize]) -> Self {
        let device = default_device();
        let data = device.zeros(shape);
        let storage = CPUOwnedStorage::new(data.clone());
        Self {
            data,
            device,
            storage: Some(Box::new(storage)),
        }
    }

    pub fn zeros_with_device(shape: &[usize], device: Device) -> Self {
        let data = device.zeros(shape);
        let storage = CPUOwnedStorage::new(data.clone());
        Self {
            data,
            device,
            storage: Some(Box::new(storage)),
        }
    }
}

// Implementation for tensor creation with One trait
impl<T> CPUTensor<T>
where
    T: GPUFloat,
{
    // Ones
    pub fn ones(shape: &[usize]) -> Self {
        let device = default_device();
        let data = device.ones(shape);
        let storage = CPUOwnedStorage::new(data.clone());
        Self {
            data,
            device,
            storage: Some(Box::new(storage)),
        }
    }

    pub fn ones_with_device(shape: &[usize], device: Device) -> Self {
        let data = device.ones(shape);
        let storage = CPUOwnedStorage::new(data.clone());
        Self {
            data,
            device,
            storage: Some(Box::new(storage)),
        }
    }
}

// Implement equality for testing, and because will be useful in the future.
impl<T> PartialEq for CPUTensor<T>
where
    T: GPUFloat + Clone + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.device == other.device
    }
}

impl<T> Eq for CPUTensor<T> where T: GPUFloat + Clone + PartialEq {}

pub struct CPUTensorIterator<T> {
    data: ndarray::ArrayD<T>,
    index: usize,
}

impl<T> Iterator for CPUTensorIterator<T>
where
    T: Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.data.len() {
            let item = self.data.as_slice().unwrap()[self.index];
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.data.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for CPUTensorIterator<T>
where
    T: Copy,
{
    fn len(&self) -> usize {
        self.data.len().saturating_sub(self.index)
    }
}

// Implementation for owned Tensor (consumes the tensor)
impl<T> IntoIterator for CPUTensor<T>
where
    T: GPUFloat + Clone + Copy,
{
    type Item = T;
    type IntoIter = CPUTensorIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        CPUTensorIterator {
            data: self.data,
            index: 0,
        }
    }
}

// Implementation for borrowed Tensor (&Tensor)
impl<'a, T> IntoIterator for &'a CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    type Item = &'a T;
    type IntoIter = ndarray::iter::Iter<'a, T, ndarray::IxDyn>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

// Implementation for mutable borrowed Tensor (&mut Tensor)
impl<'a, T> IntoIterator for &'a mut CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    type Item = &'a mut T;
    type IntoIter = ndarray::iter::IterMut<'a, T, ndarray::IxDyn>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

// Implementation for single usize index (flat indexing for any dimensional tensor)
// This attempts to mimic the behavior of NumPy's flat indexing,
// therefore you could access elements in a multi-dimensional tensor as it was a flat array.
impl<T> Index<usize> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        // Convert flat index to multi-dimensional coordinates
        let flat_data = self
            .data
            .as_slice()
            .expect("Tensor data should be contiguous");
        &flat_data[index]
    }
}

impl<T> IndexMut<usize> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // Convert flat index to multi-dimensional coordinates
        let flat_data = self
            .data
            .as_slice_mut()
            .expect("Tensor data should be contiguous");
        &mut flat_data[index]
    }
}

// Implementation for slice of usize (multi-dimensional indexing)
impl<T> Index<&[usize]> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize]> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

// Implementation for Vec<usize> (convenient alternative to slice)
impl<T> Index<Vec<usize>> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    type Output = T;

    fn index(&self, indices: Vec<usize>) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<Vec<usize>> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    fn index_mut(&mut self, indices: Vec<usize>) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

// Implementation for arrays of different sizes (up to 6D for common use cases)
impl<T> Index<[usize; 1]> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    type Output = T;

    fn index(&self, indices: [usize; 1]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<[usize; 1]> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    fn index_mut(&mut self, indices: [usize; 1]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

impl<T> Index<[usize; 2]> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    type Output = T;

    fn index(&self, indices: [usize; 2]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<[usize; 2]> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    fn index_mut(&mut self, indices: [usize; 2]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

impl<T> Index<[usize; 3]> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    type Output = T;

    fn index(&self, indices: [usize; 3]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<[usize; 3]> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    fn index_mut(&mut self, indices: [usize; 3]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

impl<T> Index<[usize; 4]> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    type Output = T;

    fn index(&self, indices: [usize; 4]) -> &Self::Output {
        &self.data[IxDyn(&indices)]
    }
}

impl<T> IndexMut<[usize; 4]> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    fn index_mut(&mut self, indices: [usize; 4]) -> &mut Self::Output {
        &mut self.data[IxDyn(&indices)]
    }
}

// Implementation for tuples (more ergonomic for 2D and 3D)
impl<T> Index<(usize, usize)> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.data[[i, j]]
    }
}

impl<T> IndexMut<(usize, usize)> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.data[[i, j]]
    }
}

impl<T> Index<(usize, usize, usize)> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    type Output = T;

    fn index(&self, (i, j, k): (usize, usize, usize)) -> &Self::Output {
        &self.data[[i, j, k]]
    }
}

impl<T> IndexMut<(usize, usize, usize)> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    fn index_mut(&mut self, (i, j, k): (usize, usize, usize)) -> &mut Self::Output {
        &mut self.data[[i, j, k]]
    }
}

impl<T> Index<(usize, usize, usize, usize)> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    type Output = T;

    fn index(&self, (i, j, k, l): (usize, usize, usize, usize)) -> &Self::Output {
        &self.data[[i, j, k, l]]
    }
}

impl<T> IndexMut<(usize, usize, usize, usize)> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    fn index_mut(&mut self, (i, j, k, l): (usize, usize, usize, usize)) -> &mut Self::Output {
        &mut self.data[[i, j, k, l]]
    }
}

// Implementation for references to arrays of different sizes
impl<T> Index<&[usize; 1]> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize; 1]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize; 1]> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    fn index_mut(&mut self, indices: &[usize; 1]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

impl<T> Index<&[usize; 2]> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize; 2]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize; 2]> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    fn index_mut(&mut self, indices: &[usize; 2]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

impl<T> Index<&[usize; 3]> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize; 3]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize; 3]> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    fn index_mut(&mut self, indices: &[usize; 3]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}

impl<T> Index<&[usize; 4]> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize; 4]) -> &Self::Output {
        &self.data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize; 4]> for CPUTensor<T>
where
    T: GPUFloat + Clone,
{
    fn index_mut(&mut self, indices: &[usize; 4]) -> &mut Self::Output {
        &mut self.data[IxDyn(indices)]
    }
}
