use crate::backend::manager::get_backend;
#[cfg(feature = "cuda")]
use crate::backend::manager::with_cuda_context;
use crate::backend::number::FerroxCudaF;
use crate::backend::storage::{CPUStorage, StorageBackend};
use crate::backend::{default_device, Device};
use crate::ops::scalar;
use ndarray::{Array, ArrayD, IxDyn};
use rand::distr::StandardUniform;
use rand_distr::Distribution;
use std::ops::{Index, IndexMut};

#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaContextManager;
#[cfg(feature = "cuda")]
use crate::backend::storage::CUDAStorage;

// Tensor wrapper to handle dynamic arrays more elegantly
#[derive(Debug)]
pub struct Tensor<T>
where
    T: FerroxCudaF,
{
    pub device: Device,
    storage: Option<Box<dyn StorageBackend<T>>>,
}

impl<T> Clone for Tensor<T>
where
    T: FerroxCudaF + Clone,
{
    /// Clone creates a new tensor with same storage type and device, copying data
    /// Uses storage backend's clone_storage() to preserve device placement
    fn clone(&self) -> Self {
        let storage = self
            .storage
            .as_ref()
            .expect("Tensor must have storage backend for cloning");

        // Use storage backend's clone method to create independent copy
        let cloned_storage = storage
            .clone_storage()
            .expect("Failed to clone storage backend");

        // Create new tensor with cloned storage on same device
        Self::from_storage_backend(cloned_storage, self.device)
            .expect("Failed to create tensor from cloned storage")
    }
}

impl<T> Tensor<T>
where
    T: FerroxCudaF,
{
    pub fn zeros(shape: &[usize]) -> Result<Self, String> {
        let backend = get_backend::<T>();
        let (device, storage) = backend.create_storage_auto(shape)?;
        Self::from_storage_backend(storage, device)
    }

    pub fn zeros_with_device(shape: &[usize], device: Device) -> Result<Self, String> {
        let backend = get_backend::<T>();
        let (validated_device, storage) = backend.create_storage(shape, device)?;
        Self::from_storage_backend(storage, validated_device)
    }

    // New ones implementation using storage backend
    pub fn ones(shape: &[usize]) -> Result<Self, String> {
        let backend = get_backend::<T>();
        let (device, storage) = backend.create_ones_storage(shape, backend.best_device())?;
        Self::from_storage_backend(storage, device)
    }

    pub fn ones_with_device(shape: &[usize], device: Device) -> Result<Self, String> {
        let backend = get_backend::<T>();
        let (validated_device, storage) = backend.create_ones_storage(shape, device)?;
        Self::from_storage_backend(storage, validated_device)
    }

    // Additional initialization method for completeness
    pub fn full(shape: &[usize], value: T) -> Result<Self, String> {
        let backend = get_backend::<T>();
        let (device, storage) = backend.create_full_storage(shape, backend.best_device(), value)?;
        Self::from_storage_backend(storage, device)
    }

    pub fn full_with_device(shape: &[usize], device: Device, value: T) -> Result<Self, String> {
        let backend = get_backend::<T>();
        let (validated_device, storage) = backend.create_full_storage(shape, device, value)?;
        Self::from_storage_backend(storage, validated_device)
    }

    // Random normal distribution initialization using storage backend
    pub fn randn(shape: &[usize]) -> Result<Self, String>
    where
        StandardUniform: Distribution<T>,
    {
        let backend = get_backend::<T>();
        let (device, storage) = backend.create_randn_storage(shape, backend.best_device())?;
        Self::from_storage_backend(storage, device)
    }

    pub fn randn_with_device(shape: &[usize], device: Device) -> Result<Self, String>
    where
        StandardUniform: Distribution<T>,
    {
        let backend = get_backend::<T>();
        let (validated_device, storage) = backend.create_randn_storage(shape, device)?;
        Self::from_storage_backend(storage, validated_device)
    }

    pub fn to_cpu(self) -> Result<Self, String> {
        self.to_device(Device::CPU)
    }

    /// Move tensor to different device
    pub fn to_device(self, target_device: Device) -> Result<Self, String> {
        if self.device == target_device {
            // Do not do anything
            return Ok(self);
        }

        let backend = get_backend::<T>();
        if let Some(storage) = self.storage {
            let (new_device, new_storage) = backend.move_storage(storage, target_device)?;
            Self::from_storage_backend(new_storage, new_device)
        } else {
            Err("Empty tensor cannot be moved".to_string())
        }
    }
}

// Main implementation block with basic operations
impl<T> Tensor<T>
where
    T: FerroxCudaF + Clone,
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
        let backend = get_backend::<T>();
        let (device, storage) = backend
            .create_storage_from_data_auto(&data)
            .expect("Error while accessing backend manager");
        Self {
            device,
            storage: Some(storage),
        }
    }

    // Creates a new tensor with the given data and device.
    // This is useful when you want to create a tensor with a specific device, for example, when you want to use a GPU.
    // In the future, we could also add a method to create a tensor with a specific data type, but for now we will keep it simple.
    // The device is set to the default device if not provided.
    // This is similar to how PyTorch and TensorFlow work, where the device is set to the default device if not specified.
    // Ideally, we should not be bound to ndarray backend here because it defaults to CPU, but it is okay for now as I prefer to focus more on the automatic differentiation engine thing.
    pub fn new_with_device(data: ArrayD<T>, device: Device) -> Self {
        let backend = get_backend::<T>();
        let (validated_device, storage) = backend
            .create_storage_from_data(&data, device)
            .expect("Error while accessing backend manager");
        Self {
            device: validated_device, // Use validated_device, not device
            storage: Some(storage),
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

        match Array::from_shape_vec(IxDyn(shape), data) {
            Ok(array) => {
                // Create storage backend for the array
                let storage = CPUStorage::new(array.clone());
                Ok(Self {
                    device: default_device(),
                    storage: Some(Box::new(storage)),
                })
            }
            Err(e) => Err(format!("Failed to create tensor: {e}")),
        }
    }

    /// Creates a tensor from a Rust vector on specified device
    /// This function is device-agnostic and uses the backend manager to handle device placement
    pub fn from_vec_with_device(
        data: Vec<T>,
        shape: &[usize],
        device: Device,
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

        // First create ndarray from vec and shape - this validates the shape
        let array = match Array::from_shape_vec(IxDyn(shape), data) {
            Ok(array) => array,
            Err(e) => return Err(format!("Failed to create array from vec: {}", e)),
        };

        // Use backend manager to create storage on the specified device
        let backend = get_backend::<T>();
        let (validated_device, storage) = backend.create_storage_from_data(&array, device)?;

        Ok(Self {
            device: validated_device,
            storage: Some(storage),
        })
    }

    // Create tensor from existing storage backend (internal use)
    // This is the most direct way to create tensors and will become primary after migration
    pub(crate) fn from_storage_backend(
        storage: Box<dyn StorageBackend<T>>,
        device: crate::backend::Device,
    ) -> Result<Self, String> {
        Ok(Self {
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
    /// Note: This method currently panics if the data is on cpu. Prefer into_data() to convert any type of tensor to a ndarray ArrayD. It mimics PyTorch's to_numpy().
    /// Get CPU data reference for CPU tensors only
    /// For GPU tensors, use cpu_data_owned() to get a copy or cpu_data_mut() for in-place conversion
    pub fn cpu_data(&self) -> Result<&ArrayD<T>, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        // Only works for CPU storage - return borrowed reference
        if !storage.is_gpu() {
            return storage.cpu_data();
        }

        // For GPU storage, direct access not possible without mutation
        Err("Cannot get CPU data reference from GPU tensor. Use cpu_data_owned() for copy or cpu_data_mut() for in-place conversion".to_string())
    }

    /// Get owned CPU data, creating copy for GPU tensors using move_storage
    /// This method handles both CPU and GPU tensors by returning owned data
    pub fn cpu_data_owned(&self) -> Result<ArrayD<T>, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        // For CPU storage - clone the data
        if !storage.is_gpu() {
            return Ok(storage.cpu_data()?.clone());
        }

        // For GPU storage, create CPU copy using move_storage
        #[cfg(feature = "cuda")]
        {
            // Clone the GPU storage to create an independent copy
            let gpu_storage_copy = storage.clone_storage()?;

            let manager = get_backend::<T>();
            // Move the cloned storage to CPU, consuming the GPU copy
            let (_device, cpu_storage) = manager.move_storage(gpu_storage_copy, Device::CPU)?;

            // Extract owned CPU data
            Ok(cpu_storage.cpu_data()?.clone())
        }

        #[cfg(not(feature = "cuda"))]
        Err("Cannot access GPU data without CUDA feature".to_string())
    }

    /// Get mutable CPU data reference, moving from GPU if necessary
    /// Uses manager's move_storage to convert storage in-place
    pub fn cpu_data_mut(&mut self) -> Result<&mut ArrayD<T>, String> {
        // Check if we need to move storage
        let needs_move = self.storage.as_ref().map(|s| s.is_gpu()).unwrap_or(false);

        if needs_move {
            #[cfg(feature = "cuda")]
            {
                // Take ownership of storage to move it
                let storage = self.storage.take().ok_or("Tensor has no storage backend")?;

                let manager = get_backend::<T>();
                // Move GPU storage to CPU, consuming the GPU storage
                let (new_device, cpu_storage) = manager.move_storage(storage, Device::CPU)?;

                // Update tensor with CPU storage
                self.storage = Some(cpu_storage);
                self.device = new_device;
            }

            #[cfg(not(feature = "cuda"))]
            {
                return Err("Cannot move GPU data without CUDA feature".to_string());
            }
        }

        // Now get mutable reference to CPU data
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

    /// Get number of elements using storage backend
    pub fn len(&self) -> usize {
        self.storage.as_ref().map(|s| s.size()).unwrap_or(0)
    }

    /// Check if tensor is empty using storage backend
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get device reference
    pub fn device(&self) -> Device {
        self.device
    }

    /// Extract data consuming tensor, handling both CPU and GPU tensors
    /// Converts CUDA storage to CPU if needed using manager's move_storage
    pub fn into_data(self) -> Result<ArrayD<T>, String> {
        let storage = self.storage.ok_or("Tensor has no storage backend")?;

        // If already CPU storage, extract directly
        if !storage.is_gpu() {
            return storage.cpu_data().cloned();
        }

        // For GPU storage, use manager to move to CPU then extract
        #[cfg(feature = "cuda")]
        {
            let manager = get_backend::<T>();
            // Move storage consumes the GPU storage, returns CPU storage
            let (_device, cpu_storage) = manager.move_storage(storage, Device::CPU)?;

            // Extract data from CPU storage
            cpu_storage.cpu_data().cloned()
        }

        #[cfg(not(feature = "cuda"))]
        Err("Cannot extract GPU data without CUDA feature".to_string())
    }

    /* pub fn execute_custom<R>(&self, op: Box<dyn CustomOperation<T, R>>) -> Result<R, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        // Execute custom operation using storage backend trait
        // Creates new results consistent with other tensor operations
        storage.execute_custom_op(op)
    }*/
}

impl<T> Tensor<T>
where
    T: FerroxCudaF + Clone,
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
        Self::from_storage_backend(result_storage, self.device)
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
        Self::from_storage_backend(result_storage, self.device)
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
        Self::from_storage_backend(result_storage, self.device)
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
        Self::from_storage_backend(result_storage, self.device)
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
        Self::from_storage_backend(result_storage, self.device)
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
        Self::from_storage_backend(result_storage, self.device)
    }

    /// Scalar addition - more efficient than broadcasting
    pub fn add_scalar(&self, scalar: T) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.add_scalar(scalar)?;
        Self::from_storage_backend(result_storage, self.device)
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: T) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.mul_scalar(scalar)?;
        Self::from_storage_backend(result_storage, self.device)
    }

    /// Scalar substraction
    pub fn sub_scalar(&self, scalar: T) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.sub_scalar(scalar)?;
        Self::from_storage_backend(result_storage, self.device)
    }

    /// Scalar division
    pub fn div_scalar(&self, scalar: T) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.div_scalar(scalar)?;
        Self::from_storage_backend(result_storage, self.device)
    }

    /// Unary negation - replaces your negate() method
    pub fn neg(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.neg()?;
        Self::from_storage_backend(result_storage, self.device)
    }

    /// Element-wise absolute value using storage trait
    pub fn abs(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.abs()?;
        Self::from_storage_backend(result_storage, self.device)
    }

    /// Element-wise clamp using storage trait
    pub fn clamp(&self, min_val: T, max_val: T) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.clamp(min_val, max_val)?;
        Self::from_storage_backend(result_storage, self.device)
    }

    /// Element-wise square root
    /// Returns a new tensor with square root values
    /// Validates that all values are non-negative
    pub fn sqrt(&self) -> Result<Self, String> {
        // Check for negative values first
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.sqrt()?;
        Self::from_storage_backend(result_storage, self.device)
    }

    pub fn reciprocal(&self) -> Result<Self, String> {
        // Check for negative values first
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.reciprocal()?;
        Self::from_storage_backend(result_storage, self.device)
    }
}

impl<T> Tensor<T>
where
    T: FerroxCudaF + Clone,
{
    /// Element-wise greater than  comparison using storage trait
    pub fn greater(&self, other: &Self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;
        let other_storage = other
            .storage
            .as_ref()
            .ok_or("Other tensor has no storage backend")?;

        let result_storage = storage.greater(other_storage.as_ref())?;
        Self::from_storage_backend(result_storage, self.device)
    }

    pub fn greater_scalar(&self, scalar: T) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.greater_scalar(scalar)?;
        Self::from_storage_backend(result_storage, self.device)
    }

    /// Element-wise less than  comparison using storage trait
    pub fn less(&self, other: &Self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;
        let other_storage = other
            .storage
            .as_ref()
            .ok_or("Other tensor has no storage backend")?;

        let result_storage = storage.less(other_storage.as_ref())?;
        Self::from_storage_backend(result_storage, self.device)
    }

    pub fn less_scalar(&self, scalar: T) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.less_scalar(scalar)?;
        Self::from_storage_backend(result_storage, self.device)
    }

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
        Self::from_storage_backend(result_storage, self.device)
    }

    pub fn greater_equal_scalar(&self, scalar: T) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.greater_equal_scalar(scalar)?;
        Self::from_storage_backend(result_storage, self.device)
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
        Self::from_storage_backend(result_storage, self.device)
    }

    pub fn less_equal_scalar(&self, scalar: T) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.less_equal_scalar(scalar)?;
        Self::from_storage_backend(result_storage, self.device)
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
        Self::from_storage_backend(result_storage, self.device)
    }

    // Check if all elements are equal between tensors - efficient shortcut for PartialEq
    /// Returns true if all corresponding elements are equal, false otherwise
    pub fn all_equal(&self, other: &Self) -> Result<bool, String> {
        // Early exit checks
        if self.device != other.device {
            return Ok(false);
        }

        if self.shape() != other.shape() {
            return Ok(false);
        }

        // For small tensors, direct comparison might be faster
        if self.size() <= 1000 {
            return match (self.cpu_data(), other.cpu_data()) {
                (Ok(self_data), Ok(other_data)) => Ok(self_data == other_data),
                _ => {
                    // For GPU tensors or other cases, use the equal method
                    let equal_tensor = self.equal(other)?;
                    let equal_data = equal_tensor.cpu_data()?;

                    // Check if all elements are 1.0 (true)
                    Ok(equal_data
                        .iter()
                        .all(|&x| x == crate::backend::number::FerroxN::one()))
                }
            };
        }

        // For larger tensors, use the optimized equal method
        let equal_tensor = self.equal(other)?;
        let equal_data = equal_tensor.cpu_data()?;

        // Check if all elements are 1.0 (true)
        Ok(equal_data
            .iter()
            .all(|&x| x == crate::backend::number::FerroxN::one()))
    }

    // Utility for tests
    /// TODO: MOVE THIS TO THE STORAGE MODULE (MUST CREATE A KERNEL)
    pub fn all<F>(&self, predicate: F) -> Result<bool, String>
    where
        F: Fn(T) -> bool,
    {
        let cpu_data = self.cpu_data()?;
        Ok(cpu_data.iter().all(|&x| predicate(x)))
    }

    /// Zip and check all elements with predicate.
    /// TODO: MOVE THIS TO THE STORAGE MODULE (MUST CREATE A KERNEL)
    pub fn zip_all<F>(&self, other: &Self, predicate: F) -> Result<bool, String>
    where
        F: Fn(T, T) -> bool,
    {
        if self.shape() != other.shape() {
            return Ok(false);
        }

        let self_data = self.cpu_data()?;
        let other_data = other.cpu_data()?;

        Ok(self_data
            .iter()
            .zip(other_data.iter())
            .all(|(&a, &b)| predicate(a, b)))
    }

    /// Logical NOT operation using storage trait
    pub fn logical_not(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.logical_not()?;
        Self::from_storage_backend(result_storage, self.device)
    }

    /// Range check operation using storage trait
    pub fn in_range(&self, min_val: T, max_val: T) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.in_range(min_val, max_val)?;
        Self::from_storage_backend(result_storage, self.device)
    }

    /// Sign function using storage trait
    pub fn sign(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.sign()?;
        Self::from_storage_backend(result_storage, self.device)
    }

    // Detach operation - creates a new tensor that shares data but detaches from graph
    // Need to check if this is the right way to do it.
    // In Pytorch i think the detach operation sets the requires_grad flag to false, but we don't have that concept at the tensor level.
    // We can just return a new tensor with the same data and device, but without any gradient tracking.
    // Detach from computational graph - uses storage backend
    pub fn detach(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let detached_storage = storage.clone_storage()?;
        Self::from_storage_backend(detached_storage, self.device)
    }

    /// Iterator access through storage - avoids direct data field usage
    pub fn iter(&self) -> Result<ndarray::iter::Iter<'_, T, IxDyn>, String> {
        let cpu_data = self.cpu_data()?;
        Ok(cpu_data.iter())
    }

    /// Mutable iterator through storage
    pub fn iter_mut(&mut self) -> Result<ndarray::iter::IterMut<'_, T, IxDyn>, String> {
        let cpu_data = self.cpu_data_mut()?;
        Ok(cpu_data.iter_mut())
    }

    /// Conditional selection using storage backend
    #[allow(deprecated)]
    pub fn where_condition(
        condition: &Self,
        true_vals: &Self,
        false_vals: &Self,
    ) -> Result<Self, String> {
        let condition_storage = condition
            .storage
            .as_ref()
            .ok_or("Condition tensor has no storage")?;
        let true_storage = true_vals
            .storage
            .as_ref()
            .ok_or("True values tensor has no storage")?;
        let false_storage = false_vals
            .storage
            .as_ref()
            .ok_or("False values tensor has no storage")?;

        // Case 1: All CPU storage
        if !condition_storage.is_gpu() && !true_storage.is_gpu() && !false_storage.is_gpu() {
            if let (Some(cond_cpu), Some(true_cpu), Some(false_cpu)) = (
                condition_storage.as_any().downcast_ref::<CPUStorage<T>>(),
                true_storage.as_any().downcast_ref::<CPUStorage<T>>(),
                false_storage.as_any().downcast_ref::<CPUStorage<T>>(),
            ) {
                let result_storage = CPUStorage::where_condition(cond_cpu, true_cpu, false_cpu)?;
                return Self::from_storage_backend(result_storage, condition.device);
            }
        }

        #[cfg(feature = "cuda")]
        {
            // Case 2: All GPU storage
            if condition_storage.is_gpu() && true_storage.is_gpu() && false_storage.is_gpu() {
                if let (Some(cond_gpu), Some(true_gpu), Some(false_gpu)) = (
                    condition_storage.as_any().downcast_ref::<CUDAStorage<T>>(),
                    true_storage.as_any().downcast_ref::<CUDAStorage<T>>(),
                    false_storage.as_any().downcast_ref::<CUDAStorage<T>>(),
                ) {
                    let result_storage =
                        CUDAStorage::where_condition(cond_gpu, true_gpu, false_gpu)?;
                    return Self::from_storage_backend(result_storage, condition.device);
                }
            }
        }

        // Case 3: Mixed or unsupported storage types - fallback to CPU
        //let cond_cpu = condition.to_cpu()?;
        //let true_cpu = true_vals.to_cpu()?;
        //let false_cpu = false_vals.to_cpu()?;
        //Self::where_condition(&cond_cpu, &true_cpu, &false_cpu)
        panic!("Tensors are not on the same device")
    }
}

impl<T> Tensor<T>
where
    T: FerroxCudaF + Clone,
{
    /// SLICING SUPPORT
    ///
    /// Get immutable slice view, moving from GPU if necessary
    /// This method automatically handles device transfer using manager
    pub fn as_slice(&self) -> Result<&[T], String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        // Only works for CPU storage - GPU requires mutation
        if storage.is_gpu() {
            return Err("Cannot get slice from GPU tensor without mutation. Use as_slice_mut() or into_data()".to_string());
        }

        let cpu_data = storage.cpu_data()?;
        cpu_data
            .as_slice()
            .ok_or("ArrayD is not contiguous - cannot convert to slice".to_string())
    }

    /// Get mutable slice view, automatically moving from GPU using manager
    /// Uses move_storage to convert GPU storage to CPU in-place
    pub fn as_slice_mut(&mut self) -> Result<&mut [T], String> {
        // Check if we need to move from GPU to CPU
        let needs_move = self.storage.as_ref().map(|s| s.is_gpu()).unwrap_or(false);

        if needs_move {
            #[cfg(feature = "cuda")]
            {
                // Take ownership of storage to move it
                let storage = self.storage.take().ok_or("Tensor has no storage backend")?;

                let manager = get_backend::<T>();
                // Move GPU storage to CPU using manager, consuming GPU storage
                let (new_device, cpu_storage) = manager.move_storage(storage, Device::CPU)?;

                // Update tensor with CPU storage
                self.storage = Some(cpu_storage);
                self.device = new_device;
            }

            #[cfg(not(feature = "cuda"))]
            {
                return Err("Cannot move GPU data without CUDA feature".to_string());
            }
        }

        let storage = self
            .storage
            .as_mut()
            .ok_or("Tensor has no storage backend")?;

        // Check if storage owns its data (required for mutable slice)
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

    /// Check if tensor data is contiguous (for slice operations)
    pub fn is_contiguous(&self) -> bool {
        if let Ok(data) = self.cpu_data() {
            data.as_slice().is_some()
        } else {
            // GPU tensors are always considered contiguous
            true
        }
    }
    // Consumes self and obtains the scalar at the first position.
    pub fn first(self) -> Result<T, String> {
        if let Ok(data) = self.into_data() {
            // Handle 0-dimensional arrays (scalars from reductions)
            let item = if data.ndim() == 0 {
                // For 0-dimensional arrays, use iter().next() instead of indexing
                data.iter()
                    .next()
                    .copied()
                    .ok_or_else(|| "Empty 0-dimensional array".to_string())?
            } else {
                // For regular arrays, get first element
                data.iter()
                    .next()
                    .copied()
                    .ok_or_else(|| "Empty array".to_string())?
            };
            return Ok(item);
        }
        Err("Cannot take out of an empty tensor!".to_string())
    }
}

// Implementation for floating-point operations
impl<T> Tensor<T>
where
    T: FerroxCudaF,
{
    /// Matrix multiplication using storage trait
    pub fn matmul(&self, other: &Self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;
        let other_storage = other
            .storage
            .as_ref()
            .ok_or("Other tensor has no storage backend")?;

        let result_storage = storage.matmul(other_storage.as_ref())?;
        Self::from_storage_backend(result_storage, self.device)
    }

    /// Sigmoid activation function using storage trait
    pub fn sigmoid(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.sigmoid()?;
        Self::from_storage_backend(result_storage, self.device)
    }

    /// ReLU activation function using storage trait
    pub fn relu(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.relu()?;
        Self::from_storage_backend(result_storage, self.device)
    }

    /// Exponential function using storage trait
    pub fn exp(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.exp()?;
        Self::from_storage_backend(result_storage, self.device)
    }

    /// Natural logarithm using storage trait
    pub fn log(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.log()?;
        Self::from_storage_backend(result_storage, self.device)
    }

    /// Hyperbolic tangent using storage trait
    pub fn tanh(&self) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.tanh()?;
        Self::from_storage_backend(result_storage, self.device)
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
        Self::from_storage_backend(result_storage, self.device)
    }

    /// Scalar power using storage trait
    pub fn power_scalar(&self, scalar: T) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.power_scalar(scalar)?;
        Self::from_storage_backend(result_storage, self.device)
    }
}

// Implementation for reduction operations and tensor manipulations
impl<T> Tensor<T>
where
    T: FerroxCudaF,
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
        Self::from_storage_backend(result_storage, self.device)
    }

    /// Mean reduction along multiple axes using storage backend
    pub fn mean(&self, axes: Option<&[usize]>) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.mean(axes)?;
        Self::from_storage_backend(result_storage, self.device)
    }

    /// Maximum values reduction along multiple axes using storage backend
    pub fn max_reduce(&self, axes: Option<&[usize]>) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.max_reduce(axes)?;
        Self::from_storage_backend(result_storage, self.device)
    }

    /// Minimum values reduction along multiple axes using storage backend
    pub fn min_reduce(&self, axes: Option<&[usize]>) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;

        let result_storage = storage.min_reduce(axes)?;
        Self::from_storage_backend(result_storage, self.device)
    }
}

impl<T> Tensor<T>
where
    T: FerroxCudaF,
{
    /// Broadcasting for gradient computation and tensor operations
    /// Uses storage backend for consistent behavior across CPU/GPU
    pub fn broadcast_to(&mut self, target_shape: &[usize]) -> Result<(), String> {
        self.storage
            .as_mut()
            .ok_or("Tensor has no storage backend")?
            .broadcast_to(target_shape)?;

        Ok(())
    }

    /// Reshape operation - change tensor shape while preserving total elements
    /// Validates element count consistency before reshaping
    pub fn reshape(&mut self, new_shape: &[usize]) -> Result<(), String> {
        self.storage
            .as_mut()
            .ok_or("Tensor has no storage backend")?
            .reshape(new_shape)?;
        Ok(())
    }

    /// Transpose operation - permute tensor axes
    /// If axes is None, performs default transpose (reverse all axes)
    /// If axes provided, must be valid permutation of 0..ndim
    pub fn transpose(&mut self, axes: Option<&[usize]>) -> Result<(), String> {
        self.storage
            .as_mut()
            .ok_or("Tensor has no storage backend")?
            .transpose(axes)?;

        Ok(())
    }

    /// Add dimension of size 1 at specified axis
    /// Similar to tf.expand_dims - axis can be 0..ndim (inclusive)
    pub fn unsqueeze(&mut self, axis: usize) -> Result<(), String> {
        self.storage
            .as_mut()
            .ok_or("Tensor has no storage backend")?
            .unsqueeze(axis)?;

        Ok(())
    }

    /// Remove dimensions of size 1 from tensor
    /// If axis is Some(ax), removes only specified axis if it has size 1
    /// If axis is None, removes all dimensions with size 1
    pub fn squeeze(&mut self, axis: Option<usize>) -> Result<(), String> {
        self.storage
            .as_mut()
            .ok_or("Tensor has no storage backend")?
            .squeeze(axis)?;

        Ok(())
    }

    /// TensorFlow-style expand_dims - alias for unsqueeze
    /// Adds dimension of size 1 at specified axis
    /// Implemented through unsqueeze to avoid code duplication
    pub fn expand_dims(&mut self, axis: usize) -> Result<(), String> {
        self.unsqueeze(axis)
    }
}

// -------------------------------------------------------------------
// CONVOLUTION OPERATIONS
// --------------------------------------------------------------------
// I decided to separate this on a different block to aid for readability.
//-----------------------------------------------------------------------
impl<T> Tensor<T>
where
    T: FerroxCudaF,
{
    /// 2D Convolution using storage backend - replaces old direct implementation
    pub fn conv2d(
        &self,
        filter: &Self,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self, String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("Tensor has no storage backend")?;
        let filter_storage = filter
            .storage
            .as_ref()
            .ok_or("Filter tensor has no storage backend")?;

        let result_storage = storage.conv2d(filter_storage.as_ref(), stride, padding)?;
        Self::from_storage_backend(result_storage, self.device)
    }
}

/// ITERATOR METHODS
pub struct TensorIterator<T> {
    values: Vec<T>,
    index: usize,
}

impl<T> Iterator for TensorIterator<T>
where
    T: Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.values.len() {
            let item = self.values[self.index];
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.values.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for TensorIterator<T>
where
    T: Copy,
{
    fn len(&self) -> usize {
        self.values.len().saturating_sub(self.index)
    }
}

// Owned tensor iteration using storage backend
impl<T> IntoIterator for Tensor<T>
where
    T: FerroxCudaF + Clone + Copy,
{
    type Item = T;
    type IntoIter = TensorIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        let storage = self.storage.expect("Tensor must have storage backend");
        // Use storage backend to get values instead of direct data access
        let values = storage.iter_values().unwrap_or_else(|_| Vec::new());

        TensorIterator { values, index: 0 }
    }
}

// For borrowed tensors, we must work with CPU data through storage backend
impl<'a, T> IntoIterator for &'a Tensor<T>
where
    T: FerroxCudaF + Clone,
{
    type Item = &'a T;
    type IntoIter = ndarray::iter::Iter<'a, T, IxDyn>;

    fn into_iter(self) -> Self::IntoIter {
        // Must use storage backend - no fallback since data field is removed
        self.cpu_data()
            .expect("Tensor must have valid CPU storage for iteration")
            .iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Tensor<T>
where
    T: FerroxCudaF + Clone,
{
    type Item = &'a mut T;
    type IntoIter = ndarray::iter::IterMut<'a, T, IxDyn>;

    fn into_iter(self) -> Self::IntoIter {
        // Must use storage backend - no fallback since data field is removed
        self.cpu_data_mut()
            .expect("Tensor must have valid mutable CPU storage for iteration")
            .iter_mut()
    }
}
// Implementation for single usize index (flat indexing for any dimensional tensor)
// This attempts to mimic the behavior of NumPy's flat indexing,
// therefore you could access elements in a multi-dimensional tensor as it was a flat array.

// Flat indexing implementation using storage backend only
impl<T> Index<usize> for Tensor<T>
where
    T: FerroxCudaF + Clone,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        // Must use storage backend - data field is removed
        let cpu_data = self
            .cpu_data()
            .expect("Tensor must have valid CPU storage for indexing");

        let flat_data = cpu_data
            .as_slice()
            .expect("CPU tensor data should be contiguous for flat indexing");

        &flat_data[index]
    }
}

impl<T> IndexMut<usize> for Tensor<T>
where
    T: FerroxCudaF + Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // Must use storage backend - data field is removed
        let cpu_data = self
            .cpu_data_mut()
            .expect("Tensor must have valid mutable CPU storage for indexing");

        let flat_data = cpu_data
            .as_slice_mut()
            .expect("CPU tensor data should be contiguous for flat indexing");

        &mut flat_data[index]
    }
}

// Multi-dimensional indexing using storage backend only
impl<T> Index<&[usize]> for Tensor<T>
where
    T: FerroxCudaF + Clone,
{
    type Output = T;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        // Must use storage backend - data field is removed
        let cpu_data = self
            .cpu_data()
            .expect("Tensor must have valid CPU storage for indexing");

        &cpu_data[IxDyn(indices)]
    }
}

impl<T> IndexMut<&[usize]> for Tensor<T>
where
    T: FerroxCudaF + Clone,
{
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        // Must use storage backend - data field is removed
        let cpu_data = self
            .cpu_data_mut()
            .expect("Tensor must have valid mutable CPU storage for indexing");

        &mut cpu_data[IxDyn(indices)]
    }
}

impl<T> PartialEq for Tensor<T>
where
    T: FerroxCudaF + Clone + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        // Use the optimized all_equal method which leverages CUDA kernels when available
        // This provides the best performance for both CPU and GPU tensors
        match self.all_equal(other) {
            Ok(result) => result,
            Err(_) => {
                // Fallback to basic comparison if all_equal fails
                // This should rarely happen but provides safety
                self.device == other.device
                    && self.shape() == other.shape()
                    && match (self.cpu_data(), other.cpu_data()) {
                        (Ok(self_data), Ok(other_data)) => self_data == other_data,
                        _ => false,
                    }
            }
        }
    }
}

impl<T> Eq for Tensor<T> where T: FerroxCudaF + Clone + PartialEq {}

#[cfg(test)]
mod tensor_ops_tests {
    use super::*;
    use crate::backend::{best_device, Device};

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];

        let device = best_device::<f32>();
        let tensor = Tensor::from_vec_with_device(data, &shape, device).unwrap();

        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.size(), 4);
    }

    #[test]
    fn test_zeros_creation() {
        let tensor = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let data = tensor.into_data().unwrap();

        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones_creation() {
        let tensor = Tensor::<f32>::ones(&[2, 2]).unwrap();
        let data = tensor.into_data().unwrap();

        assert!(data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_fill_creation() {
        let tensor = Tensor::full(&[2, 2], 5.0f32).unwrap();
        let data = tensor.into_data().unwrap();

        assert!(data.iter().all(|&x| x == 5.0));
    }

    #[test]
    fn test_add_operation() {
        let device = best_device::<f32>();
        let a = Tensor::from_vec_with_device(vec![1.0f32, 2.0], &[2], device).unwrap();
        let b = Tensor::from_vec_with_device(vec![3.0f32, 4.0], &[2], device).unwrap();

        let result = a.add(&b).unwrap();
        let data = result.into_data().unwrap();

        assert_eq!(data[0], 4.0); // 1 + 3
        assert_eq!(data[1], 6.0); // 2 + 4
    }

    #[test]
    fn test_sub_operation() {
        let device = best_device::<f32>();
        let a = Tensor::from_vec_with_device(vec![5.0f32, 8.0], &[2], device).unwrap();
        let b = Tensor::from_vec_with_device(vec![2.0f32, 3.0], &[2], device).unwrap();

        let result = a.sub(&b).unwrap();
        let data = result.into_data().unwrap();

        assert_eq!(data[0], 3.0); // 5 - 2
        assert_eq!(data[1], 5.0); // 8 - 3
    }

    #[test]
    fn test_mul_operation() {
        let device = best_device::<f32>();
        let a = Tensor::from_vec_with_device(vec![2.0f32, 3.0], &[2], device).unwrap();
        let b = Tensor::from_vec_with_device(vec![4.0f32, 5.0], &[2], device).unwrap();

        let result = a.mul(&b).unwrap();
        let data = result.into_data().unwrap();

        assert_eq!(data[0], 8.0); // 2 * 4
        assert_eq!(data[1], 15.0); // 3 * 5
    }

    #[test]
    fn test_div_operation() {
        let device = best_device::<f32>();
        let a = Tensor::from_vec_with_device(vec![8.0f32, 15.0], &[2], device).unwrap();
        let b = Tensor::from_vec_with_device(vec![2.0f32, 3.0], &[2], device).unwrap();

        let result = a.div(&b).unwrap();
        let data = result.into_data().unwrap();

        assert_eq!(data[0], 4.0); // 8 / 2
        assert_eq!(data[1], 5.0); // 15 / 3
    }

    #[test]
    fn test_add_scalar() {
        let device = best_device::<f32>();
        let tensor = Tensor::from_vec_with_device(vec![1.0f32, 2.0], &[2], device).unwrap();

        let result = tensor.add_scalar(3.0).unwrap();
        let data = result.into_data().unwrap();

        assert_eq!(data[0], 4.0); // 1 + 3
        assert_eq!(data[1], 5.0); // 2 + 3
    }

    #[test]
    fn test_mul_scalar() {
        let device = best_device::<f32>();
        let tensor = Tensor::from_vec_with_device(vec![2.0f32, 3.0], &[2], device).unwrap();

        let result = tensor.mul_scalar(4.0).unwrap();
        let data = result.into_data().unwrap();

        assert_eq!(data[0], 8.0); // 2 * 4
        assert_eq!(data[1], 12.0); // 3 * 4
    }

    #[test]
    fn test_div_scalar() {
        let device = best_device::<f32>();
        let tensor = Tensor::from_vec_with_device(vec![8.0f32, 12.0], &[2], device).unwrap();

        let result = tensor.div_scalar(2.0).unwrap();
        let data = result.into_data().unwrap();

        assert_eq!(data[0], 4.0); // 8 / 2
        assert_eq!(data[1], 6.0); // 12 / 2
    }

    #[test]
    fn test_power_operation() {
        let device = best_device::<f32>();
        let base = Tensor::from_vec_with_device(vec![2.0f32, 3.0], &[2], device).unwrap();
        let exp = Tensor::from_vec_with_device(vec![2.0f32, 3.0], &[2], device).unwrap();

        let result = base.powf(&exp).unwrap();
        let data = result.into_data().unwrap();

        assert_eq!(data[0], 4.0); // 2^2
        assert_eq!(data[1], 27.0); // 3^3
    }

    #[test]
    fn test_power_scalar() {
        let device = best_device::<f32>();
        let tensor = Tensor::from_vec_with_device(vec![2.0f32, 3.0], &[2], device).unwrap();

        let result = tensor.power_scalar(3.0).unwrap();
        let data = result.into_data().unwrap();

        assert_eq!(data[0], 8.0); // 2^3
        assert_eq!(data[1], 27.0); // 3^3
    }

    #[test]
    fn test_matmul_operation() {
        let device = best_device::<f32>();
        // A: 2x2, B: 2x2 -> C: 2x2
        let a = Tensor::from_vec_with_device(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();
        let b = Tensor::from_vec_with_device(vec![2.0f32, 0.0, 1.0, 3.0], &[2, 2], device).unwrap();

        let result = a.matmul(&b).unwrap();

        let data: Vec<f32> = result.into_data().unwrap().iter().cloned().collect();

        // [[1,2],[3,4]] * [[2,0],[1,3]] = [[4,6],[10,12]]
        assert_eq!(data[0], 4.0); // 1*2 + 2*1
        assert_eq!(data[1], 6.0); // 1*0 + 2*3
        assert_eq!(data[2], 10.0); // 3*2 + 4*1
        assert_eq!(data[3], 12.0); // 3*0 + 4*3
    }

    #[test]
    fn test_sigmoid_activation() {
        let device = best_device::<f32>();
        let tensor =
            Tensor::from_vec_with_device(vec![0.0f32, 1000.0, -1000.0], &[3], device).unwrap();

        let result = tensor.sigmoid().unwrap();
        let data = result.into_data().unwrap();

        assert!((data[0] - 0.5).abs() < 1e-6); // sigmoid(0) = 0.5
        assert!(data[1] > 0.99); // sigmoid(large) ≈ 1
        assert!(data[2] < 0.01); // sigmoid(-large) ≈ 0
    }

    #[test]
    fn test_tanh_activation() {
        let device = best_device::<f32>();
        let tensor =
            Tensor::from_vec_with_device(vec![0.0f32, 1000.0, -1000.0], &[3], device).unwrap();

        let result = tensor.tanh().unwrap();
        let data = result.into_data().unwrap();

        println!("Data: {}", data);

        assert!(data[0].abs() < 1e-6); // tanh(0) = 0
        assert!(data[1] > 0.99); // tanh(large) ≈ 1
        assert!(data[2] < -0.99); // tanh(-large) ≈ -1
    }

    #[test]
    fn test_relu_activation() {
        let device = best_device::<f32>();
        let tensor = Tensor::from_vec_with_device(vec![-2.0f32, 0.0, 3.0], &[3], device).unwrap();

        let result = tensor.relu().unwrap();
        let data = result.into_data().unwrap();

        assert_eq!(data[0], 0.0); // ReLU(-2) = 0
        assert_eq!(data[1], 0.0); // ReLU(0) = 0
        assert_eq!(data[2], 3.0); // ReLU(3) = 3
    }

    #[test]
    fn test_exp_function() {
        let device = best_device::<f32>();
        let tensor = Tensor::from_vec_with_device(vec![0.0f32, 1.0, 2.0], &[3], device).unwrap();

        let result = tensor.exp().unwrap();
        let data = result.into_data().unwrap();

        assert!((data[0] - 1.0).abs() < 1e-6); // e^0 = 1
        assert!((data[1] - std::f32::consts::E).abs() < 1e-6); // e^1 = e
        assert!((data[2] - std::f32::consts::E.powi(2)).abs() < 1e-5); // e^2
    }

    #[test]
    fn test_sum_reduction() {
        let device = best_device::<f32>();
        let tensor =
            Tensor::from_vec_with_device(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

        let result = tensor.sum(None).unwrap();
        let data: Vec<f32> = result.into_data().unwrap().iter().cloned().collect();

        assert_eq!(data[0], 10.0); // 1 + 2 + 3 + 4
    }

    #[test]
    fn test_mean_reduction() {
        let device = best_device::<f32>();
        let tensor =
            Tensor::from_vec_with_device(vec![2.0f32, 4.0, 6.0, 8.0], &[2, 2], device).unwrap();

        let result = tensor.mean(None).unwrap();
        let data: Vec<f32> = result.into_data().unwrap().iter().cloned().collect();

        // (2 + 4 + 6 + 8) / 4
        assert_eq!(data[0], 5.0);
    }

    #[test]
    fn test_max_reduction() {
        let device = best_device::<f32>();
        let tensor =
            Tensor::from_vec_with_device(vec![3.0f32, 1.0, 4.0, 2.0], &[2, 2], device).unwrap();

        let result = tensor.max_reduce(None).unwrap();
        let data: Vec<f32> = result.into_data().unwrap().iter().cloned().collect();

        assert_eq!(data[0], 4.0); // Maximum value
    }

    #[test]
    fn test_min_reduction() {
        let device = best_device::<f32>();
        let tensor =
            Tensor::from_vec_with_device(vec![3.0f32, 1.0, 4.0, 2.0], &[2, 2], device).unwrap();

        let result = tensor.min_reduce(None).unwrap();
        let data: Vec<f32> = result.into_data().unwrap().iter().cloned().collect();

        assert_eq!(data[0], 1.0); // Minimum value
    }

    #[test]
    fn test_reshape_operation() {
        let device = best_device::<f32>();
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut tensor = Tensor::from_vec_with_device(data, &[2, 3], device).unwrap();

        tensor.reshape(&[3, 2]).unwrap();

        assert_eq!(tensor.shape(), &[3, 2]);
        assert_eq!(tensor.size(), 6);
    }

    #[test]
    fn test_transpose_operation() {
        let device = best_device::<f32>();
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut tensor = Tensor::from_vec_with_device(data, &[2, 3], device).unwrap();

        tensor.transpose(None).unwrap();

        assert_eq!(tensor.shape(), &[3, 2]); // Dimensions swapped
    }

    #[test]
    fn test_unsqueeze_operation() {
        let device = best_device::<f32>();
        let mut tensor = Tensor::from_vec_with_device(vec![1.0f32, 2.0], &[2], device).unwrap();

        tensor.unsqueeze(0).unwrap();

        assert_eq!(tensor.shape(), &[1, 2]); // Added dimension at axis 0
    }

    #[test]
    fn test_squeeze_operation() {
        let device = best_device::<f32>();
        let mut tensor =
            Tensor::from_vec_with_device(vec![1.0f32, 2.0], &[1, 2, 1], device).unwrap();

        tensor.squeeze(None).unwrap();

        assert_eq!(tensor.shape(), &[2]); // Removed size-1 dimensions
    }

    #[test]
    fn test_broadcast_to() {
        let device = best_device::<f32>();
        let mut tensor = Tensor::from_vec_with_device(vec![1.0f32, 2.0], &[2], device).unwrap();

        tensor.broadcast_to(&[3, 2]).unwrap();

        assert_eq!(tensor.shape(), &[3, 2]);
        assert_eq!(tensor.size(), 6);
    }

    #[test]
    fn test_greater_comparison() {
        let device = best_device::<f32>();
        let a = Tensor::from_vec_with_device(vec![3.0f32, 1.0], &[2], device).unwrap();
        let b = Tensor::from_vec_with_device(vec![2.0f32, 2.0], &[2], device).unwrap();

        let result = a.greater(&b).unwrap();
        let data = result.into_data().unwrap();

        assert_eq!(data[0], 1.0); // 3 > 2 = true
        assert_eq!(data[1], 0.0); // 1 > 2 = false
    }

    #[test]
    fn test_less_comparison() {
        let device = best_device::<f32>();
        let a = Tensor::from_vec_with_device(vec![1.0f32, 3.0], &[2], device).unwrap();
        let b = Tensor::from_vec_with_device(vec![2.0f32, 2.0], &[2], device).unwrap();

        let result = a.less(&b).unwrap();
        let data = result.into_data().unwrap();

        assert_eq!(data[0], 1.0); // 1 < 2 = true
        assert_eq!(data[1], 0.0); // 3 < 2 = false
    }

    #[test]
    fn test_equal_comparison() {
        let device = best_device::<f32>();
        let a = Tensor::from_vec_with_device(vec![2.0f32, 3.0], &[2], device).unwrap();
        let b = Tensor::from_vec_with_device(vec![2.0f32, 4.0], &[2], device).unwrap();

        let result = a.equal(&b).unwrap();
        let data = result.into_data().unwrap();

        assert_eq!(data[0], 1.0); // 2 == 2 = true
        assert_eq!(data[1], 0.0); // 3 == 4 = false
    }

    #[test]
    fn test_all_equal_check() {
        let device = best_device::<f32>();
        let a = Tensor::from_vec_with_device(vec![1.0f32, 2.0], &[2], device).unwrap();
        let a = a.to_cpu().expect("Move out of empty tensor not allowed");
        let b = Tensor::from_vec_with_device(vec![1.0f32, 2.0], &[2], device).unwrap();
        let b = b.to_cpu().expect("Move out of empty tensor not allowed");
        let c = Tensor::from_vec_with_device(vec![1.0f32, 3.0], &[2], device).unwrap();
        let c = c.to_cpu().expect("Move out of empty tensor not allowed");

        assert!(a.all_equal(&b).unwrap()); // Same tensors
        assert!(!a.all_equal(&c).unwrap()); // Different tensors
    }

    #[test]
    fn test_slice_access() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();

        let slice = tensor.as_slice().unwrap();

        assert_eq!(slice.len(), 4);
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[3], 4.0);
    }

    #[test]
    fn test_range_slice() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();

        let slice = tensor.slice_range(1, 2).unwrap();

        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0], 2.0);
        assert_eq!(slice[1], 3.0);
    }
}
