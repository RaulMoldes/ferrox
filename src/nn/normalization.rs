use crate::backend::number::{CPUNumber, GPUFloat};
use crate::graph::Engine;
use crate::graph::node::NodeId;
use crate::nn::{Module, Parameter};
use crate::tensor::Tensor;

/// NOTE FROM THE AUTHOR: When in this module I refer to an "affine transformation", I mean the geometric definition of affine transformation,
/// applied to a vector space. These affine transformations or affine applications can be defined as a linear transformation followed by a translation.
///
/// ```text
/// y = Ax + b
/// ```
/// where `A` is a matrix, `x` is the input vector, `b` is a bias vector, and `y` is the output vector.
/// These can be understood also as a rotation, scaling, shearing, or translation of the input vector.
///
/// In the case of batch normalization, the affine transformation is applied to the normalized input.
/// As described in the paper "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
/// The affine parameters gamma (γ) and beta (β) are learnable parameters that scale and shift the normalized input.
///
/// The objective of these parameters is to
/// allow the model to learn the optimal scale and shift for each feature,
/// which can help the model to adapt better to the data distribution.
///
/// Batch Normalization layer.
///
/// Applies batch normalization over a batch of inputs as described in the paper
/// "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift".
///
/// # Mathematical Definition
///
/// During training:
/// ```text
/// μ = (1/N) * Σᵢ xᵢ                    // batch mean
/// σ² = (1/N) * Σᵢ (xᵢ - μ)²           // batch variance
/// x̂ᵢ = (xᵢ - μ) / √(σ² + ε)           // normalized input
/// yᵢ = γ * x̂ᵢ + β                     // scaled and shifted output --> Affine transformation
/// ```
///
/// During inference:
/// ```text
/// x̂ᵢ = (xᵢ - running_mean) / √(running_var + ε)
/// yᵢ = γ * x̂ᵢ + β
/// ```
///
/// # Parameters
///
/// * `weight` (γ): Learnable scale parameter of shape (num_features,)
/// * `bias` (β): Learnable shift parameter of shape (num_features,)
/// * `running_mean`: Running average of batch means (not learnable)
/// * `running_var`: Running average of batch variances (not learnable)
///
/// # Properties
///
/// - Reduces internal covariate shift
/// - Allows higher learning rates
/// - Acts as regularizer (reduces need for dropout)
/// - Makes networks less sensitive to initialization
#[derive(Debug)]
pub struct BatchNorm1d<T>
where
    T: GPUFloat,
{
    /// Number of features/channels
    pub num_features: usize,

    /// Learnable scale parameter (γ)
    pub weight: Parameter<T>,

    /// Learnable shift parameter (β)
    pub bias: Parameter<T>,

    /// Running mean for inference (not learnable)
    pub running_mean: Tensor<T>,

    /// Running variance for inference (not learnable)
    pub running_var: Tensor<T>,

    /// Small constant added to variance for numerical stability
    pub eps: T,

    /// Momentum for running statistics update
    pub momentum: T,

    /// Whether to track running statistics
    pub track_running_stats: bool,

    /// Training mode flag
    training: bool,

    /// Cached parameter nodes
    weight_node_cache: std::cell::RefCell<Option<NodeId>>,
    bias_node_cache: std::cell::RefCell<Option<NodeId>>,
}

impl<T> BatchNorm1d<T>
where
    T: GPUFloat + From<f64>,
{
    /// Creates a new BatchNorm1d layer.
    ///
    /// # Arguments
    ///
    /// * `num_features` - Number of features in the input
    /// * `eps` - Small constant for numerical stabilitydefault: 1e-5)
    /// * `momentum` - Momentum for running statistics (default: 0.1)
    /// * `affine` - Whether to use learnable affinity parameters (default: true)
    /// * `track_running_stats` - Whether to track running statistics (default: true)
    pub fn new(
        num_features: usize,
        eps: f64,
        momentum: f64,
        affine: bool,
        track_running_stats: bool,
    ) -> Self {
        let eps_val = T::from(eps);
        let momentum_val = T::from(momentum);

        // Initialize weight and bias
        let (weight, bias) = if affine {
            let weight_data = Tensor::ones(&[num_features]).expect("Failed to create ones tensor");; // Initialize γ to 1
            let bias_data = Tensor::zeros(&[num_features]).expect("Failed to create zeroed tensor");; // Initialize β to 0
            (
                Parameter::new_named(weight_data, "weight".to_string()),
                Parameter::new_named(bias_data, "bias".to_string()),
            )
        } else {
            // Create dummy parameters that won't be used
            let dummy_weight = Tensor::ones(&[num_features]).expect("Failed to create ones tensor");;
            let dummy_bias = Tensor::zeros(&[num_features]).expect("Failed to create zeroed tensor");;
            (Parameter::new(dummy_weight), Parameter::new(dummy_bias))
        };

        // Initialize running statistics
        let running_mean = Tensor::zeros(&[num_features]).expect("Failed to create zeroed tensor");;
        let running_var = Tensor::ones(&[num_features]).expect("Failed to create ones tensor");; // Initialize to 1

        Self {
            num_features,
            weight,
            bias,
            running_mean,
            running_var,
            eps: eps_val,
            momentum: momentum_val,
            track_running_stats,
            training: true,
            weight_node_cache: std::cell::RefCell::new(None),
            bias_node_cache: std::cell::RefCell::new(None),
        }
    }

    /// Creates a BatchNorm1d with default parameters.
    pub fn default_config(num_features: usize) -> Self {
        Self::new(num_features, 1e-5, 0.1, true, true)
    }

    /// Returns the number of features.
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Returns the epsilon value.
    pub fn eps(&self) -> T {
        self.eps
    }

    /// Returns the momentum value.
    pub fn momentum(&self) -> T {
        self.momentum
    }

    /// Updates running statistics during training.
    ///
    /// This should be called after computing batch statistics during training.
    /// Updates: running_mean = (1 - momentum) * running_mean + momentum * batch_mean
    /// We are not able to track running statistics inside the forward pass currently due to the actual architecture.
    /// Rust prohibits mutable access to self during forward pass, as we already have a mutable borrow of self in the forward method.
    // As a workaroud, we can use a separate method to update running statistics after the forward pass.´
    #[allow(dead_code)]
    fn update_running_stats(&mut self, batch_mean: &Tensor<T>, batch_var: &Tensor<T>) {
        if !self.track_running_stats || !self.training {
            return;
        }

        let one_minus_momentum = <T as CPUNumber>::one() - self.momentum;

        // Update running mean: (1 - momentum) * old + momentum * new
        for i in 0..self.num_features {
            let old_mean = self.running_mean[i];
            let new_mean = batch_mean[i];
            self.running_mean.data.as_slice_mut().unwrap()[i] =
                one_minus_momentum * old_mean + self.momentum * new_mean;
        }

        // Update running variance: (1 - momentum) * old + momentum * new
        for i in 0..self.num_features {
            let old_var = self.running_var[i];
            let new_var = batch_var[i];
            self.running_var.data.as_slice_mut().unwrap()[i] =
                one_minus_momentum * old_var + self.momentum * new_var;
        }
    }
}

impl<T> Module<T> for BatchNorm1d<T>
where
    T: GPUFloat,
{
    fn forward(&self, graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
        let input_shape = graph.get_shape(input);

        // Validate input shape: should be [batch_size, num_features] or [batch_size, num_features, ...]
        if input_shape.len() < 2 {
            return Err("BatchNorm1d requires input with at least 2 dimensions".to_string());
        }

        let batch_size = input_shape[0];
        let features = input_shape[1];

        if features != self.num_features {
            return Err(format!(
                "Input feature size {} doesn't match BatchNorm1d feature size {}",
                features, self.num_features
            ));
        }

        // Get or create parameter nodes
        let weight_node = {
            let mut cache = self.weight_node_cache.borrow_mut();
            if let Some(node_id) = *cache {
                node_id
            } else {
                let new_node = Parameter::create_in_graph(graph, self.weight.data.clone());
                *cache = Some(new_node);
                new_node
            }
        };

        let bias_node = {
            let mut cache = self.bias_node_cache.borrow_mut();
            if let Some(node_id) = *cache {
                node_id
            } else {
                let new_node = Parameter::create_in_graph(graph, self.bias.data.clone());
                *cache = Some(new_node);
                new_node
            }
        };

        if self.training {
            // Training mode: compute batch statistics

            // For multi-dimensional inputs, we need to compute statistics over all dimensions except the feature dimension
            // For [N, C, H, W], we compute over [N, H, W] for each channel C
            // For [N, C], we compute over [N] for each channel C
            // Where N is batch size, C is number of features, H and W are spatial dimensions

            // Reshape input to [batch_size * other_dims, num_features] for easier computation
            let total_elements = input_shape.iter().product::<usize>();
            let other_dims = total_elements / (batch_size * features);
            let reshaped_input = graph.reshape(input, vec![batch_size * other_dims, features])?;

            // Compute batch mean: mean over batch dimension (dimension 0)
            // As we have reshaped to [batch_size * other_dims, features], we sum over dimension 0
            // This gives us a mean for each feature across the batch
            // batch_mean = (1 / N) * Σ x_i
            let batch_mean = graph.summation(reshaped_input, Some(vec![0]))?;
            let n_elements =
                <T as CPUNumber>::from_f64(batch_size as f64 * other_dims as f64).unwrap();
            let mean = graph.mul_scalar(batch_mean, <T as CPUNumber>::one() / n_elements)?;

            // Compute batch variance
            // Broadcast mean to input shape
            let broadcasted_mean =
                graph.broadcast_to(mean, vec![batch_size * other_dims, features])?;
            let negated = graph.negate(broadcasted_mean)?;
            // Center the input: x - mean
            let centered = graph.add(reshaped_input, negated)?;

            // Compute squared differences ( Variance  = 1 / N * Σ (x - μ)² )
            // squared = (x - mean) * (x - mean)
            let squared = graph.mul(centered, centered)?;
            let sum_squared = graph.summation(squared, Some(vec![0]))?;
            let variance = graph.mul_scalar(sum_squared, <T as CPUNumber>::one() / n_elements)?;

            // Normalize: (x - mean) / sqrt(var + eps)
            let eps_tensor = graph.tensor_from_vec(vec![self.eps; features], &[features], false)?;
            let var_plus_eps = graph.add(variance, eps_tensor)?;
            let sqrt_var = graph.sqrt(var_plus_eps)?;

            // Broadcast for division
            let broadcasted_sqrt_var =
                graph.broadcast_to(sqrt_var, vec![batch_size * other_dims, features])?;
            let normalized = graph.div(centered, broadcasted_sqrt_var)?;

            // Apply affinity transformation: γ * normalized + β
            let broadcasted_weight =
                graph.broadcast_to(weight_node, vec![batch_size * other_dims, features])?;
            let broadcasted_bias =
                graph.broadcast_to(bias_node, vec![batch_size * other_dims, features])?;

            let scaled = graph.mul(normalized, broadcasted_weight)?;
            let output = graph.add(scaled, broadcasted_bias)?;

            // Reshape back to original shape
            let final_output = graph.reshape(output, input_shape)?;

            // Here we should update running statistics (this would require mutable access to self during forward)
            // A workararound would be to add a separate method to update running statistics after the forward pass.
            // Look at the [`update_running_stats`] method.

            // let variance_data = graph.get_data(variance);
            // let mean_data = graph.get_data(mean);
            // self.update_running_stats(&mean_data, &variance_data);  --> This requires mutable access to self,
            // so we cannot do it here directly.

            Ok(final_output)
        } else {
            // Inference mode: use running statistics.
            // Basically in inference mode we don't compute batch statistics,
            // INSTEAD we should use the running statistics that were computed during training.
            // This is why it is very important to ensure that running statistics are updated during training. after the forward pass.

            if !self.track_running_stats {
                return Err("Running statistics are not tracked in inference mode".to_string());
            }

            // Create tensors for running statistics
            let running_mean_node =
                graph.tensor_from_vec(self.running_mean.to_vec()?, &[self.num_features], false)?;

            let running_var_node =
                graph.tensor_from_vec(self.running_var.to_vec()?, &[self.num_features], false)?;

            // Normalize using running statistics
            let eps_tensor = graph.tensor_from_vec(
                vec![self.eps; self.num_features],
                &[self.num_features],
                false,
            )?;
            let var_plus_eps = graph.add(running_var_node, eps_tensor)?;
            let sqrt_var = graph.sqrt(var_plus_eps)?;

            // Broadcast statistics to input shape
            let total_elements = input_shape.iter().product::<usize>();
            let other_dims = total_elements / (batch_size * features);
            let broadcast_shape = vec![batch_size * other_dims, features];

            let reshaped_input = graph.reshape(input, broadcast_shape.clone())?;
            let broadcasted_mean =
                graph.broadcast_to(running_mean_node, broadcast_shape.clone())?;
            let broadcasted_sqrt_var = graph.broadcast_to(sqrt_var, broadcast_shape.clone())?;

            // Normalize
            let negated = graph.negate(broadcasted_mean)?;
            let centered = graph.add(reshaped_input, negated)?;
            let normalized = graph.div(centered, broadcasted_sqrt_var)?;

            // Apply affine transformation
            let broadcasted_weight = graph.broadcast_to(weight_node, broadcast_shape.clone())?;
            let broadcasted_bias = graph.broadcast_to(bias_node, broadcast_shape)?;

            let scaled = graph.mul(normalized, broadcasted_weight)?;
            let output = graph.add(scaled, broadcasted_bias)?;

            // Reshape back to original shape
            graph.reshape(output, input_shape)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![&mut self.weight, &mut self.bias]
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// Layer Normalization layer.
///
/// Applies layer normalization over the last dimension(s) of the input as described in
/// "Layer Normalization" by Ba et al.
///
/// The key difference from Batch Normalization is that LayerNorm normalizes across features for each sample independently,
/// making it suitable for variable-length sequences and recurrent architectures (RNNs, Transformers, LSTMs).
///
/// On the other hand, Batch Normalization normalizes across the batch dimension, which can be problematic for small batch sizes or variable-length inputs.
/// Batch norm is typically used in convolutional networks, while layer norm is preferred for sequential models.
///
/// Additionally, layer normalization does not require running statistics, as it computes the mean and variance for each sample independently.
/// This makes it also more stable for small batch sizes. I think it can be considered as a generalization of Batch Normalization, as it has a similar purpose but operates differently.
///
/// # Mathematical Definition
///
/// ```text
/// μ = (1/H) * Σᵢ xᵢ                    // layer mean (over normalized dimensions)
/// σ² = (1/H) * Σᵢ (xᵢ - μ)²           // layer variance
/// x̂ᵢ = (xᵢ - μ) / √(σ² + ε)           // normalized input
/// yᵢ = γ * x̂ᵢ + β                     // scaled and shifted output
/// ```
/// # Parameters
///
/// * `weight` (γ): Learnable scale parameter
/// * `bias` (β): Learnable shift parameter
#[derive(Debug)]
pub struct LayerNorm<T>
where
    T: GPUFloat,
{
    /// Shape of normalized dimensions
    pub normalized_shape: Vec<usize>,

    /// Learnable scale parameter (γ)
    pub weight: Parameter<T>,

    /// Learnable shift parameter (β)
    pub bias: Parameter<T>,

    /// Small constant added to variance for numerical stability
    pub eps: T,

    /// Whether to use learnable affine parameters
    pub elementwise_affine: bool,

    /// Training mode flag
    training: bool,

    /// Cached parameter nodes
    weight_node_cache: std::cell::RefCell<Option<NodeId>>,
    bias_node_cache: std::cell::RefCell<Option<NodeId>>,
}

impl<T> LayerNorm<T>
where
    T: GPUFloat + From<f64>,
{
    /// Creates a new LayerNorm layer.
    ///
    /// # Arguments
    ///
    /// * `normalized_shape` - Shape of the dimensions to normalize over
    /// * `eps` - Small constant for numerical stabilitydefault: 1e-5)
    /// * `elementwise_affine` - Whether to use learnable affine parameters (default: true)
    pub fn new(normalized_shape: Vec<usize>, eps: f64, elementwise_affine: bool) -> Self {
        let eps_val = T::from(eps);

        // Calculate total size of normalized dimensions
        let total_size: usize = normalized_shape.iter().product();

        // Initialize weight and bias
        let (weight, bias) = if elementwise_affine {
            let weight_data = Tensor::ones(&[total_size]).expect("Failed to create ones tensor"); // Initialize γ to 1
            let bias_data = Tensor::zeros(&[total_size]).expect("Failed to create zeroed tensor"); // Initialize β to 0
            (
                Parameter::new_named(weight_data, "weight".to_string()),
                Parameter::new_named(bias_data, "bias".to_string()),
            )
        } else {
            // Create dummy parameters
            let dummy_weight = Tensor::ones(&[total_size]).expect("Failed to create zeroed tensor");
            let dummy_bias = Tensor::zeros(&[total_size]).expect("Failed to create zeroed tensor");
            (Parameter::new(dummy_weight), Parameter::new(dummy_bias))
        };

        Self {
            normalized_shape,
            weight,
            bias,
            eps: eps_val,
            elementwise_affine,
            training: true,
            weight_node_cache: std::cell::RefCell::new(None),
            bias_node_cache: std::cell::RefCell::new(None),
        }
    }

    /// Creates a LayerNorm for the last dimension only.
    ///
    /// This is the most common use case.
    pub fn new_1d(normalized_size: usize, eps: f64) -> Self {
        Self::new(vec![normalized_size], eps, true)
    }

    /// Creates a LayerNorm with default parameters.
    pub fn default_config(normalized_shape: Vec<usize>) -> Self {
        Self::new(normalized_shape, 1e-5, true)
    }

    /// Returns the normalized shape.
    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }

    /// Returns the epsilon value.
    pub fn eps(&self) -> T {
        self.eps
    }
}

impl<T> Module<T> for LayerNorm<T>
where
    T: GPUFloat,
{
    fn forward(&self, graph: &mut Engine<T>, input: NodeId) -> Result<NodeId, String> {
        let input_shape = graph.get_shape(input);

        // Validate that input shape ends with normalized_shape
        if input_shape.len() < self.normalized_shape.len() {
            return Err(format!(
                "Input has {} dimensions but normalized_shape requires at least {}",
                input_shape.len(),
                self.normalized_shape.len()
            ));
        }

        let start_idx = input_shape.len() - self.normalized_shape.len();
        if input_shape[start_idx..] != self.normalized_shape {
            return Err(format!(
                "Input shape {:?} doesn't end with normalized_shape {:?}",
                input_shape, self.normalized_shape
            ));
        }

        // Calculate dimensions for normalization
        let batch_dims: Vec<usize> = input_shape[..start_idx].to_vec();
        let norm_dims: Vec<usize> = input_shape[start_idx..].to_vec();
        let norm_size: usize = norm_dims.iter().product();

        // Reshape for easier computation: [batch_dims..., norm_size]
        let mut reshaped_dims = batch_dims.clone();
        reshaped_dims.push(norm_size);
        let reshaped_input = graph.reshape(input, reshaped_dims.clone())?;

        // Compute mean and variance over the last dimension (normalized dimensions)
        let last_dim = reshaped_dims.len() - 1;

        // Mean over normalized dimensions
        let sum = graph.summation(reshaped_input, Some(vec![last_dim]))?;
        let n_elements = <T as CPUNumber>::from_f64(norm_size as f64).unwrap();
        let mean = graph.mul_scalar(sum, <T as CPUNumber>::one() / n_elements)?;

        // Broadcast mean back for subtraction
        let mean_expanded = graph.expand_dims_at(mean, last_dim)?;
        let mean_broadcasted = graph.broadcast_to(mean_expanded, reshaped_dims.clone())?;

        // Center the input
        let negated = graph.negate(mean_broadcasted)?;
        // Centered input: x - mean
        let centered = graph.add(reshaped_input, negated)?;

        // Compute variance
        let squared = graph.mul(centered, centered)?;
        let var_sum = graph.summation(squared, Some(vec![last_dim]))?;
        let variance = graph.mul_scalar(var_sum, <T as CPUNumber>::one() / n_elements)?;

        // Add epsilon and take square root
        let eps_tensor = {
            let variance_shape = graph.get_shape(variance);
            let num_elements = variance_shape.iter().product::<usize>();

            let eps_vec = vec![self.eps; num_elements];
            graph.tensor_from_vec(eps_vec, &variance_shape, false)?
        };

        let var_plus_eps = graph.add(variance, eps_tensor)?;
        let std = graph.sqrt(var_plus_eps)?;

        // Broadcast std for division
        let std_expanded = graph.expand_dims_at(std, last_dim)?;
        let std_broadcasted = graph.broadcast_to(std_expanded, reshaped_dims.clone())?;

        // Normalize
        let normalized = graph.div(centered, std_broadcasted)?;

        // Apply affine transformation if enabled
        let output = if self.elementwise_affine {
            // Get parameter nodes
            let weight_node = {
                let mut cache = self.weight_node_cache.borrow_mut();
                if let Some(node_id) = *cache {
                    node_id
                } else {
                    let new_node = Parameter::create_in_graph(graph, self.weight.data.clone());
                    *cache = Some(new_node);
                    new_node
                }
            };

            let bias_node = {
                let mut cache = self.bias_node_cache.borrow_mut();
                if let Some(node_id) = *cache {
                    node_id
                } else {
                    let new_node = Parameter::create_in_graph(graph, self.bias.data.clone());
                    *cache = Some(new_node);
                    new_node
                }
            };

            // Broadcast parameters to match normalized shape
            let weight_broadcasted = graph.broadcast_to(weight_node, reshaped_dims.clone())?;
            let bias_broadcasted = graph.broadcast_to(bias_node, reshaped_dims)?;

            // Apply: γ * normalized + β
            let scaled = graph.mul(normalized, weight_broadcasted)?;
            graph.add(scaled, bias_broadcasted)?
        } else {
            normalized
        };

        // Reshape back to original input shape
        graph.reshape(output, input_shape)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        if self.elementwise_affine {
            vec![&self.weight, &self.bias]
        } else {
            vec![]
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        if self.elementwise_affine {
            vec![&mut self.weight, &mut self.bias]
        } else {
            vec![]
        }
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}
