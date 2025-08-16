#[allow(dead_code)]
use ferrox::backend::FerroxCudaF;
use ferrox::FerroxN;

/// Configuration for MLP model architecture
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MLPConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub num_samples: usize,
}
#[allow(dead_code)]
impl MLPConfig {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        num_samples: usize,
    ) -> Self {
        Self {
            input_size,
            hidden_size,
            output_size,
            num_samples,
        }
    }

    /// Configuration for regression tasks
    pub fn regression() -> Self {
        Self {
            input_size: 4,
            hidden_size: 8,
            output_size: 1,
            num_samples: 500,
        }
    }

    /// Configuration for binary classification
    pub fn binary_classification() -> Self {
        Self {
            input_size: 4,
            hidden_size: 16,
            output_size: 1,
            num_samples: 1000,
        }
    }

    /// Configuration for multiclass classification
    pub fn multiclass_classification(num_classes: usize) -> Self {
        Self {
            input_size: 4,
            hidden_size: 16,
            output_size: num_classes,
            num_samples: 1000,
        }
    }


}

/// Training configuration supporting multiple optimizers
#[derive(Debug, Clone)]
pub struct TrainingConfig<T>
where
    T: FerroxCudaF,
{
    // Common training parameters
    pub batch_size: usize,
    pub num_epochs: usize,
    pub learning_rate: Option<T>,
    pub print_every: usize,

    // Optimizer selection and parameters
    pub optimizer: &'static str,

    // SGD-specific parameters
    pub momentum: Option<T>,
    pub nesterov: bool,
    pub decay: Option<T>,

    // Adam-specific parameters
    pub beta1: Option<T>,
    pub beta2: Option<T>,
    pub eps: Option<T>,
    pub amsgrad: bool,
}
#[allow(dead_code)]
impl<T> TrainingConfig<T>
where
    T: FerroxCudaF + FerroxN,
{
    /// Default configuration for stable training
    pub fn default_stable() -> Self {
        Self {
            batch_size: 32,
            num_epochs: 100,
            learning_rate: Some(FerroxN::from_f32(0.0001).unwrap()),
            print_every: 10,
            optimizer: "Adam",

            // SGD parameters
            momentum: Some(FerroxN::from_f32(0.9).unwrap()),
            nesterov: false,
            decay: Some(FerroxN::from_f32(0.0001).unwrap()),

            // Adam parameters
            beta1: Some(FerroxN::from_f32(0.9).unwrap()),
            beta2: Some(FerroxN::from_f32(0.999).unwrap()),
            eps: Some(FerroxN::from_f32(1e-8).unwrap()),
            amsgrad: false,
        }
    }

    pub fn cnn_training() -> Self {
        Self {
            batch_size: 32,
            num_epochs: 100,
            learning_rate: Some(FerroxN::from_f32(0.0001).unwrap()), // Lower LR for CNN
            print_every: 5,
            optimizer: "Adam",

            momentum: Some(FerroxN::from_f32(0.9).unwrap()),
            nesterov: false,
            decay: Some(FerroxN::from_f32(0.0001).unwrap()),

            beta1: Some(FerroxN::from_f32(0.9).unwrap()),
            beta2: Some(FerroxN::from_f32(0.999).unwrap()),
            eps: Some(FerroxN::from_f32(1e-8).unwrap()),
            amsgrad: false,
        }
    }

    /// Fast training configuration for quick testing
    pub fn fast() -> Self {
        Self {
            batch_size: 100,
            num_epochs: 50,
            learning_rate: Some(FerroxN::from_f32(0.001).unwrap()),
            print_every: 5,
            optimizer: "Adam",

            momentum: Some(FerroxN::from_f32(0.9).unwrap()),
            nesterov: false,
            decay: Some(FerroxN::from_f32(0.0001).unwrap()),

            beta1: Some(FerroxN::from_f32(0.9).unwrap()),
            beta2: Some(FerroxN::from_f32(0.999).unwrap()),
            eps: Some(FerroxN::from_f32(1e-8).unwrap()),
            amsgrad: false,
        }
    }

    /// SGD-specific configuration
    pub fn sgd(learning_rate: T) -> Self {
        Self {
            batch_size: 32,
            num_epochs: 100,
            learning_rate: Some(learning_rate),
            print_every: 10,
            optimizer: "SGD",

            momentum: Some(FerroxN::from_f32(0.9).unwrap()),
            nesterov: true,
            decay: Some(FerroxN::from_f32(0.0001).unwrap()),

            beta1: None,
            beta2: None,
            eps: None,
            amsgrad: false,
        }
    }
}

impl<T> Default for TrainingConfig<T>
where
    T: FerroxCudaF + FerroxN,
{
    fn default() -> Self {
        Self::default_stable()
    }
}
