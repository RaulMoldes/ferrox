use ferrox::backend::{Device, FerroxCudaF};
use ferrox::graph::{AutoFerroxEngine, NodeId};
use ferrox::nn::{
    layers::{BatchNorm, Conv2d, Dropout, Flatten, GlobalAvgPool2d, Linear, ReLU},
    Module,
};
use std::collections::HashMap;

/// Multi-Layer Perceptron for regression tasks
#[derive(Debug)]
#[allow(dead_code)]
pub struct MLPRegressor<T>
where
    T: FerroxCudaF,
{
    hidden1: Linear<T>,
    activation1: ReLU<T>,
    hidden2: Linear<T>,
    activation2: ReLU<T>,
    output: Linear<T>,
    training: bool,
}
#[allow(dead_code)]
impl<T> MLPRegressor<T>
where
    T: FerroxCudaF,
{
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, device: Device) -> Self {
        Self {
            hidden1: Linear::new_with_device(input_size, hidden_size, true, device),
            activation1: ReLU::new(),
            hidden2: Linear::new_with_device(hidden_size, hidden_size, true, device),
            activation2: ReLU::new(),
            output: Linear::new_with_device(hidden_size, output_size, false, device),
            training: true,
        }
    }

    pub fn get_parameters(&self) -> Vec<&ferrox::nn::parameter::Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.hidden1.parameters());
        params.extend(self.hidden2.parameters());
        params.extend(self.output.parameters());
        params
    }
}

impl<T> Module<T> for MLPRegressor<T>
where
    T: FerroxCudaF + rand_distr::num_traits::FromPrimitive,
{
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        let hidden1_out = self.hidden1.forward(graph, input)?;
        let activated1 = self.activation1.forward(graph, hidden1_out)?;
        let hidden2_out = self.hidden2.forward(graph, activated1)?;
        let activated2 = self.activation2.forward(graph, hidden2_out)?;
        let output = self.output.forward(graph, activated2)?;
        Ok(output)
    }

    fn create_parameters_in_graph(
        &self,
        engine: &mut AutoFerroxEngine<T>,
    ) -> HashMap<String, NodeId> {
        let mut param_map = HashMap::new();

        let hidden1_params = self.hidden1.create_parameters_in_graph(engine);
        for (param_name, node_id) in hidden1_params {
            param_map.insert(format!("hidden1_{}", param_name), node_id);
        }

        let hidden2_params = self.hidden2.create_parameters_in_graph(engine);
        for (param_name, node_id) in hidden2_params {
            param_map.insert(format!("hidden2_{}", param_name), node_id);
        }

        let output_params = self.output.create_parameters_in_graph(engine);
        for (param_name, node_id) in output_params {
            param_map.insert(format!("output_{}", param_name), node_id);
        }

        param_map
    }

    fn parameters(&self) -> Vec<&ferrox::nn::parameter::Parameter<T>> {
        self.get_parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut ferrox::nn::parameter::Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.hidden1.parameters_mut());
        params.extend(self.hidden2.parameters_mut());
        params.extend(self.output.parameters_mut());
        params
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.hidden1.set_training(training);
        self.activation1.set_training(training);
        self.hidden2.set_training(training);
        self.activation2.set_training(training);
        self.output.set_training(training);
    }
}

/// Multi-Layer Perceptron for classification tasks
#[derive(Debug)]
#[allow(dead_code)]
pub struct MLPClassifier<T>
where
    T: FerroxCudaF,
{
    hidden1: Linear<T>,
    activation1: ReLU<T>,
    hidden2: Linear<T>,
    activation2: ReLU<T>,
    output: Linear<T>,
    training: bool,
}

#[allow(dead_code)]
impl<T> MLPClassifier<T>
where
    T: FerroxCudaF,
{
    /// Create new MLP classifier
    /// num_classes: 1 for binary, >1 for multiclass
    pub fn new(input_size: usize, hidden_size: usize, num_classes: usize, device: Device) -> Self {
        Self {
            hidden1: Linear::new_with_device(input_size, hidden_size, true, device),
            activation1: ReLU::new(),
            hidden2: Linear::new_with_device(hidden_size, hidden_size, true, device),
            activation2: ReLU::new(),
            output: Linear::new_with_device(hidden_size, num_classes, true, device),
            training: true,
        }
    }

    pub fn get_parameters(&self) -> Vec<&ferrox::nn::parameter::Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.hidden1.parameters());
        params.extend(self.hidden2.parameters());
        params.extend(self.output.parameters());
        params
    }
}

impl<T> Module<T> for MLPClassifier<T>
where
    T: FerroxCudaF + rand_distr::num_traits::FromPrimitive,
{
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        let hidden1_out = self.hidden1.forward(graph, input)?;
        let activated1 = self.activation1.forward(graph, hidden1_out)?;
        let hidden2_out = self.hidden2.forward(graph, activated1)?;
        let activated2 = self.activation2.forward(graph, hidden2_out)?;
        // Output logits (no activation for classification)
        let logits = self.output.forward(graph, activated2)?;
        Ok(logits)
    }

    fn create_parameters_in_graph(
        &self,
        engine: &mut AutoFerroxEngine<T>,
    ) -> HashMap<String, NodeId> {
        let mut param_map = HashMap::new();

        let hidden1_params = self.hidden1.create_parameters_in_graph(engine);
        for (param_name, node_id) in hidden1_params {
            param_map.insert(format!("hidden1_{}", param_name), node_id);
        }

        let hidden2_params = self.hidden2.create_parameters_in_graph(engine);
        for (param_name, node_id) in hidden2_params {
            param_map.insert(format!("hidden2_{}", param_name), node_id);
        }

        let output_params = self.output.create_parameters_in_graph(engine);
        for (param_name, node_id) in output_params {
            param_map.insert(format!("output_{}", param_name), node_id);
        }

        param_map
    }

    fn set_training(&mut self, training: bool) {
        self.training = training
    }
}

/// Convolutional Neural Network for image classification
/// Simple CNN with conv layers, pooling, and fully connected output
#[derive(Debug)]
#[allow(dead_code)]
pub struct CNNClassifier<T>
where
    T: FerroxCudaF,
{
    // First convolutional block
    conv1: Conv2d<T>,
    batch_norm1: BatchNorm<T>,
    activation1: ReLU<T>,

    // Second convolutional block
    conv2: Conv2d<T>,
    batch_norm2: BatchNorm<T>,
    activation2: ReLU<T>,

    // Global average pooling instead of MaxPool2d
    global_pool: GlobalAvgPool2d<T>,

    // Flatten and fully connected layers
    flatten: Flatten<T>,
    fc1: Linear<T>,
    activation3: ReLU<T>,
    dropout: Dropout<T>,
    output: Linear<T>,
    training: bool,
}

#[allow(dead_code)]
impl<T> CNNClassifier<T>
where
    T: FerroxCudaF,
{
    /// Create new CNN classifier for image classification
    /// input_channels: number of input channels (1 for grayscale, 3 for RGB)
    /// num_classes: number of output classes
    /// image_size: input image height and width (assumed square)
    pub fn new(
        input_channels: usize,
        num_classes: usize,
        image_size: usize,
        device: Device,
    ) -> Self {
        // Scale network complexity based on image size
        let base_filters = if image_size <= 16 {
            16
        } else if image_size <= 32 {
            32
        } else {
            64
        };
        let conv1_filters = base_filters;
        let conv2_filters = base_filters * 2;
        let fc_hidden = if image_size <= 32 { 64 } else { 128 };

        Self {
            conv1: Conv2d::new(
                input_channels,
                conv1_filters,
                (3, 3),
                (1, 1),
                (1, 1),
                true,
                device,
            ),
            batch_norm1: BatchNorm::new(conv1_filters, 1e-3, 0.1, true, device),
            activation1: ReLU::new(),

            conv2: Conv2d::new(
                conv1_filters,
                conv2_filters,
                (3, 3),
                (1, 1),
                (1, 1),
                true,
                device,
            ),
            batch_norm2: BatchNorm::new(conv2_filters, 1e-3, 0.1, true, device),
            activation2: ReLU::new(),

            global_pool: GlobalAvgPool2d::new(false),
            flatten: Flatten::new_batch_first(),
            fc1: Linear::new_with_device(conv2_filters, fc_hidden, true, device),
            activation3: ReLU::new(),
            dropout: Dropout::new(0.3),
            output: Linear::new_with_device(fc_hidden, num_classes, true, device),
            training: true,
        }
    }

    pub fn get_parameters(&self) -> Vec<&ferrox::nn::parameter::Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.batch_norm1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.batch_norm2.parameters());
        params.extend(self.fc1.parameters());
        params.extend(self.output.parameters());
        params
    }
}

impl<T> Module<T> for CNNClassifier<T>
where
    T: FerroxCudaF,
{
    fn forward(&self, graph: &mut AutoFerroxEngine<T>, input: NodeId) -> Result<NodeId, String> {
        // First convolutional block

        let conv1_out = self.conv1.forward(graph, input)?;
        let bn1_out = self.batch_norm1.forward(graph, conv1_out)?;
        let act1_out = self.activation1.forward(graph, bn1_out)?;

        // Second convolutional block
        let conv2_out = self.conv2.forward(graph, act1_out)?;
        let bn2_out = self.batch_norm2.forward(graph, conv2_out)?;
        let act2_out = self.activation2.forward(graph, bn2_out)?;

        // Global average pooling and classification
        let pooled = self.global_pool.forward(graph, act2_out)?;
        let flattened = self.flatten.forward(graph, pooled)?;
        let fc1_out = self.fc1.forward(graph, flattened)?;
        let act3_out = self.activation3.forward(graph, fc1_out)?;
        let dropout_out = self.dropout.forward(graph, act3_out)?;
        let logits = self.output.forward(graph, dropout_out)?;

        Ok(logits)
    }

    fn create_parameters_in_graph(
        &self,
        engine: &mut AutoFerroxEngine<T>,
    ) -> HashMap<String, NodeId> {
        let mut param_map = HashMap::new();

        let conv1_params = self.conv1.create_parameters_in_graph(engine);
        for (param_name, node_id) in conv1_params {
            param_map.insert(format!("conv1_{}", param_name), node_id);
        }

        let bn1_params = self.batch_norm1.create_parameters_in_graph(engine);
        for (param_name, node_id) in bn1_params {
            param_map.insert(format!("batch_norm1_{}", param_name), node_id);
        }

        let conv2_params = self.conv2.create_parameters_in_graph(engine);
        for (param_name, node_id) in conv2_params {
            param_map.insert(format!("conv2_{}", param_name), node_id);
        }

        let bn2_params = self.batch_norm2.create_parameters_in_graph(engine);
        for (param_name, node_id) in bn2_params {
            param_map.insert(format!("batch_norm2_{}", param_name), node_id);
        }

        let fc1_params = self.fc1.create_parameters_in_graph(engine);
        for (param_name, node_id) in fc1_params {
            param_map.insert(format!("fc1_{}", param_name), node_id);
        }

        let output_params = self.output.create_parameters_in_graph(engine);
        for (param_name, node_id) in output_params {
            param_map.insert(format!("output_{}", param_name), node_id);
        }

        param_map
    }

    fn parameters(&self) -> Vec<&ferrox::nn::parameter::Parameter<T>> {
        self.get_parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut ferrox::nn::parameter::Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters_mut());
        params.extend(self.batch_norm1.parameters_mut());
        params.extend(self.conv2.parameters_mut());
        params.extend(self.batch_norm2.parameters_mut());
        params.extend(self.fc1.parameters_mut());
        params.extend(self.output.parameters_mut());
        params
    }

    fn training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.conv1.set_training(training);
        self.batch_norm1.set_training(training);
        self.activation1.set_training(training);
        self.conv2.set_training(training);
        self.batch_norm2.set_training(training);
        self.activation2.set_training(training);
        self.global_pool.set_training(training);
        self.flatten.set_training(training);
        self.fc1.set_training(training);
        self.activation3.set_training(training);
        self.dropout.set_training(training);
        self.output.set_training(training);
    }
}
