use ferrox::backend::{Device, FerroxCudaF};
use ferrox::graph::{AutoFerroxEngine, NodeId};
use ferrox::nn::{layers::{Linear, ReLU}, Module};
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

    fn create_parameters_in_graph(&self, engine: &mut AutoFerroxEngine<T>) -> HashMap<String, NodeId> {
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

    fn create_parameters_in_graph(&self, engine: &mut AutoFerroxEngine<T>) -> HashMap<String, NodeId> {
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
