use crate::activation::Activation;
use crate::layer::Layer;

use rand_distr::Distribution;
use ndarray::Array1;
struct NetworkState {
    neuron_values: Vec<Array1<f64>>,
    deltas: Vec<Array1<f64>>
}

impl NetworkState {
    pub fn new(c_inputs: usize, layers: &[Layer]) -> Self {
        let mut neuron_values = Vec::with_capacity(layers.len() + 1);
        let mut deltas = Vec::with_capacity(layers.len());

        neuron_values.push(Array1::zeros(c_inputs));

        for l in layers {
            neuron_values.push(Array1::zeros(l.biases.len()));
            deltas.push(Array1::zeros(l.biases.len()));
        }

        Self {
            neuron_values,
            deltas
        }
    }
}

pub struct Network {
    layers: Vec<Layer>,
    state: NetworkState
}


impl Network {
    pub fn new(
        c_inputs: usize,
        c_hidden_layers: &[usize],
        hidden_activation: Activation,
        c_outputs: usize,
        outputs_activation: Activation
    ) -> Self {

        let mut layers: Vec<Layer> = Vec::with_capacity(c_hidden_layers.len() + 1);
        let mut c_prev: usize = c_inputs;

        // Hidden layers neurons
        for &c_curr in c_hidden_layers {
            let distr = hidden_activation.weights_distr(c_prev, c_curr);
            let weights: Vec<f64> = (0..(c_prev*c_curr)).map(|_| distr.sample(&mut rand::rng())).collect();

            layers.push(
                Layer::new(
                    weights,
                    vec![0.0; c_curr],
                    c_prev,
                    c_curr,
                    hidden_activation
                )
            );

            c_prev = c_curr;
        }

        // Output layer neurons
        let distr = outputs_activation.weights_distr(c_prev, c_outputs);
        let weights: Vec<f64> = (0..(c_prev*c_outputs)).map(|_| distr.sample(&mut rand::rng())).collect();

        layers.push(
            Layer::new(
                weights,
                vec![0.0; c_outputs],
                c_prev,
                c_outputs,
                outputs_activation
            )
        );

        
        let network_state = NetworkState::new(c_inputs, &layers);

        Self {
            layers: layers,
            state: network_state
        }
    }

    fn forward(&mut self, inputs: &[f64]) {
        self.state.neuron_values[0].assign(&Array1::from_vec(inputs.to_vec()));

        for i in 0..self.layers.len() {
            let input_val = self.state.neuron_values[i].view();
            self.state.neuron_values[i + 1] = self.layers[i].forward(&input_val);
        }
    }

    fn get_output(&self) -> &[f64] {
        self.state.neuron_values.last().unwrap().as_slice().unwrap()
    }

    fn backpropagate(&mut self, targets: &[f64], learning_rate: f64) {
        let targets = Array1::from_vec(targets.to_vec());
        
        let last_idx = self.layers.len() - 1;
        let outputs = &self.state.neuron_values[last_idx + 1];
        let layer = &self.layers[last_idx];
        
        let mut last_deltas = outputs - &targets;
        if layer.activation != Activation::Softmax {
            last_deltas *= &layer.activation.derivative_array(outputs);
        }
        self.state.deltas[last_idx] = last_deltas;

        for i in (0..self.layers.len() - 1).rev() {
            let next_layer = &self.layers[i + 1];
            let curr_layer = &self.layers[i];
            
            let delta = next_layer.weights.t().dot(&self.state.deltas[i + 1]) 
                        * &curr_layer.activation.derivative_array(&self.state.neuron_values[i + 1]);
            
            self.state.deltas[i] = delta;
        }

        for i in 0..self.layers.len() {
            let delta = &self.state.deltas[i];
            let inputs = &self.state.neuron_values[i];
            let layer = &mut self.layers[i];

            layer.biases -= &(learning_rate * delta);

            use ndarray::Axis;
            let delta_mat = delta.view().insert_axis(Axis(1));
            let inputs_mat = inputs.view().insert_axis(Axis(0));
            
            layer.weights -= &(learning_rate * delta_mat.dot(&inputs_mat));
        }
    }

    pub fn train(&mut self, inputs: &[f64], targets: &[f64], learning_rate: f64) -> &[f64] {
        self.forward(inputs);
        self.backpropagate(targets, learning_rate);
        self.get_output()
    }

    pub fn predict(&mut self, inputs: &[f64]) -> &[f64] {
        self.forward(inputs);
        self.get_output()
    }
}
