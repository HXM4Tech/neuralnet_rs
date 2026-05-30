use crate::activation::Activation;
use crate::layer::Layer;

use std::fs::File;

use rand_distr::Distribution;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

pub struct NetworkState {
    neuron_values: Vec<Array1<f32>>,
    deltas: Vec<Array1<f32>>
}

impl NetworkState {
    pub fn new(net: &Network) -> Self {
        let c_inputs = net.layers[0].weights.shape()[1];
        let layers = &net.layers;

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

#[derive(Serialize, Deserialize)]
pub struct Network {
    layers: Vec<Layer>
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
            let weights: Vec<f32> = (0..(c_prev*c_curr)).map(|_| distr.sample(&mut rand::rng())).collect();

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
        let weights: Vec<f32> = (0..(c_prev*c_outputs)).map(|_| distr.sample(&mut rand::rng())).collect();

        layers.push(
            Layer::new(
                weights,
                vec![0.0; c_outputs],
                c_prev,
                c_outputs,
                outputs_activation
            )
        );

        Self {
            layers: layers
        }
    }

    fn forward(&mut self, net_state: &mut NetworkState, inputs: &[f32]) {
        net_state.neuron_values[0].assign(&Array1::from_vec(inputs.to_vec()));

        for i in 0..self.layers.len() {
            let input_val = net_state.neuron_values[i].view();
            net_state.neuron_values[i + 1] = self.layers[i].forward(&input_val);
        }
    }

    fn get_output<'ns_lifetime>(&self, net_state: &'ns_lifetime NetworkState) -> &'ns_lifetime [f32] {
        net_state.neuron_values.last().unwrap().as_slice().unwrap()
    }

    fn backpropagate(&mut self, net_state: &mut NetworkState, targets: &[f32], learning_rate: f32) {
        let targets = Array1::from_vec(targets.to_vec());
        
        let last_idx = self.layers.len() - 1;
        let outputs = &net_state.neuron_values[last_idx + 1];
        let layer = &self.layers[last_idx];
        
        let mut last_deltas = outputs - &targets;
        if layer.activation != Activation::Softmax {
            last_deltas *= &layer.activation.derivative_array(outputs);
        }
        net_state.deltas[last_idx] = last_deltas;

        for i in (0..self.layers.len() - 1).rev() {
            let next_layer = &self.layers[i + 1];
            let curr_layer = &self.layers[i];
            
            let delta = next_layer.weights.t().dot(&net_state.deltas[i + 1]) 
                        * &curr_layer.activation.derivative_array(&net_state.neuron_values[i + 1]);
            
            net_state.deltas[i] = delta;
        }

        for i in 0..self.layers.len() {
            let delta = &net_state.deltas[i];
            let inputs = &net_state.neuron_values[i];
            let layer = &mut self.layers[i];

            layer.biases -= &(learning_rate * delta);

            let delta_mat = delta.view().insert_axis(ndarray::Axis(1));
            let inputs_mat = inputs.view().insert_axis(ndarray::Axis(0));
            
            layer.weights -= &(learning_rate * delta_mat.dot(&inputs_mat));
        }
    }

    #[allow(dead_code)]
    pub fn train<'ns_lifetime>(
        &mut self,
        net_state: &'ns_lifetime mut NetworkState,
        inputs: &[f32],
        targets: &[f32],
        learning_rate: f32
    ) -> &'ns_lifetime [f32] {

        self.forward(net_state, inputs);
        self.backpropagate(net_state, targets, learning_rate);
        self.get_output(net_state)
    }

    #[allow(dead_code)]
    pub fn run<'ns_lifetime>(
        &mut self,
        net_state: &'ns_lifetime mut NetworkState,
        inputs: &[f32]
    ) -> &'ns_lifetime [f32] {

        self.forward(net_state, inputs);
        self.get_output(net_state)
    }

    #[allow(dead_code)]
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        rmp_serde::encode::write(
            &mut file,
            self
        ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    #[allow(dead_code)]
    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;

        rmp_serde::decode::from_read(
            file
        ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}
