use crate::network::Network;

use ndarray::{Array1, Array2};

#[derive(Clone)]
pub struct Gradients {
    pub weight_grads: Vec<Array2<f32>>,
    pub bias_grads: Vec<Array1<f32>>
}

impl Gradients {
    fn new(net: &Network) -> Self {
        let mut weight_grads = Vec::with_capacity(net.layers.len());
        let mut bias_grads = Vec::with_capacity(net.layers.len());

        for l in &net.layers {
            weight_grads.push(Array2::zeros(l.weights.raw_dim()));
            bias_grads.push(Array1::zeros(l.biases.raw_dim()));
        }

        Self {
            weight_grads,
            bias_grads
        }
    }

    pub fn clear(&mut self) {
        for w in &mut self.weight_grads {
            w.fill(0.0);
        }

        for b in &mut self.bias_grads {
            b.fill(0.0);
        }
    }
}

pub struct TrainingState {
    // for neuron_values and deltas dims are (batch size, neuron count)
    pub neuron_values: Vec<Array2<f32>>,
    pub deltas: Vec<Array2<f32>>,
    pub grads_buffer: Gradients
}

impl TrainingState {
    pub fn new(net: &Network) -> Self {
        let c_inputs = net.layers[0].weights.shape()[1];
        let layers = &net.layers;

        let mut neuron_values = Vec::with_capacity(layers.len() + 1);
        let mut deltas = Vec::with_capacity(layers.len());

        neuron_values.push(Array2::zeros((0, c_inputs)));

        for l in layers {
            neuron_values.push(Array2::zeros((0, l.biases.len())));
            deltas.push(Array2::zeros((0, l.biases.len())));
        }

        Self {
            neuron_values,
            deltas,
            grads_buffer: Gradients::new(net)
        }
    }

}
