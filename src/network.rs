use crate::activation::Activation;
use crate::layer::Layer;

use std::fs::File;

use rand_distr::Distribution;
use ndarray::{Array1, ArrayView1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};

#[derive(Clone)]
struct Gradients {
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

    fn clear(&mut self) {
        for w in &mut self.weight_grads {
            w.fill(0.0);
        }

        for b in &mut self.bias_grads {
            b.fill(0.0);
        }
    }
}

pub struct NetworkState {
    // for neuron_values and deltas dims are (batch size, neuron count)
    neuron_values: Vec<Array2<f32>>,
    deltas: Vec<Array2<f32>>,
    grads_buffer: Gradients
}

impl NetworkState {
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

    fn forward(&self, net_state: &mut NetworkState, inputs: ArrayView2<f32>) {
        net_state.neuron_values[0] = inputs.to_owned();

        for i in 0..self.layers.len() {
            let input_val = net_state.neuron_values[i].view();
            net_state.neuron_values[i + 1] = self.layers[i].forward(&input_val);
        }
    }

    fn backpropagate(
        &self,
        neuron_values: &[Array2<f32>],
        deltas: &mut [Array2<f32>],
        targets: ArrayView2<f32>,
        grads: &mut Gradients
    ) {
        
        let last_idx = self.layers.len() - 1;
        let outputs = &neuron_values[last_idx + 1];
        let layer = &self.layers[last_idx];
        
        let mut last_deltas = outputs - &targets;
        if layer.activation != Activation::Softmax {
            last_deltas *= &layer.activation.derivative_array(&outputs);
        }
        deltas[last_idx] = last_deltas;

        for i in (0..self.layers.len() - 1).rev() {
            let next_layer = &self.layers[i + 1];
            let curr_layer = &self.layers[i];
            
            let delta = deltas[i + 1].dot(&next_layer.weights) 
                * &curr_layer.activation.derivative_array(&neuron_values[i + 1]);
            
            deltas[i] = delta;
        }

        for i in 0..self.layers.len() {
            let delta = &deltas[i];
            let inputs = &neuron_values[i];

            grads.weight_grads[i] += &delta.t().dot(inputs);
            grads.bias_grads[i] += &delta.sum_axis(ndarray::Axis(0));
        }
    }

    fn apply_gradients(&mut self, grads: &Gradients, learning_rate: f32) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.weights -= &(learning_rate * &grads.weight_grads[i]);
            layer.biases -= &(learning_rate * &grads.bias_grads[i]);
        }
    }

    #[allow(dead_code)]
    pub fn train(
        &mut self,
        net_state: &mut NetworkState,
        inputs: ArrayView1<f32>,
        targets: ArrayView1<f32>,
        learning_rate: f32
    ) -> f32 {
        net_state.grads_buffer.clear();

        let inputs_2d = inputs.insert_axis(ndarray::Axis(0));
        let targets_2d = targets.insert_axis(ndarray::Axis(0));

        self.forward(net_state, inputs_2d);

        let NetworkState {
            neuron_values,
            deltas,
            grads_buffer
        } = net_state;

        self.backpropagate(neuron_values, deltas, targets_2d, grads_buffer);

        let mut loss = 0.0;
        let output = neuron_values.last().unwrap().as_slice().unwrap();

        for (o, t) in output.iter().zip(targets_2d.iter()) {
            loss -= (o + 1e-15).ln() * t;
        }

        self.apply_gradients(&grads_buffer, learning_rate);

        loss
    }

    #[allow(dead_code)]
    pub fn train_batch(
        &mut self,
        net_state: &mut NetworkState,
        batch_inputs: ArrayView2<f32>,
        batch_targets: ArrayView2<f32>,
        learning_rate: f32
    ) -> f32 {

        let batch_size = batch_inputs.nrows();
        let target_dim = batch_targets.ncols();

        net_state.grads_buffer.clear();

        self.forward(net_state, batch_inputs);

        let NetworkState { neuron_values, deltas, grads_buffer } = net_state;
        self.backpropagate(neuron_values, deltas, batch_targets, grads_buffer);

        let mut batch_loss = 0.0;
        let outputs = neuron_values.last().unwrap();
        for i in 0..batch_size {
            for j in 0..target_dim {
                let o = outputs[[i, j]];
                let t = batch_targets[[i, j]];
                batch_loss -= (o + 1e-15).ln() * t;
            }
        }

        self.apply_gradients(&net_state.grads_buffer, learning_rate / batch_size as f32);

        batch_loss
    }

    #[allow(dead_code)]
    pub fn run<'ns_lifetime>(
        &self,
        net_state: &'ns_lifetime mut NetworkState,
        inputs: ArrayView1<f32>
    ) -> ArrayView1<'ns_lifetime, f32> {
        let inputs_2d = inputs.insert_axis(ndarray::Axis(0));

        self.forward(net_state, inputs_2d);
        net_state.neuron_values.last().unwrap().row(0)
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
