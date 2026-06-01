use crate::activation::Activation;
use crate::layer::Layer;
use crate::loss::LossFunction;
use crate::training_state::{TrainingState, Gradients};

use std::fs::File;

use rand_distr::Distribution;
use ndarray::{ArrayView1, Array2, ArrayView2, linalg::general_mat_mul};
use serde::{Serialize, Deserialize};


#[derive(Serialize, Deserialize)]
pub struct Network {
    pub layers: Vec<Layer>,
    loss_function: LossFunction
}


impl Network {
    pub fn new(
        c_inputs: usize,
        c_hidden_layers: &[usize],
        hidden_activation: Activation,
        c_outputs: usize,
        outputs_activation: Activation,
        loss_function: LossFunction
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
            layers: layers,
            loss_function
        }
    }

    fn forward(&self, net_state: &mut TrainingState, inputs: ArrayView2<f32>) {
        net_state.neuron_values[0] = inputs.to_owned();

        for i in 0..self.layers.len() {
            let layer = &self.layers[i];

            for mut row in net_state.neuron_values[i+1].rows_mut() {
                row.assign(&layer.biases);
            }

            let (a, b) = net_state.neuron_values.split_at_mut(i+1);

            if &b[0].shape() != &[inputs.nrows(), layer.biases.len()] {
                b[0] = Array2::zeros((inputs.nrows(), layer.biases.len()));
            }

            // equivalent to net_state.neuron_values[i+1] += net_state.neuron_values[i].dot(&layer.weights.t());
            general_mat_mul(
                1.0,
                &a[i],
                &layer.weights.t(),
                1.0,
                &mut b[0]
            );

            layer.activation.apply(&mut net_state.neuron_values[i+1]);
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
        
        deltas[last_idx] = self.loss_function.calculate_deltas(
            outputs.view(),
            targets,
            &layer.activation
        );

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

            // equivalent to grads.weight_grads[i] += &delta.t().dot(inputs);
            general_mat_mul(1.0, &delta.t(), inputs, 1.0, &mut grads.weight_grads[i]);

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
        net_state: &mut TrainingState,
        inputs: ArrayView1<f32>,
        targets: ArrayView1<f32>,
        learning_rate: f32
    ) -> f32 {
        net_state.grads_buffer.clear();

        let inputs_2d = inputs.insert_axis(ndarray::Axis(0));
        let targets_2d = targets.insert_axis(ndarray::Axis(0));

        self.forward(net_state, inputs_2d);

        let TrainingState {
            neuron_values,
            deltas,
            grads_buffer
        } = net_state;

        self.backpropagate(neuron_values, deltas, targets_2d, grads_buffer);

        let loss = self.loss_function.calculate_loss(
            neuron_values.last().unwrap().view(),
            targets_2d.view()
        );

        self.apply_gradients(&grads_buffer, learning_rate);

        loss
    }

    #[allow(dead_code)]
    pub fn train_batch(
        &mut self,
        net_state: &mut TrainingState,
        batch_inputs: ArrayView2<f32>,
        batch_targets: ArrayView2<f32>,
        learning_rate: f32
    ) -> f32 {

        let batch_size = batch_inputs.nrows();

        net_state.grads_buffer.clear();

        self.forward(net_state, batch_inputs);

        let TrainingState { neuron_values, deltas, grads_buffer } = net_state;
        self.backpropagate(neuron_values, deltas, batch_targets, grads_buffer);

        let batch_loss = self.loss_function.calculate_loss(
            neuron_values.last().unwrap().view(),
            batch_targets.view()
        );

        self.apply_gradients(&net_state.grads_buffer, learning_rate / batch_size as f32);

        batch_loss
    }

    #[allow(dead_code)]
    pub fn run<'ns_lifetime>(
        &self,
        net_state: &'ns_lifetime mut TrainingState,
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
