use crate::activation::Activation;
use crate::layer::Layer;
use crate::loss::Loss;
use crate::state::{ForwardCache, TrainingState, Gradients};

use rand_distr::Distribution;
use ndarray::{ArrayView1, Array2, ArrayView2, linalg::general_mat_mul};
use serde::{Serialize, Deserialize};


#[derive(Serialize, Deserialize)]
pub struct Network {
    pub layers: Vec<Layer>,
    pub loss_function: Loss
}


impl Network {
    pub fn new(
        c_inputs: usize,
        c_hidden_layers: &[usize],
        hidden_activation: &[Activation],
        c_outputs: usize,
        outputs_activation: Activation,
        loss_function: Loss
    ) -> Self {

        assert_eq!(
            c_hidden_layers.len(),
            hidden_activation.len(),
            "Lenght of hidden layers neuron counts slice and hidden layers activation functions slice do not match!"
        );

        let mut layers: Vec<Layer> = Vec::with_capacity(c_hidden_layers.len() + 1);
        let mut c_prev: usize = c_inputs;

        // Hidden layers neurons
        for (i,&c_curr) in c_hidden_layers.iter().enumerate() {
            let distr = hidden_activation[i].weights_distr(c_prev, c_curr);
            let weights: Vec<f32> = (0..(c_prev*c_curr)).map(|_| distr.sample(&mut rand::rng())).collect();

            layers.push(
                Layer::new(
                    weights,
                    vec![0.0; c_curr],
                    c_prev,
                    c_curr,
                    hidden_activation[i]
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

    fn forward(&self, cache: &mut ForwardCache, inputs: ArrayView2<f32>) {
        cache.neuron_values[0] = inputs.to_owned();

        for i in 0..self.layers.len() {
            let layer = &self.layers[i];

            for mut row in cache.neuron_values[i+1].rows_mut() {
                row.assign(&layer.biases);
            }

            let (a, b) = cache.neuron_values.split_at_mut(i+1);

            if &b[0].shape() != &[inputs.nrows(), layer.biases.len()] {
                b[0] = Array2::zeros((inputs.nrows(), layer.biases.len()));
            }

            // equivalent to cache.neuron_values[i+1] += cache.neuron_values[i].dot(&layer.weights.t());
            general_mat_mul(
                1.0,
                &a[i],
                &layer.weights.t(),
                1.0,
                &mut b[0]
            );

            layer.activation.apply(&mut cache.neuron_values[i+1]);
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
        training_state: &mut TrainingState,
        inputs: ArrayView1<f32>,
        targets: ArrayView1<f32>,
        learning_rate: f32
    ) -> f32 {

        let TrainingState {
            forward_cache,
            deltas,
            gradients
        } = training_state;

        gradients.clear();

        let inputs_2d = inputs.insert_axis(ndarray::Axis(0));
        let targets_2d = targets.insert_axis(ndarray::Axis(0));

        self.forward(forward_cache, inputs_2d);
        self.backpropagate(&forward_cache.neuron_values, deltas, targets_2d, gradients);

        let loss = self.loss_function.calculate_loss(
            forward_cache.neuron_values.last().unwrap().view(),
            targets_2d.view()
        );

        self.apply_gradients(&gradients, learning_rate);

        loss
    }

    #[allow(dead_code)]
    pub fn train_batch(
        &mut self,
        training_state: &mut TrainingState,
        batch_inputs: ArrayView2<f32>,
        batch_targets: ArrayView2<f32>,
        learning_rate: f32
    ) -> f32 {
        
        let TrainingState {
            forward_cache,
            deltas,
            gradients
        } = training_state;

        gradients.clear();

        self.forward(forward_cache, batch_inputs);
        self.backpropagate(&forward_cache.neuron_values, deltas, batch_targets, gradients);

        let batch_loss = self.loss_function.calculate_loss(
            forward_cache.neuron_values.last().unwrap().view(),
            batch_targets.view()
        );

        self.apply_gradients(gradients, learning_rate / batch_inputs.nrows() as f32);

        batch_loss
    }

    #[allow(dead_code)]
    pub fn infer<'cache_lifetime>(
        &self,
        cache: &'cache_lifetime mut ForwardCache,
        inputs: ArrayView1<f32>
    ) -> ArrayView1<'cache_lifetime, f32> {

        let inputs_2d = inputs.insert_axis(ndarray::Axis(0));
        self.forward(cache, inputs_2d);
        cache.neuron_values.last().unwrap().row(0)
    }

    #[allow(dead_code)]
    pub fn infer_batch<'cache_lifetime>(
        &self,
        cache: &'cache_lifetime mut ForwardCache,
        batch_inputs: ArrayView2<f32>
    ) -> ArrayView2<'cache_lifetime, f32> {

        self.forward(cache, batch_inputs);
        cache.neuron_values.last().unwrap().view()
    }
}
