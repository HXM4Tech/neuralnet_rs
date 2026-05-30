use crate::activation::Activation;

use ndarray::{Array2, Array1, ArrayView1};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Layer {
    pub weights: Array2<f64>,    // length: neuron count * inputs count
    pub biases: Array1<f64>,     // length: neuron count
    pub activation: Activation
}

impl Layer {
    pub fn new(weights_flat: Vec<f64>, biases: Vec<f64>, c_inputs:usize, c_outputs: usize, activation: Activation) -> Self {
        assert_eq!(weights_flat.len(), c_inputs * c_outputs);
        assert_eq!(biases.len(), c_outputs);

        let weights = Array2::from_shape_vec((c_outputs, c_inputs), weights_flat).unwrap();
        let biases = Array1::from(biases);

        Self {
            weights,
            biases,
            activation
         }
     }


    pub fn forward(&self, inputs: &ArrayView1<f64>) -> Array1<f64> {
        let mut outputs = inputs.dot(&self.weights.t()) + &self.biases;
        self.activation.apply(&mut outputs);

        outputs
    }
}
