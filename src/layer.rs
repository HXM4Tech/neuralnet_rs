use crate::activation::Activation;

use ndarray::{Array1, Array2};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct Layer {
    pub weights: Array2<f32>,    // shape: (neuron count, input count)
    pub biases: Array1<f32>,     // shape: (neuron count)
    pub activation: Activation
}

impl Layer {
    pub fn new(weights_flat: Vec<f32>, biases: Vec<f32>, c_inputs:usize, c_outputs: usize, activation: Activation) -> Self {
        debug_assert_eq!(weights_flat.len(), c_inputs * c_outputs);
        debug_assert_eq!(biases.len(), c_outputs);

        let weights = Array2::from_shape_vec((c_outputs, c_inputs), weights_flat).unwrap();
        let biases = Array1::from(biases);

        Self {
            weights,
            biases,
            activation
        }
    }
}
