use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum Activation {
    Relu,
    LeakyRelu,
    Sigmoid,
    Softmax
}

impl Activation {
    pub fn apply(&self, inputs: &mut Array2<f32>) {
        match self {
            Activation::Relu => inputs.mapv_inplace(|x| x.max(0.0)),
            Activation::LeakyRelu => inputs.mapv_inplace(|x| if x > 0.0 { x } else { 0.01 * x }),
            Activation::Sigmoid => inputs.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp())),

            Activation::Softmax => {
                for mut row in inputs.rows_mut() {
                    // prevent overflow in exp leading to NaNs; this will not treat saturation
                    row.mapv_inplace(|x| x.clamp(-88.72, 88.72)); // ln(f32::MAX) = ~88.723

                    let max_input = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    row.mapv_inplace(|x| (x-max_input).exp());

                    let exps_sum: f32 = row.iter().sum();
                    row.mapv_inplace(|x| x / (exps_sum + 1e-15));
                }
            }
        }
    }

    pub fn weights_distr(&self, inputs_count: usize, outputs_count: usize) -> impl rand_distr::Distribution<f32> {
        match self {
            Activation::Relu | Activation::LeakyRelu => {
                // He Normal
                let std = (2.0 / inputs_count as f32).sqrt();
                rand_distr::Normal::new(0.0, std).unwrap()
            },
            Activation::Sigmoid | Activation::Softmax => {
                // Xavier Normal
                let std = (2.0 / (inputs_count + outputs_count) as f32).sqrt();
                rand_distr::Normal::new(0.0, std).unwrap()
            },
        }
    }

    pub fn derivative(&self, y: f32) -> f32 {
        match self {
            Activation::Relu => if y > 0.0 { 1.0 } else { 0.0 },
            Activation::LeakyRelu => if y > 0.0 { 1.0 } else { 0.01 },
            Activation::Sigmoid => y * (1.0 - y),
            Activation::Softmax => panic!("Softmax wrongly used for the hidden layers!")
        }
    }

    pub fn derivative_array(&self, y: &Array2<f32>) -> Array2<f32> {
        y.mapv(|val| self.derivative(val))
    }
}
