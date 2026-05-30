use ndarray::Array1;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum Activation {
    ReLU,
    LeakyReLU,
    Sigmoid,
    Softmax
}

impl Activation {
    pub fn apply(&self, inputs: &mut Array1<f64>) {
        match self {
            Activation::ReLU => inputs.iter_mut().for_each(|x| *x = (*x).max(0.0)),
            Activation::LeakyReLU => inputs.iter_mut().for_each(|x| *x = (*x).max(0.01 * (*x))),
            Activation::Sigmoid => inputs.iter_mut().for_each(|x| *x = 1.0 / (1.0 + (- (*x)).exp())),

            Activation::Softmax => {
                // prevent overflow in exp leading to NaNs; this will not treat saturation
                inputs.iter_mut().for_each(|x| *x = x.clamp(-709.78, 709.78));

                let max_input = inputs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                
                inputs.iter_mut().for_each(|x| *x = ((*x) - max_input).exp());

                let exps_sum: f64 = inputs.iter().sum();
                inputs.iter_mut().for_each(|x| *x /= exps_sum + 1e-15);
            }
        }
    }

    pub fn weights_distr(&self, inputs_count: usize, outputs_count: usize) -> impl rand_distr::Distribution<f64> {
        match self {
            Activation::ReLU | Activation::LeakyReLU => {
                // He
                let std = (2.0 / inputs_count as f64).sqrt();
                rand_distr::Normal::new(0.0, std).unwrap()
            },
            Activation::Sigmoid | Activation::Softmax => {
                // Xavier
                let std = (2.0 / (inputs_count + outputs_count) as f64).sqrt();
                rand_distr::Normal::new(0.0, std).unwrap()
            },
        }
    }

    pub fn derivative(&self, y: f64) -> f64 {
        match self {
            Activation::ReLU => if y > 0.0 { 1.0 } else { 0.0 },
            Activation::LeakyReLU => if y > 0.0 { 1.0 } else { 0.01 },
            Activation::Sigmoid => y * (1.0 - y),
            Activation::Softmax => panic!("Softmax wrongly used for the hidden layers!")
        }
    }

    pub fn derivative_array(&self, y: &Array1<f64>) -> Array1<f64> {
        y.mapv(|val| self.derivative(val))
    }
}
