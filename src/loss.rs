use crate::activation::Activation;

use ndarray::{Array2, ArrayView2};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[allow(dead_code)]
pub enum Loss {
    CrossEntropy,
    Mse
}

impl Loss {
    pub(crate) fn calculate_loss(&self, outputs: ArrayView2<f32>, targets: ArrayView2<f32>) -> f32 {
        let batch_size = outputs.nrows();
        let target_dim = outputs.ncols();
        let mut batch_loss = 0.0;

        match self {
            Loss::CrossEntropy => {
                for i in 0..batch_size {
                    for j in 0..target_dim {
                        batch_loss -= (outputs[[i, j]] + 1e-15).ln() * targets[[i, j]];
                    }
                }
                batch_loss
            },
            Loss::Mse => {
                for i in 0..batch_size {
                    for j in 0..target_dim {
                        let error = outputs[[i, j]] - targets[[i, j]];
                        batch_loss += error * error;
                    }
                }
                batch_loss / (batch_size * target_dim) as f32
            }
        }
    }

    pub(crate) fn calculate_deltas(
        &self,
        outputs: ArrayView2<f32>,
        targets: ArrayView2<f32>,
        output_activation: &Activation,
    ) -> Array2<f32> {

        match self {
            Loss::CrossEntropy => {
                // Skrócona pochodna dla Softmax + CrossEntropy
                &outputs - &targets
            },
            Loss::Mse => {
                // Pochodna MSE pomnożona przez pochodną funkcji aktywacji
                (&outputs - &targets) * &output_activation.derivative_array(&outputs.to_owned())
            }
        }
    }
}
