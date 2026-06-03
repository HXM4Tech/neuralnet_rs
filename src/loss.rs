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

        match self {
            Loss::CrossEntropy => {
                let loss_matrix = -(&targets * (&outputs + 1e-15).mapv(f32::ln));

                loss_matrix.sum()
            },
            Loss::Mse => {
                let errors = &outputs - &targets;

                (&errors * &errors).sum() / (batch_size * target_dim) as f32
            }
        }
    }

    pub(crate) fn calculate_deltas(
        &self,
        outputs: ArrayView2<f32>,
        targets: ArrayView2<f32>,
        output_activation: &Activation,
    ) -> Array2<f32> {

        match (self, output_activation) {
            (Loss::CrossEntropy, Activation::Softmax) => {
                &outputs - &targets
            },
            (Loss::CrossEntropy, _) => {
                // 1e-7 to prevent division by zero
                let loss_derivative = -&targets / (&outputs + 1e-7);

                loss_derivative * &output_activation.derivative_array(&outputs)
            },
            (Loss::Mse, _) => (&outputs - &targets) * &output_activation.derivative_array(&outputs)
        }
    }
}
