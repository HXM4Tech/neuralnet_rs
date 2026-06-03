use crate::activation::Activation;
use crate::loss::Loss;
use crate::network::Network;
use crate::state::{ForwardCache, TrainingState};

use std::fs::File;
use std::io::{BufReader, Read};
use rand::seq::SliceRandom;

#[test]
fn simple_passthrough_network() {
    let mut net = Network::new(
        1,
        &[],
        &[],
        1,
        Activation::Identity,
        Loss::Mse
    );

    net.layers[0].weights[[0, 0]] = 1.0;
    net.layers[0].biases[0] = 0.0;

    let mut fc = ForwardCache::new(&net);
    let inputs = ndarray::Array2::from_shape_vec(
        (10, 1),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ).unwrap();

    let outputs = net.infer_batch(&mut fc, inputs.view());

    assert_eq!(
        outputs.ncols(),
        1,
        "Output dimension mismatch: expected 1, got {}",
        outputs.ncols()
    );
    
    for i in 0..10 {
        assert!(
            (outputs[[i, 0]] - (i as f32 + 1.0)).abs() < 1e-6,
            "Output differs from input for row {}: expected {}, got {}", i, (i as f32) + 1.0,
            outputs[[i, 0]]
        );
    }
}

#[test]
fn xor_problem() {
    let mut net = Network::new(
        2,
        &[4, 4],
        &[Activation::Tanh, Activation::Tanh],
        1,
        Activation::Sigmoid,
        Loss::Mse
    );

    let mut state = TrainingState::new(&net);

    let inputs = ndarray::Array2::from_shape_vec(
        (4, 2),
        vec![
            0.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0
        ]
    ).unwrap();

    let targets = ndarray::Array2::from_shape_vec(
        (4, 1),
        vec![
            0.0,
            1.0,
            1.0,
            0.0
        ]
    ).unwrap();

    for _ in 0..10000 {
        net.train_batch(&mut state, inputs.view(), targets.view(), 0.1);
    }

    let outputs = net.infer_batch(&mut state.forward_cache, inputs.view());

    for i in 0..4 {
        assert!(
            (outputs[[i, 0]] - targets[[i, 0]]).abs() < 0.25,
            "Incorrect (or poor) inference for input {:?}: expected {}, got {}",
            inputs.slice(ndarray::s![i, ..]),
            targets[[i, 0]],
            outputs[[i, 0]]
        );
    }
}

#[test]
#[should_panic(expected = "Softmax activation incorrectly paired with Mse loss function!")]
fn softmax_mse() {
    Network::new(
        2,
        &[2],
        &[Activation::Tanh],
        1,
        Activation::Softmax,
        Loss::Mse
    );
}

#[test]
#[should_panic(expected = "Softmax activation incorrectly used in the hidden layers!")]
fn softmax_hidden() {
    Network::new(
        2,
        &[2],
        &[Activation::Softmax],
        1,
        Activation::Sigmoid,
        Loss::CrossEntropy
    );
}

#[test]
#[should_panic(expected = "Lengths of hidden layers neuron counts slice and hidden layers activation functions slice do not match!")]
fn hidden_layer_mismatch() {
    Network::new(
        2,
        &[2, 2],
        &[Activation::LeakyRelu],
        1,
        Activation::Tanh,
        Loss::Mse
    );
}

#[test]
fn mnist_95percent_accuracy() {
    let (net, epoch_losses) = train_mnist().unwrap();

    let l0 = epoch_losses[0];
    let l_last = epoch_losses[epoch_losses.len() - 1];

    assert!(
        &l0 > &l_last,
        "Loss did not decrease during training! Initial loss: {}, final loss: {}",
        l0,
        l_last
    );

    let test_images = read_mnist_images("tests/training_data/t10k-images.idx3-ubyte");
    let test_labels = read_mnist_labels("tests/training_data/t10k-labels.idx1-ubyte");
    let test_images_flat: Vec<f32> = test_images.into_iter().flatten().collect();
    let test_inputs = ndarray::Array2::from_shape_vec((10000, 28 * 28), test_images_flat).unwrap();

    let mut fc = ForwardCache::new(&net);
    let test_outputs = net.infer_batch(&mut fc, test_inputs.view());
    let mut correct = 0;

    for i in 0..10000 {
        let predicted_label = argmax(test_outputs.slice(ndarray::s![i, ..]));
        if predicted_label == test_labels[i] as usize {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / 10000.0;

    assert!(
        accuracy > 0.95,
        "Accuracy is too low: {:.2}%",
        accuracy * 100.0
    );
}

#[test]
fn save_load_nnrs() {
    let net = Network::new(
        2,
        &[2],
        &[Activation::LeakyRelu],
        1,
        Activation::Sigmoid,
        Loss::Mse
    );

    std::fs::create_dir_all("target/test_temp").unwrap();
    net.save("target/test_temp/mnist_model.nnrs").unwrap();
    let loaded_net = Network::load("target/test_temp/mnist_model.nnrs").unwrap();

    let mut fc1 = ForwardCache::new(&net);
    let mut fc2 = ForwardCache::new(&loaded_net);

    let inputs = ndarray::Array2::from_shape_vec(
        (4, 2),
        vec![
            0.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0
        ]
    ).unwrap();

    let outputs = net.infer_batch(&mut fc1, inputs.view());
    let loaded_outputs = loaded_net.infer_batch(&mut fc2, inputs.view());

    assert_eq!(
        outputs.shape(),
        loaded_outputs.shape(),
        "Output shape mismatch between original and loaded network: expected {:?}, got {:?}",
        outputs.shape(),
        loaded_outputs.shape()
    );

    for i in 0..4 {
        assert!(
            (outputs[[i, 0]] - loaded_outputs[[i, 0]]).abs() < 1e-6,
            "Output mismatch for input {:?}: expected {}, got {}",
            inputs.slice(ndarray::s![i, ..]), outputs[[i, 0]],
            loaded_outputs[[i, 0]]
        );
    }
}

#[test]
#[cfg_attr(no_python_env, ignore = "Python environment or required libraries are missing")]
fn save_mnist_onnx_plus_python_onnxruntime_inference() {
    let (net, _) = train_mnist().unwrap();

    std::fs::create_dir_all("target/test_temp").unwrap();
    net.save_onnx("target/test_temp/mnist_model.onnx").unwrap();

    let output = std::process::Command::new("python3")
        .arg("tests/onnx_mnist_test.py")
        .arg("target/test_temp/mnist_model.onnx")
        .output()
        .expect("Failed to execute onnx_mnist_test.py");

    assert!(
        output.status.success(),
        "ONNX MNIST model test failed with exit code {}. Stderr:\n{}",
        output.status.code().unwrap_or(1),
        String::from_utf8_lossy(&output.stderr)
    );
}



/// Helper functions for tests

fn train_mnist() -> std::io::Result<(Network , Vec<f32>)> {
    // Load MNIST dataset
    let train_images = read_mnist_images("tests/training_data/train-images.idx3-ubyte");
    let train_labels = read_mnist_labels("tests/training_data/train-labels.idx1-ubyte");

    let train_images_flat: Vec<f32> = train_images.into_iter().flatten().collect();
    let train_inputs = ndarray::Array2::from_shape_vec((60000, 28 * 28), train_images_flat).unwrap();

    let mut train_targets = ndarray::Array2::zeros((60000, 10));
    for (i, &label) in train_labels.iter().enumerate() {
        train_targets[[i, label as usize]] = 1.0;
    }

    let mut net = Network::new(
        28 * 28,
        &[128, 64],
        &[Activation::LeakyRelu, Activation::Tanh],
        10,
        Activation::Softmax,
        Loss::CrossEntropy
    );
    
    let mut net_state = TrainingState::new(&net);

    let epochs = 10;
    let learning_rate = 0.05;
    let batch_size = 64;

    let mut indices: Vec<usize> = (0..train_inputs.nrows()).collect();
    let mut rng = rand::rng();

    let mut epoch_losses: Vec<f32> = Vec::with_capacity(epochs);

    for _ in 1..=epochs {
        indices.shuffle(&mut rng);

        let mut epoch_loss = 0.0;

        for chunk_start in (0..train_inputs.nrows()).step_by(batch_size) {
            let chunk_end = (chunk_start + batch_size).min(train_inputs.nrows());

            let batch_inputs = train_inputs.slice(ndarray::s![chunk_start..chunk_end, ..]);
            let batch_targets = train_targets.slice(ndarray::s![chunk_start..chunk_end, ..]);

            let batch_loss = net.train_batch(
                &mut net_state,
                batch_inputs.view(),
                batch_targets.view(),
                learning_rate
            );

            epoch_loss += batch_loss;
        }

        epoch_losses.push(epoch_loss);
    }

    Ok((net, epoch_losses))
}


fn argmax(data: ndarray::ArrayView1<f32>) -> usize {
    data.iter().enumerate().max_by(
        |(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    ).map(|(index, _)| index).unwrap()
}


fn read_mnist_images(path: &str) -> Vec<Vec<f32>> {
    let file = File::open(path).unwrap_or_else(|_| panic!("File not found: {}", path));
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 4];
    let mut count = [0u8; 4];
    let mut rows = [0u8; 4];
    let mut cols = [0u8; 4];

    reader.read_exact(&mut magic).unwrap();
    reader.read_exact(&mut count).unwrap();
    reader.read_exact(&mut rows).unwrap();
    reader.read_exact(&mut cols).unwrap();

    let count = u32::from_be_bytes(count) as usize;
    let rows = u32::from_be_bytes(rows) as usize;
    let cols = u32::from_be_bytes(cols) as usize;
    let image_size = rows * cols;

    let mut dataset = Vec::with_capacity(count);
    let mut buffer = vec![0u8; image_size];

    for _ in 0..count {
        reader.read_exact(&mut buffer).unwrap();

        let normalized: Vec<f32> = buffer.iter().map(|&x| x as f32 / 255.0).collect();
        dataset.push(normalized);
    }

    dataset
}

fn read_mnist_labels(path: &str) -> Vec<u8> {
    let file = File::open(path).unwrap_or_else(|_| panic!("File not found: {}", path));
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 4];
    let mut count = [0u8; 4];

    reader.read_exact(&mut magic).unwrap();
    reader.read_exact(&mut count).unwrap();

    let count = u32::from_be_bytes(count) as usize;
    let mut labels = vec![0u8; count];
    reader.read_exact(&mut labels).unwrap();
    
    labels
}
