mod activation;
mod layer;
mod loss;
mod network;
mod training_state;

use activation::Activation;
use loss::LossFunction;
use network::Network;
use training_state::TrainingState;

use std::fs::File;
use std::io::{BufReader, Read};
use rand::seq::SliceRandom;

fn main() {
    // Load MNIST dataset
    let train_images = read_mnist_images("training_data/train-images.idx3-ubyte");
    let train_labels = read_mnist_labels("training_data/train-labels.idx1-ubyte");
    let test_images = read_mnist_images("training_data/t10k-images.idx3-ubyte");
    let test_labels = read_mnist_labels("training_data/t10k-labels.idx1-ubyte");

    let train_images_flat: Vec<f32> = train_images.into_iter().flatten().collect();
    let train_inputs = ndarray::Array2::from_shape_vec((60000, 28 * 28), train_images_flat).unwrap();

    let mut train_targets = ndarray::Array2::zeros((60000, 10));
    for (i, &label) in train_labels.iter().enumerate() {
        train_targets[[i, label as usize]] = 1.0;
    }

    let test_images_flat: Vec<f32> = test_images.into_iter().flatten().collect();
    let test_inputs = ndarray::Array2::from_shape_vec((10000, 28 * 28), test_images_flat).unwrap();

    println!("Loaded {} training samples and {} test samples", train_inputs.nrows(), test_inputs.nrows());

    let mut net = Network::new(
        28 * 28,
        &[128, 64],
        &[Activation::LeakyRelu, Activation::Tanh],
        10,
        Activation::Softmax,
        LossFunction::CrossEntropy
    );
    
    let mut net_state = TrainingState::new(&net);

    let epochs = 10;
    let learning_rate = 0.05;
    let batch_size = 64;

    let mut indices: Vec<usize> = (0..train_inputs.nrows()).collect();
    let mut rng = rand::rng();

    println!("Starting training");

    for epoch in 1..=epochs {
        indices.shuffle(&mut rng);

        let mut epoch_loss = 0.0;

        for chunk in indices.chunks(batch_size) {
            let batch_inputs = train_inputs.select(ndarray::Axis(0), chunk);
            let batch_targets = train_targets.select(ndarray::Axis(0), chunk);


            let loss = net.train_batch(
                &mut net_state,
                batch_inputs.view(),
                batch_targets.view(),
                learning_rate
            );

            epoch_loss += loss;
        }

        let mut correct = 0;
        for i in 0..test_inputs.nrows() {
            let output = net.run(&mut net_state, test_inputs.row(i));
            let prediction = argmax(output);
            
            if prediction == test_labels[i] as usize {
                correct += 1;
            }
        }

        let accuracy = (correct as f32 / test_inputs.nrows() as f32) * 100.0;
        let avg_loss = epoch_loss / train_inputs.nrows() as f32;
        
        println!("Epoch {}/{} -> Loss: {:.4} | Test Accuracy: {:.2}%", epoch, epochs, avg_loss, accuracy);
    }

    // Save the trained model
    std::fs::create_dir_all("saved_models").expect("Failed to create directory for saved models");

    net.save("saved_models/mnist.bin").expect("Failed to save the model");
}


fn argmax(data: ndarray::ArrayView1<f32>) -> usize {
    data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap()
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
