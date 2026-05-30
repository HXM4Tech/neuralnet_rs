mod activation;
mod layer;
mod network;

use activation::Activation;
use network::{Network, NetworkState};

use std::fs::File;
use std::io::{BufReader, Read};
use rand::seq::SliceRandom;

fn main() {
    // Load MNIST dataset
    let train_images = read_mnist_images("training_data/train-images.idx3-ubyte");
    let train_labels = read_mnist_labels("training_data/train-labels.idx1-ubyte");
    let test_images = read_mnist_images("training_data/t10k-images.idx3-ubyte");
    let test_labels = read_mnist_labels("training_data/t10k-labels.idx1-ubyte");

    println!("Loaded {} training samples and {} test samples", train_images.len(), test_images.len());

    let mut net = Network::new(
        28 * 28,
        &[128, 64],
        Activation::LeakyReLU,
        10,
        Activation::Softmax
    );
    
    let mut net_state = NetworkState::new(&net);

    let epochs = 10;
    let learning_rate = 0.01;

    let mut indices: Vec<usize> = (0..train_images.len()).collect();
    let mut rng = rand::rng();

    println!("Starting training");

    for epoch in 1..=epochs {
        indices.shuffle(&mut rng);

        let mut epoch_loss = 0.0;

        for &idx in &indices {
            let input = &train_images[idx];
            let label = train_labels[idx] as usize;

            let mut target = vec![0.0; 10];
            target[label] = 1.0;

            let output = net.train(
                &mut net_state,
                &input,
                &target,
                learning_rate
            );

            epoch_loss -= (output[label] + 1e-15).ln();
        }

        let mut correct = 0;
        for i in 0..test_images.len() {
            let output = net.run(&mut net_state, &test_images[i]);
            let prediction = argmax(output);
            
            if prediction == test_labels[i] as usize {
                correct += 1;
            }
        }

        let accuracy = (correct as f64 / test_images.len() as f64) * 100.0;
        let avg_loss = epoch_loss / train_images.len() as f64;
        
        println!("Epoch {}/{} -> Loss: {:.4} | Test Accuracy: {:.2}%", epoch, epochs, avg_loss, accuracy);
    }

    // Save the trained model
    std::fs::create_dir_all("saved_models").expect("Failed to create directory for saved models");

    net.save("saved_models/mnist_model.bin").expect("Failed to save the model");
}


fn argmax(slice: &[f64]) -> usize {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap()
}


fn read_mnist_images(path: &str) -> Vec<Vec<f64>> {
    let file = File::open(path).unwrap_or_else(|_| panic!("Nie znaleziono pliku: {}", path));
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

        let normalized: Vec<f64> = buffer.iter().map(|&x| x as f64 / 255.0).collect();
        dataset.push(normalized);
    }
    dataset
}

fn read_mnist_labels(path: &str) -> Vec<u8> {
    let file = File::open(path).unwrap_or_else(|_| panic!("Nie znaleziono pliku: {}", path));
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
