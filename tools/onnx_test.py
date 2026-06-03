#!/usr/bin/env python3

import struct
import numpy as np
import onnxruntime as ort

def load_mnist_images(filepath):
    with open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8)
        return images.reshape(num_images, rows * cols).astype(np.float32) / 255

def load_mnist_labels(filepath):
    with open(filepath, 'rb') as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

def main():
    model_path = "saved_models/mnist.onnx"
    images_path = "training_data/t10k-images.idx3-ubyte"
    labels_path = "training_data/t10k-labels.idx1-ubyte"

    test_images = load_mnist_images(images_path)
    test_labels = load_mnist_labels(labels_path)
    print(f"Loaded {test_images.shape[0]} test images")

    indices = np.random.permutation(test_images.shape[0])
    test_images = test_images[indices]
    test_labels = test_labels[indices]

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    num_samples = 10
    print(f"\nRunning inference for {num_samples} samples...\n")
    
    for i in range(num_samples):
        single_image = np.expand_dims(test_images[i], axis=0)
        true_label = test_labels[i]

        raw_outputs = session.run(None, {input_name: single_image})
        logits = raw_outputs[0]

        predicted_label = np.argmax(logits)

        status = "✅ CORRECT" if predicted_label == true_label else "❌ WRONG"
        print(f"Sample #{i} | True Label: {true_label} | Predicted: {predicted_label} -> {status}")

if __name__ == "__main__":
    main()
