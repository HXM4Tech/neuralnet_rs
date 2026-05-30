#!/usr/bin/env python3

import argparse
import os
import msgpack
import onnx

def main():
    parser = argparse.ArgumentParser(description="Convert a .bin file (MessagePack) to ONNX format.")
    parser.add_argument("input_file", help="Path to the neuralnet_rs generated .bin file")
    parser.add_argument("output_file", help="Path to the output ONNX file")
    args = parser.parse_args()

    with open(args.input_file, "rb") as f:
        data = msgpack.unpack(f)

    onnx_model = convert_to_onnx(data)

    onnx.save(onnx_model, args.output_file)

class Layer:
    def __init__(self, data):
        self.inputs = data[0][1][1]
        self.outputs = data[1][1][0]

        self.weights = data[0][2]
        self.biases = data[1][2]
        self.activation = data[2]

    def __repr__(self):
        return f"Layer(inputs={self.inputs}, outputs={self.outputs}, activation={self.activation})"

class Network:
    def __init__(self, data):
        self.layers = [Layer(layer_data) for layer_data in data[0]]

    def __repr__(self):
        return "Network(\n  layers=[\n    " + ",\n    ".join(repr(layer) for layer in self.layers) + "\n  ]\n)"

def convert_to_onnx(data):
    nn = Network(data)
    print(nn)
    
    nodes = []
    initializers = []

    input_info = onnx.helper.make_tensor_value_info(
        name="input",
        elem_type=onnx.TensorProto.FLOAT,
        shape=[None, nn.layers[0].inputs]
    )

    curr_input_name = "input"

    for i, layer in enumerate(nn.layers):
        weights_name = f"w_{i}"
        biases_name = f"b_{i}"
        gemm_output = f"gemm_{i}"
        output_name = f"output_{i}" if i < len(nn.layers) - 1 else "output"

        w_tensor = onnx.helper.make_tensor(
            name=weights_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=[layer.outputs, layer.inputs],
            vals=layer.weights
        )

        b_tensor = onnx.helper.make_tensor(
            name=biases_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=[layer.outputs],
            vals=layer.biases
        )

        initializers.extend([w_tensor, b_tensor])

        nodes.append(
            onnx.helper.make_node(
                "Gemm",
                inputs=[curr_input_name, weights_name, biases_name],
                outputs=[gemm_output],
                transB=1
            )
        )

        nodes.append(
            onnx.helper.make_node(
                layer.activation,
                inputs=[gemm_output],
                outputs=[output_name],
                **({"alpha": 0.01} if layer.activation == "LeakyRelu" else {})
            )
        )

        curr_input_name = output_name

    output_info = onnx.helper.make_tensor_value_info(
        name="output",
        elem_type=onnx.TensorProto.FLOAT,
        shape=[None, nn.layers[-1].outputs]
    )

    graph = onnx.helper.make_graph(
        nodes=nodes,
        name="NetworkGraph",
        inputs=[input_info],
        outputs=[output_info],
        initializer=initializers
    )

    model = onnx.helper.make_model(
        graph,
        producer_name="neuralnet_rs",
        opset_imports=[onnx.helper.make_operatorsetid("", 13)]
    )
    onnx.checker.check_model(model)

    return model


if __name__ == "__main__":
    main()
