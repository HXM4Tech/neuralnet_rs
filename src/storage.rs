use crate::network::Network;

use std::fs::File;

impl Network {
    #[allow(dead_code)]
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        rmp_serde::encode::write(
            &mut file,
            self
        ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    #[allow(dead_code)]
    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;

        rmp_serde::decode::from_read(
            file
        ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    #[allow(dead_code)]
    pub fn save_onnx(&self, path: &str) -> std::io::Result<()> {
        use crate::activation::Activation;

        use onnx_pb::{
            AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, ValueInfoProto,
            TypeProto, type_proto, TensorShapeProto, tensor_shape_proto
        };
        use prost::Message;
        use std::fs::File;
        use std::io::Write;

        let mut graph = GraphProto::default();
        graph.name = "NetworkGraph".to_string();

        let input_features = self.layers[0].weights.shape()[1] as i64;
        let input_dims = vec![
            tensor_shape_proto::Dimension {
                value: Some(tensor_shape_proto::dimension::Value::DimValue(-1)), // -1 means dynamic batch size
                ..Default::default()
            },
            tensor_shape_proto::Dimension {
                value: Some(tensor_shape_proto::dimension::Value::DimValue(input_features)),
                ..Default::default()
            },
        ];

        graph.input.push(ValueInfoProto {
            name: "input".to_string(),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: 1, // 1 = FLOAT
                    shape: Some(TensorShapeProto { dim: input_dims }),
                })),
                ..Default::default()
            }),
            ..Default::default()
        });

        let mut curr_input_name = "input".to_string();
        let num_layers = self.layers.len();

        for (i, layer) in self.layers.iter().enumerate() {
            let weights_name = format!("w_{}", i);
            let biases_name = format!("b_{}", i);
            let gemm_output = format!("gemm_{}", i);
            
            let output_name = if i == num_layers - 1 {
                "output".to_string()
            } else {
                format!("output_{}", i)
            };

            // weights and biases initializers
            let weights_shape = layer.weights.shape().iter().map(|&d| d as i64).collect();
            graph.initializer.push(TensorProto {
                name: weights_name.clone(),
                data_type: 1, // 1 = FLOAT
                dims: weights_shape,
                float_data: layer.weights.as_slice().unwrap().to_vec(),
                ..Default::default()
            });

            let biases_shape = layer.biases.shape().iter().map(|&d| d as i64).collect();
            graph.initializer.push(TensorProto {
                name: biases_name.clone(),
                data_type: 1, // 1 = FLOAT
                dims: biases_shape,
                float_data: layer.biases.as_slice().unwrap().to_vec(),
                ..Default::default()
            });

            // Gemm node
            let mut gemm_node = NodeProto::default();
            gemm_node.op_type = "Gemm".to_string();
            gemm_node.input = vec![curr_input_name.clone(), weights_name, biases_name];
            gemm_node.output = vec![gemm_output.clone()];
            
            gemm_node.attribute.push(AttributeProto {
                name: "transB".to_string(),
                i: 1,
                r#type: 2, // 2 = INT
                ..Default::default()
            });
            graph.node.push(gemm_node);

            // Activation node
            let mut act_node = NodeProto::default();
            act_node.op_type = layer.activation.as_ref().to_string();
            act_node.input = vec![gemm_output];
            act_node.output = vec![output_name.clone()];

            if layer.activation == Activation::LeakyRelu {
                act_node.attribute.push(AttributeProto {
                    name: "alpha".to_string(),
                    f: 0.01,
                    r#type: 1, // 1 = FLOAT
                    ..Default::default()
                });
            }

            graph.node.push(act_node);
            curr_input_name = output_name;
        }

        // Output value info
        let output_features = self.layers.last().unwrap().biases.shape()[0] as i64;
        let output_dims = vec![
            tensor_shape_proto::Dimension {
                value: Some(tensor_shape_proto::dimension::Value::DimValue(-1)), // -1 means dynamic batch size
                ..Default::default()
            },
            tensor_shape_proto::Dimension {
                value: Some(tensor_shape_proto::dimension::Value::DimValue(output_features)),
                ..Default::default()
            },
        ];

        graph.output.push(ValueInfoProto {
            name: "output".to_string(),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: 1, // 1 = FLOAT
                    shape: Some(TensorShapeProto { dim: output_dims }),
                })),
                ..Default::default()
            }),
            ..Default::default()
        });

        // Model metadata and opset import
        let mut model = ModelProto::default();
        model.ir_version = 7;
        model.producer_name = "neuralnet_rs".to_string();
        model.producer_version = env!("CARGO_PKG_VERSION").to_string();
        model.graph = Some(graph);
        model.opset_import.push(OperatorSetIdProto {
            domain: "".to_string(), // empty domain means the default ONNX opset
            version: 13,
        });

        let mut buf = Vec::new();

        model.encode(&mut buf).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e)
        })?;

        // Save the ONNX model to file
        let mut file = File::create(path)?;
        file.write_all(&buf)?;

        Ok(())
    }
}
