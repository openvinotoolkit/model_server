# Classic Models {#ovms_docs_models_repository_classic}

```{toctree}
---
maxdepth: 1
hidden:
---
ovms_docs_cloud_storage

```

Classic models perform data analysis in a single inference operation. They can be served using the KServe API. 

The AI models served by OpenVINO&trade; Model Server must be in either of the five formats:
- [OpenVINO IR](https://docs.openvino.ai/2026/documentation/openvino-ir-format.html), where the graph is represented in .bin and .xml files
- [ONNX](https://onnx.ai/), using the .onnx file
- [PaddlePaddle](https://www.paddlepaddle.org.cn/en), using .pdiparams and .pdmodel files
- [TensorFlow](https://www.tensorflow.org/), using SavedModel, MetaGraph or frozen Protobuf formats.
- [TensorFlow Lite](https://www.tensorflow.org/lite), using the .tflite file

To use models trained in other formats you need to convert them first. To do so, use
OpenVINO’s [conversion tool](https://docs.openvino.ai/2026/openvino-workflow/model-preparation/convert-model-to-ir.html) for IR, or different
[converters](https://onnx.ai/supported-tools.html) for ONNX.

The models need to be placed and mounted in a particular directory structure according to the following rules:

```
tree models/
models/
├── model1
│   ├── 1
│   │   ├── ir_model.bin
│   │   └── ir_model.xml
│   └── 2
│       ├── ir_model.bin
│       └── ir_model.xml
├── model2
│   └── 1
│       ├── ir_model.bin
│       ├── ir_model.xml
│       └── mapping_config.json
├── model3
│    └── 1
│        └── model.onnx
├── model4
│      └── 1
│        ├── model.pdiparams
│        └── model.pdmodel
├── model5
│      └── 1
│        ├── model.pdiparams
│        └── model.pdmodel
└── model6
       └── 1
         ├── variables
         └── saved_model.pb

```

- Each model should be stored in a dedicated directory, e.g. model1 and model2.
- Each model directory should include a sub-folder for each of its versions (1,2, etc). The versions and their folder names should be positive integer values.
**Note:** In execution, the versions are enabled according to a pre-defined version policy. If the client does not specify
the version number in parameters, by default, the latest version is served.
- As an alternative for local filesystem only, `model_path` / `base_path` can point directly to a single model file (`.xml`, `.onnx`, `.pdmodel`, `.pdiparams`, `.pb`, `.tflite`). In this mode, Model Server exposes synthetic version `1`.
- Every version folder _must_ include model files, that is, .bin and .xml for IR, .onnx for ONNX, .pdiparams and .pdmodel for Paddlepaddle. The file name can be arbitrary.
- Each model defines input and output tensors in the AI graph. The client passes data to model input tensors by filling appropriate entries in the request input map.
- Prediction results can be read from the response output map. By default, OpenVINO™ Model Server uses model tensor names as input and output names in prediction requests and responses. The client passes the input values to the request and reads the results by referring to the corresponding output names.
- You can optionally add a `mapping_config.json` file to customize input and output names. This file maps tensor names to user-friendly keys, which is particularly useful for models with complex tensor naming. Here is an example:
```json
{
       "inputs":{
          "tensor_name":"grpc_custom_input_name"
       },
       "outputs":{
          "tensor_name1":"grpc_output_key_name1",
          "tensor_name2":"grpc_output_key_name2"
       }
}
```

## Serving multiple models

Serving many models at the same time requires preparing a `config.json` file with a list of models to be deployed. This can be done using the OVMS CLI

```text
ovms --add_to_config --config_path config.json --model_name model1 --target_device GPU --batch_size 1 --model_path models/model1
```

Models can also be removed from the configuration file:
```text
ovms --remove_from_config --config_path config.json --model_name model1
```

All models can be started with the following command:
```

For more information on how to use cloud-hosted models, refer to the [cloud storage guide](./using_cloud_storage.md).

For additional information, see how to [start the model server](./starting_server.md).

