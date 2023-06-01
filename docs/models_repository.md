# Preparing a Model Repository {#ovms_docs_models_repository}

The AI models served by OpenVINO&trade; Model Server must be in either of the four formats:
- [OpenVINO IR](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_IR_and_opsets.html#doxid-openvino-docs-m-o-d-g-i-r-and-opsets), where the graph is represented in .bin and .xml files 
- [ONNX](https://onnx.ai/), using the .onnx file
- [PaddlePaddle](https://www.paddlepaddle.org.cn/en), using .pdiparams and .pdmodel files
- [TensorFlow](https://www.tensorflow.org/), using SavedModel, MetaGraph or frozen Protobuf formats.

To use models trained in other formats you need to convert them first. To do so, use 
OpenVINO’s [Model Optimizer](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) for IR, or different
[converters](https://onnx.ai/supported-tools.html) for ONNX.

The models need to be placed and mounted in a particular directory structure and according to the following rules:
When the models are hosted on the cloud storage, they should be frozen to be imported successfully.

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
- Every version folder _must_ include model files, that is, .bin and .xml for IR, .onnx for ONNX, .pdiparams and .pdmodel for Paddlepaddle. The file name can be arbitrary.


Each model defines input and output tensors in the AI graph. The client passes data to model input tensors by filling appropriate entries in the request input map. 
Prediction results can be read from the response output map. By default, OpenVINO™ Model Server uses model tensor names as input and output names in 
prediction requests and responses. The client passes the input values to the request and reads the results by referring to the corresponding output names.

Here is an example of client code:

```python
input_tensorname = 'input'
request.inputs[input_tensorname].CopyFrom(make_tensor_proto(img, shape=(1, 3, 224, 224)))

.....

output_tensorname = 'resnet_v1_50/predictions/Reshape_1'
predictions = make_ndarray(result.outputs[output_tensorname])
```


- It is possible to adjust this behavior by adding an optional .json file named `mapping_config.json`. 
It can map the input and output keys to the appropriate tensors. This extra mapping can be used to enable user-friendly names for models with difficult tensor names.
Here is an example of mapping_config.json:

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

