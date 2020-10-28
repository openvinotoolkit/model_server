## Preparing the Models Repository

After the Docker image is built, you can use it to start the model server container. The first step is to prepare models to be served.

To serve artificial intelligence (AI) models with OpenVINO Model Server, they should be provided in Intermediate Representation (IR) format (a pair of files with .bin and .xml extensions). 
OpenVINO&trade; toolkit includes a `model_optimizer` tool for converting  TensorFlow, Caffe, MXNet, Kaldi and ONNX trained models into IR format.  
Refer to the [model optimizer documentation](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer) for more details.

Selected models should be placed and mounted in a folder structure as depicted below:

**Note:** folder names - if valid - can be arbitrary

```bash
tree models/
models/
├── model1
│   ├── 1
│   │   ├── ir_model.bin
│   │   └── ir_model.xml
│   └── 2
│       ├── ir_model.bin
│       └── ir_model.xml
└── model2
    └── 1
        ├── ir_model.bin
        ├── ir_model.xml
        └── mapping_config.json
``` 

Each model should have a dedicated folder (model1 and model2 in the examples above) and should include subfolders representing different versions of a model. Model files should be placed in proper version subfolder. The subfolder names with model versions should be positive integer values.

Every version folder _must_ include a pair of model files with .bin and .xml extensions; however, the file name can be arbitrary.

Each model in IR format defines input and output tensors in the AI graph. By default, OpenVINO&trade; model server is using 
tensors names as the input and output dictionary keys.  The client is passing the input values to the gRPC request and 
reads the results by referring to the correspondent tensor names. 

Below is the snippet of the example client code:
```python
# Target model's input name
input_tensorname = 'input'

# Filling gRPC requests with tensor proto for particular input of the target model 
request.inputs[input_tensorname].CopyFrom(make_tensor_proto(img, shape=(1, 3, 224, 224)))

.....
# Target model's output name
output_tensorname = 'resnet_v1_50/predictions/Reshape_1'

# Extracting numpy array from gRPC response with results from particular output of the target model 
predictions = make_ndarray(result.outputs[output_tensorname])

# NOTE: Models can have multiple inputs and outputs.
```

It is possible to adjust this behavior by adding an optional json file with name `mapping_config.json` 
which can map the input and output keys to the appropriate tensors.

```json
{
       "inputs": 
           { "tensor_name":"grpc_custom_input_name"},
       "outputs":{
        "tensor_name1":"grpc_output_key_name1",
        "tensor_name2":"grpc_output_key_name2"
       }
}
```
This extra mapping can be handy to enable model user friendly names on the client when the model has cryptic tensor names. Mapping config file shall be placed in model version subfolder. Mapping set in this file is effective only for that one particular version.

OpenVINO&trade; model server is enabling the versions present in the configured model folder according to the defined
[version policy](docker_container.md#model-version-policy).
By default, the latest version is served.

While the client _is not_ defining the model version in the request specification or the requested version is 0, OpenVINO&trade; Model Server will run the inference
 on the default one which is the latest.
