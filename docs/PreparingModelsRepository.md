# Preparing the Models Repository for OpenVINO&trade; Model Server

## Introduction 
This guide will help you to create a models repository for serving with the OpenVINO&trade; Model Server. This will help you to serve any model with the OpenVINO&trade; Model Server.


## Creating Repository for Models in **Intermediate Representation (IR)** format
The AI models to be served with OpenVINO&trade; Model Server should be in Intermediate Representation (IR) format (where the graph is represented in .bin and .xml format). Tensorflow, Caffe and MXNet trained models can be converted using Model_Optimizer from  OpenVINO&trade; toolkit. Follow the steps from  [model optimizer documentation](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer) to convert your model.


The IR models should be placed and mounted in a folder structure as depicted below:
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

- Each model should be stored in a dedicated directory (model1 and model2 in the examples above) and should include sub-folders
representing its versions (1,2, etc). The versions and the sub-folder names should be positive integer values. 

- Every version folder _must_ include a pair of model files with .bin and .xml extensions; however, the file name can be arbitrary.

- Each model in IR format defines input and output tensors in the AI graph. By default OpenVINO&trade; model server is using 
tensors names as the input and output dictionary keys.  The client passes the input values to the gRPC request and 
reads the results by referring to the correspondent tensor names. 

Below is the snippet of the example client code:
```python
input_tensorname = 'input'
request.inputs[input_tensorname].CopyFrom(make_tensor_proto(img, shape=(1, 3, 224, 224)))

.....

output_tensorname = 'resnet_v1_50/predictions/Reshape_1'
predictions = make_ndarray(result.outputs[output_tensorname])
```

- It is possible to adjust this behavior by adding an optional json file with name `mapping_config.json` 
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
- This extra mapping can be handy to enable model `user friendly` names on the client when the model has cryptic 
tensor names.

- OpenVINO&trade; model server enables the versions present in the configured model folder according to the defined
[version policy](./ModelVersionPolicy.md).

- If the client does not specify the version number in parameters, by default the latest version is served
