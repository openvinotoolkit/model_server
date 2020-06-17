## OpemVINO Model Server Troubleshooting


### Model Import Issues
OpenVINO&trade; Model Server loads all defined models versions according 
to set [version policy](docs/docker_container.md#model-version-policy). 
A model version is represented by a numerical directory in a model path, 
containing OpenVINO model files with .bin and .xml extensions.

Below are examples of incorrect structure:
```bash
models/
├── model1
│   ├── 1
│   │   ├── ir_model.bin
│   │   └── ir_model.xml
│   └── 2
│       ├── somefile.bin
│       └── anotherfile.txt
└── model2
    ├── ir_model.bin
    ├── ir_model.xml
    └── mapping_config.json
```

In above scenario, server will detect only version `1` of `model1`.
Directory `2` does not contain valid OpenVINO model files, so it won't 
be detected as a valid model version. 
For `model2`, there are correct files, but they are not in a numerical directory. 
The server will not detect any version in `model2`.

When new model version is detected, the server loads the model files 
and starts serving new model version. This operation might fail for the following reasons:
- there is a problem with accessing model files (i. e. due to network connectivity issues
to the  remote storage or insufficient permissions)
- model files are malformed and can not be imported by the Inference Engine
- model requires custom CPU extension

In all those situations, the root cause is reported in the server logs or in the response from a call
to GetModelStatus function. 

Detected but not loaded model version will not be served and will report status
`LOADING` with error message: `Error occurred while loading version`.
When model files become accessible or fixed, server will try to 
load them again on the next [version update](docs/docker_container.md#updating-model-versions) 
attempt.

At startup, the server will enable gRPC and REST API endpoint, after all configured models and detected model versions
are loaded successfully (in AVAILABLE state).

The server will fail to start if it can not list the content of configured model paths.


### Client Request Issues
When the model server starts successfully and all the models are imported, there could be a couple of reasons for errors 
in the request handling. 
The information about the failure reason is passed to the gRPC client in the response. It is also logged on the 
model server in the DEBUG mode.

The possible issues could be:
* Incorrect shape of the input data.
* Incorrect input key name which does not match the tensor name or set input key name in `mapping_config.json`.
* Incorrectly serialized data on the client side.

### Resource Allocation
RAM consumption might depend on the size and volume of the models configured for serving. It should be measured experimentally, 
however it can be estimated that each model will consume RAM size equal to the size of the model weights file (.bin file).
Every version of the model creates a separate inference engine object, so it is recommended to mount only the desired model versions.

OpenVINO&trade; model server consumes all available CPU resources unless they are restricted by operating system, docker or 
kubernetes capabilities.

### Usage Monitoring
It is possible to track the usage of the models including processing time while DEBUG mode is enabled.
With this setting model server logs will store information about all the incoming requests.
You can parse the logs to analyze: volume of requests, processing statistics and most used models.
