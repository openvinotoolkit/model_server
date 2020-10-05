## OpenVINO Model Server Troubleshooting


### Model Import Issues
OpenVINO&trade; Model Server loads all defined models versions according 
to set [version policy](docker_container.md#model-version-policy). 
A model version is represented by a numerical directory in a model path, 
containing OpenVINO model files with .bin and .xml extensions.

An example of an incorrect structure::
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

In the above scenario, server will detect only version `1` of `model1`.
Directory `2` does not contain valid OpenVINO model files, so it won't 
be detected as a valid model version. 
For `model2`, there are correct files, but they are not in a numerical directory. 
The server will not detect any version in `model2`.

When a new model version is detected, the server loads the model files 
and starts serving new model version according the defined [version policy](docker_container.md#model-version-policy). 
This load operation might fail for the following reasons:
- there is a problem with accessing model files (i.e. due to network connectivity issues
to the  remote storage or insufficient permissions)
- OVMS is started with a model, which doesn't support defined configuration like changing the model shape in runtime
- target device different from CPU does not support at least one layer in the model 
- model files are malformed and cannot be imported by the Inference Engine
- OpenVINO is not able to load network into device due to lack of available RAM

In all those situations, the root cause is reported in the server logs or in the response from a call
to GetModelStatus function. 

Detected but not loaded model version will not be served and will report status
`LOADING` with error message: `Error occurred while loading version`.
When model files become accessible or fixed, server will try to 
load them again on the next [version update](docker_container.md#updating-model-versions) 
attempt.

OpenVINO Model Server enables gRPC and REST API endpoint (when `--rest_port` includes a valid port number), 
after all model versions are loaded successfully (in `AVAILABLE` state).
The server will fail to start if it cannot list the content of configured model path.

An exception is when OVMS starts with the configuration file. It that case, gRPC and REST servers will start
and OVMS will wait for the models to show up in configured paths or for updating the configuration file.


### Client Request Issues
When the model server starts successfully and all the models are imported, there could be a couple of reasons for errors 
in the request handling. 
The information about the failure reason is passed to the gRPC or REST client in the response. It is also logged on the 
model server in the `DEBUG` mode.

The possible issues could be:
* Incorrect shape of the input data.
* Incorrect input key name which does not match the tensor name or input key name set in `mapping_config.json`.
* Incorrectly serialized data on the client side.

### Resource Allocation
RAM consumption might depend on the size of the models configured for serving and serving parameters. It should be measured experimentally.
Every version of the model creates a separate inference engine object, so it is recommended to mount only the desired model versions.
Increasing the number of processing streams might also impact the RAM allocation.

OpenVINO&trade; Model Server consumes all available CPU resources unless they are restricted by operating system, Docker or 
Kubernetes capabilities.

### Usage Monitoring
It is possible to track the usage of the models including processing time while DEBUG mode is enabled.
With this setting model server logs will store information about all the incoming requests.
You can parse the logs to analyze: volume of requests, processing statistics and most used models.

### Configuring AWS For Use With a Proxy
To use AWS behind a proxy, an environment variable should be configured. The AWS storage module is using the following format
```
http://user:password@hostname:port
or
https://user:password@hostname:port
```

where user and password are optional. The OVMS will try to use the following environment variables:
```
https_proxy
HTTPS_PROXY
http_proxy
HTTP_proxy
```

Note that neither `no_proxy` or `NO_PROXY` is used.

### Using GCS model behind a proxy
If your environment is required to use proxy but `http_proxy`/`https_proxy` is not passed to server container there will be 15 minutes timeout when accessing GCS models.
During that time no logs will be captured by OVMS. Currently there is no option to change timeout duration for GCS.

### Unable to load network into device with: `can't protect` in server logs
Since this is known bug, please refer OpenVINO&trade; [release notes](https://software.intel.com/content/www/us/en/develop/articles/openvino-relnotes.html).
