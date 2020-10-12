# OpenVINO&trade; Model Server Troubleshooting

## Introduction
This document gives information about troubleshooting following issues while using OpenVINO&trade; Model Server:
* <a href="#model-import">Model Import Issues</a>
* <a href="#client-request">Client Request Issues</a>
* <a href="#resource-allocation">Resource Allocation</a>
* <a href="#usage-monitoring">Usage Monitoring</a>
* <a href="#configure-aws">Configuring AWS For Use With a Proxy</a>
* <a href="#gcs">Using GCS model behind a proxy </a>
* <a href="#load-network-issue">Unable to load network into device with: `can't protect` in server logs </a>


## Model Import Issues<a name="model-import"></a>

OpenVINO&trade; Model Server loads all defined models versions according to set [version policy](./ModelVersionPolicy.md). A model version is represented by a numerical directory in a model path, containing OpenVINO model files with .bin and .xml extensions.

When new model version is detected, the server loads the model files and starts serving new model version. This operation might fail for the following reasons:
- There is a problem with accessing model files (i. e. due to network connectivity issues to the  remote storage or insufficient permissions)
- Model files are malformed and can not be imported by the Inference Engine
- Model requires custom CPU extension


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
- In the above example, the server will detect only Directory `1` of `model1`. It will not detect Directory `2` as valid model version because it does not contain valid OpenVINO model files.

- The server will not detect any version in `model2`because although the files in `model2` are correct, but they are not in a numerical directory 

- The root cause is reported in the server logs or in the response from a call to GetModelStatus function. 

- A model version that is detected but not loaded will not be served. It will report status `LOADING` with error message: `Error occurred while loading version`.

- When model files become accessible or fixed, server will try to load them again on the next [version update](./ModelVersionPolicy.md) attempt.

- At startup, the server will enable gRPC and REST API endpoint, after all configured models and detected model versions are loaded successfully (in AVAILABLE state).

- The server will fail to start if it can not list the content of configured model paths.


## Client Request Issues<a name="client-request"></a>
- When the model server starts successfully and all the models are imported, there could be a couple of reasons for errors in the request handling. 
- The information about the failure reason is passed to the gRPC client in the response. It is also logged on the model server in the DEBUG mode.

The possible issues could be:
* Incorrect shape of the input data.
* Incorrect input key name which does not match the tensor name or set input key name in `mapping_config.json`.
* Incorrectly serialized data on the client side.

## Resource Allocation<a name="resource-allocation"></a>
- RAM consumption might depend on the size and volume of the models configured for serving. It should be measured experimentally, however it can be estimated that each model will consume RAM size equal to the size of the model weights file (.bin file).

- Every version of the model creates a separate inference engine object, so it is recommended to mount only the desired model versions.

- OpenVINO&trade; model server consumes all available CPU resources unless they are restricted by operating system, docker or 
kubernetes capabilities.

## Usage Monitoring<a name="usage-monitoring"></a>
- It is possible to track the usage of the models including processing time while DEBUG mode is enabled.
- With this setting model server logs will store information about all the incoming requests.
- You can parse the logs to analyze: volume of requests, processing statistics and most used models.

## Configuring AWS For Use With a Proxy<a name="configure-aws"></a>
- To use AWS behind a proxy, an environment variable should be configured. The AWS storage module is using the following format
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

> **Note**: that neither `no_proxy` or `NO_PROXY` is used.

## Using GCS model behind a proxy <a name="gcs"></a>

- If your environment is required to use proxy but `http_proxy`/`https_proxy` is not passed to server container there will be 15 minutes timeout when accessing GCS models.
- During that time no logs will be captured by OVMS. Currently there is no option to change timeout duration for GCS.

## Unable to load network into device with: `can't protect` in server logs <a name="load-network-issue"></a>
- Since this is known bug, please refer OpenVINO&trade; [release notes](https://software.intel.com/content/www/us/en/develop/articles/openvino-relnotes.html).
