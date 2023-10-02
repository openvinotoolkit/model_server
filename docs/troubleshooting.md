# Troubleshooting {#ovms_docs_troubleshooting}

## Introduction
This document gives information about troubleshooting the following issues while using the OpenVINO&trade; Model Server:
* <a href="#model-import">Model Import Issues</a>
* <a href="#client-request">Client Request Issues</a>
* <a href="#resource-allocation">Resource Allocation</a>
* <a href="#usage-monitoring">Usage Monitoring</a>
* <a href="#configure-aws">Configuring S3 storage For Use With a Proxy</a>
* <a href="#gcs">Using GCS storage behind a proxy </a>
* <a href="#model-cache">Model Cache Issues </a>


## Model Import Issues<a name="model-import"></a>

OpenVINO&trade; Model Server loads all defined models versions according to set [version policy](./model_version_policy.md). A model version is represented by a numerical directory in a model path, containing OpenVINO model files with .bin and .xml extensions.

When a new model version is detected, the server loads the model files and starts serving a new model version. This operation might fail for the following reasons :
- There is a problem with accessing model files (due to network connectivity issues to the remote storage or insufficient permissions).
- Model files are malformed and can not be imported by the OpenVINO&trade; Runtime.
- Model requires a custom CPU extension.


Below are examples of incorrect structure :
```
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
- In the above example, the server will detect only Directory `1` of `model1`. It will not detect Directory `2` as a valid model version because it does not contain valid OpenVINO model files.

- The server will not detect any version in `model2` because, although the files in `model2` are correct, they are not in a numerical directory.

- The root cause is reported in the server logs or the response from a call to GetModelStatus function. 

- A model version that is detected but not loaded will not be served. It will report status `LOADING` with the error message: `Error occurred while loading version`.

- When model files become accessible or fixed, the server will try to load them again on the next version update attempt.

- Model import will fail if the OVMS process does not have read permissions to the model files and list permissions on the model folder and model version subfolder. 


## Client Request Issues<a name="client-request"></a>
- When the model server starts successfully and all the models are imported, there could be a couple of reasons for errors in the request handling. 
- The information about the failure reason is passed to the client in the response. It is also logged on the model server in the DEBUG mode.

The possible issues could be :
* Incorrect shape of the input data.
* Incorrect input key name which does not match the tensor name or set input key name in `mapping_config.json`.
* Incorrectly serialized data on the client-side.

## Resource Allocation<a name="resource-allocation"></a>
- RAM consumption might depend on the size and volume of the models configured for serving. It should be measured experimentally, however it can be estimated that each model will consume RAM size equal to the size of the model weights file (.bin file).

- Every version of the model enabled in the version policy creates a separate OpenVINO&trade; Runtime `ov::Model` and `ov::CompiledModel` object. By default, only the latest version is enabled.

- OpenVINO&trade; model server consumes all available CPU resources unless they are restricted by the operating system, Docker or Kubernetes capabilities.

*Note* When insufficient memory is allocated to the container, it might get terminated by the Docker engine [OOM Killer](https://docs.docker.com/config/containers/resource_constraints/). There will be no termination root cause
mentioned in the OVMS logs but such a situation can be confirmed by `docker inspect <terminated_container>` and reported State `"OOMKilled": true`.
It will be also included in the host system logs like `Memory cgroup out of memory: Killed process`.


## Usage Monitoring<a name="usage-monitoring"></a>
- Prometheus standard metrics are available via the REST /metrics endpoint.
- It is possible to track the usage of the models including processing time while DEBUG mode is enabled.
- With this setting model server logs will store information about all the incoming requests.
- You can parse the logs to analyze: the volume of requests, processing statistics, and most used models.

## Configuring S3 Storage For Use With a Proxy<a name="configure-aws"></a>
- To use S3 storage behind a proxy, an environment variable should be configured. The S3 loader module is using the following format
```
http://user:password@hostname:port
or
https://user:password@hostname:port
```
where user and password are optional. The OVMS will try to use the following environment variables :
```
https_proxy
HTTPS_PROXY
http_proxy
HTTP_proxy
```

> **Note**: that neither `no_proxy` or `NO_PROXY` is used.

## Using GCS Model Behind a Proxy <a name="gcs"></a>

- If your environment is required to use proxy but `http_proxy`/`https_proxy` is not passed to the server container there will be 15 minutes timeout when accessing GCS models.
- During that time no logs will be captured by OVMS. Currently, there is no option to change the timeout duration for GCS.

## Model Cache Issues <a name="model-cache"></a>

- Cache folder (by default `/opt/cache` or defined by `--cache_dir`) should be mounted into docker container with read-write access. Unless changed by the docker run command, the model server has a security context of ovms account with uid 5000.
- The biggest speedup in the model loading time is expected for GPU device. For CPU device the gain will depend on the model topology. In some rare cases, it is possible the load time will not be improved noticeably or it might be even slightly slower.
