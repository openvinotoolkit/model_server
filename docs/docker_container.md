# Deploying OpenVINO&trade; Model Server using Docker Container

## Introduction

OpenVINO&trade; Model Server is a serving system for machine learning models. OpenVINO&trade; Model Server makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs. This guide will help you deploy OpenVINO&trade; Model Server through docker containers.

## System Requirements

### Hardware 
* Required:
    * 6th to 11th generation Intel® Core™ processors and Intel® Xeon® processors.
* Optional:
    * Intel® Neural Compute Stick 2.
    * Intel® Iris® Pro & Intel® HD Graphics
    * Intel® Vision Accelerator Design with Intel® Movidius™ VPUs.

## Overview 

This guide provides step-by-step instructions on how to deploy OpenVINO&trade; Model Server for Linux using Docker Container including a Quick Start guide. Links are provided for different compatible hardwares. Following instructions are covered in this:

- <a href="#quickstart">Quick Start Guide for OpenVINO&trade; Model Server</a>
- <a href="#sourcecode">Building the OpenVINO&trade; Model Server Image </a>
- <a href="#singlemodel">Starting Docker Container with a Single Model
- <a href="#configfile">Starting Docker container with a configuration file for multiple models</a>
- <a href="#params">Configuration Parameters</a>
- <a href="#storage">Cloud Storage Requirements</a>
- <a href="#ai">Running OpenVINO&trade; Model Server with AI Accelerators NCS, HDDL and GPU</a>
- <a href="#sec">Security Considerations</a>


## Quick Start Guide <a name="quickstart"></a>

A quick start guide to download models and run OpenVINO&trade; Model Server is provided below. 
It allows you to setup OpenVINO&trade; Model Server and run a Face Detection Example.

Refer [Quick Start guide](./ovms_quickstart.md) to set up OpenVINO&trade; Model Server.


## Detailed steps to deploy OpenVINO&trade; Model Server using Docker container

### Install Docker

Install Docker using the following link:

- [Install Docker Engine](https://docs.docker.com/engine/install/)

### Pulling OpenVINO&trade; Model Server Image

After Docker installation you can pull the OpenVINO&trade; Model Server image. Open Terminal and run following command:

```bash
docker pull openvino/model_server:latest
```

Alternatively pull the image from [RedHat Ecosystem Catalog](https://catalog.redhat.com/software/containers/intel/openvino-model-server/607833052937385fc98515de)
```bash
docker pull registry.connect.redhat.com/intel/openvino-model-server:latest
```

###  Building the OpenVINO&trade; Model Server Docker Image<a name="sourcecode"></a>

<details><summary>Building a Docker image</summary>


To build your own image, use the following command in the [git repository root folder](https://github.com/openvinotoolkit/model_server), 

```bash
   make docker_build
```

It will generate the images, tagged as:

- openvino/model_server:latest - with CPU, NCS and HDDL support,
- openvino/model_server-gpu:latest - with CPU, NCS, HDDL and iGPU support,
- openvino/model_server:latest-nginx-mtls - with CPU, NCS and HDDL support and a reference nginx setup of mTLS integration,
as well as a release package (.tar.gz, with ovms binary and necessary libraries), in a ./dist directory.
</details>

*Note:* Latest images include OpenVINO 2021.4 release.

*Note:* OVMS docker image could be created with ubi8-minimal base image, centos7 or the default ubuntu20.
Use command `make docker_build BASE_OS=redhat` or `make docker_build BASE_OS=centos`. OVMS with ubi base image doesn't support NCS and HDDL accelerators.

Additionally you can set version of GPU driver used by the produced image. Currently following versions are available:
- 19.41.14441
- 20.35.17767

Provide version from the list above as INSTALL_DRIVER_VERSION argument in make command to build image with specific version of the driver like 
`make docker_build INSTALL_DRIVER_VERSION=19.41.14441`. 
If not provided, version 20.35.17767 is used.

Docker image can be built also with experimental support for CUDA GPU cards.
It requires placing manually built plugins libCUDAPlugin.so, libAutoPlugin.so, libinterpreter_backend.so and libngraph_backend.so ([How to build plugins](./building_plugins.md))
in the folder release_files before initiating the command: 
```make docker_build CUDA=1```

### Running the OpenVINO&trade; Model Server Image for **Single** Model <a name="singlemodel"></a>

Follow the [Preparation of Model guide](models_repository.md) before running the docker image 

Run the OpenVINO&trade; Model Server by running the following command: 

```
docker run -d --rm -v <models_repository>:/models -p 9000:9000 -p 9001:9001 openvino/model_server:latest \
--model_path <path_to_model> --model_name <model_name> --port 9000 --rest_port 9001 --log_level DEBUG
```

#### Configuration Arguments for Running the OpenVINO&trade; Model Server:

- --rm - Remove the container when exiting the Docker container.
- -d - Run the container in the background.
- -v - Defines how to mount the models folder in the Docker container.
- -p - Exposes the model serving port outside the Docker container.
- openvino/model_server:latest - Represents the image name. This varies by tag and build process. The ovms binary is the Docker entry point. See the full list of [ovms tags](https://hub.docker.com/r/openvino/model_server/tags).
- --model_path - Model location. This can be a Docker container path that is mounted during start-up or a Google* Cloud Storage path in format gs://<bucket>/<model_path> or AWS S3 path s3://<bucket>/<model_path> or az://<container>/<model_path> for Azure blob. See the requirements below for using a cloud storage.
- --model_name - The name of the model in the model_path.
- --port - gRPC server port.
- --rest_port - REST server port.


*Note:*

- Publish the container's port to your host's **open ports**.
- In above command port 9000 is exposed for gRPC and port 9001 is exposed for REST API calls.
- For preparing and saving models to serve with OpenVINO&trade; Model Server refer [models_repository documentation](models_repository.md).
- Add model_name for the client gRPC/REST API calls.

### Starting Docker Container with a Configuration File for **Multiple** Models <a name="configfile"></a>

To use a container that has several models, you must use a model server configuration file that defines each model. The configuration file is in JSON format.
In the configuration file, provide an array, model_config_list, that includes a collection of config objects for each served model. For each config object include, at a minimum, values for the model name and the base_path attributes.

Example configuration file:
```json
{
   "model_config_list":[
      {
         "config":{
            "name":"model_name1",
            "base_path":"/opt/ml/models/model1",
            "batch_size": "16"
         }
      },
      {
         "config":{
            "name":"model_name2",
            "base_path":"/opt/ml/models/model2",
            "batch_size": "auto",
            "model_version_policy": {"all": {}}
         }
      },
      {
         "config":{
            "name":"model_name3",
            "base_path":"gs://bucket/models/model3",
            "model_version_policy": {"specific": { "versions":[1, 3] }},
            "shape": "auto"
         }
      },
      {
         "config":{
             "name":"model_name4",
             "base_path":"s3://bucket/models/model4",
             "shape": {
                "input1": "(1,3,200,200)",
                "input2": "(1,3,50,50)"
             },
         "plugin_config": {"CPU_THROUGHPUT_STREAMS": "CPU_THROUGHPUT_AUTO"}
         }
      },
      {
         "config":{
             "name":"model_name5",
             "base_path":"s3://bucket/models/model5",
             "shape": "auto",
             "nireq": 32,
             "target_device": "HDDL",
         }
      }
   ]
}
```

When the config file is present, the Docker container can be started in a similar manner as a single model. Keep in mind that models with cloud storage path require specific environmental variables set. Refer to cloud storage requirements below.

```bash

docker run --rm -d -v /models/:/opt/ml:ro -p 9001:9001 -p 8001:8001 -v <config.json>:/opt/ml/config.json openvino/model_server:latest \
--config_path /opt/ml/config.json --port 9001 --rest_port 8001

```

*Note:* Follow the below model repository structure for multiple models:

```bash
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

Here the numerical values depict the version number of the model.

### Configuration Parameters<a name="params"></a>:

<details><summary>Model configuration options</summary>

| Option  | Value format | Description | Required |
|---|---|---|---|
| `"model_name"/"name"` | `string` | model name exposed over gRPC and REST API.(use `model_name` in command line, `name` in json config)   | &check;|
| `"model_path"/"base_path"` | `"/opt/ml/models/model"`<br>"gs://bucket/models/model"<br>"s3://bucket/models/model"<br>"azure://bucket/models/model" | If using a Google Cloud Storage, Azure Storage or S3 path, see the requirements below.(use `model_path` in command line, `base_path` in json config)  | &check;|
| `"shape"` | `tuple, json or "auto"` | `shape` is optional and takes precedence over `batch_size`. The `shape` argument changes the model that is enabled in the model server to fit the parameters. <br><br>`shape` accepts three forms of the values:<br>* `auto` - The model server reloads the model with the shape that matches the input data matrix.<br>* a tuple, such as `(1,3,224,224)` - The tuple defines the shape to use for all incoming requests for models with a single input.<br>* A dictionary of shapes, such as `{"input1":"(1,3,224,224)","input2":"(1,3,50,50)", "input3":"auto"}` - This option defines the shape of every included input in the model.<br><br>Some models don't support the reshape operation.<br><br>If the model can't be reshaped, it remains in the original parameters and all requests with incompatible input format result in an error. See the logs for more information about specific errors.<br><br>Learn more about supported model graph layers including all limitations at [Shape Inference Document](https://docs.openvinotoolkit.org/2021.4/_docs_IE_DG_ShapeInference.html). ||
| `"batch_size"` | `integer / "auto"` | Optional. By default, the batch size is derived from the model, defined through the OpenVINO Model Optimizer. `batch_size` is useful for sequential inference requests of the same batch size.<br><br>Some models, such as object detection, don't work correctly with the `batch_size` parameter. With these models, the output's first dimension doesn't represent the batch size. You can set the batch size for these models by using network reshaping and setting the `shape` parameter appropriately.<br><br>The default option of using the Model Optimizer to determine the batch size uses the size of the first dimension in the first input for the size. For example, if the input shape is `(1, 3, 225, 225)`, the batch size is set to `1`. If you set `batch_size` to a numerical value, the model batch size is changed when the service starts.<br><br>`batch_size` also accepts a value of `auto`. If you use `auto`, then the served model batch size is set according to the incoming data at run time. The model is reloaded each time the input data changes the batch size. You might see a delayed response upon the first request.<br>  ||
| `"layout" `| `json / string` | `layout` is optional argument which allows to change the layout of model input and output tensors. Only `NCHW` and `NHWC` layouts are supported.<br><br>When specified with single string value - layout change is only applied to single model input. To change multiple model inputs or outputs, you can specify json object with mapping, such as: `{"input1":"NHWC","input2":"NHWC","output1":"NHWC"}`.<br><br>If not specified, layout is inherited from model. ||
| `"model_version_policy"` | `{ "all": {} }`<br>`{ "latest": { "num_versions":2 } }`<br>`{ "specific": { "versions":[1, 3] } }`</code> | Optional.<br><br>The model version policy lets you decide which versions of a model that the OpenVINO Model Server is to serve. By default, the server serves the latest version. One reason to use this argument is to control the server memory consumption.<br><br>The accepted format is in json.<br><br>Examples:<br><code>{"latest": { "num_versions":2 } # server will serve only two latest versions of model<br><br>{"specific": { "versions":[1, 3] } } # server will serve only versions 1 and 3 of given model<br><br>{"all": {} } # server will serve all available versions of given model ||
| `"plugin_config"` | json with plugin config mappings like`{"CPU_THROUGHPUT_STREAMS": "CPU_THROUGHPUT_AUTO"}` |  List of device plugin parameters. For full list refer to [OpenVINO documentation](https://docs.openvinotoolkit.org/2021.4/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html) and [performance tuning guide](./performance_tuning.md)  ||
| `"nireq"` | `integer` | The size of internal request queue. When set to 0 or no value is set value is calculated automatically based on available resources.||
| `"target_device"` | `"CPU"/"HDDL"/"GPU"/"NCS"/"MULTI"/"HETERO"` | Device name to be used to execute inference operations. Refer to AI accelerators support below. ||
| `stateful` | `bool` | If set to true, model is loaded as stateful. ||
| `idle_sequence_cleanup` | `bool` | If set to true, model will be subject to periodic sequence cleaner scans. <br> See [idle sequence cleanup](stateful_models.md#stateful_cleanup). ||
| `max_sequence_number` | `uint32` | Determines how many sequences can be handled concurrently by a model instance. ||
| `low_latency_transformation` | `bool` | If set to true, model server will apply [low latency transformation](https://docs.openvinotoolkit.org/2021.4/openvino_docs_IE_DG_network_state_intro.html#lowlatency_transformation) on model load. ||

#### To know more about batch size, shape and layout parameters refer [Batch Size, Shape and Layout document](shape_batch_size_and_layout.md)

</details>


<details><summary>Server configuration options</summary>

Configuration options for server are defined only via command line options and determine configuration common for all served models. 

| Option  | Value format  | Description  | Required  |
|---|---|---|---|
| `port` | `integer` | Number of the port used by gRPC sever. | &check;|
| `rest_port` | `integer` | Number of the port used by HTTP server (if not provided or set to 0, HTTP server will not be launched). ||
| `grpc_bind_address` | `string` | Network interface address or a hostname, to which gRPC server will bind to. Default: all interfaces: 0.0.0.0 ||
| `rest_bind_address` | `string` | Network interface address or a hostname, to which REST server will bind to. Default: all interfaces: 0.0.0.0 ||
| `grpc_workers` | `integer` | Number of the gRPC server instances (must be from 1 to CPU core count). Default value is 1 and it's optimal for most use cases. Consider setting higher value while expecting heavy load. ||
| `rest_workers` | `integer` | Number of HTTP server threads. Effective when `rest_port` > 0. Default value is set based on the number of CPUs. ||
| `file_system_poll_wait_seconds` | `integer` | Time interval between config and model versions changes detection in seconds. Default value is 1. Zero value disables changes monitoring. ||
| `sequence_cleaner_poll_wait_minutes` | `integer` | Time interval (in minutes) between next sequence cleaner scans. Sequences of the models that are subjects to idle sequence cleanup that have been inactive since the last scan are removed. Zero value disables sequence cleaner.<br> See [idle sequence cleanup](stateful_models.md#stateful_cleanup). ||
| `cpu_extension` | `string` | Optional path to a library with [custom layers implementation](https://docs.openvinotoolkit.org/2021.4/openvino_docs_IE_DG_Extensibility_DG_Intro.html) (preview feature in OVMS).
| `log_level` | `"DEBUG"/"INFO"/"ERROR"` | Serving logging level ||
| `log_path` | `string` | Optional path to the log file. ||


</details>

### Cloud Storage Requirements<a name="storage"></a>:

OVMS supports a range of cloud storage types. In general OVMS requires "read" and "list" permissions on the model repository side.
Below are specific steps for every storage provider:

<details><summary>Azure Cloud Storage path requirements</summary>

Add the Azure Storage path as the model_path and pass the Azure Storage credentials to the Docker container.

To start a Docker container with support for Azure Storage paths to your model use the AZURE_STORAGE_CONNECTION_STRING variable. This variable contains the connection string to the AS authentication storage account.

Example connection string is: 
```
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=azure_account_name;AccountKey=smp/hashkey==;EndpointSuffix=core.windows.net"
```

Example command with blob storage `az://<container_name>/<model_path>:`
```
docker run --rm -d -p 9001:9001 \
-e AZURE_STORAGE_CONNECTION_STRING=“${AZURE_STORAGE_CONNECTION_STRING}” \
openvino/model_server:latest \
--model_path az://container/model_path --model_name az_model --port 9001
```

Example command with file storage `azfs://<share>/<model_path>:`

```
docker run --rm -d -p 9001:9001 \
-e AZURE_STORAGE_CONNECTION_STRING=“${AZURE_STORAGE_CONNECTION_STRING}” \
openvino/model_server:latest \
--model_path azfs://share/model_path --model_name az_model --port 9001
```
Add `-e "http_proxy=$http_proxy" -e "https_proxy=$https_proxy"` to docker run command for proxy cloud storage connection.

By default, the `https_proxy` variable will be used. If you want to use `http_proxy` please set the `AZURE_STORAGE_USE_HTTP_PROXY` environment variable to any value and pass it to the container.

</details>

<details><summary>Google Cloud Storage path requirements</summary>

Add the Google Cloud Storage path as the model_path and pass the Google Cloud Storage credentials to the Docker container.
Exception: This is not required if you use GKE kubernetes cluster. GKE kubernetes clusters handle authorization.

To start a Docker container with support for Google Cloud Storage paths to your model use the GOOGLE_APPLICATION_CREDENTIALS variable. This variable contains the path to the GCP authentication key.

Example command with `gs://<bucket>/<model_path>:`
```
docker run --rm -d -p 9001:9001 \
-e GOOGLE_APPLICATION_CREDENTIALS=“${GOOGLE_APPLICATION_CREDENTIALS}” \
-v ${GOOGLE_APPLICATION_CREDENTIALS}:${GOOGLE_APPLICATION_CREDENTIALS} \
openvino/model_server:latest \
--model_path gs://bucket/model_path --model_name gs_model --port 9001
```
</details>

<details><summary>AWS S3 and Minio storage path requirements</summary>

Add the S3 path as the model_path and pass the credentials as environment variables to the Docker container.
S3_ENDPOINT is optional for AWS s3 storage and mandatory for Minio and other s3 compatible storage types.

Example command with `s3://<bucket>/<model_path>:`

```
docker run --rm -d -p 9001:9001 \
-e AWS_ACCESS_KEY_ID=“${AWS_ACCESS_KEY_ID}” \
-e AWS_SECRET_ACCESS_KEY=“${AWS_SECRET_ACCESS_KEY}” \
-e AWS_REGION=“${AWS_REGION}” \
-e S3_ENDPOINT=“${S3_ENDPOINT}” \
openvino/model_server:latest \
--model_path s3://bucket/model_path --model_name s3_model --port 9001
```

You can also use anonymous access to s3 public paths.

Example command with `s3://<public_bucket>/<model_path>:`

```
docker run --rm -d -p 9001:9001 \
openvino/model_server:latest \
--model_path s3://public_bucket/model_path --model_name s3_model --port 9001
```

or setup a profile credentials file in the docker image described here
[AWS Named profiles](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html)

Example command with `s3://<bucket>/<model_path>:`

```
docker run --rm -d -p 9001:9001 \
-e AWS_PROFILE=“${AWS_PROFILE}” \
-v ${HOME}/.aws/credentials:/home/ovms/.aws/credentials \
openvino/model_server:latest \
--model_path s3://bucket/model_path --model_name s3_model --port 9001
```
</details>

### Model Version Policy

OpenVINO Model Server can manage the versions of the models in runtime. It includes a model manager, which monitors 
newly added and deleted versions in the models repository and applies the model version policy.
To know more about it, refer to [Version Policy](./model_version_policy.md) document.


### Updating Configuration File
OpenVINO Model Server monitors the changes in its configuration and applies required modifications in runtime in two ways:

- Automatically, with an interval defined by the parameter --file_system_poll_wait_seconds. (introduced in release 2021.1)
- On demand, by using [Config Reload API](./model_server_rest_api.md#config-reload). (introduced in release 2021.3)

Configuration reload triggers the following operations:

- new model or [DAGs](./dag_scheduler.md) added to the configuration file will be loaded and served by OVMS.
- changes made in the configured model storage (e.g. new model version is added) will be applied. 
- changes in the configuration of deployed models and [DAGs](./dag_scheduler.md) will be applied. 
- all model version will be reloaded when there is a change in model configuration.
- when a deployed model, [DAG](./dag_scheduler.md) is deleted from config.json, it will be unloaded completely from OVMS after already started inference operations are completed.
- [DAGs](./dag_scheduler.md) that depends on changed or removed models will also be reloaded.
- changes in [custom loaders](./custom_model_loader.md) and custom node libraries configs will also be applied.

OVMS behavior in case of errors during config reloading:

- if the new config.json is not compliant with json schema, no changes will be applied to the served models.
- if the new model, [DAG](./dag_scheduler.md) or [custom loader](./custom_model_loader.md) has invalid configuration it will be ignored till next configuration reload. Configuration may be invalid because of invalid paths(leading to non-existing directories), forbidden values in config, invalid structure of [DAG](./dag_scheduler.md) (e.g. found cycle in a graph), etc.
- an error during one model reloading, [DAG](./dag_scheduler.md) or [custom loader](./custom_model_loader.md) does not prevent the reload of the remaining updated models.
- errors from configuration reloads triggered internally are saved in the logs. If [Config Reload API](./model_server_rest_api.md#config-reload) was used, also the response contains an error message. 

### Running OpenVINO&trade; Model Server with AI Accelerators NCS, HDDL and GPU <a name="ai"></a>

<details><summary>Using an Intel® Movidius™ Neural Compute Stick</summary>

#### Prepare to use an Intel® Movidius™ Neural Compute Stick

[Intel® Movidius™ Neural Compute Stick 2](https://software.intel.com/en-us/neural-compute-stick) can be employed by OVMS via a [MYRIAD
plugin](https://docs.openvinotoolkit.org/2021.4/openvino_docs_IE_DG_supported_plugins_MYRIAD.html). 

The Intel® Movidius™ Neural Compute Stick must be visible and accessible on host machine. 

Follow steps to update the udev rules if necessary</summary>
</br>

1. Create a file named `97-usbboot.rules` that includes the following content:

```
   SUBSYSTEM=="usb", ATTRS{idProduct}=="2150", ATTRS{idVendor}=="03e7", GROUP="users", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1" 
   SUBSYSTEM=="usb", ATTRS{idProduct}=="2485", ATTRS{idVendor}=="03e7", GROUP="users", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
   SUBSYSTEM=="usb", ATTRS{idProduct}=="f63b", ATTRS{idVendor}=="03e7", GROUP="users", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
```
 
2. In the same directory execute these commands: 

```
   sudo cp 97-usbboot.rules /etc/udev/rules.d/
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   sudo ldconfig
   rm 97-usbboot.rules
```
NCS devices should be reported by `lsusb` command, which should print out `ID 03e7:2485`.<br>

</br>

#### Start the server with an Intel® Movidius™ Neural Compute Stick

To start server with Neural Compute Stick:

```
docker run --rm -it --net=host -u root --privileged -v /opt/model:/opt/model -v /dev:/dev -p 9001:9001 openvino/model_server \
--model_path /opt/model --model_name my_model --port 9001 --target_device MYRIAD
```

`--net=host` and `--privileged` parameters are required for USB connection to work properly. 

`-v /dev:/dev` mounts USB drives.

A single stick can handle one model at a time. If there are multiple sticks plugged in, OpenVINO Toolkit 
chooses to which one the model is loaded. 
</details>

<details><summary>Starting docker container with HDDL</summary>

In order to run container that is using HDDL accelerator, _hddldaemon_ must
 run on host machine. It's required to set up environment 
 (the OpenVINO package must be pre-installed) and start _hddldaemon_ on the
 host before starting a container. Refer to the steps from [OpenVINO documentation](https://docs.openvinotoolkit.org/2021.4/_docs_install_guides_installing_openvino_docker_linux.html#build_docker_image_for_intel_vision_accelerator_design_with_intel_movidius_vpus).

To start server with HDDL you can use command similar to:

```
docker run --rm -it --device=/dev/ion:/dev/ion -v /var/tmp:/var/tmp -v /opt/model:/opt/model -p 9001:9001 openvino/model_server:latest \
--model_path /opt/model --model_name my_model --port 9001 --target_device HDDL
```

`--device=/dev/ion:/dev/ion` mounts the accelerator device.

`-v /var/tmp:/var/tmp` enables communication with _hddldaemon_ running on the
 host machine

Check out our recommendations for [throughput optimization on HDDL](performance_tuning.md#hddl-accelerators)

*Note:* OpenVINO Model Server process in the container communicates with hddldaemon via unix sockets in /var/tmp folder.
It requires RW permissions in the docker container security context. It is recommended to start docker container in the
same context like the account starting hddldaemon. For example if you start the hddldaemon as root, add `--user root` to 
the `docker run` command.

</details>

<details><summary>Starting docker container with GPU</summary>

The [GPU plugin](https://docs.openvinotoolkit.org/2021.4/openvino_docs_IE_DG_supported_plugins_GPU.html) uses the Intel® Compute Library for Deep Neural Networks ([clDNN](https://01.org/cldnn)) to infer deep neural networks. 
It employs for inference execution Intel® Processor Graphics including Intel® HD Graphics and Intel® Iris® Graphics.

Before using GPU as OVMS target device, you need to install the required drivers. Refer to [OpenVINO installation steps](https://docs.openvinotoolkit.org/2021.4/openvino_docs_install_guides_installing_openvino_linux.html).
Next, start the docker container with additional parameter --device /dev/dri to pass the device context and set OVMS parameter --target_device GPU. 
The command example is listed below:

```
docker run --rm -it --device=/dev/dri -v /opt/model:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name my_model --port 9001 --target_device GPU
```

Running the inference operation on GPU requires the ovms process security context account to have correct permissions.
It has to belong to the render group identified by the command:
```
stat -c "group_name=%G group_id=%g" /dev/dri/render*
```
The default account in the docker image is already preconfigured. In case you change the security context, use the following command
to start the ovms container:
```
docker run --rm -it  --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render*) -u $(id -u):$(id -g) \
-v /opt/model:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name my_model --port 9001 --target_device GPU
```

*Note:* The public docker image includes the OpenCL drivers for GPU in version 20.35.17767. Older version could be used 
via building the image with a parameter INSTALL_DRIVER_VERSION:
`make docker_build INSTALL_DRIVER_VERSION=19.41.14441`. The older GPU driver will not support the latest Intel GPU platforms like TigerLake or newer.
</details>

<details><summary>Using Multi-Device Plugin</summary>

If you have multiple inference devices available (e.g. Myriad VPUs and CPU) you can increase inference throughput by enabling the Multi-Device Plugin. 
With Multi-Device Plugin enabled, inference requests will be load balanced between multiple devices. 
For more detailed information read [OpenVino's Multi-Device plugin documentation](https://docs.openvinotoolkit.org/2021.4/_docs_IE_DG_supported_plugins_MULTI.html).

In order to use this feature in OpenVino™ Model Server, following steps are required:

Set target_device for the model in configuration json file to MULTI:<DEVICE_1>,<DEVICE_2> (e.g. MULTI:MYRIAD,CPU, order of the devices defines their priority, so MYRIAD devices will be used first in this example)

When running the inference load on multiple devices, it might help to increase the default size of inference queue size with a parameter `nireq`.
It should be higher than the expected amount of parallel requests from all clients.

Below is exemplary config.json setting up Multi-Device Plugin for resnet model, using Intel® Movidius™ Neural Compute Stick and CPU devices:

```json
{"model_config_list": [
   {"config": {
      "name": "resnet",
      "base_path": "/opt/ml/resnet",
      "batch_size": "1",
      "nireq": 20
      "target_device": "MULTI:MYRIAD,CPU"}
   }]
}

```
Starting OpenVINO™ Model Server with config.json (placed in ./models/config.json path) defined as above config.json:
```
docker run -d --net=host -u root --privileged --rm -v $(pwd)/models/:/opt/ml:ro -v /dev:/dev -p 9001:9001 \
openvino/model_server:latest --config_path /opt/ml/config.json --port 9001 
```
Or alternatively, when you are using just a single model, start OpenVINO™ Model Server using this command (config.json is not needed in this case):
```
docker run -d --net=host -u root --privileged --name ie-serving --rm -v $(pwd)/models/:/opt/ml:ro -v \
 /dev:/dev -p 9001:9001 openvino/model_server:latest model --nireq 20 --model_path /opt/ml/resnet --model_name resnet --port 9001 --target_device 'MULTI:MYRIAD,CPU'
 ```
After these steps, deployed model will perform inference on both Intel® Movidius™ Neural Compute Stick and CPU.
Total throughput will be roughly equal to sum of CPU and Intel® Movidius™ Neural Compute Stick throughput.

While using CUDA and CPU devices, the command starting the container with MULTI plugin_config could look like below:

```
docker run -it -p 9000:9000 --gpus all -v $(pwd)/models/:/opt/ml:ro openvino/model_server:latest-cuda 
--model_name renset --model_path /opt/ml/resnet --target_device MULTI:CPU,CUDA --port 9000 
--plugin_config '{"CPU_THROUGHPUT_STREAMS":"CPU_THROUGHPUT_AUTO"}' --nireq 20
```

</details>

<details><summary>Using Heterogeneous Plugin</summary>

[HETERO plugin](https://docs.openvinotoolkit.org/2021.4/openvino_docs_IE_DG_supported_plugins_HETERO.html) makes it possible to distribute a single inference processing and model between several AI accelerators.
That way different parts of the DL network can split and executed on optimized devices.
OpenVINO automatically divides the network to optimize the execution.

Similarly to the MULTI plugin, Heterogenous plugin can be configured by using `--target_device` parameter using the pattern: `HETERO:<DEVICE_1>,<DEVICE_2>`.
The order of devices defines their priority. The first one is the primary device while the second is the fallback.<br>
Below is a config example using heterogeneous plugin with GPU as a primary device and CPU as a fallback.

```json
{"model_config_list": [
   {"config": {
      "name": "resnet",
      "base_path": "/opt/ml/resnet",
      "batch_size": "1",
      "target_device": "HETERO:GPU,CPU"}
   }]
}
```
Starting OpenVINO™ Model Server docker container with CUDA requires Nvidia GPU drivers and NVIDIA Container Toolkit to be installed on the host.
Docker command requires adding `--gpus` parameter to pass the device to the container like below:

```
docker run -d --gpus all --rm -v $(pwd)/models/:/opt/ml:ro -p 9001:9001 \
openvino/model_server:latest-cuda --config_path /opt/ml/config.json --port 9001 
```
Alternatively, for serving a single model, the server can be started without the configuration file:
```
docker run -d --gpus all --rm -v $(pwd)/models/:/opt/ml:ro -p 9001:9001 \
openvino/model_server:latest-cuda --model_name resnet --model_path /opt/ml/resnet --port 9001 --target_device CUDA
```


</details>

<details><summary>AUTO Plugin</summary>

[AUTO plugin](https://docs.openvinotoolkit.org/2021.4/openvino_docs_IE_DG_supported_plugins_AUTO.html) makes it possible to assign automatically the target device to the model.
The plugin detects all available devices on the host and checks if on which the model is supported. Some model layers and data precisions might 
be not implemented on all accelerators.

Auto plugin can be configured by using `--target_device AUTO` parameter. 
Alternatively, it is possible to to limit the selection of devices like `AUTO:CPU,CUDA`.
Optionally, you can also pass tuning device settings in the plugin_config parameter. By default, Auto plugin assigned one execution stream for CPU device 
to optimize the latency in a single client mode.

```json
{"model_config_list": [
   {"config": {
      "name": "resnet",
      "base_path": "/opt/ml/resnet",
      "batch_size": "1",
      "target_device": "AUTO",
      "plugin_config": {"CPU_THROUGHPUT_STREAMS": "CPU_THROUGHPUT_AUTO"}
      }
   }]
}
```
</details>

<details><summary>CUDA Plugin</summary>

CUDA plugin is currently in experimental version. 

```json
{"model_config_list": [
   {"config": {
      "name": "resnet",
      "base_path": "/opt/ml/resnet",
      "batch_size": "1",
      "target_device": "CUDA"}
   }]
}
```


</details>

## Security Considerations <a name="sec"></a>

OpenVINO Model Server docker containers, by default, starts with the security context of local account ovms with linux uid 5000. It ensure docker container has not elevated permissions on the host machine. This is in line with best practices to use minimal permissions to run docker applications. You can change the security context by adding --user parameter to docker run command. It might be needed for example to load mounted models with restricted access. 
For additional security hardening, you might also consider preventing write operations on the container root filesystem by adding a --read-only flag. It might prevent undesired modification of the container files. It case the cloud storage is used for the models (S3, GoogleStorage or Azure storage), restricting root filesystem should be combined with `--tmpfs /tmp` flag.

```
docker run --rm -d --user $(id -u):$(id -g) --read-only --tmpfs /tmp -v ${pwd}/model/:/model -p 9178:9178 openvino/model_server:latest \
--model_path /model --model_name my_model

``` 
OpenVINO Model Server currently doesn't provide access restrictions and traffic encryption on gRPC and REST API endpoints. The endpoints can be secured using network settings like docker network settings or network firewall on the host. The recommended configuration is to place OpenVINO Model Server behind any reverse proxy component or load balancer, which provides traffic encryption and user authorization.
