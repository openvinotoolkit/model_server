# OpenVINO&trade; Model Server Quickstart

## Deploying Model Server

Model Server can be deployed using any of the following options:
1. Download a pre-built container image from Docker Hub
2. Build a container image with a Dockerfile
3. Build the application locally from source 

Each approach is described below.

### Pre-built Docker Container

A pre-built container with OpenVINO&trade; Model Server is available for download on [Docker Hub](https://hub.docker.com/r/openvino/ubuntu18_model_server). The image
is based on Ubuntu 18.04 and contains all required components needed to perform inference. 

To start using this image, you can pull it with the following command:

```bash
docker pull openvino/ubuntu18_model_server
```

### Build with a Dockerfile

Three Dockerfiles are provided that can be used to build a container image locally. There is also a Makefile to simplify building an image. The Dockerfiles are listed below:

#### 1. [Ubuntu 18 Dockerfile](../Dockerfile)

To build this image use following command from the `model_server` directory:

```bash
make docker_build_apt_ubuntu
```

#### 2. [Clear Linux Dockerfile](../Dockerfile_clearlinux)

To build this image use following command from the `model_server` directory:

```bash
make docker_build_clearlinux
```

#### 3. [Based on OpenVINO Toolkit](../Dockerfile_binary_openvino)

In this case an image is built based on OpenVINO Toolkit - to build it you need to obtaing an URL to the toolkit package. It can be done by registering on the [OpenVINO&trade; Toolkit website](https://software.intel.com/en-us/openvino-toolkit/choose-download).

When you have this URL, please issue the following command - passing there just obtained URL:

```bash
make docker_build_bin dldt_package_url=<url-to-openvino-package-after-registration>/l_openvino_toolkit_p_2020.1.023_online.tgz
```

### Build Locally from Source

Instructions to build locally are [here](../host.md).


## Running Model Server

### Model Repository

Model Server uses models in Intermediate Representation (IR) format (a pair of files with .bin and .xml extensions). These models can be downloaded from the [Open Model Zoo](https://github.com/opencv/open_model_zoo) or by converting TensorFlow, ONNX, Caffe, MXNet or other supported model formats by using [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) which is a part of the OpenVINO Toolkit.

Predefined IR models should be placed in a folder structure as depicted below:
```bash
tree models/
models/
├── model1
│   ├── 1
│   │   ├── ir_model.bin
│   │   └── ir_model.xml
│   └── 2
│       ├── ir_model.bin
│       └── ir_model.xml
└── model2
    └── 1
        ├── ir_model.bin
        ├── ir_model.xml
        └── mapping_config.json
``` 

Each model should be stored in a dedicated folder (model1 and model2 in the examples above) and should include subfolders
representing its versions. The versions and the subfolder names should be positive integer values. 

Every version folder _must_ include a pair of model files with .bin and .xml extensions; however, the filename can be arbitrary.

### Starting Model Server Container with a Single Model

When models are ready and stored in the correct directory structure, you are ready to start the Docker container with the Model Server. To serve just a single model, no configuration file is required, just a single command:

```docker run --rm -d  -v /models/:/opt/ml:ro -p 9001:9001 -p 8001:8001 ie-serving-py:latest \
/ie-serving-py/start_server.sh ie_serving model --model_path /opt/ml/model1 --model_name my_model --port 9001 --rest_port 8001
```

Where:
* option `-v` defines how the models folder should be mounted inside the docker container.
* option `-p` exposes the model serving port outside the docker container.
* `ie-serving-py:latest` represent the name of the image with OpenVINO&trade; Model Server. 
* `start_server.sh` script activates the python virtual environment inside the docker container.
* `ie_serving` command starts the model server with the following parameters:
	* `--model_path` - path to the folder's structure with models
	* `--mode_name` - name of the model
	* `--port` - port on which gRpc service listens 
	* `--rest_port` - port on which REST service listens

### Starting Model Server Container with Multiple Models

Model Server can serve multiple models at once. To start in this mode, a configuration file with the location of models is required. An example of this file is included below:

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
      }
   ]
}

```
The file is in JSON format and contains a mandatory array `model_config_list`, which includes a collection of `config` objects for each served model. Each config object includes values for the model `name` and the `base_path` attributes.

To run Model Server in this mode - please run the following command:

```docker run --rm -d  -v /models/:/opt/ml:ro -p 9001:9001 -p 8001:8001 ie-serving-py:latest \
/ie-serving-py/start_server.sh ie_serving config /opt/ml/config.json --port 9001 --rest_port 8001
```

**Note** do not forget to copy the configuration file to the `/models/` directory.

### Starting Model Server on Bare Metal/Virtual Machine

Model Server can be run locally on a machine after it has been installed as described [here](./host.md).

To start Model Server locally please run the following command:

```
ie_serving model --model_path /opt/ml/model1 --model_name my_model --port 9001 --rest_port 8001
```

To run Model Server exposing one model located in the `/opt/ml/model1` folder. Server can be accessed via ports `9001` (gRPC) and `8001` (HTTP/REST).

To run Model Server exposing multiple models - please use the following command:

```
ie_serving config config /opt/ml/config.json --port 9001 --rest_port 8001
```

Model Server exposes models described in the `/opt/ml/config.json` configuration file. The inference service can be reached on ports `9001` (gRPC) and `8001` (HTTP/REST).

## Checking the Status of Served Models

Model Server provides a service to check a status of the served models. To use it just send a GET request to the following address `http://<ovms_host>:<rest_port>/v1/models/<model_name>` for example:

```
curl http://localhost:8001/v1/models/my_model
```

This service returns information about the state of a served model. A short description of a problem is also returned when an error occurs.
