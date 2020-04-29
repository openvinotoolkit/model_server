# OpenVINO&trade; Model Server quickstart

## Installing the Server

The OpenVINO&trade; Model Server can be installed in three ways - using a prebuilt docker image (preffered way), by building image on your own and by building the application locally from sources. All approaches are described below.

### Prebuilt containers

A prebuilt container with OpenVINO&trade; Model Server is available for download on [dockerhub](https://hub.docker.com/r/openvino/ubuntu18_model_server). This image
is based on Ubuntu 18 operating system and contains all components needed to perform an inference. 

To start using this image, you can pull it using the following command:

```bash
docker pull openvino/ubuntu18_model_server
```

### Build container from sources

In the [repository](https://github.com/openvinotoolkit/model_server) with OpenVINO&trade; model server there are three different Dockerfiles that can be used to build an image with model server locally. In this repository there is also a Makefile that makes building the image far easier. Those Dockerfiles are:

#### 1.[Ubuntu based](https://github.com/openvinotoolkit/model_server/blob/master/Dockerfile)

To build it just issue the following command from a folder with OpenVINO&trade; Model Server sources:

```bash
make docker_build_apt_ubuntu
```

#### 2.[Clearlinux based](https://github.com/openvinotoolkit/model_server/blob/master/Dockerfile_clearlinux)

This one can be built with the following command:

```bash
make docker_build_clearlinux
```

#### 3.[Based on OpenVINO&trade; Toolkit](https://github.com/openvinotoolkit/model_server/blob/master/Dockerfile_binary_openvino)

In this case an image is built based on OpenVINO&trade; Toolkit - to build it you need to obtaing an URL to the toolkit package. It can be done
by registering on the [OpenVINO&trade; Toolkit website](https://software.intel.com/en-us/openvino-toolkit/choose-download).

When you have this URL, please issue the following command - passing there just obtained URL:

```bash
make docker_build_bin dldt_package_url=<url-to-openvino-package-after-registration>/l_openvino_toolkit_p_2020.1.023_online.tgz
```

### Build OpenVINO&trade; Model Server locally from sources

The description how to build model server locally can be found [here](https://github.com/openvinotoolkit/model_server/blob/master/docs/host.md).

## Running the Server

### Model repository

OpenVINO&trade; Model Server uses models in Intermediate Representation (IR) format (a pair of files with .bin and .xml extensions). Such models can be downloaded
using instructions from [open model zoo](https://github.com/opencv/open_model_zoo) or converted from other formats using Model Optimizer which is a part of OpenVINO&trade; Toolkit.

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

Every version folder _must_ include a pair of model files with .bin and .xml extensions; however, the file name can be arbitrary.

### Starting OpenVINO&trade; Model Server Docker Container with a Single Model

When the models are ready and stored in correct folders structure, you are ready to start the Docker container with the OpenVINO™ model server. To enable just a single model, you do not need any extra configuration file, so this process can be completed with just one command like below:

```docker run --rm -d  -v /models/:/opt/ml:ro -p 9001:9001 -p 8001:8001 ie-serving-py:latest \
/ie-serving-py/start_server.sh ie_serving model --model_path /opt/ml/model1 --model_name my_model --port 9001 --rest_port 8001
```

where:
* option `-v` defines how the models folder should be mounted inside the docker container.
* option `-p` exposes the model serving port outside the docker container.
* `ie-serving-py:latest` represent the name of the image with OpenVINO&trade; Model Server. 
* `start_server.sh` script activates the python virtual environment inside the docker container.
* `ie_serving` command starts the model server with the following parameters:
	* `--model_path` - path to the folder's structure with models
	* `--mode_name` - name of the model
	* `--port` - port on which gRpc service listens 
	* `--rest_port` - port on which REST service listens

### Starting OpenVINO&trade; Model Server Docker Container with Multiple Models

OpenVINO&trade; Model server can handle more than one model at the same moment. To start it in this mode, a special configuration file with location of models has to be preapred. Below there is an example of such file:

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
The file is in JSON format and contains a mandatory array `model_config_list`, which includes a collection of `config` objects for each served model. 
Each config object includes values for the model `name` and the `base_path` attributes.

To run the OpenVINO&trade; Model Server in this mode - please execute the following command:

```docker run --rm -d  -v /models/:/opt/ml:ro -p 9001:9001 -p 8001:8001 ie-serving-py:latest \
/ie-serving-py/start_server.sh ie_serving config config /opt/ml/config.json --port 9001 --rest_port 8001
```

**Note** do not forget to copy the configuration file to the `/models/` (in this case) folder on the host machine.

### Starting OpenVINO&trade; Model Server locally

OpenVINO&trade; Model Server can be run locally on a machine where it was previously installed in the way described [here](https://github.com/openvinotoolkit/model_server/blob/master/docs/host.md)[here](https://github.com/openvinotoolkit/model_server/blob/master/docs/host.md).

To start Model Server please execute the following commands:

```
ie_serving model --model_path /opt/ml/model1 --model_name my_model --port 9001 --rest_port 8001
```

to run Model Server exposing one model located in the `/opt/ml/model1` folder. Server can be accessed via ports `9001` (gRpc) and `8001` (REST).

To run Model Server exposing many models - please use the following command:

```
ie_serving config config /opt/ml/config.json --port 9001 --rest_port 8001
```

this Model Server exposes models described in the `/opt/ml/config.json` configuration file. Inference service can be reached on ports `9001` (gRpc) and `8001` (REST).

## Checking the status of the served models

OpenVINO&trade; Model Server provides a service to check a status of the served models. To use it just send a GET request to the following address `http://<ovms_host>:<rest_port>/v1/models/<model_name>` for example:

```
curl http://localhost:8001/v1/models/my_model
```

This service returns information about a state of a service serving a model with the given name and information about a version of the served model. In case of errors on the OVMS side - a short description of a problem is also returned.


