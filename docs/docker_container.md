# Using Containers {#ovms_docs_docker_container}

## Overview 

This guide provides step-by-step instructions for deploying OpenVINO&trade; Model Server on Linux using Docker Containers. Links are provided for different compatible hardware. 

## System Requirements

- [Docker Engine](https://docs.docker.com/engine/) installed
- 6th to 12th generation Intel® Core™ processors or Intel® Xeon® processors
- (optional) AI accelerators [supported by OpenVINO](https://docs.openvino.ai/latest/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html)
- Linux, macOS or Windows via [WSL](https://docs.microsoft.com/en-us/windows/wsl/) 

NOTE: accelerators are only tested on bare metal Linux hosts.

## Quick Start Guide <a name="quickstart"></a>

Start a Docker container with OpenVINO Model Server and a ResNet-50 model from public cloud storage:
```bash
docker run -p 9000:9000 openvino/model_server:latest \ 
--model_name resnet --model_path gs://ovms-public-eu/resnet50-binary \ 
--layout NHWC --port 9000 
```
Download a JPEG image to classify and list of classes with label mappings:
```
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/main/example_client/images/zebra.jpeg 
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/main/example_client/classes.py 
```
Install the Python-based ovmsclient package:
```
pip3 install ovmsclient
```
Run predication using ovmsclient
```python
import numpy as np
from classes import imagenet_classes
from ovmsclient import make_grpc_client

client = make_grpc_client("localhost:9000")

with open("path/to/img.jpeg", "rb") as f:
   img = f.read()

output = client.predict({"0": img}, "resnet")
result_index = np.argmax(output[0])
predicted_class = imagenet_classes[result_index]
```


Refer to the [Quick Start guide](./ovms_quickstart.md) to set up OpenVINO&trade; Model Server.

## Steps to Build OpenVINO&trade; Model Server from Source

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

To build your own image, use the following command in the [git repository root folder](https://github.com/openvinotoolkit/model_server), 

```bash
   make docker_build
```

It will generate the images, tagged as:

- openvino/model_server:latest - with CPU, NCS and HDDL support,
- openvino/model_server-gpu:latest - with CPU, NCS, HDDL and iGPU support,
- openvino/model_server:latest-nginx-mtls - with CPU, NCS and HDDL support and a reference nginx setup of mTLS integration,
as well as a release package (.tar.gz, with ovms binary and necessary libraries), in a ./dist directory.

*Note:* Latest images include OpenVINO 2021.4 release.

*Note:* OVMS docker image could be created with ubi8-minimal base image, centos7 or the default ubuntu20.
Use command `make docker_build BASE_OS=redhat` or `make docker_build BASE_OS=centos`. OVMS with ubi base image doesn't support NCS and HDDL accelerators.

Additionally you can set version of GPU driver used by the produced image. Currently following versions are available:
- 19.41.14441
- 20.35.17767

Provide version from the list above as INSTALL_DRIVER_VERSION argument in make command to build image with specific version of the driver like 
`make docker_build INSTALL_DRIVER_VERSION=19.41.14441`. 
If not provided, version 20.35.17767 is used.
