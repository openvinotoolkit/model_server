# Model Server in Docker Containers {#ovms_docs_docker_container}

This is a step-by-step guide on how to deploy OpenVINO&trade; Model Server on Linux, using a Docker Container. Links are provided for different compatible hardware. 

**Before you start, make sure you have:**

- [Docker Engine](https://docs.docker.com/engine/) installed ([How to Install Docker Engine](https://docs.docker.com/engine/install/))
- Intel® Core™ processor (6-12th gen.) or Intel® Xeon® processor
- (optional) AI accelerators [supported by OpenVINO](https://docs.openvino.ai/latest/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html)
- Linux, macOS or Windows via [WSL](https://docs.microsoft.com/en-us/windows/wsl/) 

**NOTE:** accelerators are only tested on bare-metal Linux hosts.


## Starting with a container <a name="quickstart"></a>

- Pull OpenVINO&trade; Model Server Image.
- Start a Docker Container with OVMS and your chosen model from cloud storage.
- Provide the input files, (arrange an input Dataset).
- Prepare a client package.
- Run the prediction using ovmsclient.

Here is an example of this process using a ResNet50 model for image classification:

Pull an image from Docker or [RedHat Ecosystem Catalog](https://catalog.redhat.com/software/containers/intel/openvino-model-server/607833052937385fc98515de)

```bash
docker pull openvino/model_server:latest

# or, alternatively 

docker pull registry.connect.redhat.com/intel/openvino-model-server:latest
```

Start the container
```bash
# start the container 
docker run -p 9000:9000 openvino/model_server:latest \ 
--model_name resnet --model_path gs://ovms-public-eu/resnet50-binary \ 
--layout NHWC --port 9000 

# download input files, an image, and a label mapping file
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/v2021.4.2/example_client/images/zebra.jpeg 
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/v2021.4.2/example_client/classes.py 

# Install the Python-based ovmsclient package
pip3 install ovmsclient
```

Run prediction
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

To learn how to set up OpenVINO Model Server, refer to the [Quick Start guide](./ovms_quickstart.md).



## Building an OpenVINO&trade; Model Server Docker Image <a name="sourcecode"></a>

You can build your own Docker image executing the `make docker_build` command in the [git repository root folder](https://github.com/openvinotoolkit/model_server).
In the `./dist` directory it will generate: 

- image tagged as openvino/model_server:latest - with CPU, NCS, and HDDL support
- image tagged as openvino/model_server-gpu:latest - with CPU, NCS, HDDL, and iGPU support
- image tagged as openvino/model_server:latest-nginx-mtls - with CPU, NCS, and HDDL support and a reference nginx setup of mTLS integration
- release package (.tar.gz, with ovms binary and necessary libraries)

**Note:** OVMS docker image can be created with ubi8-minimal base image, centos7, or the default ubuntu20. 
Note that OVMS with the ubi base image doesn’t support NCS and HDDL accelerators.

To do so, use either of these commands:

```bash
make docker_build BASE_OS=redhat

make docker_build BASE_OS=centos
```

Additionally, you can use the `INSTALL_DRIVER_VERSION` argument command to choose which GPU driver version is used by the produced image. 
If not provided, most recent version is used.

Currently, the following versions are available:
- 19.41.14441
- 20.35.17767

Example:
```bash
make docker_build INSTALL_DRIVER_VERSION=19.41.14441
```
