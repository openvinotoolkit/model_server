# Model Server in Docker Containers {#ovms_docs_docker_container}

This is a step-by-step guide on how to deploy OpenVINO&trade; Model Server on Linux, using a Docker Container. Links are provided for different compatible hardware. 

**Before you start, make sure you have:**

- [Docker Engine](https://docs.docker.com/engine/) installed ([How to Install Docker Engine](https://docs.docker.com/engine/install/))
- Intel® Core™ processor (6-12th gen.) or Intel® Xeon® processor
- (optional) AI accelerators [supported by OpenVINO](https://docs.openvino.ai/2022.1/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html)
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
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/1/demos/common/static/images/zebra.jpeg
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/1/demos/common/python/classes.py

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

**Note:** OVMS docker image can be created with ubi8-minimal base image or the default ubuntu20. 
Note that OVMS with the ubi base image doesn’t support NCS and HDDL accelerators.

To do so, use either of these commands:

Running the inference operation on GPU requires the ovms process security context account to have correct permissions.
It has to belong to the render group identified by the command:
```
stat -c "group_name=%G group_id=%g" /dev/dri/render*
```
The default account in the docker image is already preconfigured. In case you change the security context, use the following command
to start the ovms container:
```
docker run --rm -it  --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
-v /opt/model:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
--model_path /opt/model --model_name my_model --port 9001 --target_device GPU
```

*Note:* The public docker image includes the OpenCL drivers for GPU in version 21.38.21026.

## Using Multi-Device Plugin

If you have multiple inference devices available (e.g. Myriad VPUs and CPU) you can increase inference throughput by enabling the Multi-Device Plugin. 
With Multi-Device Plugin enabled, inference requests will be load balanced between multiple devices. 
For more detailed information read [OpenVino's Multi-Device plugin documentation](https://docs.openvino.ai/2022.1/openvino_docs_OV_UG_Running_on_multiple_devices.html).

In order to use this feature in OpenVino™ Model Server, following steps are required:

Set target_device for the model in configuration json file to MULTI:DEVICE_1,DEVICE_2 (e.g. MULTI:MYRIAD,CPU, order of the devices defines their priority, so MYRIAD devices will be used first in this example)

Below is exemplary config.json setting up Multi-Device Plugin for resnet model, using Intel® Movidius™ Neural Compute Stick and CPU devices:
```
make docker_build BASE_OS=ubuntu
```

Additionally, you can use the `INSTALL_DRIVER_VERSION` argument command to choose which GPU driver version is used by the produced image. 
If not provided, most recent version is used.

Currently, the following versions are available:
- 21.38.21026 - Redhat
- 21.48.21782 - Ubuntu

Example:
```bash
make docker_build INSTALL_DRIVER_VERSION=21.38.21026
```
If not provided, version 21.38.21026 is used for Redhat and 21.48.21782 is used for Ubuntu.
