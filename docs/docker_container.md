# Deploying Model Server inside Docker Container {#ovms_docs_docker_container}

OpenVINO Model Server is hosted inside a docker container. You can either download a pre-build container or build a container from source. It is also suitable for landing in the [Kubernetes environment](installations_kubernetes.md).

* <a href="#prebuild-container">Use Pre-build Model Server Container</a>
* <a href="#model-server-installation">Build Container from Source</a>

## Use Pre-build Model Server Container <a name="prebuild-container"></a>

This is a step-by-step guide on how to deploy OpenVINO&trade; Model Server on Linux, using a pre-build Docker Container. 

**Before you start, make sure you have:**

- [Docker Engine](https://docs.docker.com/engine/) installed 
- Intel® Core™ processor (6-12th gen.) or Intel® Xeon® processor
- Linux, macOS or Windows via [WSL](https://docs.microsoft.com/en-us/windows/wsl/) 
- (optional) AI accelerators [supported by OpenVINO](https://docs.openvino.ai/2022.2/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html)

> **NOTE**: Accelerators are tested only on bare-metal Linux hosts.

### Launch Model Server Container <a name="quickstart"></a>

1. Pull OpenVINO&trade; Model Server Image.
2. Prepare data for serving:
   - Start a Docker Container with OVMS and your model.
   - Provide the input files.
   - Prepare a client package.
3. Run the prediction using ovmsclient.

Here is an example of launching the model server using a ResNet50 image classification model from a cloud storage:

#### Step 1. Pull Model Server Image

Pull an image from Docker: 

```bash
docker pull openvino/model_server:latest
```

or [RedHat Ecosystem Catalog](https://catalog.redhat.com/software/containers/intel/openvino-model-server/607833052937385fc98515de):

```
docker pull registry.connect.redhat.com/intel/openvino-model-server:latest
```

#### Step 2. Prepare Data for Serving

```bash
# start the container with the model
docker run -p 9000:9000 openvino/model_server:latest \ 
--model_name resnet --model_path gs://ovms-public-eu/resnet50-binary \ 
--layout NHWC:NCHW --port 9000 

# download input files: an image and a label mapping file
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/1/demos/common/static/images/zebra.jpeg
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/1/demos/common/python/classes.py

# install the Python-based ovmsclient package
pip3 install ovmsclient
```

#### Step 3. Run Prediction


```bash
echo 'import numpy as np
from classes import imagenet_classes
from ovmsclient import make_grpc_client

client = make_grpc_client("localhost:9000")

with open("zebra.jpeg", "rb") as f:
   img = f.read()

output = client.predict({"0": img}, "resnet")
result_index = np.argmax(output[0])
print(imagenet_classes[result_index])' >> predict.py

python predict.py
```
If everything is set up correctly, you will see 'zebra' prediction in the output.

### Next Steps

- To serve your own model, [prepare it for serving](models_repository.md) and proceed to serve [single](single_model_mode.md) or [multiple](multiple_models_mode.md) models.
- To see another example of setting up the model server with a face-detection model, refer to the [Quickstart guide](./ovms_quickstart.md).
- Learn more about model server [starting parameters](parameters.md).

## Build Container from Source <a name="model-server-installation"></a> 

Before starting the server, make sure your hardware is [supported](https://docs.openvino.ai/2022.2/_docs_IE_DG_supported_plugins_Supported_Devices.html) by OpenVINO.

> **NOTE**: OpenVINO Model Server execution on baremetal is tested on Ubuntu 20.04.x. For other operating systems, starting model server in a [docker container](./docker_container.md) is recommended.
   
1. Clone model server git repository.
2. Navigate to the model server directory.
3. Use a precompiled binary or build it in a Docker container.
4. Navigate to the folder containing the binary package and unpack the `tar.gz` file.

Run the following commands to build a model server Docker image:

```bash

git clone https://github.com/openvinotoolkit/model_server.git

cd model_server   
   
# automatically build a container from source
# it places a copy of the binary package in the `dist` subfolder in the Model Server root directory
make docker_build

# unpack the `tar.gz` file
cd dist/ubuntu && tar -xzvf ovms.tar.gz

```
In the `./dist` directory it will generate: 

- image tagged as openvino/model_server:latest - with CPU, NCS, and HDDL support
- image tagged as openvino/model_server:latest-gpu - with CPU, NCS, HDDL, and iGPU support
- image tagged as openvino/model_server:latest-nginx-mtls - with CPU, NCS, and HDDL support and a reference nginx setup of mTLS integration
- release package (.tar.gz, with ovms binary and necessary libraries)

> **NOTE**: Model Server docker image can be created with ubi8-minimal base image or the default ubuntu20. Model Server with the ubi base image does not support NCS and HDDL accelerators.

### Running the Server

The server can be started in two ways:

- using the ```./ovms/bin/ovms --help``` command in the folder, where OVMS was is installed
- in the interactive mode - as a background process or a daemon initiated by ```systemctl/initd``` depending on the Linux distribution and specific hosting requirements


> **NOTE**:
> When [AI accelerators](accelerators.md)are used for inference execution, additional steps may be required to install their drivers and dependencies. 
> Learn more in the [OpenVINO installation guide](https://docs.openvino.ai/2022.2/openvino_docs_install_guides_installing_openvino_linux.html).

### Next Steps

- To serve your own model, [prepare it for serving](models_repository.md) and proceed to serve [single](single_model_mode.md) or [multiple](multiple_models_mode.md) models.
- Learn more about the model server [parameters](parameters.md).

### Additional Resources

- [Configure AI accelerators](accelerators.md)
- [Model server parameters](parameters.md)
- [Quickstart guide](./ovms_quickstart.md)
- [Troubleshooting](troubleshooting.md)