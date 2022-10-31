# Model Server in Docker Containers {#ovms_docs_docker_container}

OpenVINO™ Model Server includes a C++ implementation of gRPC and RESTful API interfaces defined by TensorFlow Serving. 
In the backend, it uses OpenVINO&trade; Runtime libraries from OpenVINO&trade; toolkit, which speeds up the execution on CPU, and enables it on iGPU and Movidius devices.

OpenVINO Model Server can be hosted on a bare metal server, virtual machine, or inside a docker container. It is also suitable for landing in the Kubernetes environment.

This is a step-by-step guide on how to deploy OpenVINO&trade; Model Server on Linux, using a Docker Container. 

**Before you start, make sure you have:**

- [Docker Engine](https://docs.docker.com/engine/) installed 
- Intel® Core™ processor (6-12th gen.) or Intel® Xeon® processor
- Linux, macOS or Windows via [WSL](https://docs.microsoft.com/en-us/windows/wsl/) 
- (optional) AI accelerators [supported by OpenVINO](https://docs.openvino.ai/2022.2/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html)
> **NOTE**: Accelerators are only tested on bare-metal Linux hosts.


## Launch Model Server Container <a name="quickstart"></a>

1. Pull OpenVINO&trade; Model Server Image.
2. Prepare data for serving:
   - Start a Docker Container with OVMS and your chosen model from cloud storage.
   - Provide the input files.
   - Prepare a client package.
3. Run the prediction using ovmsclient.

Here is an example of launching the model server using a ResNet50 model for image classification:

### Step 1. Pull Model Server Image

Pull an image from Docker: 

```bash
docker pull openvino/model_server:latest
```

or [RedHat Ecosystem Catalog](https://catalog.redhat.com/software/containers/intel/openvino-model-server/607833052937385fc98515de):

```
docker pull registry.connect.redhat.com/intel/openvino-model-server:latest
```

### Step 2. Prepare Data for Serving

```bash
# start the container 
docker run -p 9000:9000 openvino/model_server:latest \ 
--model_name resnet --model_path gs://ovms-public-eu/resnet50-binary \ 
--layout NHWC:NCHW --port 9000 

# download input files, an image, and a label mapping file
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/1/demos/common/static/images/zebra.jpeg
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/1/demos/common/python/classes.py

# Install the Python-based ovmsclient package
pip3 install ovmsclient
```

### Step 3. Run prediction

Run prediction
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

To see another example of setting up the model server with a face-detection model, refer to the [Quickstart guide](./ovms_quickstart.md).

To serve your own model, [prepare it for serving](model_repository.md) and proceed to serve [single](single_model_mode.md) or [multiple](multiple_models_mode.md) models.
