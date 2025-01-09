## Deploying Model Server in Docker Container

This is a step-by-step guide on how to deploy OpenVINO&trade; Model Server on Linux, using a pre-build Docker Container.

**Before you start, make sure you have:**

- [Docker Engine](https://docs.docker.com/engine/) installed
- Intel® Core™ processor (6-13th gen.) or Intel® Xeon® processor (1st to 4th gen.)
- Linux, macOS or Windows via [WSL](https://docs.microsoft.com/en-us/windows/wsl/)
- (optional) AI accelerators [supported by OpenVINO](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes.html). Accelerators are tested only on bare-metal Linux hosts.

### Launch Model Server Container

This example shows how to launch the model server with a ResNet50 image classification model from a cloud storage:

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

##### 2.1 Start the container with the model

```bash
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.{xml,bin} -P models/resnet50/1
docker run -u $(id -u) -v $(pwd)/models:/models -p 9000:9000 openvino/model_server:latest \
--model_name resnet --model_path /models/resnet50 \
--layout NHWC:NCHW --port 9000
```

##### 2.2 Download input files: an image and a label mapping file

```bash
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/common/static/images/zebra.jpeg
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/common/python/classes.py
```

##### 2.3 Install the Python-based ovmsclient package

```bash
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
zebra
```
If everything is set up correctly, you will see 'zebra' prediction in the output.