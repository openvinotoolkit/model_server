## Deploying Model Server in Docker Container {#ovms_docs_deploying_server_docker}

This is a step-by-step guide on how to deploy OpenVINO&trade; Model Server on Linux, using Docker.

**Before you start, make sure you have:**

- [Docker Engine](https://docs.docker.com/engine/) installed
- Intel® Core™ processor or Intel® Xeon® processor
- Linux, macOS or Windows via [WSL](https://docs.microsoft.com/en-us/windows/wsl/)
- (optional) AI accelerators [supported by OpenVINO](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes.html). Accelerators are tested only on bare-metal Linux hosts.

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

> **NOTE**: You can also pull public image `openvino/model_server:weekly` with development version of the model server, which is built from the main branch. It allow you to evaluate the latest features ahead of official releases.

##### Optional: Compose a Custom OVMS Image

Use the image composer to build a smaller release image from an existing OVMS image without recompiling the server. By default, the composed image supports CPU inference with OpenVINO IR models. Feature flags retain optional components from the source image and install any required system packages; they cannot add OVMS components that the source image does not contain.

Run the composer from the root of the OVMS source checkout. It requires Python 3, Docker with Buildx, and an OVMS source image based on Ubuntu 22.04, 24.04, or 26.04. Check the source image's Ubuntu version before using a rolling tag such as `latest` or `weekly`.

From the checkout root, build an image with ONNX model loading and OpenCV, then check that the server starts:

```text
python3 docker_compose.py --full-image openvino/model_server:2026.4 \
    --onnx --opencv --output-name ovms-custom:onnx
docker run --rm ovms-custom:onnx --help
```

To serve a model, use the resulting image tag in place of `openvino/model_server:latest` in Step 2. That example uses an IR model; to serve an ONNX model instead, follow the [ONNX demo](../demos/using_onnx_model/python/README.md).

| Flag | Feature |
| --- | --- |
| `--gpu` | Intel GPU inference, including GPU drivers. |
| `--npu` | Intel NPU inference, including the compiler and drivers. |
| `--opencv` | OpenCV shared libraries for server-side OpenCV features. |
| `--python` | Python runtime and Python custom nodes. |
| `--tensorflow` | Direct loading of TensorFlow and TensorFlow Lite models. |
| `--paddle` | Direct loading of PaddlePaddle models. |
| `--onnx` | Direct loading of ONNX models. |
| `--espeak` | eSpeak library and data for Kokoro text-to-speech. |

Combine flags as needed. If the source OVMS executable depends on Python or OpenCV, the composer requires the corresponding flag rather than removing those libraries. GPU and NPU use still requires compatible hardware and [device access for the container](accelerators.md).

The composer pulls the source image by default; use `--skip-pull` for a local image. Use `--dry-run` to generate `Dockerfile.compose` without building it, or run `python3 docker_compose.py --help` for the full option list.

#### Step 2. Prepare Data for Serving

##### 2.1 Start the container with the model

```bash
mkdir -p ${HOME}/models
curl -L https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.bin -o ${HOME}/models/resnet50.bin
curl -L https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.xml -o ${HOME}/models/resnet50.xml
docker run -u $(id -u) -v ${HOME}/models:/models -p 9000:9000 openvino/model_server:latest \
--model_name resnet --model_path /models/resnet50.xml \
--mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" --layout "NHWC:NCHW" --port 9000
```

##### 2.2 Download input files: an image and a label mapping file

```bash
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/common/static/images/zebra.jpeg
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/common/python/classes.py
```

##### 2.3 Install the Python-based client package

```bash
pip3 install tritonclient[grpc] numpy
```


#### Step 3. Run Prediction


```bash
echo 'import numpy as np
from classes import imagenet_classes
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient(url="localhost:9000")

# Get model metadata to discover input/output names
metadata = client.get_model_metadata("resnet")
input_name = metadata.inputs[0].name
output_name = metadata.outputs[0].name

# Load image as binary
with open("zebra.jpeg", "rb") as f:
    img_bytes = f.read()

# Run inference
infer_input = grpcclient.InferInput(input_name, [1], "BYTES")
infer_input.set_data_from_numpy(np.array([img_bytes], dtype=np.object_))
result = client.infer("resnet", [infer_input])
output = result.as_numpy(output_name)
result_index = np.argmax(output[0])
print(imagenet_classes[result_index])' >> predict.py

python predict.py
zebra
```
If everything is set up correctly, you will see 'zebra' prediction in the output.
