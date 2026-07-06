# Dynamic Shape with Binary Inputs{#ovms_docs_dynamic_shape_binary_inputs}

## Introduction
This guide shows how to use the binary inputs feature to send data in binary format. This means you can load just a JPEG or PNG image and run inference on it without any data preprocessing.

To run inference on binary encoded data, make sure your model accepts the NHWC layout. When preparing the request, we need to let the Model Server know that the data is in binary format.

Learn more about the [binary inputs](binary_input.md) feature.

## Steps

#### Download the Pretrained Model
Download the model files and store them in the `models` directory
```bash
mkdir -p models/resnet/1
curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml -o models/resnet/1/resnet50-binary-0001.bin -o models/resnet/1/resnet50-binary-0001.xml
```

#### Pull the Latest Model Server Image from Docker Hub
Pull the latest version of OpenVINO&trade; Model Server from Docker Hub :
```bash
docker pull openvino/model_server:latest
```

#### Start the Container with Downloaded Model
Start the container with the image pulled in the previous step and mount the `models` directory :
```bash
docker run --rm -d -v $(pwd)/models:/models -p 9000:9000 openvino/model_server:latest --model_name resnet --model_path /models/resnet --layout NHWC:NCHW --port 9000
```

### Download Client Package

```bash
pip3 install tritonclient[grpc] numpy opencv-python-headless
```

### Download a Sample Image and Label Mappings
```bash
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/common/static/images/zebra.jpeg

wget https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/common/python/classes.py
```

### Run Inference

```bash
echo '
import numpy as np
import cv2
from classes import imagenet_classes
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient(url="localhost:9000")
metadata = client.get_model_metadata("resnet")
input_name = metadata.inputs[0].name
output_name = metadata.outputs[0].name

img = cv2.imread("zebra.jpeg")
img = cv2.resize(img, (224, 224))
img = img.astype(np.float32)
img = img.reshape(1, img.shape[0], img.shape[1], 3)

infer_input = grpcclient.InferInput(input_name, img.shape, "FP32")
infer_input.set_data_from_numpy(img)
result = client.infer("resnet", [infer_input])
output = result.as_numpy(output_name)
result_index = np.argmax(output[0])
print(imagenet_classes[result_index])' >> predict.py

python predict.py
zebra
```