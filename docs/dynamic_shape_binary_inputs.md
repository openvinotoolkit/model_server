# Dynamic shape with binary inputs{#ovms_docs_dynamic_shape_binary_inputs}

## Introduction
This document guides how to use binary inputs feature to send data in binary format. This way you can just load an JPEG/PNG image and run inference on it without preprocessing.

To run inference on binary encoded data make sure your model accepts NHWC layout and prepare your request in a certain way to let model server know that data in the request is binary.

Learn more about [binary inputs feature](binary_input.md).

## Steps
Clone OpenVINO&trade; Model Server github repository and enter `model_server` directory.
#### Download the pretrained model
Download model files and store it in `models` directory
```Bash
mkdir -p models/resnet/1
curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml -o models/resnet/1/resnet50-binary-0001.bin -o models/resnet/1/resnet50-binary-0001.xml
```

#### Pull the latest OVMS image from dockerhub
Pull the latest version of OpenVINO&trade; Model Server from Dockerhub :
```Bash
docker pull openvino/model_server:latest
```

#### Start ovms docker container with downloaded model
Start ovms container with image pulled in previous step and mount `models` directory :
```Bash
docker run --rm -d -v $(pwd)/models:/models -v $(pwd)/config.json:/config.json -p 9000:9000 openvino/model_server:latest --model_name resnet --model_path /models/resnet --layout NHWC --port 9000
```

### Download ovmsclient package

``` 
pip3 install ovmsclient 
```

### Download sample image and file with label mapping
```
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/main/example_client/images/zebra.jpeg 

wget https://raw.githubusercontent.com/openvinotoolkit/model_server/main/example_client/classes.py 
```

### Run inference and see the result

```
import numpy as np 
from classes import imagenet_classes 
from ovmsclient import make_grpc_client 
client = make_grpc_client("localhost:9000") 

with open("zebra.jpeg", "rb") as f: 
    img = f.read() 

output = client.predict(inputs={ "0": img}, model_name= "resnet") 
result_index = np.argmax(output[0]) 
print(imagenet_classes[result_index]) 
```