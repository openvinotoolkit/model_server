# Prediction with ONNX Models {#ovms_docs_demo_onnx}

Similar to the steps in the [quick start](ovms_quickstart.md) guide using an OpenVINO IR model format. Model Server accepts ONNX models with the same versioning structure. Similar to IR, place each ONNX model file in a separate model version subdirectory.
Below is a complete functional use case using python 3.6 or higher.

Download the model:
```bash
curl -L --create-dir https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx -o resnet/1/resnet50-caffe2-v1-9.onnx
```
```bash
tree resnet/
resnet/
└── 1
    └── resnet50-caffe2-v1-9.onnx
```
Note that the downloaded model requires an additional [preprocessing function](https://github.com/onnx/models/tree/master/vision/classification/resnet#preprocessing). Preprocessing can be performed in the client by manipulating data before sending the request. Preprocessing can be also delegated to the server by creating a DAG, and using a custom processing node. Both methods will be explained below.

<a href="#client-side">Option 1: Adding preprocessing to the client side</a>  
<a href="#server-side">Option 2: Adding preprocessing to the server side (building DAG)</a>

Get an image to classify:
```
wget -q https://github.com/openvinotoolkit/model_server/raw/main/example_client/images/bee.jpeg
```
Install python libraries:
```bash
pip install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/main/example_client/client_requirements.txt
```
Get the list of imagenet classes:
```bash
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/main/example_client/classes.py
```

## Option 1: Adding preprocessing to the client side <a name="client-side"></a>

Start the OVMS container with single model instance:
```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd)/resnet:/model -p 9001:9001 openvino/model_server:latest \
--model_path /model --model_name resnet --port 9001
```

Run inference request with a client containing `preprocess` function presented below:
```python
import numpy as np
import cv2
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow import make_tensor_proto, make_ndarray
import classes

def preprocess(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data

def getJpeg(path, size):
    with open(path, mode='rb') as file:
        content = file.read()

    img = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # BGR format
    # format of data is HWC
    # add image preprocessing if needed by the model
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32')
    #convert to NCHW
    img = img.transpose(2,0,1)
    # normalize to adjust to model training dataset
    img = preprocess(img)
    img = img.reshape(1,3,size,size)
    print(path, img.shape, "; data range:",np.amin(img),":",np.amax(img))
    return img

img1 = getJpeg('bee.jpeg', 224)

channel = grpc.insecure_channel("localhost:9001")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = "resnet"
request.inputs["gpu_0/data_0"].CopyFrom(make_tensor_proto(img1, shape=(img1.shape)))
result = stub.Predict(request, 10.0) # result includes a dictionary with all model outputs

output = make_ndarray(result.outputs["gpu_0/softmax_1"])
max = np.argmax(output)
print("Class is with highest score: {}".format(max))
print("Detected class name: {}".format(classes.imagenet_classes[max]))
```

It shows the following output:
```bash
bee.jpeg (1, 3, 224, 224) ; data range: -2.117904 : 2.64
Class is with highest score: 309
Detected class name: bee
```


## Option 2: Adding preprocessing to the server side (building a DAG) <a name="server-side"></a>

Create a configuration file with DAG containing two sequential nodes: one being the _image transformation node_ and one _DL model node_ resnet. The job of the image transformation node will be to preprocess the image data to match format required by the ONNX model `resnet50-caffe2-v1-9.onnx`.

The example [configuration file](https://github.com/openvinotoolkit/model_server/blob/main/src/custom_nodes/image_transformation/config_with_preprocessing_node.json) is available in _image transformation_ custom node directory.  
Image transformation custom node library building steps can be found [here](https://github.com/openvinotoolkit/model_server/tree/main/src/custom_nodes/image_transformation).

Prepare workspace with the model, preprocessing node library and configuration file.
```
$ tree workspace

workspace
├── config_with_preprocessing_node.json
├── lib
│   └── libcustom_node_image_transformation.so
└── models
    └── resnet50-caffe2-v1
        └── 1
            └── resnet50-caffe2-v1-9.onnx
```

Start the OVMS container with a configuration file option:
```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd)/workspace:/workspace -p 9001:9001 openvino/model_server:latest \
--config_path /workspace/config_with_preprocessing_node.json --port 9001
```

Use a sample client to send JPEG images to the server:
```
$ cd model_server/example_client

$ python3 grpc_binary_client.py --grpc_port 9001 --input_name 0 --output_name 1463 --model_name resnet --batchsize 1
```
Below is the client output including performance and accuracy results:
```
Start processing:
        Model name: resnet
        Images list file: input_images.txt
Batch: 0; Processing time: 21.52 ms; speed 46.47 fps
         1 airliner 404 ; Correct match.
Batch: 1; Processing time: 17.50 ms; speed 57.14 fps
         2 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Batch: 2; Processing time: 14.91 ms; speed 67.05 fps
         3 bee 309 ; Correct match.
Batch: 3; Processing time: 11.66 ms; speed 85.76 fps
         4 golden retriever 207 ; Correct match.
Batch: 4; Processing time: 12.88 ms; speed 77.66 fps
         5 gorilla, Gorilla gorilla 366 ; Correct match.
Batch: 5; Processing time: 14.34 ms; speed 69.74 fps
         6 magnetic compass 635 ; Correct match.
Batch: 6; Processing time: 13.14 ms; speed 76.12 fps
         7 peacock 84 ; Correct match.
Batch: 7; Processing time: 14.48 ms; speed 69.04 fps
         8 pelican 144 ; Correct match.
Batch: 8; Processing time: 12.71 ms; speed 78.71 fps
         9 snail 113 ; Correct match.
Batch: 9; Processing time: 17.23 ms; speed 58.03 fps
         10 zebra 340 ; Correct match.
Overall accuracy= 100.0 %
Average latency= 14.5 ms
```

## Node parameters explanation
Additional preprocessing step applies a division and an subtraction to each pixel value in the image. This calculation is configured by passing two parameters to _image transformation_ custom node:
```
"params": {
  ...
  "mean_values": "[123.675,116.28,103.53]",
  "scale_values": "[58.395,57.12,57.375]",
  ...
}
```
For each pixel, the custom node subtracts `123.675` from blue value, `116.28` from green value and `103.53` from red value. Next, it divides in the same color order using `58.395`, `57.12`, `57.375` values. This way we match the image data to the input required by the ONNX model.
