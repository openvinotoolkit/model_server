# Prediction Example with an ONNX Model

Steps are similar to when you work with IR model format. Model Server accepts ONNX models as well with no differences in versioning. Locate ONNX model file in separate model version directory.
Below is a complete functional use case using Python 3.6 or higher.

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
Note that the downloaded model requires additional [preprocessing function](https://github.com/onnx/models/tree/master/vision/classification/resnet#preprocessing). Preprocessing can be performed in the client by manipulating data before sending the request. Preprocessing can be also delegated to the server by creating a DAG and using a custom processing node. Both methods will be explained below.

<a href="#client-side">Option 1: Adding preprocessing to the client side</a>  
<a href="#server-side">Option 2: Adding preprocessing to the server side (building DAG)</a>

## Option 1: Adding preprocessing to the client side <a name="client-side"></a>

Prepare workspace with the model by running: 
```make client_preprocessing```

You should see `workspace` directory created with the following content:
```
workspace/
└── resnet50-onnx
    └── 1
        └── resnet50-caffe2-v1-9.onnx

```

Start the OVMS container with single model instance:
```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd)/workspace:/workspace -p 9001:9001 openvino/model_server:latest \
--model_path /workspace/resnet50-onnx --model_name resnet --port 9001
```

Install python client dependencies:
```bash
pip3 install -r requirements.txt
```

Run the client with preprocessing:
```bash
python3 onnx_model_demo.py --run_preprocessing
Running with preprocessing on client side
../../common/static/images/bee.jpeg (1, 3, 224, 224) ; data range: -2.117904 : 2.64
Class is with highest score: 309
Detected class name: bee
```

## Option 2: Adding preprocessing to the server side (building a DAG) <a name="server-side"></a>

Prepare workspace with the model, preprocessing node library and configuration file by running:
```make server_preprocessing```

You should see `workspace` directory created with the following content:
```
workspace/
├── config.json
├── lib
│   └── libcustom_node_image_transformation.so
└── resnet50-onnx
    └── 1
        └── resnet50-caffe2-v1-9.onnx
```

Start the OVMS container with a configuration file option:
```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd)/workspace:/workspace -p 9001:9001 openvino/model_server:latest \
--config_path /workspace/config.json --port 9001
```

Run the client without preprocessing:
```bash
python3 onnx_model_demo.py
Running without preprocessing on client side
Class is with highest score: 309
Detected class name: bee
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
For each pixel, the custom node subtractes `123.675` from blue value, `116.28` from green value and `103.53` from red value. Next, it divides in the same color order using `58.395`, `57.12`, `57.375` values. This way we match the image data to the input required by onnx model.
