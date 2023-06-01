# Dynamic Shape with dynamic IR/ONNX Model {#ovms_docs_dynamic_shape_dynamic_model}

## Introduction
This guide explains how to leverage OpenVINO dynamic shape feature to work within OVMS. Configure a model to accept dynamic input data shape. Starting with 2022.1 release, it is possible to have dynamic dimensions in model shape natively for models in IR format or ONNX format.

Enable dynamic shape by setting the `shape` parameter to range or undefined:
- `--shape "(1,3,-1,-1)"` when model is supposed to support any value of height and width. Note that any dimension can be dynamic, height and width are only examples here.
- `--shape "(1,3,200:500,200:500)"` when model is supposed to support height and width values in a range of 200-500. Note that any dimension can support range of values, height and width are only examples here.

> Note that some models do not support dynamic dimensions. Learn more about supported model graph layers including all limitations
on [Shape Inference Document](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_ShapeInference.html).

Another option to use dynamic shape feature is to export the model with dynamic dimension using Model Optimizer. OpenVINO Model Server will inherit the dynamic shape and no additional settings are needed.

To the demonstrate dynamic dimensions, take advantage of:

- Example client in Python [face_detection.py](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/demos/face_detection/python/face_detection.py) that can be used to request inference with the desired input shape.

- An example [face_detection_retail_0004](https://docs.openvinotoolkit.org/2021.4/omz_models_model_face_detection_retail_0004.html) model.

When using the `face_detection_retail_0004` model with the `face_detection.py` script, images are reloaded and resized to the desired width and height. Then, the output is processed from the server, and the inference results are displayed with bounding boxes drawn around the predicted faces. 

## Steps
Clone OpenVINO&trade; Model Server GitHub repository and enter `model_server` directory.
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
```
#### Download the Pretrained Model
Download the model files and store them in the `models` directory
```bash
mkdir -p models/face_detection/1
curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.xml -o models/face_detection/1/face-detection-retail-0004.bin -o models/face_detection/1/face-detection-retail-0004.xml
```

#### Pull the Latest Model Server Image
Pull the latest version of OpenVINO&trade; Model Server from Docker Hub:
```bash
docker pull openvino/model_server:latest
```

#### Start the Model Server Container with the Model and Dynamic Batch Size
Start the container using the image pulled in the previous step and mount the `models` directory:
```bash
docker run --rm -d -v $(pwd)/models:/models -p 9000:9000 openvino/model_server:latest --model_name face-detection --model_path /models/face_detection --shape "(1,3,-1,-1)" --port 9000
```

#### Run the Client
```bash
cd demos/face_detection/python
virtualenv .venv
. .venv/bin/activate
pip install -r ../../common/python/requirements.txt
mkdir results_500x500

python face_detection.py --grpc_port 9000 --width 500 --height 500 --input_images_dir ../../common/static/images/people --output_dir results_500x500

mkdir results_600x400

python face_detection.py --grpc_port 9000 --width 600 --height 400 --input_images_dir ../../common/static/images/people --output_dir results_600x400
```
The results from running the client will be saved in the directory specified by `--output_dir`
