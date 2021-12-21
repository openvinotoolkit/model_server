# Dynamic shape with automatic model reloading{#ovms_docs_dynamic_shape_auto_reload}

## Introduction
This document guides how to configure model to accept input data in different shapes. In this case it's done by reloading the model with new shape every time it receives the request with shape different than the one currently set. 

The enabling of dynamic shape via model reload is as simple as setting `shape` parameter to `auto`. To show how to configure dynamic batch size and make use of it let's take adventage of:

- Example client in python [face_detection.py](https://github.com/openvinotoolkit/model_server/blob/main/example_client/face_detection.py), that can be used to request inference on desired input shape.

- The example model [face_detection_retail_0004](https://docs.openvinotoolkit.org/2021.4/omz_models_model_face_detection_retail_0004.html).

- While using face_detection_retail_0004 model with face_detection.py the script loads images and resizes them to desired width and height. Then it processes the output from the server and displays the inference results by drawing bounding boxes around predicted faces. 

## Steps
Clone OpenVINO&trade; Model Server github repository and enter `model_server` directory.
```
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
```
#### Download the pretrained model
Download model files and store it in `models` directory
```Bash
mkdir -p models/face_detection/1
curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.xml -o models/face_detection/1/face-detection-retail-0004.bin -o models/face_detection/1/face-detection-retail-0004.xml
```

#### Pull the latest OVMS image from dockerhub
Pull the latest version of OpenVINO&trade; Model Server from Dockerhub :
```Bash
docker pull openvino/model_server:latest
```

#### Start ovms docker container with downloaded model and dynamic batch size
Start ovms container with image pulled in previous step and mount `models` directory :
```Bash
docker run --rm -d -v $(pwd)/models:/models -p 9000:9000 openvino/model_server:latest --model_name face_detection --model_path /models/face_detection --shape auto --port 9000
```

#### Run the client
```Bash
cd example_client
virtualenv .venv
. .venv/bin/activate
pip install -r client_requirements.txt
mkdir results_500x500 results_600x400

python face_detection.py --width 500 --height 500 --input_images_dir images/people --output_dir results_500x500

python face_detection.py --width 600 --height 400 --input_images_dir images/people --output_dir results_600x400
```
Results of running the client will be available in directories specified in `--output_dir`


> Note that reloading the model takes time and during the reload new requests get queued up. Therefore, frequent model reloading may negatively affect overall performance. 