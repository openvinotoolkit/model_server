## Overview

This guide demonstrates how to run inference requests for PaddlePaddle model with OpenVINO Model Server.
As an example, we will use [ocrnet-hrnet-w48-paddle](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ocrnet-hrnet-w48-paddle) to perform segmentation on an image.

## Prerequisites

- [Docker](https://docs.docker.com/engine/install/) installed

- Python 3.6 or newer installed

## Preparing to Run

Clone the repository and enter segmentation_using_paddlepaddle_model directory

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/segmentation_using_paddlepaddle_model/python
```

Install [paddlepaddle](https://pypi.org/project/paddlepaddle/) and other packages required to export the model to deployable format
```bash
pip3 install paddlepaddle
pip3 install pyyaml
pip3 install scipy
pip3 install opencv-python
pip3 install requests
```

You can prepare the workspace by just running

```bash
make
```

## Deploying OVMS

Deploy OVMS with vehicles analysis pipeline using the following command:

```bash
docker run -p 9000:9000 -d -v ${PWD}/model:/models openvino/model_server --port 9000 --model_path /models --model_name ocrnet
```
## Requesting the Service

Install python dependencies:
```bash
pip3 install -r requirements.txt
``` 

Now you can run the client:
```bash
python3 segmentation_using_paddlepaddle_model.py --grpc_port 9000 --image_input_path ../../common/static/images/cars/road1.jpg --image_output_path ./road2.jpg
```
Exemplary result of running the demo:

![Road](road2.jpg)