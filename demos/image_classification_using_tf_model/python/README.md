# Model Server demo with a direct import of TensorFlow model {#ovms_demo_tf_classification}

This guide demonstrates how to run inference requests for TensorFlow model with OpenVINO Model Server.
As an example, we will use [InceptionResNetV2](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz) to perform classification of an image.

## Prerequisites

- [Docker](https://docs.docker.com/engine/install/) installed

- Python 3.7 or newer installed

## Preparing to Run

Clone the repository and enter image_classification_using_tf_model directory

```console
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/image_classification_using_tf_model/python
```

## Download the InceptionResNetV2 model

```console
curl --create-dirs -o model/1/inception_resnet_v2_2018_04_27.tgz https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz
tar xzf model/1/inception_resnet_v2_2018_04_27.tgz -C model/1
```

## Run Openvino Model Server

```bash
chmod -R 755 model
docker run -d -v $PWD/model:/models -p 9000:9000 openvino/model_server:latest --model_path /models --model_name resnet --port 9000
```

Alternatively see (instructions)[https://github.com/openvinotoolkit/model_server/blob/releases/2025/2/docs/deploying_server_baremetal.md] for deployment on bare metal.

Make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

on every shell that will start OpenVINO Model Server.

And start Model Server using the following command:
```bat
ovms --model_name resnet --model_path model/ --port 9000
```

## Run the client

Install python dependencies:
```console
pip3 install -r requirements.txt
``` 

Now you can run the client:
```console
python image_classification_using_tf_model.py --help
usage: image_classification_using_tf_model.py [-h] [--grpc_address GRPC_ADDRESS] [--grpc_port GRPC_PORT] --image_input_path IMAGE_INPUT_PATH

Client for OCR pipeline

optional arguments:
  -h, --help            show this help message and exit
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT
                        Specify port to grpc service. default: 9000
  --image_input_path IMAGE_INPUT_PATH
                        Image input path
```

Exemplary result of running the demo:
```console
python image_classification_using_tf_model.py --grpc_port 9000 --image_input_path ../../common/static/images/zebra.jpeg
Image classified as zebra
```
