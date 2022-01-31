# Face Detection Demo in OVMS

## Overview

The script [face_detection.py](face_detection.py) run face detection inference requests for all the images
saved in `input_images_dir` directory. 

The script can adjust the input image size and change the batch size in the request. It demonstrates how to use
the functionality of dynamic shape in OpenVINO Model Server and how to process the output from the server.

The example relies on the model [face-detection-retail-0004](https://docs.openvinotoolkit.org/2021.4/omz_models_model_face_detection_retail_0004.html).

```bash
python face_detection.py --help
usage: face_detection.py [-h] [--input_images_dir INPUT_IMAGES_DIR]
                         [--output_dir OUTPUT_DIR] [--batch_size BATCH_SIZE]
                         [--width WIDTH] [--height HEIGHT]
                         [--grpc_address GRPC_ADDRESS] [--grpc_port GRPC_PORT]
                         [--model_name MODEL_NAME] [--tls]
                         [--server_cert SERVER_CERT]
                         [--client_cert CLIENT_CERT] [--client_key CLIENT_KEY]

Demo for face detection requests via TFS gRPC API.analyses input images and
saves with with detected faces.it relies on model face_detection...

Arguments:
  -h, --help            show this help message and exit
  --input_images_dir INPUT_IMAGES_DIR
                        Directory with input images
  --output_dir OUTPUT_DIR
                        Directory for storing images with detection results
  --batch_size BATCH_SIZE
                        how many images should be grouped in one batch
  --width WIDTH         how the input image width should be resized in pixels
  --height HEIGHT       how the input height should be resized in pixels
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT
                        Specify port to grpc service. default: 9000
  --model_name MODEL_NAME
                        Specify the model name
  --tls                 use TLS communication with gRPC endpoint
  --server_cert SERVER_CERT
                        Path to server certificate
  --client_cert CLIENT_CERT
                        Path to client certificate
  --client_key CLIENT_KEY
                        Path to client key
```

## Usage example

Start the OVMS service locally:

```bash
mkdir -p model/1
wget -P model/1 https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.bin
wget -P model/1 https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.xml
docker run -d -u $(id -u):$(id -g) -v `pwd`/model:/models -p 9000:9000 openvino/model_server:latest --model_path /models --model_name face-detection --port 9000  --shape auto --log_level DEBUG
```

Run the client:
```bash
pip install -r ../../common/python/requirements.txt
mkdir results

python face_detection.py --batch_size 1 --width 300 --height 300

python face_detection.py --batch_size 4 --width 600 --height 400 --input_images_dir ../../common/static/images/people --output_dir results
```

The scipt will visualize the inference results on the images saved in the directory `output_dir`. Saved images have the
following naming convention:

<#iteration>_<#image_in_batch>.jpeg



