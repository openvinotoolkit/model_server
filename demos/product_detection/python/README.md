# Face Detection Demo {#ovms_demo_face_detection}

## Overview

The script product_detection.py runs product detection inference requests for all the images
saved in `input_images_dir` directory. 

The script can adjust the input image size and change the batch size in the request. It demonstrates how to use
the functionality of dynamic shape in OpenVINO Model Server and how to process the output from the server.


Clone the repository and enter face_detection directory
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/product_detection/python
```

Prepare environment:
```bash
pip install -r ../../common/python/requirements.txt
```

```bash
python product_detection.py --help
usage: product_detection.py [-h] [--input_images_dir INPUT_IMAGES_DIR]
                         [--output_dir OUTPUT_DIR] [--batch_size BATCH_SIZE]
                         [--width WIDTH] [--height HEIGHT]
                         [--grpc_address GRPC_ADDRESS] [--grpc_port GRPC_PORT]
                         [--model_name MODEL_NAME] [--tls]
                         [--server_cert SERVER_CERT]
                         [--client_cert CLIENT_CERT] [--client_key CLIENT_KEY]

optional arguments:
  -h, --help            show this help message and exit
  --input_images_dir INPUT_IMAGES_DIR
                        Directory with input images
  --output_dir OUTPUT_DIR
                        Directory for storing images with detection results
  --batch_size BATCH_SIZE
                        How many images should be grouped in one batch
  --width WIDTH         How the input image width should be resized in pixels
  --height HEIGHT       How the input image width should be resized in pixels
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

Place product detection model in ./model/1 dir

Run the client:
```bash
mkdir results

python product_detection.py --batch_size 1 --width 300 --height 300 --grpc_port 9000
