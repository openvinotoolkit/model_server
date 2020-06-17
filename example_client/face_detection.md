# Exemplary client for face detection model with configurable input shape and batch size

## Overview

The script [face_detection.py](face_detection.py) run face detection inference requests for all the images
saved in `image_input_dir` directory. 

The script can adjust the input image size and change the batch size in the request. It demonstrate how to use
the functionality of dynamic shape in OpenVINO Model Server and how to process the output from the server.

The example relies on the model [face_detection_0004](https://docs.openvinotoolkit.org/2018_R5/_docs_Retail_object_detection_face_sqnet10modif_ssd_0004_caffe_desc_face_detection_retail_0004.html).

```bash
python face_detection.py --help
usage: face_detection.py [-h] [--input_images_dir INPUT_IMAGES_DIR]
                         [--output_dir OUTPUT_DIR] [--batch_size BATCH_SIZE]
                         [--width WIDTH] [--height HEIGHT]
                         [--grpc_address GRPC_ADDRESS] [--grpc_port GRPC_PORT]

Demo for face detection requests via TFS gRPC API.analyses input images and
saves with with detected faces.it relies on model face_detection...

optional arguments:
  -h, --help            show this help message and exit
  --input_images_dir INPUT_IMAGES_DIR
                        Directory with input images
  --output_dir OUTPUT_DIR
                        Directory for staring images with detection results
  --batch_size BATCH_SIZE
                        how many images should be grouped in one batch
  --width WIDTH         how the input image width should be resized in pixels
  --height HEIGHT       how the input height should be resized in pixels
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT
                        Specify port to grpc service. default: 9000
```

## Usage example

Start the OVMS service locally:

```bash
mkdir -p model/1
wget -P model/1 https://download.01.org/opencv/2019/open_model_zoo/R2/20190628_180000_models_bin/face-detection-retail-0004/FP32/face-detection-retail-0004.bin
wget -P model/1 https://download.01.org/opencv/2019/open_model_zoo/R2/20190628_180000_models_bin/face-detection-retail-0004/FP32/face-detection-retail-0004.xml
docker run -d -v `pwd`/model:/models -e LOG_LEVEL=DEBUG -p 9000:9000 ie-serving-py:latest \
/ie-serving-py/start_server.sh ie_serving model --model_path /models --model_name face-detection --port 9000  --shape auto
```

Run the client:
```bash
cd example_client
virtualenv .venv
. .venv/bin/activate
pip install -r client_requirements.txt
mkdir results

python face_detection.py --batch_size 1 --width 300 --height 300

python face_detection.py --batch_size 4 --width 600 --height 400 --input_images_dir images/people --output_dir results
```

The scipt will visualize the inference results on the images saved in the directory `output_dir`. Saved images have the
following naming convention:

<#iteration>_<#image_in_batch>.jpeg



