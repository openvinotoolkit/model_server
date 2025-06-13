# Image Classification Demo (Python) {#ovms_demo_image_classification}

## Overview

The script [image_classification.py](https://github.com/openvinotoolkit/model_server/blob/releases/2025/2/demos/image_classification/python/image_classification.py) reads all images and their labels specified in the text file. It then classifies them with [ResNet50](https://github.com/openvinotoolkit/open_model_zoo/blob/releases/2023/1/models/intel/resnet50-binary-0001/README.md) model and presents accuracy results.


## Download ResNet50 model

```bash
mkdir -p model/1
wget -P model/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin
wget -P model/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml
```

## Run OpenVINO Model Server
```bash
docker run -d -v $PWD/model:/models -p 9000:9000 openvino/model_server:latest --model_path /models --model_name resnet --port 9000
```

## Run the client:
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/image_classification/python

python image_classification.py --help
usage: image_classification.py [-h] [--images_list IMAGES_LIST]
                              [--grpc_address GRPC_ADDRESS]
                              [--grpc_port GRPC_PORT]
                              [--input_name INPUT_NAME]
                              [--output_name OUTPUT_NAME]
                              [--model_name MODEL_NAME] [--size SIZE]
                              [--rgb_image RGB_IMAGE]
```

### Arguments

| Argument      | Description |
| :---        |    :----   |
| -h, --help       | Show help message and exit       |
| --images_list   |   Path to a file with a list of labeled images      |
| --grpc_address GRPC_ADDRESS | Specify url to grpc service. Default:localhost |
| --grpc_port GRPC_PORT | Specify port to grpc service. Default: 9000 |
| --input_name | Specify input tensor name. Default: input |
| --output_name | Specify output name. Default: resnet_v1_50/predictions/Reshape_1 |
| --model_name | Define model name, must be same as is in service. Default: resnet|
| --size SIZE  | The size of the image in the model|
| --rgb_image RGB_IMAGE | Convert BGR channels to RGB channels in the input image |

### Usage example

```bash
python image_classification.py --grpc_port 9000 --input_name 0 --output_name 1463 --images_list ../input_images.txt

Start processing:
        Model name: resnet
        Images list file: ../input_images.txt
../../common/static/images/airliner.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 25.08 ms; speed 39.87 fps
         1 airliner 404 ; Correct match.
../../common/static/images/arctic-fox.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 22.97 ms; speed 43.53 fps
         2 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
../../common/static/images/bee.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 24.45 ms; speed 40.90 fps
         3 bee 309 ; Correct match.
../../common/static/images/golden_retriever.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 23.93 ms; speed 41.78 fps
         4 golden retriever 207 ; Correct match.
../../common/static/images/gorilla.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 24.72 ms; speed 40.46 fps
         5 gorilla, Gorilla gorilla 366 ; Correct match.
../../common/static/images/magnetic_compass.jpeg (1, 3, 224, 224) ; data range: 0.0 : 247.0
Processing time: 24.74 ms; speed 40.43 fps
         6 magnetic compass 635 ; Correct match.
../../common/static/images/peacock.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 22.39 ms; speed 44.66 fps
         7 peacock 84 ; Correct match.
../../common/static/images/pelican.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 25.96 ms; speed 38.53 fps
         8 pelican 144 ; Correct match.
../../common/static/images/snail.jpeg (1, 3, 224, 224) ; data range: 0.0 : 248.0
Processing time: 23.68 ms; speed 42.23 fps
         9 snail 113 ; Correct match.
../../common/static/images/zebra.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 23.68 ms; speed 42.24 fps
         10 zebra 340 ; Correct match.
Overall accuracy= 100.0 %
Average latency= 23.5 ms
```




