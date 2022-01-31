# Image Classification Demo with OVMS

## Overview

The script [image_classification.py](image_classification.py) reads all images and their labels specified in the text file. It then classifies them with [ResNet50](https://docs.openvino.ai/latest/omz_models_model_resnet50_binary_0001.html) model and presents accuracy results.


## Download ResNet50 model

```bash
mkdir -p model/1
wget -P model/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin
wget -P model/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml
```

## Run OpenVINO Model Server
```bash
docker run -d -v `pwd`/model:/models -p 9000:9000 openvino/model_server:latest --model_path /models --model_name resnet --port 9000
```

## Run the client:
```bash
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
python image_classification.py --grpc_port 9000 --input_name 0 --output_name prob --images_list ../input_images.txt

Start processing:
	Model name: resnet
	Images list file: input_images.txt
images/airliner.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 23.40 ms; speed 42.73 fps
	 1 airliner 404 ; Correct match.
images/arctic-fox.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 21.09 ms; speed 47.42 fps
	 2 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
images/bee.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 21.04 ms; speed 47.52 fps
	 3 bee 309 ; Correct match.
images/golden_retriever.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 20.03 ms; speed 49.92 fps
	 4 golden retriever 207 ; Correct match.
images/gorilla.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 21.95 ms; speed 45.57 fps
	 5 gorilla, Gorilla gorilla 366 ; Correct match.
images/magnetic_compass.jpeg (1, 3, 224, 224) ; data range: 0.0 : 247.0
Processing time: 21.51 ms; speed 46.48 fps
	 6 magnetic compass 635 ; Correct match.
images/peacock.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 20.81 ms; speed 48.05 fps
	 7 peacock 84 ; Correct match.
images/pelican.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 21.90 ms; speed 45.66 fps
	 8 pelican 144 ; Correct match.
images/snail.jpeg (1, 3, 224, 224) ; data range: 0.0 : 248.0
Processing time: 22.38 ms; speed 44.68 fps
	 9 snail 113 ; Correct match.
images/zebra.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 22.22 ms; speed 45.00 fps
	 10 zebra 340 ; Correct match.
Overall accuracy= 100.0 %
Average latency= 21.2 ms
```




