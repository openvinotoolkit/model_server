# OpenVINO&trade; Model Server Client Library Samples

This document contains examples to run *GetModelStatus*, *GetModelMetadata*, *Predict* functions over gRPC API.
Samples are based on [ovmsclient](https://pypi.org/project/ovmsclient/) package. 

It covers following topics:
- <a href="#grpc-api">gRPC API Example Clients </a>
  - <a href="#grpc-model-status">grpc_get_model_status.py</a>
  - <a href="#grpc-model-metadata">grpc_get_model_metadata.py</a>
  - <a href="#grpc-predict-numeric">grpc_predict_resnet.py</a>
  - <a href="#grpc-predict-binary">grpc_predict_binary_resnet.py</a>
  - <a href="#grpc-detect-vehicle">grpc_predict_binary_vehicle_detection.py</a>
- <a href="#http-api">HTTP API Example Clients </a>
  - <a href="#http-model-status">http_get_model_status.py</a>
  - <a href="#http-model-metadata">http_get_model_metadata.py</a>
  - <a href="#http-predict-numeric">http_predict_resnet.py</a>
  - <a href="#http-predict-binary">http_predict_binary_resnet.py</a>
  - <a href="#http-detect-vehicle">http_predict_binary_vehicle_detection.py</a>

## Requirement

Clone the repository and enter directory:

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/client/python/ovmsclient/samples
```

Install samples dependencies:
```bash
pip3 install -r requirements.txt
```

Download [Resnet50-tf Model](https://docs.openvino.ai/2022.2/omz_models_model_resnet_50_tf.html) and convert it into Intermediate Representation format:
```bash
mkdir models
docker run -u $(id -u):$(id -g) -v ${PWD}/models:/models openvino/ubuntu20_dev:latest omz_downloader --name resnet-50-tf --output_dir /models
docker run -u $(id -u):$(id -g) -v ${PWD}/models:/models:rw openvino/ubuntu20_dev:latest omz_converter --name resnet-50-tf --download_dir /models --output_dir /models --precisions FP32
mv ${PWD}/models/public/resnet-50-tf/FP32 ${PWD}/models/public/resnet-50-tf/1
```

OVMS can be started using a command:
```bash
docker run -d --rm -v ${PWD}/models/public/resnet-50-tf:/models/public/resnet-50-tf -p 8000:8000 -p 9000:9000 openvino/model_server:latest --model_name resnet --model_path /models/public/resnet-50-tf --port 9000 --rest_port 8000
```


## gRPC Client Examples <a name="grpc-api"></a>

### Model Status <a name="grpc-model-status">

#### **Get information about the status of served models over gRPC interface:**

- Command

```bash
python grpc_get_model_status.py -h
usage: grpc_get_model_status.py [-h] [--service_url SERVICE_URL]
                                [--model_name MODEL_NAME]
                                [--model_version MODEL_VERSION]
                                [--timeout TIMEOUT]


optional arguments:
  -h, --help            show this help message and exit
  --service_url SERVICE_URL
                        Specify url to grpc service. default:localhost:9000
  --model_name MODEL_NAME
                        Model name to query. default: resnet
  --model_version MODEL_VERSION
                        Model version to query. Lists all versions if omitted
  --timeout TIMEOUT     Request timeout. default: 10.0
```

- Usage Example

```bash
python grpc_get_model_status.py --model_name resnet --service_url localhost:9000
{1: {'state': 'AVAILABLE', 'error_code': 0, 'error_message': 'OK'}}
```


### Model Metadata <a name="grpc-model-metadata">

#### **Get information about the status of served models over gRPC interface:**

- Command

```bash
python grpc_get_model_metadata.py --help
usage: grpc_get_model_metadata.py [-h] [--service_url SERVICE_URL]
                                  [--model_name MODEL_NAME]
                                  [--model_version MODEL_VERSION]
                                  [--timeout TIMEOUT]

Get information about the status of served models over gRPC interface

optional arguments:
  -h, --help            show this help message and exit
  --service_url SERVICE_URL
                        Specify url to grpc service. default:localhost:9000
  --model_name MODEL_NAME
                        Model name to query. default: resnet
  --model_version MODEL_VERSION
                        Model version to query. If omitted or set to 0 returns
                        result for latest version
  --timeout TIMEOUT     Request timeout. default: 10.0
```
- Usage Example

```bash
python grpc_get_model_metadata.py --model_name resnet --model_version 1 --service_url localhost:9000
{'model_version': 1, 'inputs': {'map/TensorArrayStack/TensorArrayGatherV3': {'shape': [1, 224, 224, 3], 'dtype': 'DT_FLOAT'}}, 'outputs': {'softmax_tensor': {'shape': [1, 1001], 'dtype': 'DT_FLOAT'}}}
```


### Predict numeric format <a name="grpc-predict-numeric">

#### **Make prediction using images in numerical format:**

- Command

```bash
python grpc_predict_resnet.py --help
usage: grpc_predict_resnet.py [-h] --images_numpy IMAGES_NUMPY
                              [--service_url SERVICE_URL]
                              [--model_name MODEL_NAME]
                              [--model_version MODEL_VERSION]
                              [--iterations ITERATIONS] [--timeout TIMEOUT]

Make prediction using images in numerical format

optional arguments:
  -h, --help            show this help message and exit
  --images_numpy IMAGES_NUMPY
                        Path to a .npy file with data to infer
  --service_url SERVICE_URL
                        Specify url to grpc service. default:localhost:9000
  --model_name MODEL_NAME
                        Model name to query. default: resnet
  --model_version MODEL_VERSION
                        Model version to query. default: latest available
  --iterations ITERATIONS
                        Total number of requests to be sent. default: 0 - all
                        elements in numpy
  --timeout TIMEOUT     Request timeout. default: 10.0

```
- Usage example

```bash
python grpc_predict_resnet.py --images_numpy ../../imgs_nhwc.npy --model_name resnet --service_url localhost:9000
Image #0 has been classified as airliner
Image #1 has been classified as Arctic fox, white fox, Alopex lagopus
Image #2 has been classified as bee
Image #3 has been classified as golden retriever
Image #4 has been classified as gorilla, Gorilla gorilla
Image #5 has been classified as magnetic compass
Image #6 has been classified as peacock
Image #7 has been classified as pelican
Image #8 has been classified as snail
Image #9 has been classified as zebra
```

### Predict binary format<a name="grpc-predict-binary">

#### **Make prediction using images in binary format:**

- Command

```bash
python grpc_predict_binary_resnet.py --help
usage: grpc_predict_binary_resnet.py [-h] --images_dir IMAGES_DIR
                              [--service_url SERVICE_URL]
                              [--model_name MODEL_NAME]
                              [--model_version MODEL_VERSION]
                              [--timeout TIMEOUT]

Make prediction using images in binary format

optional arguments:
  -h, --help            show this help message and exit
  --images_dir IMAGES_DIR
                        Path to a directory with images in JPG or PNG format
  --service_url SERVICE_URL
                        Specify url to grpc service. default:localhost:9000
  --model_name MODEL_NAME
                        Model name to query. default: resnet
  --model_version MODEL_VERSION
                        Model version to query. default: latest available
  --timeout TIMEOUT     Request timeout. default: 10.0
```
- Usage example

```bash
python grpc_predict_binary_resnet.py --images_dir ../../../../demos/common/static/images --model_name resnet --service_url localhost:9000
Image ../../../../demos/common/static/images/magnetic_compass.jpeg has been classified as magnetic compass
Image ../../../../demos/common/static/images/pelican.jpeg has been classified as pelican
Image ../../../../demos/common/static/images/gorilla.jpeg has been classified as gorilla, Gorilla gorilla
Image ../../../../demos/common/static/images/snail.jpeg has been classified as snail
Image ../../../../demos/common/static/images/zebra.jpeg has been classified as zebra
Image ../../../../demos/common/static/images/arctic-fox.jpeg has been classified as Arctic fox, white fox, Alopex lagopus
Image ../../../../demos/common/static/images/bee.jpeg has been classified as bee
Image ../../../../demos/common/static/images/peacock.jpeg has been classified as peacock
Image ../../../../demos/common/static/images/airliner.jpeg has been classified as warplane, military plane
Image ../../../../demos/common/static/images/golden_retriever.jpeg has been classified as golden retriever
```

## Prepare the model from OpenVINO Model Zoo

### Vehicle detection model
```bash
mkdir -p models
curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/vehicle-detection-0202/FP32/vehicle-detection-0202.xml -o ${PWD}/models/vehicle-detection/1/vehicle-detection-0202.xml
curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/vehicle-detection-0202/FP32/vehicle-detection-0202.bin -o ${PWD}/models/vehicle-detection/1/vehicle-detection-0202.bin
chmod -R 755 ${PWD}/models/vehicle-detection
```
OVMS container can be started using a command:
```bash
docker run -d --rm -v ${PWD}/models/vehicle-detection:/models/vehicle-detection -p 8000:8000 -p 9000:9000 openvino/model_server:latest --model_name vehicle-detection --model_path /models/vehicle-detection --port 9000 --rest_port 8000 --layout NHWC:NCHW
```


### Detect vehicle  <a name="grpc-detect-vehicle">


#### **Make vehicle detection prediction using images in binary format:**

- Command

```bash
python grpc_predict_binary_vehicle_detection.py --help
usage: grpc_predict_binary_vehicle_detection.py [-h] --images_dir IMAGES_DIR
                                           [--service_url SERVICE_URL]
                                           [--model_name MODEL_NAME]
                                           [--model_version MODEL_VERSION]
                                           --output_dir OUTPUT_DIR
                                           [--timeout TIMEOUT]

Make vehicle detection prediction using images in binary format

optional arguments:
  -h, --help            show this help message and exit
  --images_dir IMAGES_DIR
                        Path to a directory with images in JPG or PNG format
  --service_url SERVICE_URL
                        Specify url to grpc service. default:localhost:9000
  --model_name MODEL_NAME
                        Model name to query. default: vehicle-detection
  --model_version MODEL_VERSION
                        Model version to query. default: latest available
  --output_dir OUTPUT_DIR
                        Path to store output.
  --timeout TIMEOUT     Request timeout. default: 10.0
```

- Usage example

```bash
python grpc_predict_binary_vehicle_detection.py --images_dir ../../../../demos/common/static/images/cars/ --output_dir ./output --service_url localhost:9000
Making directory for output: ./output
Detection results in file:  ./output/road1.jpg
```

## HTTP Client Examples <a name="http-api"></a>

OVMS can be started using a command:
```bash
docker run -d --rm -v ${PWD}/models/public/resnet-50-tf:/models/public/resnet-50-tf -p 8000:8000 -p 9000:9000 openvino/model_server:latest --model_name resnet --model_path /models/public/resnet-50-tf --port 9000 --rest_port 8000 
```

### Model Status <a name="http-model-status">

#### **Get information about the status of served models over HTTP interface:**

- Command

```bash
python http_get_model_status.py --help
usage: http_get_model_status.py [-h] [--service_url SERVICE_URL]
                                [--model_name MODEL_NAME]
                                [--model_version MODEL_VERSION]
                                [--timeout TIMEOUT]

Get information about the status of served models over HTTP interface

optional arguments:
  -h, --help            show this help message and exit
  --service_url SERVICE_URL
                        Specify url to http service. default:localhost:8000
  --model_name MODEL_NAME
                        Model name to query. default: resnet
  --model_version MODEL_VERSION
                        Model version to query. Lists all versions if omitted
  --timeout TIMEOUT     Request timeout. default: 10.0
```

- Usage Example

```bash
python http_get_model_status.py --model_name resnet --service_url localhost:8000
{1: {'state': 'AVAILABLE', 'error_code': 0, 'error_message': 'OK'}}
```


### Model Metadata <a name="http-model-metadata">

#### **Get information about the status of served models over HTTP interface:**

- Command

```bash
python http_get_model_metadata.py --help
usage: http_get_model_metadata.py [-h] [--service_url SERVICE_URL]
                                  [--model_name MODEL_NAME]
                                  [--model_version MODEL_VERSION]
                                  [--timeout TIMEOUT]

Get information about the status of served models over HTTP interface

optional arguments:
  -h, --help            show this help message and exit
  --service_url SERVICE_URL
                        Specify url to http service. default:localhost:8000
  --model_name MODEL_NAME
                        Model name to query. default: resnet
  --model_version MODEL_VERSION
                        Model version to query. If omitted or set to 0 returns
                        result for latest version
  --timeout TIMEOUT     Request timeout. default: 10.0
```
- Usage Example

```bash
python http_get_model_metadata.py --model_name resnet --model_version 1 --service_url localhost:8000
{'inputs': {'map/TensorArrayStack/TensorArrayGatherV3': {'dtype': 'DT_FLOAT', 'shape': [1, 3, 224, 224]}}, 'outputs': {'softmax_tensor': {'dtype': 'DT_FLOAT', 'shape': [1, 1001]}}, 'model_version': 1}
```


### Predict numeric format <a name="http-predict-numeric">

#### **Make prediction using images in numerical format:**

- Command

```bash
python http_predict_resnet.py --help
usage: http_predict_resnet.py [-h] --images_numpy IMAGES_NUMPY
                              [--service_url SERVICE_URL]
                              [--model_name MODEL_NAME]
                              [--model_version MODEL_VERSION]
                              [--iterations ITERATIONS] [--timeout TIMEOUT]

Make prediction using images in numerical format

optional arguments:
  -h, --help            show this help message and exit
  --images_numpy IMAGES_NUMPY
                        Path to a .npy file with data to infer
  --service_url SERVICE_URL
                        Specify url to http service. default:localhost:8000
  --model_name MODEL_NAME
                        Model name to query. default: resnet
  --model_version MODEL_VERSION
                        Model version to query. default: latest available
  --iterations ITERATIONS
                        Total number of requests to be sent. default: 0 - all
                        elements in numpy
  --timeout TIMEOUT     Request timeout. default: 10.0
```
- Usage example

```bash
python http_predict_resnet.py --images_numpy ../../imgs_nhwc.npy --model_name resnet --service_url localhost:8000
Image #0 has been classified as airliner
Image #1 has been classified as Arctic fox, white fox, Alopex lagopus
Image #2 has been classified as bee
Image #3 has been classified as golden retriever
Image #4 has been classified as gorilla, Gorilla gorilla
Image #5 has been classified as magnetic compass
Image #6 has been classified as peacock
Image #7 has been classified as pelican
Image #8 has been classified as snail
Image #9 has been classified as zebra
```

### Predict binary format<a name="http-predict-binary">

#### **Make prediction using images in binary format:**

- Command

```bash
python http_predict_binary_resnet.py --help
usage: http_predict_binary_resnet.py [-h] --images_dir IMAGES_DIR
                                     [--service_url SERVICE_URL]
                                     [--model_name MODEL_NAME]
                                     [--model_version MODEL_VERSION]
                                     [--timeout TIMEOUT]

Make prediction using images in binary format

optional arguments:
  -h, --help            show this help message and exit
  --images_dir IMAGES_DIR
                        Path to a directory with images in JPG or PNG format
  --service_url SERVICE_URL
                        Specify url to http service. default:localhost:8000
  --model_name MODEL_NAME
                        Model name to query. default: resnet
  --model_version MODEL_VERSION
                        Model version to query. default: latest available
  --timeout TIMEOUT     Request timeout. default: 10.0
```
- Usage example

```bash
python http_predict_binary_resnet.py --images_dir ../../../../demos/common/static/images --model_name resnet --service_url localhost:8000
Image ../../../../demos/common/static/images/magnetic_compass.jpeg has been classified as magnetic compass
Image ../../../../demos/common/static/images/pelican.jpeg has been classified as pelican
Image ../../../../demos/common/static/images/gorilla.jpeg has been classified as gorilla, Gorilla gorilla
Image ../../../../demos/common/static/images/snail.jpeg has been classified as snail
Image ../../../../demos/common/static/images/zebra.jpeg has been classified as zebra
Image ../../../../demos/common/static/images/arctic-fox.jpeg has been classified as Arctic fox, white fox, Alopex lagopus
Image ../../../../demos/common/static/images/bee.jpeg has been classified as bee
Image ../../../../demos/common/static/images/peacock.jpeg has been classified as peacock
Image ../../../../demos/common/static/images/airliner.jpeg has been classified as warplane, military plane
Image ../../../../demos/common/static/images/golden_retriever.jpeg has been classified as golden retriever
```

## Prepare the model from OpenVINO Model Zoo

### Vehicle detection model

OVMS container can be started using a command:
```bash
docker run -d --rm -v ${PWD}/models/vehicle-detection:/models/vehicle-detection -p 8000:8000 -p 9000:9000 openvino/model_server:latest --model_name vehicle-detection --model_path /models/vehicle-detection --port 9000 --rest_port 8000 --layout NHWC:NCHW
```


### Detect vehicle  <a name="http-detect-vehicle">


#### **Make vehicle detection prediction using images in binary format:**

- Command

```bash
python http_predict_binary_vehicle_detection.py --help
usage: http_predict_binary_vehicle_detection.py [-h] --images_dir IMAGES_DIR
                                                [--service_url SERVICE_URL]
                                                [--model_name MODEL_NAME]
                                                [--model_version MODEL_VERSION]
                                                --output_dir OUTPUT_DIR
                                                [--timeout TIMEOUT]

Make vehicle detection prediction using images in binary format

optional arguments:
  -h, --help            show this help message and exit
  --images_dir IMAGES_DIR
                        Path to a directory with images in JPG or PNG format
  --service_url SERVICE_URL
                        Specify url to http service. default:localhost:8000
  --model_name MODEL_NAME
                        Model name to query. default: vehicle-detection
  --model_version MODEL_VERSION
                        Model version to query. default: latest available
  --output_dir OUTPUT_DIR
                        Path to store output.
  --timeout TIMEOUT     Request timeout. default: 10.0
```

- Usage example

```bash
python http_predict_binary_vehicle_detection.py --images_dir ../../../../demos/common/static/images/cars/ --output_dir ./output --service_url localhost:8000
Detection results in file:  ./output/road1.jpg
```
