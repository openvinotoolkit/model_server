# OpenVINO&trade; Model Server Client Library Example Clients

This document contains examples to run *GetModelStatus*, *GetModelMetadata*, *Predict* functions over gRPC API.

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

Start Model Server Client Library virtual environment

If using Python virtual environment, activate it:
```
. ../lib/.venv/bin/activate
```

Install client dependencies using the command below:
```
pip3 install -r requirements.txt
```
Build and install [Model Server Client Library](../lib)
`pip3 install dist/ovmsclient-0.1-py3-none-any.whl`

Download [Resnet50-tf Model](https://docs.openvinotoolkit.org/latest/omz_models_model_resnet_50_tf.html) and convert it into Intermediate Representation format:
```bash
mkdir models
docker run -u $(id -u):$(id -g) -v ${PWD}/models:/models openvino/ubuntu18_dev:latest deployment_tools/open_model_zoo/tools/downloader/downloader.py --name resnet-50-tf --output_dir /models
docker run -u $(id -u):$(id -g) -v ${PWD}/models:/models:rw openvino/ubuntu18_dev:latest deployment_tools/open_model_zoo/tools/downloader/converter.py --name resnet-50-tf --download_dir /models --output_dir /models --precisions FP32
mv ${PWD}/models/public/resnet-50-tf/FP32 ${PWD}/models/public/resnet-50-tf/1
```

OVMS can be started using a command:
```bash
docker run -d --rm -v ${PWD}/models/public/resnet-50-tf:/models/public/resnet-50-tf -p 8000:8000 -p 9000:9000 openvino/model_server:latest --model_name resnet --model_path /models/public/resnet-50-tf --port 9000 --rest_port 8000 
```


## gRPC Client Examples <a name="grpc-api"></a>

### Model Status <a name="grpc-model-status">

#### **Get information about the status of served models over gRPC interace:**

- Command

```bash
python grpc_get_model_status.py -h
usage: grpc_get_model_status.py [-h] [--service_url SERVICE_URL]
                                [--model_name MODEL_NAME]
                                [--model_version MODEL_VERSION]
                                [--timeout TIMEOUT]


optional arguments:
  -h, --help            show this help message and exit
  --service_url SERVICE_URL Specify url to grpc service. default:localhost:9000
  --model_name MODEL_NAME Model name to query. default: resnet
  --model_version MODEL_VERSION Model version to query. Lists all versions if omitted
  --timeout TIMEOUT Request timeout. default: 10.0
```

- Usage Example

```
python grpc_get_model_status.py --model_name resnet
{1: {'state': 'AVAILABLE', 'error_code': 0, 'error_message': 'OK'}}
```


### Model Metadata <a name="grpc-model-metadata">

#### **Get information about the status of served models over gRPC interace:**

- Command

```bash
python grpc_get_model_metadata.py --help
usage: grpc_get__model_metadata.py [-h] [--service_url SERVICE_URL]
                                  [--model_name MODEL_NAME]
                                  [--model_version MODEL_VERSION]
                                  [--timeout TIMEOUT]


optional arguments:
  -h, --help            show this help message and exit
  --service_url SERVICE_URL Specify url to grpc service. default:localhost:9000
  --model_name MODEL_NAME Model name to query. default: resnet
  --model_version MODEL_VERSION Model version to query. If ommited or set to 0 returns result for latest version
  --timeout TIMEOUT Request timeout. default: 10.0
```
- Usage Example

```
python grpc_get_model_metadata.py --model_name resnet --model_version 1
{'inputs': {'map/TensorArrayStack/TensorArrayGatherV3': {'shape': [1, 224, 224, 3], 'dtype': 'DT_FLOAT'}}, 'outputs': {'softmax_tensor': {'shape': [1, 1001], 'dtype': 'DT_FLOAT'}}}
```


### Predict numeric format <a name="grpc-predict-numeric">

#### **Make prediction using images in numerical format:**

- Command

```bash
usage: grpc_predict_resnet.py [-h] [--images_numpy IMAGES_NUMPY]
                              [--service_url SERVICE_URL]
                              [--model_name MODEL_NAME]
                              [--model_version MODEL_VERSION]
                              [--timeout TIMEOUT]


optional arguments:
  -h, --help            show this help message and exit 
  --images_numpy IMAGES_NUMPY Path to a .npy file with data to infer
  --service_url SERVICE_URL Specify url to grpc service. default:localhost:9000
  --model_name MODEL_NAME Model name to query. default: resnet
  --model_version MODEL_VERSION Model version to query. default: latest available
  --timeout TIMEOUT Request timeout. default: 10.0
```
- Usage example

```
python grpc_predict_resnet.py --images_dir images --model_name resnet
Image images/magnetic_compass.jpeg has been classified as magnetic compass
Image images/pelican.jpeg has been classified as pelican
Image images/gorilla.jpeg has been classified as gorilla, Gorilla gorilla
Image images/snail.jpeg has been classified as snail
Image images/zebra.jpeg has been classified as zebra
Image images/arctic-fox.jpeg has been classified as Arctic fox, white fox, Alopex lagopus
Image images/bee.jpeg has been classified as bee
Image images/peacock.jpeg has been classified as peacock
Image images/airliner.jpeg has been classified as airliner
Image images/golden_retriever.jpeg has been classified as golden retriever
```

To serve Resnet with support for binary input data, the model needs to be configured with NHWC layout. That can be acheived by starting the OVMS container with `--layout NHWC` parameter.
new OVMS instance with --layout MHWC parameter.
```bash
docker run -d --rm -v ${PWD}/models/public/resnet-50-tf:/models/public/resnet-50-tf -p 8000:8000 -p 9000:9000 openvino/model_server:latest --model_name resnet --model_path /models/public/resnet-50-tf --port 9000 --rest_port 8000 --layout NHWC
```

### Predict binary format<a name="grpc-predict-binary">

#### **Make prediction using images in binary format:**

- Command

```bash
usage: grpc_predict_binary_resnet.py [-h] [--images_dir IMAGES_DIR]
                              [--service_url SERVICE_URL]
                              [--model_name MODEL_NAME]
                              [--model_version MODEL_VERSION]
                              [--timeout TIMEOUT]


optional arguments:
  -h, --help            show this help message and exit 
  --images_dir IMAGES_DIR Path to a directory with images in JPG or PNG format
  --service_url SERVICE_URL Specify url to grpc service. default:localhost:9000
  --model_name MODEL_NAME Model name to query. default: resnet
  --model_version MODEL_VERSION Model version to query. default: latest available
  --timeout TIMEOUT Request timeout. default: 10.0
```
- Usage example

```
python grpc_predict_binary_resnet.py --images_dir images --model_name resnet
Image images/magnetic_compass.jpeg has been classified as magnetic compass
Image images/pelican.jpeg has been classified as pelican
Image images/gorilla.jpeg has been classified as gorilla, Gorilla gorilla
Image images/snail.jpeg has been classified as snail
Image images/zebra.jpeg has been classified as zebra
Image images/arctic-fox.jpeg has been classified as Arctic fox, white fox, Alopex lagopus
Image images/bee.jpeg has been classified as bee
Image images/peacock.jpeg has been classified as peacock
Image images/airliner.jpeg has been classified as warplane, military plane
Image images/golden_retriever.jpeg has been classified as golden retriever
```

## Prepare the model from OpenVINO Model Zoo

### Vehicle detection model
```
mkdir -p models
curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/vehicle-detection-0202/FP32/vehicle-detection-0202.xml -o ${PWD}/models/vehicle-detection/1/vehicle-detection-0202.xml
curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/vehicle-detection-0202/FP32/vehicle-detection-0202.bin -o ${PWD}/models/vehicle-detection/1/vehicle-detection-0202.bin
chmod -R 755 ${PWD}/models/vehicle-detection
```
OVMS container can be started using a command:
```bash
docker run -d --rm -v ${PWD}/models/vehicle-detection:/models/vehicle-detection -p 8000:8000 -p 9000:9000 openvino/model_server:latest --model_name vehicle-detection --model_path /models/vehicle-detection --port 9000 --rest_port 8000 --layout NHWC
```


### Detect vehicle  <a name="grpc-detect-vehicle">


#### **Make vehicle detection prediction using images in binary format:**

- Command

```bash
usage: grpc_predict_binary_vehicle_detection.py [-h] [--images_dir IMAGES_DIR]
                                           [--service_url SERVICE_URL]
                                           [--model_name MODEL_NAME]
                                           [--model_version MODEL_VERSION]
                                           [--output_dir OUTPUT_DIR]
                                           [--timeout TIMEOUT]


optional arguments:
  -h, --help            show this help message and exit
  --images_dir IMAGES_DIR Path to a directory with images in JPG or PNG format
  --service_url SERVICE_URL Specify url to grpc service. default:localhost:9000
  --model_name MODEL_NAME Model name to query. default: vehicle-detection
  --model_version MODEL_VERSION Model version to query. default: latest available
  --output_dir OUTPUT_DIR Path to store output.
  --timeout TIMEOUT Request timeout. default: 10.0
```

- Usage example

```bash
python grpc_predict_binary_vehicle_detection.py --images_dir ./images/cars/ --output_dir ./output
Making directory for output: ./output
Detection results in file:  ./output/road1.jpg
```

## HTTP Client Examples <a name="http-api"></a>

### Model Status <a name="http-model-status">

#### **Get information about the status of served models over HTTP interace:**

- Command

```bash
python http_get_model_status.py -h
usage: http_get_model_status.py [-h] [--service_url SERVICE_URL]
                                [--model_name MODEL_NAME]
                                [--model_version MODEL_VERSION]
                                [--timeout TIMEOUT]


optional arguments:
  -h, --help            show this help message and exit
  --service_url SERVICE_URL Specify url to http service. default:localhost:9000
  --model_name MODEL_NAME Model name to query. default: resnet
  --model_version MODEL_VERSION Model version to query. Lists all versions if omitted
  --timeout TIMEOUT Request timeout. default: 10.0
```

- Usage Example

```
python http_get_model_status.py --model_name resnet
{1: {'state': 'AVAILABLE', 'error_code': 0, 'error_message': 'OK'}}
```


### Model Metadata <a name="http-model-metadata">

#### **Get information about the status of served models over HTTP interace:**

- Command

```bash
python http_get_model_metadata.py --help
usage: http_get_model_metadata.py [-h] [--service_url SERVICE_URL]
                                  [--model_name MODEL_NAME]
                                  [--model_version MODEL_VERSION]
                                  [--timeout TIMEOUT]


optional arguments:
  -h, --help            show this help message and exit
  --service_url SERVICE_URL Specify url to http service. default:localhost:9000
  --model_name MODEL_NAME Model name to query. default: resnet
  --model_version MODEL_VERSION Model version to query. If ommited or set to 0 returns result for latest version
  --timeout TIMEOUT Request timeout. default: 10.0
```
- Usage Example

```
python http_get_model_metadata.py --model_name resnet --model_version 1
{'inputs': {'map/TensorArrayStack/TensorArrayGatherV3': {'shape': [1, 224, 224, 3], 'dtype': 'DT_FLOAT'}}, 'outputs': {'softmax_tensor': {'shape': [1, 1001], 'dtype': 'DT_FLOAT'}}}
```


### Predict numeric format <a name="http-predict-numeric">

#### **Make prediction using images in numerical format:**

- Command

```bash
usage: http_predict_resnet.py [-h] [--images_numpy IMAGES_NUMPY]
                              [--service_url SERVICE_URL]
                              [--model_name MODEL_NAME]
                              [--model_version MODEL_VERSION]
                              [--timeout TIMEOUT]


optional arguments:
  -h, --help            show this help message and exit 
  --images_numpy IMAGES_NUMPY Path to a .npy file with data to infer
  --service_url SERVICE_URL Specify url to http service. default:localhost:9000
  --model_name MODEL_NAME Model name to query. default: resnet
  --model_version MODEL_VERSION Model version to query. default: latest available
  --timeout TIMEOUT Request timeout. default: 10.0
```
- Usage example

```
python http_predict_resnet.py --images_dir images --model_name resnet
Image images/magnetic_compass.jpeg has been classified as magnetic compass
Image images/pelican.jpeg has been classified as pelican
Image images/gorilla.jpeg has been classified as gorilla, Gorilla gorilla
Image images/snail.jpeg has been classified as snail
Image images/zebra.jpeg has been classified as zebra
Image images/arctic-fox.jpeg has been classified as Arctic fox, white fox, Alopex lagopus
Image images/bee.jpeg has been classified as bee
Image images/peacock.jpeg has been classified as peacock
Image images/airliner.jpeg has been classified as airliner
Image images/golden_retriever.jpeg has been classified as golden retriever
```

To serve Resnet with support for binary input data, the model needs to be configured with NHWC layout. That can be acheived by starting the OVMS container with `--layout NHWC` parameter.
new OVMS instance with --layout MHWC parameter.
```bash
docker run -d --rm -v ${PWD}/models/public/resnet-50-tf:/models/public/resnet-50-tf -p 8000:8000 -p 9000:9000 openvino/model_server:latest --model_name resnet --model_path /models/public/resnet-50-tf --port 9000 --rest_port 8000 --layout NHWC
```

### Predict binary format<a name="http-predict-binary">

#### **Make prediction using images in binary format:**

- Command

```bash
usage: http_predict_binary_resnet.py [-h] [--images_dir IMAGES_DIR]
                              [--service_url SERVICE_URL]
                              [--model_name MODEL_NAME]
                              [--model_version MODEL_VERSION]
                              [--timeout TIMEOUT]


optional arguments:
  -h, --help            show this help message and exit 
  --images_dir IMAGES_DIR Path to a directory with images in JPG or PNG format
  --service_url SERVICE_URL Specify url to http service. default:localhost:9000
  --model_name MODEL_NAME Model name to query. default: resnet
  --model_version MODEL_VERSION Model version to query. default: latest available
  --timeout TIMEOUT Request timeout. default: 10.0
```
- Usage example

```
python http_predict_binary_resnet.py --images_dir images --model_name resnet
Image images/magnetic_compass.jpeg has been classified as magnetic compass
Image images/pelican.jpeg has been classified as pelican
Image images/gorilla.jpeg has been classified as gorilla, Gorilla gorilla
Image images/snail.jpeg has been classified as snail
Image images/zebra.jpeg has been classified as zebra
Image images/arctic-fox.jpeg has been classified as Arctic fox, white fox, Alopex lagopus
Image images/bee.jpeg has been classified as bee
Image images/peacock.jpeg has been classified as peacock
Image images/airliner.jpeg has been classified as warplane, military plane
Image images/golden_retriever.jpeg has been classified as golden retriever
```

## Prepare the model from OpenVINO Model Zoo

### Vehicle detection model
```
mkdir -p models
curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/vehicle-detection-0202/FP32/vehicle-detection-0202.xml -o ${PWD}/models/vehicle-detection/1/vehicle-detection-0202.xml
curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/vehicle-detection-0202/FP32/vehicle-detection-0202.bin -o ${PWD}/models/vehicle-detection/1/vehicle-detection-0202.bin
chmod -R 755 ${PWD}/models/vehicle-detection
```
OVMS container can be started using a command:
```bash
docker run -d --rm -v ${PWD}/models/vehicle-detection:/models/vehicle-detection -p 8000:8000 -p 9000:9000 openvino/model_server:latest --model_name vehicle-detection --model_path /models/vehicle-detection --port 9000 --rest_port 8000 --layout NHWC
```


### Detect vehicle  <a name="http-detect-vehicle">


#### **Make vehicle detection prediction using images in binary format:**

- Command

```bash
usage: http_predict_binary_vehicle_detection.py [-h] [--images_dir IMAGES_DIR]
                                           [--service_url SERVICE_URL]
                                           [--model_name MODEL_NAME]
                                           [--model_version MODEL_VERSION]
                                           [--output_dir OUTPUT_DIR]
                                           [--timeout TIMEOUT]


optional arguments:
  -h, --help            show this help message and exit
  --images_dir IMAGES_DIR Path to a directory with images in JPG or PNG format
  --service_url SERVICE_URL Specify url to http service. default:localhost:9000
  --model_name MODEL_NAME Model name to query. default: vehicle-detection
  --model_version MODEL_VERSION Model version to query. default: latest available
  --output_dir OUTPUT_DIR Path to store output.
  --timeout TIMEOUT Request timeout. default: 10.0
```

- Usage example

```bash
python http_predict_binary_vehicle_detection.py --images_dir ./images/cars/ --output_dir ./output
Making directory for output: ./output
Detection results in file:  ./output/road1.jpg
```