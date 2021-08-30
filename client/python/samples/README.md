# OpenVINO&trade; Model Server Client Library Example Clients

This document contains examples to run *GetModelStatus*, *GetModelMetadata*, *Predict* functions over gRPC API.

It covers following topics:
* <a href="#grpc-api">gRPC API Example Clients </a>

## Requirement

**Note**: Provided examples and their dependencies are updated and validated for Python 3.6+ version. For older versions of Python, dependencies versions adjustment might be required.

If using Python virtual environment, activate it:
```
. ../lib/.venv/bin/activate
```

Install client dependencies using the command below in the example_client directory:
```
pip3 install -r requirements.txt
```
Build and install [Model Server Client Library](../lib)
`pip install dist/ovmsclient-0.1-py3-none-any.whl`

Download [Resnet50-tf Model](https://docs.openvinotoolkit.org/latest/omz_models_model_resnet_50_tf.html) and Convert it into Inference Engine Format
and place the xml and bin files in the <PATH_TO_MODELS>/resnet/1

For numeric format data OVMS can be started using a command:
```bash
docker run -d --rm -v /resnet/model/path:/model -e "http_proxy=$http_proxy" -e "https_proxy=$https_proxy" -p 8000:8000 -p 9000:9000 openvino/model_server:latest --model_name resnet --model_path /models --port 9000 --rest_port 8000
```


## gRPC Client Examples <a name="grpc-api"></a>

### Model Status

#### **Get information about the status of served models over gRPC interace:**

- Command

```bash
python get_grpc_model_status.py -h
usage: get_grpc_model_status.py [-h] [--grpc_address GRPC_ADDRESS]
                                [--grpc_port GRPC_PORT]
                                [--model_name MODEL_NAME]
                                [--model_version MODEL_VERSION]


optional arguments:
  -h, --help            show this help message and exit
  --grpc_address GRPC_ADDRESS Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT Specify port to grpc service. default: 9000
  --model_name MODEL_NAME Model name to query. default: resnet
  --model_version MODEL_VERSION Model version to query. Lists all versions if omitted
```

- Usage Example

```
python get_grpc_model_status.py --grpc_port 9000 --model_name resnet
{1: {'state': 'AVAILABLE', 'error_code': 0, 'error_message': 'OK'}}
```


### Model Metadata

#### **Get information about the status of served models over gRPC interace:**

- Command

```bash
python get_grpc_model_metadata.py --help
usage: get_grpc_model_metadata.py [-h] [--grpc_address GRPC_ADDRESS]
                                  [--grpc_port GRPC_PORT]
                                  [--model_name MODEL_NAME]
                                  [--model_version MODEL_VERSION]


optional arguments:
  -h, --help            show this help message and exit
  --grpc_address GRPC_ADDRESS Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT Specify port to grpc service. default: 9000
  --model_name MODEL_NAME Model name to query. default: resnet
  --model_version MODEL_VERSION Model version to query. Lists all versions if omitted
```
- Usage Example

```
python get_grpc_model_metadata.py --grpc_port 9000 --model_name resnet --model_version 1
{1: {'inputs': {'data': {'shape': [1, 3, 224, 224], 'dtype': 'DT_FLOAT'}}, 'outputs': {'prob': {'shape': [1, 1000], 'dtype': 'DT_FLOAT'}}}}
```


### Predict 

#### **Make prediction using images in numerical format:**

- Command

```bash
usage: resnet_grpc_predict.py [-h] --images_dir IMAGES_DIR
                              [--grpc_address GRPC_ADDRESS]
                              [--grpc_port GRPC_PORT]
                              [--model_name MODEL_NAME]
                              [--model_version MODEL_VERSION]


optional arguments:
  -h, --help            show this help message and exit --images_dir IMAGES_DIR
                        Path to a directory with images in JPG or PNG format
  --grpc_address GRPC_ADDRESS Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT Specify port to grpc service. default: 9000
  --model_name MODEL_NAME Model name to query. default: resnet
  --model_version MODEL_VERSION Model version to query. default: latest available
```
- Usage example

```
python resnet_grpc_predict.py --grpc_port 9000 --images_dir images --model_name resnet
Image images/magnetic_compass.jpeg has been classified as magnetic compass with 99.99372959136963% confidence
Image images/pelican.jpeg has been classified as pelican with 99.17410612106323% confidence
Image images/gorilla.jpeg has been classified as gorilla, Gorilla gorilla with 98.07604551315308% confidence
Image images/snail.jpeg has been classified as snail with 99.97051358222961% confidence
Image images/zebra.jpeg has been classified as zebra with 99.4793951511383% confidence
Image images/arctic-fox.jpeg has been classified as Arctic fox, white fox, Alopex lagopus with 93.65214705467224% confidence
Image images/bee.jpeg has been classified as bee with 96.6326653957367% confidence
Image images/peacock.jpeg has been classified as peacock with 99.97820258140564% confidence
Image images/airliner.jpeg has been classified as airliner with 49.202319979667664% confidence
Image images/golden_retriever.jpeg has been classified as golden retriever with 88.68610262870789% confidence
```

To run resnet model serving for binary input data we need to kill the preious OVMS docker instance and run
new OVMS instance with --layout MHWC parameter.
For binary format data OVMS can be started using a command:
```bash
docker run -d --rm -v /resnet/model/path:/model -e "http_proxy=$http_proxy" -e "https_proxy=$https_proxy" -p 8000:8000 -p 9000:9000 openvino/model_server:latest --model_name resnet --model_path /models --port 9000 --rest_port 8000 --layout MHWC
```

### Predict 

#### **Make prediction using images in binary format:**

- Command

```bash
usage: resnet_grpc_predict_binary.py [-h] --images_dir IMAGES_DIR
                              [--grpc_address GRPC_ADDRESS]
                              [--grpc_port GRPC_PORT]
                              [--model_name MODEL_NAME]
                              [--model_version MODEL_VERSION]


optional arguments:
  -h, --help            show this help message and exit --images_dir IMAGES_DIR
                        Path to a directory with images in JPG or PNG format
  --grpc_address GRPC_ADDRESS Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT Specify port to grpc service. default: 9000
  --model_name MODEL_NAME Model name to query. default: resnet
  --model_version MODEL_VERSION Model version to query. default: latest available
```
- Usage example

```
python resnet_grpc_predict_binary.py --grpc_port 9000 --images_dir images --model_name resnet
Image images/magnetic_compass.jpeg has been classified as magnetic compass with 99.99269247055054% confidence
Image images/pelican.jpeg has been classified as pelican with 97.38033413887024% confidence
Image images/gorilla.jpeg has been classified as gorilla, Gorilla gorilla with 96.9128966331482% confidence
Image images/snail.jpeg has been classified as snail with 99.9498724937439% confidence
Image images/zebra.jpeg has been classified as zebra with 98.35399389266968% confidence
Image images/arctic-fox.jpeg has been classified as Arctic fox, white fox, Alopex lagopus with 87.82028555870056% confidence
Image images/bee.jpeg has been classified as bee with 97.44628667831421% confidence
Image images/peacock.jpeg has been classified as peacock with 99.98815059661865% confidence
Image images/airliner.jpeg has been classified as warplane, military plane with 73.17261695861816% confidence
Image images/golden_retriever.jpeg has been classified as golden retriever with 87.05007433891296% confidence
```

## Prepare the model from OpenVINO Model Zoo
### Vehicle detection model
```
mkdir -p model_server/client/python/samples/models/vehicle-detection/1
cd model_server/client/python/samples/models/vehicle-detection/1
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/vehicle-detection-0202/FP32/vehicle-detection-0202.xml
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/vehicle-detection-0202/FP32/vehicle-detection-0202.bin
```
For numeric format data OVMS can be started using a command:
```bash
docker run -d --rm -v model_server/client/python/samples/models/vehicle-detection:/models -e "http_proxy=$http_proxy" -e "https_proxy=$https_proxy" -p 8000:8000 -p 9000:9000 openvino/model_server:latest --model_name vehicle-detection --model_path /models --port 9000 --rest_port 8000
```


### Predict 

#### **Make vehicle detection prediction using images in binary format:**
#### **Required Model Server running with vehicle detection model:**

- Command

```bash
usage: vehicle_detection_predict_binary.py [-h] --images_dir IMAGES_DIR
                                           [--grpc_address GRPC_ADDRESS]
                                           [--grpc_port GRPC_PORT]
                                           [--model_name MODEL_NAME]
                                           [--model_version MODEL_VERSION]
                                           --output_save_path OUTPUT_SAVE_PATH


optional arguments:
  -h, --help            show this help message and exit
  --images_dir IMAGES_DIR Path to a directory with images in JPG or PNG format
  --grpc_address GRPC_ADDRESS Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT Specify port to grpc service. default: 9000
  --model_name MODEL_NAME Model name to query. default: vehicle-detection
  --model_version MODEL_VERSION Model version to query. default: latest available
  --output_save_path OUTPUT_SAVE_PATH Path to store output.
```

- Usage example

```bash
python vehicle_detection_predict_binary.py --images_dir ./images/cars/ --output_save_path ./output
Making directory for output: ./output
Detection results in file:  ./output/vehicle-detection.jpg
```