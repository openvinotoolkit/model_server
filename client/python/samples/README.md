# OpenVINO&trade; Model Server Client Library Example Clients

This document contains examples to run *GetModelStatus*, *GetModelMetadata*, *Predict* functions over gRPC API.

It covers following topics:
* <a href="#grpc-api">gRPC API Example Clients </a>

## Requirement

**Note**: Provided examples and their dependencies are updated and validated for Python 3.6+ version. For older versions of Python, dependencies versions adjustment might be required.

Start Model Server Client Library virtual environment
```
. ../lib/.venv/bin/activate
```

Install client dependencies using the command below in the example_client directory:
```
pip3 install -r requirements.txt
```
Build and install [Model Server Client Library](../lib)
`pip install dist/ovmsclient-0.1-py3-none-any.whl`

Access to Google Cloud Storage might require proper configuration of https_proxy in the docker engine or in the docker container.
In the examples listed below, OVMS can be started using a command:
```bash
docker run -d --rm -e "http_proxy=$http_proxy" -e "https_proxy=$https_proxy" -p 8000:8000 -p 9000:9000 openvino/model_server:latest --model_name resnet --model_path gs://ovms-public-eu/resnet50 --port 9000 --rest_port 8000
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
```

### Predict 

#### **Make prediction using images in binary format:**

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
```