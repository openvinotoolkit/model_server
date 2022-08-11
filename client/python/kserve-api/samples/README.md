# KServe API usage samples{#ovms_docs_kserve_samples}

OpenVINO Model Server 2022.2 release introduced support for [KServe API](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2).

This guide shows how to interact with KServe API endpoints on both gRPC and HTTP interfaces. It covers following topics:
- <a href="#grpc-api">GRPC API Examples </a>
  - <a href="#grpc-server-live">grpc_server_live.py</a>
  - <a href="#grpc-server-ready">grpc_server_ready.py</a>
  - <a href="#grpc-server-metadata">grpc_server_metadata.py</a>
  - <a href="#grpc-model-ready">grpc_model_ready.py</a>
  - <a href="#grpc-model-metadata">grpc_model_metadata.py</a>
  - <a href="#grpc-model-infer">grpc_infer_resnet.py</a>
- <a href="#http-api">HTTP API Example</a>
  - <a href="#http-server-live">http_server_live.py</a>
  - <a href="#http-server-ready">http_server_ready.py</a>
  - <a href="#http-server-metadata">http_server_metadata.py</a>
  - <a href="#http-model-ready">http_model_ready.py</a>
  - <a href="#http-model-metadata">http_model_metadata.py</a>
  - <a href="#http-model-infer">http_infer_resnet.py</a>

> **Note:** Some of the samples will use [ResNet50](https://github.com/openvinotoolkit/open_model_zoo/blob/2022.1.0/models/intel/resnet50-binary-0001/README.md).

## Before you run the samples

### Clone OpenVINO&trade; Model Server GitHub repository and enter model_server directory.
```Bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
```
### Download the Pretrained Model
Download the model files and store them in the `models` directory
```Bash
mkdir -p models/resnet/1
curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml -o models/resnet/1/resnet50-binary-0001.bin -o models/resnet/1/resnet50-binary-0001.xml
```

### Pull the Latest Model Server Image
Pull the latest version of OpenVINO&trade; Model Server from Docker Hub :
```Bash
docker pull openvino/model_server:latest
```

### Start the Model Server Container with Downloaded Model and Dynamic Batch Size
Start the server container with the image pulled in the previous step and mount the `models` directory :
```Bash
docker run --rm -d -v $(pwd)/models:/models -p 9000:9000 -p 5000:5000 openvino/model_server:latest --model_name resnet --model_path /models/resnet --batch_size auto --port 9000 --rest_port 5000
```

### Prepare virtualenv
```Bash
cd client/python/kserve-api/samples
virtualenv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Once you finish above steps, you are ready to run the samples.

---

## GRPC Examples <a name="grpc-api"></a>

### Run the Client to get server liveness <a name="grpc-server-live"></a>

- Command

```Bash
python3 ./grpc_server_live.py --help
usage: grpc_server_live.py [-h] [--grpc_address GRPC_ADDRESS] [--grpc_port GRPC_PORT]

Sends request via KServe gRPC API to check if server is alive.

optional arguments:
  -h, --help            show this help message and exit
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT
                        Specify port to grpc service. default: 9000

```

- Usage Example 

```Bash
python3 ./grpc_server_live.py --grpc_port 9000 --grpc_address localhost
Server Live: True
```

### Run the Client to get server readiness <a name="grpc-server-ready"></a>

- Command

```Bash
python3 ./grpc_server_ready.py --help
usage: grpc_server_ready.py [-h] [--grpc_address GRPC_ADDRESS] [--grpc_port GRPC_PORT]

Sends request via KServe gRPC API to check if server is ready.

optional arguments:
  -h, --help            show this help message and exit
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT
                        Specify port to grpc service. default: 9000

```

- Usage Example

```Bash
python3 ./grpc_server_ready.py --grpc_port 9000 --grpc_address localhost
Server Ready: True
```


### Run the Client to get model readiness <a name="grpc-model-ready"></a>

- Command

```Bash
python3 ./grpc_model_ready.py --help
usage: grpc_model_ready.py [-h] [--grpc_address GRPC_ADDRESS] [--grpc_port GRPC_PORT] [--model_name MODEL_NAME] [--model_version MODEL_VERSION]

Sends requests via KServe gRPC API to check if model is ready for inference.

optional arguments:
  -h, --help            show this help message and exit
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT
                        Specify port to grpc service. default: 9000
  --model_name MODEL_NAME
                        Define model name, must be same as is in service. default: resnet
  --model_version MODEL_VERSION
                        Define model version. If not specified, the default version will be taken from model server
```

- Usage Example

```Bash
python3 ./grpc_model_ready.py --grpc_port 9000 --grpc_address localhost --model_name resnet
Model Ready: True
```

### Run the Client to get metadata <a name="grpc-model-metadata"></a>

- Command

```Bash
python3 ./grpc_model_metadata.py --help
usage: grpc_model_metadata.py [-h] [--grpc_address GRPC_ADDRESS] [--grpc_port GRPC_PORT] [--model_name MODEL_NAME] [--model_version MODEL_VERSION]

Sends requests via KServe gRPC API to get model metadata

optional arguments:
  -h, --help            show this help message and exit
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT
                        Specify port to grpc service. default: 9000
  --model_name MODEL_NAME
                        Define model name, must be same as is in service. default: resnet
  --model_version MODEL_VERSION
                        Define model version. If not specified, the default version will be taken from model server
```

- Usage Example

```Bash
python3 ./grpc_model_metadata.py --grpc_port 9000 --grpc_address localhost --model_name resnet
model metadata:
name: "resnet"
versions: "1"
platform: "OpenVINO"
inputs {
  name: "0"
  datatype: "FP32"
  shape: 1
  shape: 3
  shape: 224
  shape: 224
}
outputs {
  name: "1463"
  datatype: "FP32"
  shape: 1
  shape: 1000
}
```

### Run the Client to perform inference <a name="grpc-model-infer"></a>

- Command

```Bash
python grpc_infer_resnet.py --help
TODO
```

- Usage Example

```Bash
python grpc_infer_resnet.py --grpc_port 9000 --images_numpy_path ../../imgs.npy --labels_numpy_path ../../lbs.npy --input_name 0 --output_name 1463 --model_name resnet --transpose_input False
Image data range: 0.0 : 255.0
Start processing:
        Model name: resnet
        Iterations: 10
        Images numpy path: ../../imgs.npy
        Numpy file shape: (10, 3, 224, 224)

Iteration 1; Processing time: 29.98 ms; speed 33.36 fps
imagenet top results in a single batch:
         0 airliner 404 ; Correct match.
Iteration 2; Processing time: 23.50 ms; speed 42.56 fps
imagenet top results in a single batch:
         0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Iteration 3; Processing time: 23.42 ms; speed 42.71 fps
imagenet top results in a single batch:
         0 bee 309 ; Correct match.
Iteration 4; Processing time: 23.25 ms; speed 43.01 fps
imagenet top results in a single batch:
         0 golden retriever 207 ; Correct match.
Iteration 5; Processing time: 24.08 ms; speed 41.53 fps
imagenet top results in a single batch:
         0 gorilla, Gorilla gorilla 366 ; Correct match.
Iteration 6; Processing time: 26.38 ms; speed 37.91 fps
imagenet top results in a single batch:
         0 magnetic compass 635 ; Correct match.
Iteration 7; Processing time: 25.46 ms; speed 39.27 fps
imagenet top results in a single batch:
         0 peacock 84 ; Correct match.
Iteration 8; Processing time: 24.60 ms; speed 40.65 fps
imagenet top results in a single batch:
         0 pelican 144 ; Correct match.
Iteration 9; Processing time: 23.52 ms; speed 42.52 fps
imagenet top results in a single batch:
         0 snail 113 ; Correct match.
Iteration 10; Processing time: 22.40 ms; speed 44.64 fps
imagenet top results in a single batch:
         0 zebra 340 ; Correct match.

processing time for all iterations
average time: 24.20 ms; average speed: 41.32 fps
median time: 23.50 ms; median speed: 42.55 fps
max time: 29.00 ms; min speed: 34.48 fps
min time: 22.00 ms; max speed: 45.45 fps
time percentile 90: 26.30 ms; speed percentile 90: 38.02 fps
time percentile 50: 23.50 ms; speed percentile 50: 42.55 fps
time standard deviation: 1.94
time variance: 3.76
Classification accuracy: 100.00
```

---

## HTTP Examples <a name="http-api"></a>

### Run the Client to get server liveness <a name="http-server-live"></a>

- Command

```Bash
python3 ./http_server_live.py --help
usage: http_server_live.py [-h] [--http_address HTTP_ADDRESS] [--http_port HTTP_PORT]

Sends request via KServe HTTP API to check if server is alive.

optional arguments:
  -h, --help            show this help message and exit
  --http_address HTTP_ADDRESS
                        Specify url to HTTP service. default:localhost
  --http_port HTTP_PORT
                        Specify port to HTTP service. default: 5000
```

- Usage Example

```Bash
python3 ./http_server_live.py --http_port 5000 --http_address localhost
Server Live: True
```

### Run the Client to get server readiness <a name="http-server-ready"></a>

- Command

```Bash
python3 ./http_server_ready.py --help
usage: http_server_ready.py [-h] [--http_address HTTP_ADDRESS] [--http_port HTTP_PORT]

Sends request via KServe HTTP API to check if server is ready.

optional arguments:
  -h, --help            show this help message and exit
  --http_address HTTP_ADDRESS
                        Specify url to HTTP service. default:localhost
  --http_port HTTP_PORT
                        Specify port to HTTP service. default: 5000

```

- Usage Example

```Bash
python3 ./http_server_ready.py --http_port 5000 --http_address localhost
Server Ready: True
```

### Run the Client to get server metadata <a name="http-server-metadata"></a>

- Command

```Bash
python3 ./http_server_metadata.py --help
usage: http_server_metadata.py [-h] [--http_address HTTP_ADDRESS] [--http_port HTTP_PORT]

Sends request via KServe HTTP API to get server metadata.

optional arguments:
  -h, --help            show this help message and exit
  --http_address HTTP_ADDRESS
                        Specify url to HTTP service. default:localhost
  --http_port HTTP_PORT
                        Specify port to HTTP service. default: 5000
```

- Usage Example

```Bash
python3 ./http_server_metadata.py --http_port 5000 --http_address localhost
TODO
```

### Run the Client to get model readiness <a name="http-model-ready"></a>

- Command

```Bash
python3 ./http_model_ready.py --help
usage: http_model_ready.py [-h] [--http_address HTTP_ADDRESS] [--http_port HTTP_PORT] [--model_name MODEL_NAME] [--model_version MODEL_VERSION]

Sends request via KServe HTTP API to check if model is ready.

optional arguments:
  -h, --help            show this help message and exit
  --http_address HTTP_ADDRESS
                        Specify url to HTTP service. default:localhost
  --http_port HTTP_PORT
                        Specify port to HTTP service. default: 5000
  --model_name MODEL_NAME
                        Define model name, must be same as is in service. default: resnet
  --model_version MODEL_VERSION
                        Define model version. If not specified, the default version will be taken from model server
```

- Usage Example

```Bash
python3 ./http_model_ready.py --http_port 5000 --http_address localhost --model_name resnet
Model Ready: True
```

### Run the Client to get model metadata <a name="http-model-metadata"></a>

- Command

```Bash
python3 ./http_model_metadata.py --help
usage: http_model_metadata.py [-h] [--http_address HTTP_ADDRESS] [--http_port HTTP_PORT] [--model_name MODEL_NAME] [--model_version MODEL_VERSION]

Sends request via KServe HTTP API to get model metadata.

optional arguments:
  -h, --help            show this help message and exit
  --http_address HTTP_ADDRESS
                        Specify url to HTTP service. default:localhost
  --http_port HTTP_PORT
                        Specify port to HTTP service. default: 5000
  --model_name MODEL_NAME
                        Define model name, must be same as is in service. default: resnet
  --model_version MODEL_VERSION
                        Define model version. If not specified, the default version will be taken from model server
```

- Usage Example

```Bash
python3 ./http_model_metadata.py --http_port 5000 --http_address localhost --model_name resnet
TODO
```

### Run the Client to perform inference <a name="http-model-infer"></a>

- Command

```Bash
python3 ./http_infer_resnet.py --help
TODO
```

- Usage Example

```Bash
TODO
```