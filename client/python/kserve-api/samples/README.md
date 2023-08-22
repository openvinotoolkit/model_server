# KServe API usage samples

OpenVINO Model Server 2022.2 release introduced support for [KServe API](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2).

This guide shows how to interact with KServe API endpoints on both gRPC and HTTP interfaces. It covers following topics:
- <a href="#grpc-api">GRPC API Examples </a>
  - <a href="#grpc-server-live">grpc_server_live.py</a>
  - <a href="#grpc-server-ready">grpc_server_ready.py</a>
  - <a href="#grpc-server-metadata">grpc_server_metadata.py</a>
  - <a href="#grpc-model-ready">grpc_model_ready.py</a>
  - <a href="#grpc-model-metadata">grpc_model_metadata.py</a>
  - <a href="#grpc-model-infer">grpc_infer_resnet.py</a>
  - <a href="#grpc-model-infer-binary">grpc_infer_binary_resnet.py</a>
  - <a href="#grpc-model-async-infer">grpc_async_infer_resnet.py</a>
- <a href="#http-api">HTTP API Example</a>
  - <a href="#http-server-live">http_server_live.py</a>
  - <a href="#http-server-ready">http_server_ready.py</a>
  - <a href="#http-server-metadata">http_server_metadata.py</a>
  - <a href="#http-model-ready">http_model_ready.py</a>
  - <a href="#http-model-metadata">http_model_metadata.py</a>
  - <a href="#http-model-infer">http_infer_resnet.py</a>
  - <a href="#http-model-infer-binary">http_infer_binary_resnet.py</a>
  - <a href="#http-model-async-infer">http_async_infer_resnet.py</a>

> **Note:** Some of the samples will use [ResNet50](https://github.com/openvinotoolkit/open_model_zoo/blob/2022.1.0/models/intel/resnet50-binary-0001/README.md).

## Before you run the samples

### Clone OpenVINO&trade; Model Server GitHub repository and enter model_server directory.
```Bash
git clone https://github.com/openvinotoolkit/model_server.git
```

### Prepare virtualenv
```Bash
cd model_server/client/python/kserve-api/samples
virtualenv .venv
. .venv/bin/activate
pip install -r requirements.txt
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

### Start the Model Server Container with Downloaded Model
Start the server container with the image pulled in the previous step and mount the `models` directory :
```Bash
docker run --rm -d -v $(pwd)/models:/models -p 9000:9000 -p 8000:8000 openvino/model_server:latest --model_name resnet --model_path /models/resnet --port 9000 --rest_port 8000 --layout NHWC:NCHW
```

> Note: The model default setting is to accept inputs in layout NCHW, but we change it to NHWC to make it work with samples using either regular, array-like input data or JPEG encoded images. 

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

### Run the Client to get server metadata <a name="grpc-server-metadata"></a>

- Command

```Bash
python3 ./grpc_server_metadata.py --help
usage: grpc_server_metadata.py [-h] [--grpc_address GRPC_ADDRESS] [--grpc_port GRPC_PORT]

Sends request via KServe gRPC API to get server metadata.

optional arguments:
  -h, --help            show this help message and exit
  --grpc_address GRPC_ADDRESS
                        Specify url to gRPC service. default:localhost
  --grpc_port GRPC_PORT
                        Specify port to gRPC service. default: 9000
```

- Usage Example

```Bash
python3 ./grpc_server_metadata.py --grpc_port 9000 --grpc_address localhost
name: "OpenVINO Model Server"
version: "2022.2.c290da85"
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
  shape: 224
  shape: 224
  shape: 3
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
python3 grpc_infer_resnet.py --help
usage: grpc_infer_resnet.py [-h] --images_numpy_path IMAGES_NUMPY_PATH [--labels_numpy_path LABELS_NUMPY_PATH] [--grpc_address GRPC_ADDRESS]
                            [--grpc_port GRPC_PORT] [--input_name INPUT_NAME] [--output_name OUTPUT_NAME] [--transpose_input {False,True}]
                            [--transpose_method {nchw2nhwc,nhwc2nchw}] [--iterations ITERATIONS] [--batchsize BATCHSIZE] [--model_name MODEL_NAME]
                            [--pipeline_name PIPELINE_NAME] [--dag-batch-size-auto] [--tls] [--server_cert SERVER_CERT] [--client_cert CLIENT_CERT] [--client_key CLIENT_KEY]

Sends requests via KServe gRPC API using images in numpy format. It displays performance statistics and optionally the model accuracy

optional arguments:
  -h, --help            show this help message and exit
  --images_numpy_path IMAGES_NUMPY_PATH
                        numpy in shape [n,w,h,c] or [n,c,h,w]
  --labels_numpy_path LABELS_NUMPY_PATH
                        numpy in shape [n,1] - can be used to check model accuracy
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT
                        Specify port to grpc service. default: 9000
  --input_name INPUT_NAME
                        Specify input tensor name. default: input
  --output_name OUTPUT_NAME
                        Specify output name. default: resnet_v1_50/predictions/Reshape_1
  --transpose_input {False,True}
                        Set to False to skip NHWC>NCHW or NCHW>NHWC input transposing. default: True
  --transpose_method {nchw2nhwc,nhwc2nchw}
                        How the input transposition should be executed: nhwc2nchw or nchw2nhwc
  --iterations ITERATIONS
                        Number of requests iterations, as default use number of images in numpy memmap. default: 0 (consume all frames)
  --batchsize BATCHSIZE
                        Number of images in a single request. default: 1
  --model_name MODEL_NAME
                        Define model name, must be same as is in service. default: resnet
  --pipeline_name PIPELINE_NAME
                        Define pipeline name, must be same as is in service
  --dag-batch-size-auto
                        Add demultiplexer dimension at front
  --tls                 use TLS communication with gRPC endpoint
  --server_cert SERVER_CERT
                        Path to server certificate
  --client_cert CLIENT_CERT
                        Path to client certificate
  --client_key CLIENT_KEY
                        Path to client key
```

- Usage Example

```Bash
python3 grpc_infer_resnet.py --grpc_port 9000 --images_numpy_path ../../imgs_nhwc.npy --labels_numpy_path ../../lbs.npy --input_name 0 --output_name 1463 --model_name resnet --transpose_input False
Image data range: 0.0 : 255.0
Start processing:
        Model name: resnet
        Iterations: 10
        Images numpy path: ../../imgs_nhwc.npy
        Numpy file shape: (10, 224, 224, 3)

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


### Run the Client to perform inference with binary encoded image <a name="grpc-model-infer-binary"></a>
- Command

```Bash
python3 grpc_infer_binary_resnet.py --help
usage: grpc_infer_binary_resnet.py [-h] [--images_list IMAGES_LIST] [--grpc_address GRPC_ADDRESS] [--grpc_port GRPC_PORT] [--input_name INPUT_NAME] [--output_name OUTPUT_NAME] [--batchsize BATCHSIZE]
                                   [--model_name MODEL_NAME] [--pipeline_name PIPELINE_NAME] [--tls]

Sends requests via KServe gRPC API using images in format supported by OpenCV. It displays performance statistics and optionally the model accuracy

optional arguments:
  -h, --help            show this help message and exit
  --images_list IMAGES_LIST
                        path to a file with a list of labeled images
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT
                        Specify port to grpc service. default: 9000
  --input_name INPUT_NAME
                        Specify input tensor name. default: input
  --output_name OUTPUT_NAME
                        Specify output name. default: resnet_v1_50/predictions/Reshape_1
  --batchsize BATCHSIZE
                        Number of images in a single request. default: 1
  --model_name MODEL_NAME
                        Define model name, must be same as is in service. default: resnet
  --pipeline_name PIPELINE_NAME
                        Define pipeline name, must be same as is in service
  --tls                 use TLS communication with GRPC endpoint
```

- Usage Example

```Bash
python3 grpc_infer_binary_resnet.py --grpc_port 9000 --images_list ../../resnet_input_images.txt --input_name 0 --output_name 1463 --model_name resnet
Start processing:
        Model name: resnet
Iteration 0; Processing time: 27.09 ms; speed 36.92 fps
imagenet top results in a single batch:
         0 airliner 404 ; Correct match.
Iteration 1; Processing time: 27.62 ms; speed 36.20 fps
imagenet top results in a single batch:
         0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Iteration 2; Processing time: 25.33 ms; speed 39.48 fps
imagenet top results in a single batch:
         0 bee 309 ; Correct match.
Iteration 3; Processing time: 23.54 ms; speed 42.47 fps
imagenet top results in a single batch:
         0 golden retriever 207 ; Correct match.
Iteration 4; Processing time: 21.62 ms; speed 46.25 fps
imagenet top results in a single batch:
         0 gorilla, Gorilla gorilla 366 ; Correct match.
Iteration 5; Processing time: 24.28 ms; speed 41.18 fps
imagenet top results in a single batch:
         0 magnetic compass 635 ; Correct match.
Iteration 6; Processing time: 23.52 ms; speed 42.51 fps
imagenet top results in a single batch:
         0 peacock 84 ; Correct match.
Iteration 7; Processing time: 24.10 ms; speed 41.49 fps
imagenet top results in a single batch:
         0 pelican 144 ; Correct match.
Iteration 8; Processing time: 25.36 ms; speed 39.44 fps
imagenet top results in a single batch:
         0 snail 113 ; Correct match.
Iteration 9; Processing time: 32.67 ms; speed 30.61 fps
imagenet top results in a single batch:
         0 zebra 340 ; Correct match.

processing time for all iterations
average time: 25.10 ms; average speed: 39.84 fps
median time: 24.50 ms; median speed: 40.82 fps
max time: 32.00 ms; min speed: 31.25 fps
min time: 21.00 ms; max speed: 47.62 fps
time percentile 90: 27.50 ms; speed percentile 90: 36.36 fps
time percentile 50: 24.50 ms; speed percentile 50: 40.82 fps
time standard deviation: 2.88
time variance: 8.29
Classification accuracy: 100.00
```


### Run the Client to perform asynchronous inference <a name="grpc-model-async-infer"></a>

- Command

```Bash
python3 grpc_async_infer_resnet.py --help
usage: grpc_async_infer_resnet.py [-h] --images_numpy_path IMAGES_NUMPY_PATH [--labels_numpy_path LABELS_NUMPY_PATH] [--grpc_address GRPC_ADDRESS] [--grpc_port GRPC_PORT] [--input_name INPUT_NAME]
                                  [--output_name OUTPUT_NAME] [--transpose_input {False,True}] [--transpose_method {nchw2nhwc,nhwc2nchw}] [--iterations ITERATIONS] [--batchsize BATCHSIZE] [--model_name MODEL_NAME]
                                  [--pipeline_name PIPELINE_NAME] [--dag-batch-size-auto] [--tls] [--server_cert SERVER_CERT] [--client_cert CLIENT_CERT] [--client_key CLIENT_KEY] [--timeout TIMEOUT]

Sends requests via KServe gRPC API using images in numpy format. It displays performance statistics and optionally the model accuracy

optional arguments:
  -h, --help            show this help message and exit
  --images_numpy_path IMAGES_NUMPY_PATH
                        numpy in shape [n,w,h,c] or [n,c,h,w]
  --labels_numpy_path LABELS_NUMPY_PATH
                        numpy in shape [n,1] - can be used to check model accuracy
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT
                        Specify port to grpc service. default: 9000
  --input_name INPUT_NAME
                        Specify input tensor name. default: input
  --output_name OUTPUT_NAME
                        Specify output name. default: resnet_v1_50/predictions/Reshape_1
  --transpose_input {False,True}
                        Set to False to skip NHWC>NCHW or NCHW>NHWC input transposing. default: True
  --transpose_method {nchw2nhwc,nhwc2nchw}
                        How the input transposition should be executed: nhwc2nchw or nchw2nhwc
  --iterations ITERATIONS
                        Number of requests iterations, as default use number of images in numpy memmap. default: 0 (consume all frames)
  --batchsize BATCHSIZE
                        Number of images in a single request. default: 1
  --model_name MODEL_NAME
                        Define model name, must be same as is in service. default: resnet
  --pipeline_name PIPELINE_NAME
                        Define pipeline name, must be same as is in service
  --dag-batch-size-auto
                        Add demultiplexer dimension at front
  --tls                 use TLS communication with gRPC endpoint
  --server_cert SERVER_CERT
                        Path to server certificate
  --client_cert CLIENT_CERT
                        Path to client certificate
  --client_key CLIENT_KEY
                        Path to client key
  --timeout TIMEOUT     Request timeout
```

- Usage Example

```Bash
python3 grpc_async_infer_resnet.py --grpc_port 9000 --images_numpy_path ../../imgs_nhwc.npy --labels_numpy_path ../../lbs.npy --input_name 0 --output_name 1463 --transpose_input False --model_name resnet
Image data range: 0.0 : 255.0
Start processing:
        Model name: resnet
        Iterations: 10
        Images numpy path: ../../imgs_nhwc.npy
        Numpy file shape: (10, 3, 224, 224)

imagenet top results in a single batch:
         0 peacock 84 ; Correct match.
imagenet top results in a single batch:
         0 gorilla, Gorilla gorilla 366 ; Correct match.
imagenet top results in a single batch:
         0 airliner 404 ; Correct match.
imagenet top results in a single batch:
         0 snail 113 ; Correct match.
imagenet top results in a single batch:
         0 bee 309 ; Correct match.
imagenet top results in a single batch:
         0 golden retriever 207 ; Correct match.
imagenet top results in a single batch:
         0 magnetic compass 635 ; Correct match.
imagenet top results in a single batch:
         0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
imagenet top results in a single batch:
         0 pelican 144 ; Correct match.
imagenet top results in a single batch:
         0 zebra 340 ; Correct match.
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
                        Specify port to HTTP service. default: 8000
```

- Usage Example

```Bash
python3 ./http_server_live.py --http_port 8000 --http_address localhost
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
                        Specify port to HTTP service. default: 8000

```

- Usage Example

```Bash
python3 ./http_server_ready.py --http_port 8000 --http_address localhost
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
                        Specify port to HTTP service. default: 8000
```

- Usage Example

```Bash
python3 ./http_server_metadata.py --http_port 8000 --http_address localhost
{'name': 'OpenVINO Model Server', 'version': '2022.2.c290da85'}
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
                        Specify port to HTTP service. default: 8000
  --model_name MODEL_NAME
                        Define model name, must be same as is in service. default: resnet
  --model_version MODEL_VERSION
                        Define model version. If not specified, the default version will be taken from model server
```

- Usage Example

```Bash
python3 ./http_model_ready.py --http_port 8000 --http_address localhost --model_name resnet
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
                        Specify port to HTTP service. default: 8000
  --model_name MODEL_NAME
                        Define model name, must be same as is in service. default: resnet
  --model_version MODEL_VERSION
                        Define model version. If not specified, the default version will be taken from model server
```

- Usage Example

```Bash
python3 ./http_model_metadata.py --http_port 8000 --http_address localhost --model_name resnet
{'name': 'resnet', 'versions': ['1'], 'platform': 'OpenVINO', 'inputs': [{'name': '0', 'datatype': 'FP32', 'shape': [1, 224, 224, 3]}], 'outputs': [{'name': '1463', 'datatype': 'FP32', 'shape': [1, 1000]}]}
```

### Run the Client to perform inference <a name="http-model-infer"></a>

- Command

```Bash
python3 ./http_infer_resnet.py --help
usage: http_infer_resnet.py [-h] --images_numpy_path IMAGES_NUMPY_PATH [--labels_numpy_path LABELS_NUMPY_PATH] [--http_address HTTP_ADDRESS]
                            [--http_port HTTP_PORT] [--input_name INPUT_NAME] [--output_name OUTPUT_NAME] [--transpose_input {False,True}]
                            [--transpose_method {nchw2nhwc,nhwc2nchw}] [--iterations ITERATIONS] [--batchsize BATCHSIZE] [--model_name MODEL_NAME]
                            [--pipeline_name PIPELINE_NAME] [--dag-batch-size-auto] [--binary_data] [--tls] [--server_cert SERVER_CERT] [--client_cert CLIENT_CERT] [--client_key CLIENT_KEY]

Sends requests via KServe REST API using images in numpy format. It displays performance statistics and optionally the model accuracy

optional arguments:
  -h, --help            show this help message and exit
  --images_numpy_path IMAGES_NUMPY_PATH
                        numpy in shape [n,w,h,c] or [n,c,h,w]
  --labels_numpy_path LABELS_NUMPY_PATH
                        numpy in shape [n,1] - can be used to check model accuracy
  --http_address HTTP_ADDRESS
                        Specify url to http service. default:localhost
  --http_port HTTP_PORT
                        Specify port to http service. default: 8000
  --input_name INPUT_NAME
                        Specify input tensor name. default: input
  --output_name OUTPUT_NAME
                        Specify output name. default: resnet_v1_50/predictions/Reshape_1
  --transpose_input {False,True}
                        Set to False to skip NHWC>NCHW or NCHW>NHWC input transposing. default: True
  --transpose_method {nchw2nhwc,nhwc2nchw}
                        How the input transposition should be executed: nhwc2nchw or nchw2nhwc
  --iterations ITERATIONS
                        Number of requests iterations, as default use number of images in numpy memmap. default: 0 (consume all frames)
  --batchsize BATCHSIZE
                        Number of images in a single request. default: 1
  --model_name MODEL_NAME
                        Define model name, must be same as is in service. default: resnet
  --pipeline_name PIPELINE_NAME
                        Define pipeline name, must be same as is in service
  --dag-batch-size-auto
                        Add demultiplexer dimension at front
  --binary_data         Send input data in binary format
  --tls                 use TLS communication with gRPC endpoint
  --server_cert SERVER_CERT
                        Path to server certificate
  --client_cert CLIENT_CERT
                        Path to client certificate
  --client_key CLIENT_KEY
                        Path to client key
```

- Usage Example #1 - Input data placed in JSON object.

```Bash
python3 ./http_infer_resnet.py --http_port 8000 --images_numpy_path ../../imgs_nhwc.npy --labels_numpy_path ../../lbs.npy --input_name 0 --output_name 1463 --model_name resnet --transpose_input False
Image data range: 0.0 : 255.0
Start processing:
        Model name: resnet
        Iterations: 10
        Images numpy path: ../../imgs_nhwc.npy
        Numpy file shape: (10, 224, 224, 3)

Iteration 1; Processing time: 206.35 ms; speed 4.85 fps
imagenet top results in a single batch:
         0 airliner 404 ; Correct match.
Iteration 2; Processing time: 196.61 ms; speed 5.09 fps
imagenet top results in a single batch:
         0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Iteration 3; Processing time: 184.16 ms; speed 5.43 fps
imagenet top results in a single batch:
         0 bee 309 ; Correct match.
Iteration 4; Processing time: 189.03 ms; speed 5.29 fps
imagenet top results in a single batch:
         0 golden retriever 207 ; Correct match.
Iteration 5; Processing time: 188.47 ms; speed 5.31 fps
imagenet top results in a single batch:
         0 gorilla, Gorilla gorilla 366 ; Correct match.
Iteration 6; Processing time: 192.66 ms; speed 5.19 fps
imagenet top results in a single batch:
         0 magnetic compass 635 ; Correct match.
Iteration 7; Processing time: 175.03 ms; speed 5.71 fps
imagenet top results in a single batch:
         0 peacock 84 ; Correct match.
Iteration 8; Processing time: 192.43 ms; speed 5.20 fps
imagenet top results in a single batch:
         0 pelican 144 ; Correct match.
Iteration 9; Processing time: 190.80 ms; speed 5.24 fps
imagenet top results in a single batch:
         0 snail 113 ; Correct match.
Iteration 10; Processing time: 210.31 ms; speed 4.75 fps
imagenet top results in a single batch:
         0 zebra 340 ; Correct match.

processing time for all iterations
average time: 192.20 ms; average speed: 5.20 fps
median time: 191.00 ms; median speed: 5.24 fps
max time: 210.00 ms; min speed: 4.76 fps
min time: 175.00 ms; max speed: 5.71 fps
time percentile 90: 206.40 ms; speed percentile 90: 4.84 fps
time percentile 50: 191.00 ms; speed percentile 50: 5.24 fps
time standard deviation: 9.58
time variance: 91.76
Classification accuracy: 100.00
```

- Usage Example #2 - Input data placed as binary, outside JSON object.

```Bash
python3 ./http_infer_resnet.py --http_port 8000 --images_numpy_path ../../imgs_nhwc.npy --labels_numpy_path ../../lbs.npy --input_name 0 --output_name 1463 --model_name resnet --transpose_input False --binary_data
Image data range: 0.0 : 255.0
Start processing:
        Model name: resnet
        Iterations: 10
        Images numpy path: ../../imgs_nhwc.npy
        Numpy file shape: (10, 224, 224, 3)

Iteration 1; Processing time: 36.58 ms; speed 27.34 fps
imagenet top results in a single batch:
         0 airliner 404 ; Correct match.
Iteration 2; Processing time: 33.76 ms; speed 29.62 fps
imagenet top results in a single batch:
         0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Iteration 3; Processing time: 28.55 ms; speed 35.03 fps
imagenet top results in a single batch:
         0 bee 309 ; Correct match.
Iteration 4; Processing time: 28.27 ms; speed 35.37 fps
imagenet top results in a single batch:
         0 golden retriever 207 ; Correct match.
Iteration 5; Processing time: 28.83 ms; speed 34.69 fps
imagenet top results in a single batch:
         0 gorilla, Gorilla gorilla 366 ; Correct match.
Iteration 6; Processing time: 26.80 ms; speed 37.31 fps
imagenet top results in a single batch:
         0 magnetic compass 635 ; Correct match.
Iteration 7; Processing time: 27.20 ms; speed 36.76 fps
imagenet top results in a single batch:
         0 peacock 84 ; Correct match.
Iteration 8; Processing time: 26.46 ms; speed 37.80 fps
imagenet top results in a single batch:
         0 pelican 144 ; Correct match.
Iteration 9; Processing time: 29.52 ms; speed 33.87 fps
imagenet top results in a single batch:
         0 snail 113 ; Correct match.
Iteration 10; Processing time: 27.49 ms; speed 36.37 fps
imagenet top results in a single batch:
         0 zebra 340 ; Correct match.

processing time for all iterations
average time: 28.80 ms; average speed: 34.72 fps
median time: 28.00 ms; median speed: 35.71 fps
max time: 36.00 ms; min speed: 27.78 fps
min time: 26.00 ms; max speed: 38.46 fps
time percentile 90: 33.30 ms; speed percentile 90: 30.03 fps
time percentile 50: 28.00 ms; speed percentile 50: 35.71 fps
time standard deviation: 3.06
time variance: 9.36
Classification accuracy: 100.00
```


### Run the Client to perform inference with binary encoded image <a name="http-model-infer-binary"></a>
- Command

```Bash
python3 ./http_infer_binary_resnet.py --help
usage: http_infer_binary_resnet.py [-h] [--images_list IMAGES_LIST] [--http_address HTTP_ADDRESS] [--http_port HTTP_PORT] [--input_name INPUT_NAME] [--output_name OUTPUT_NAME] [--batchsize BATCHSIZE]
                                   [--model_name MODEL_NAME] [--pipeline_name PIPELINE_NAME] [--tls] [--server_cert SERVER_CERT] [--client_cert CLIENT_CERT] [--client_key CLIENT_KEY]

Sends requests via KServe REST API using binary encoded images. It displays performance statistics and optionally the model accuracy

optional arguments:
  -h, --help            show this help message and exit
  --images_list IMAGES_LIST
                        path to a file with a list of labeled images
  --http_address HTTP_ADDRESS
                        Specify url to http service. default:localhost
  --http_port HTTP_PORT
                        Specify port to http service. default: 8000
  --input_name INPUT_NAME
                        Specify input tensor name. default: input
  --output_name OUTPUT_NAME
                        Specify output name. default: resnet_v1_50/predictions/Reshape_1
  --batchsize BATCHSIZE
                        Number of images in a single request. default: 1
  --model_name MODEL_NAME
                        Define model name, must be same as is in service. default: resnet
  --pipeline_name PIPELINE_NAME
                        Define pipeline name, must be same as is in service
  --tls                 use TLS communication with HTTP endpoint
  --server_cert SERVER_CERT
                        Path to server certificate
  --client_cert CLIENT_CERT
                        Path to client certificate
  --client_key CLIENT_KEY
                        Path to client key
```

- Usage Example

```Bash
python3 ./http_infer_binary_resnet.py --http_port 8000 --images_list ../../resnet_input_images.txt --input_name 0 --output_name 1463 --model_name resnet
Start processing:
        Model name: resnet
Iteration 0; Processing time: 38.61 ms; speed 25.90 fps
imagenet top results in a single batch:
         0 airliner 404 ; Correct match.
Iteration 1; Processing time: 44.28 ms; speed 22.58 fps
imagenet top results in a single batch:
         0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Iteration 2; Processing time: 30.81 ms; speed 32.45 fps
imagenet top results in a single batch:
         0 bee 309 ; Correct match.
Iteration 3; Processing time: 31.36 ms; speed 31.89 fps
imagenet top results in a single batch:
         0 golden retriever 207 ; Correct match.
Iteration 4; Processing time: 30.21 ms; speed 33.10 fps
imagenet top results in a single batch:
         0 gorilla, Gorilla gorilla 366 ; Correct match.
Iteration 5; Processing time: 32.92 ms; speed 30.37 fps
imagenet top results in a single batch:
         0 magnetic compass 635 ; Correct match.
Iteration 6; Processing time: 36.39 ms; speed 27.48 fps
imagenet top results in a single batch:
         0 peacock 84 ; Correct match.
Iteration 7; Processing time: 33.83 ms; speed 29.56 fps
imagenet top results in a single batch:
         0 pelican 144 ; Correct match.
Iteration 8; Processing time: 32.22 ms; speed 31.03 fps
imagenet top results in a single batch:
         0 snail 113 ; Correct match.
Iteration 9; Processing time: 46.04 ms; speed 21.72 fps
imagenet top results in a single batch:
         0 zebra 340 ; Correct match.

processing time for all iterations
average time: 35.20 ms; average speed: 28.41 fps
median time: 32.50 ms; median speed: 30.77 fps
max time: 46.00 ms; min speed: 21.74 fps
min time: 30.00 ms; max speed: 33.33 fps
time percentile 90: 44.20 ms; speed percentile 90: 22.62 fps
time percentile 50: 32.50 ms; speed percentile 50: 30.77 fps
time standard deviation: 5.47
time variance: 29.96
Classification accuracy: 100.00
```

### Run the Client to perform asynchronous inference <a name="http-model-async-infer"></a>

- Command

```Bash
 python3 http_async_infer_resnet.py --help
usage: http_async_infer_resnet.py [-h] --images_numpy_path IMAGES_NUMPY_PATH [--labels_numpy_path LABELS_NUMPY_PATH] [--http_address HTTP_ADDRESS] [--http_port HTTP_PORT] [--input_name INPUT_NAME]
                                  [--output_name OUTPUT_NAME] [--transpose_input {False,True}] [--transpose_method {nchw2nhwc,nhwc2nchw}] [--iterations ITERATIONS] [--batchsize BATCHSIZE] [--model_name MODEL_NAME]
                                  [--pipeline_name PIPELINE_NAME] [--dag-batch-size-auto] [--binary_data] [--tls] [--server_cert SERVER_CERT] [--client_cert CLIENT_CERT] [--client_key CLIENT_KEY]

Sends requests via KServe REST API using images in numpy format. It displays performance statistics and optionally the model accuracy

optional arguments:
  -h, --help            show this help message and exit
  --images_numpy_path IMAGES_NUMPY_PATH
                        numpy in shape [n,w,h,c] or [n,c,h,w]
  --labels_numpy_path LABELS_NUMPY_PATH
                        numpy in shape [n,1] - can be used to check model accuracy
  --http_address HTTP_ADDRESS
                        Specify url to http service. default:localhost
  --http_port HTTP_PORT
                        Specify port to http service. default: 8000
  --input_name INPUT_NAME
                        Specify input tensor name. default: input
  --output_name OUTPUT_NAME
                        Specify output name. default: resnet_v1_50/predictions/Reshape_1
  --transpose_input {False,True}
                        Set to False to skip NHWC>NCHW or NCHW>NHWC input transposing. default: True
  --transpose_method {nchw2nhwc,nhwc2nchw}
                        How the input transposition should be executed: nhwc2nchw or nchw2nhwc
  --iterations ITERATIONS
                        Number of requests iterations, as default use number of images in numpy memmap. default: 0 (consume all frames)
  --batchsize BATCHSIZE
                        Number of images in a single request. default: 1
  --model_name MODEL_NAME
                        Define model name, must be same as is in service. default: resnet
  --pipeline_name PIPELINE_NAME
                        Define pipeline name, must be same as is in service
  --dag-batch-size-auto
                        Add demultiplexer dimension at front
  --binary_data         Send input data in binary format
  --tls                 use TLS communication with gRPC endpoint
  --server_cert SERVER_CERT
                        Path to server certificate
  --client_cert CLIENT_CERT
                        Path to client certificate
  --client_key CLIENT_KEY
                        Path to client key
```

- Usage Example

```Bash
python3 http_async_infer_resnet.py --http_port 8000 --images_numpy_path ../../imgs_nhwc.npy --labels_numpy_path ../../lbs.npy --input_name 0 --output_name 1463 --transpose_input False --model_name resnet
Image data range: 0.0 : 255.0
Start processing:
        Model name: resnet
        Iterations: 10
        Images numpy path: ../../imgs_nhwc.npy
        Numpy file shape: (10, 3, 224, 224)

imagenet top results in a single batch:
         0 airliner 404 ; Correct match.
imagenet top results in a single batch:
         0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
imagenet top results in a single batch:
         0 bee 309 ; Correct match.
imagenet top results in a single batch:
         0 golden retriever 207 ; Correct match.
imagenet top results in a single batch:
         0 gorilla, Gorilla gorilla 366 ; Correct match.
imagenet top results in a single batch:
         0 magnetic compass 635 ; Correct match.
imagenet top results in a single batch:
         0 peacock 84 ; Correct match.
imagenet top results in a single batch:
         0 pelican 144 ; Correct match.
imagenet top results in a single batch:
         0 snail 113 ; Correct match.
imagenet top results in a single batch:
         0 zebra 340 ; Correct match.
Classification accuracy: 100.00
```