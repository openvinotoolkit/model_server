# KServe API usage samples

OpenVINO Model Server introduced support for [KServe API](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2), including [Triton](https://github.com/triton-inference-server)'s raw format extension.

This guide shows how to interact with KServe API endpoints on both gRPC and HTTP interfaces using [Triton](https://github.com/triton-inference-server)'s client library. It covers following topics:
- <a href="#grpc-api">GRPC API Examples </a>
  - <a href="#grpc-server-live">grpc_server_live</a>
  - <a href="#grpc-server-ready">grpc_server_ready</a>
  - <a href="#grpc-server-metadata">grpc_server_metadata</a>
  - <a href="#grpc-model-ready">grpc_model_ready</a>
  - <a href="#grpc-model-metadata">grpc_model_metadata</a>
  - <a href="#grpc-model-infer">grpc_infer_dummy</a>
  - <a href="#grpc-model-infer-resnet">grpc_infer_resnet</a>
  - <a href="#grpc-model-async-infer-resnet">grpc_async_infer_resnet</a>
- <a href="#http-api">HTTP API Example</a>
  - <a href="#http-server-live">http_server_live</a>
  - <a href="#http-server-ready">http_server_ready</a>
  - <a href="#http-server-metadata">http_server_metadata</a>
  - <a href="#http-model-ready">http_model_ready</a>
  - <a href="#http-model-metadata">http_model_metadata</a>
  - <a href="#http-model-infer">http_infer_dummy</a>
  - <a href="#http-model-infer-resnet">http_infer_resnet</a>
  - <a href="#http-model-async-infer-resnet">http_async_infer_resnet</a>

## Before you run the samples

### Clone OpenVINO&trade; Model Server GitHub repository and go to the top directory.
```Bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
```

### Start the Model Server Container with Dummy Model
```Bash
docker run --rm -d -v $(pwd)/src/test/dummy:/models -p 9000:9000 -p 8000:8000 openvino/model_server:latest --model_name dummy --model_path /models --port 9000 --rest_port 8000 
```

### Build client library and samples
```Bash
cd client/cpp/kserve-api
cmake . && make
cd samples
```

## GRPC Examples <a name="grpc-api"></a>


## GRPC Examples with Dummy Model

This section demonstrates inference on a simple model, which increments each provided value. 

### Run the Client to get server liveness <a name="grpc-server-live"></a>

- Command

```Bash
./grpc_server_live --help
Sends requests via KServe gRPC API to check if server is alive.
Usage:
  grpc_server_live [OPTION...]

  -h, --help                    Show this help message and exit
      --grpc_address GRPC_ADDRESS
                                Specify url to grpc service.  (default: 
                                localhost)
      --grpc_port PORT          Specify port to grpc service.  (default: 
                                9000)
      --timeout TIMEOUT         Request timeout. (default: 0)
```

- Usage Example 

```Bash
./grpc_server_live --grpc_port 9000 --grpc_address localhost
Server Live: True
```

### Run the Client to get server readiness <a name="grpc-server-ready"></a>

- Command

```Bash
./grpc_server_ready --help
Sends requests via KServe gRPC API to check if server is ready.
Usage:
  grpc_server_ready [OPTION...]

  -h, --help                    Show this help message and exit
      --grpc_address GRPC_ADDRESS
                                Specify url to grpc service.  (default: 
                                localhost)
      --grpc_port PORT          Specify port to grpc service.  (default: 
                                9000)
      --timeout TIMEOUT         Request timeout. (default: 0)
```

- Usage Example

```Bash
./grpc_server_ready --grpc_port 9000 --grpc_address localhost
Server Ready: True
```

### Run the Client to get server metadata <a name="grpc-server-metadata"></a>

- Command

```Bash
./grpc_server_metadata --help
Sends requests via KServe gRPC API to get server metadata.
Usage:
  grpc_server_metadata [OPTION...]

  -h, --help                    Show this help message and exit
      --grpc_address GRPC_ADDRESS
                                Specify url to grpc service.  (default: 
                                localhost)
      --grpc_port PORT          Specify port to grpc service.  (default: 
                                9000)
      --timeout TIMEOUT         Request timeout. (default: 0)
```

- Usage Example

```Bash
./grpc_server_metadata --grpc_port 9000 --grpc_address localhost
Name: OpenVINO Model Server
Version: 2022.2.c290da85
```

### Run the Client to get model readiness <a name="grpc-model-ready"></a>

- Command

```Bash
./grpc_model_ready --help
Sends requests via KServe gRPC API to check if model is ready for inference.
Usage:
  grpc_model_ready [OPTION...]

  -h, --help                    Show this help message and exit
      --grpc_address GRPC_ADDRESS
                                Specify url to grpc service.  (default: 
                                localhost)
      --grpc_port PORT          Specify port to grpc service.  (default: 
                                9000)
      --model_name MODEL_NAME   Define model name, must be same as is in 
                                service.  (default: dummy)
      --model_version MODEL_VERSION
                                Define model version. (default: "")
      --timeout TIMEOUT         Request timeout. (default: 0)
```

- Usage Example

```Bash
./grpc_model_ready --grpc_port 9000 --grpc_address localhost --model_name dummy
Model Ready: True
```

### Run the Client to get metadata <a name="grpc-model-metadata"></a>

- Command

```Bash
./grpc_model_metadata --help
Sends requests via KServe gRPC API to get model metadata.
Usage:
  grpc_ready [OPTION...]

  -h, --help                    Show this help message and exit
      --grpc_address GRPC_ADDRESS
                                Specify url to grpc service.  (default: 
                                localhost)
      --grpc_port PORT          Specify port to grpc service.  (default: 
                                9000)
      --model_name MODEL_NAME   Define model name, must be same as is in 
                                service.  (default: dummy)
      --model_version MODEL_VERSION
                                Define model version. (default: "")
      --timeout TIMEOUT         Request timeout. (default: 0)
```

- Usage Example

```Bash
./grpc_model_metadata --grpc_port 9000 --grpc_address localhost --model_name dummy
name: "dummy"
versions: "1"
platform: "OpenVINO"
inputs {
  name: "b"
  datatype: "FP32"
  shape: 1
  shape: 10
}
outputs {
  name: "a"
  datatype: "FP32"
  shape: 1
  shape: 10
}
```
### Run the Client to perform inference <a name="grpc-model-infer"></a>

```Bash
./grpc_infer_dummy --help
Sends requests via KServe gRPC API.
Usage:
  grpc_infer_dummy [OPTION...]

  -h, --help                    Show this help message and exit
      --grpc_address GRPC_ADDRESS
                                Specify url to grpc service.  (default: 
                                localhost)
      --grpc_port PORT          Specify port to grpc service.  (default: 
                                9000)
      --input_name INPUT_NAME   Specify input tensor name.  (default: b)
      --output_name OUTPUT_NAME
                                Specify input tensor name.  (default: a)
      --model_name MODEL_NAME   Define model name, must be same as is in 
                                service.  (default: dummy)
      --model_version MODEL_VERSION
                                Define model version.
      --timeout TIMEOUT         Request timeout. (default: 0)
```

- Usage Example

```Bash
./grpc_infer_dummy --grpc_port 9000 --model_name dummy
0 => 1
1 => 2
2 => 3
3 => 4
4 => 5
5 => 6
6 => 7
7 => 8
8 => 9
9 => 10
======Client Statistics======
Number of requests: 1
Total processing time: 5.28986 ms
Latency: 5.28986 ms
Requests per second: 189.041
```

## HTTP Examples <a name="http-api"></a>

### Run the Client to get server liveness <a name="http-server-live"></a>

- Command

```Bash
./http_server_live --help
Sends requests via KServe REST API to check if server is alive.
Usage:
  http_server_live [OPTION...]

  -h, --help                    Show this help message and exit
      --http_address HTTP_ADDRESS
                                Specify url to REST service.  (default: 
                                localhost)
      --http_port PORT          Specify port to REST service.  (default: 
                                8000)
      --timeout TIMEOUT         Request timeout. (default: 0)
```

- Usage Example

```Bash
./http_server_live --http_port 8000 --http_address localhost
Server Live: True
```

### Run the Client to get server readiness <a name="http-server-ready"></a>

- Command

```Bash
./http_server_ready --help
Sends requests via KServe REST API to check if server is ready.
Usage:
  http_server_ready [OPTION...]

  -h, --help                    Show this help message and exit
      --http_address HTTP_ADDRESS
                                Specify url to REST service.  (default: 
                                localhost)
      --http_port PORT          Specify port to REST service.  (default: 
                                8000)
      --timeout TIMEOUT         Request timeout. (default: 0)
```

- Usage Example

```Bash
./http_server_ready --http_port 8000 --http_address localhost
Server Ready: True
```

### Run the Client to get server metadata <a name="http-server-metadata"></a>

- Command

```Bash
./http_server_metadata --help
Sends requests via KServe REST API to get server metadata.
Usage:
  http_server_metadata [OPTION...]

  -h, --help                    Show this help message and exit
      --http_address HTTP_ADDRESS
                                Specify url to REST service.  (default: 
                                localhost)
      --http_port PORT          Specify port to REST service.  (default: 
                                8000)
      --timeout TIMEOUT         Request timeout. (default: 0)
```

- Usage Example

```Bash
./http_server_metadata --http_port 8000 --http_address localhost
{"name":"OpenVINO Model Server","version":"2022.2.c290da85"}
```

### Run the Client to get model readiness <a name="http-model-ready"></a>

- Command

```Bash
./http_model_ready --help
Sends requests via KServe REST API to check if model is ready for inference.
Usage:
  http_model_ready [OPTION...]

  -h, --help                    Show this help message and exit
      --http_address HTTP_ADDRESS
                                Specify url to REST service.  (default: 
                                localhost)
      --http_port PORT          Specify port to REST service.  (default: 
                                8000)
      --model_name MODEL_NAME   Define model name, must be same as is in 
                                service.  (default: dummy)
      --model_version MODEL_VERSION
                                Define model version. (default: "")
      --timeout TIMEOUT         Request timeout. (default: 0)
```

- Usage Example

```Bash
./http_model_ready --http_port 8000 --http_address localhost
Model Ready: True
```

### Run the Client to get model metadata <a name="http-model-metadata"></a>

- Command

```Bash
./http_model_metadata --help
Sends requests via KServe REST API to get model metadata.
Usage:
  http_ready [OPTION...]

  -h, --help                    Show this help message and exit
      --http_address HTTP_ADDRESS
                                Specify url to REST service.  (default: 
                                localhost)
      --http_port PORT          Specify port to REST service.  (default: 
                                8000)
      --model_name MODEL_NAME   Define model name, must be same as is in 
                                service.  (default: dummy)
      --model_version MODEL_VERSION
                                Define model version. (default: "")
      --timeout TIMEOUT         Request timeout. (default: 0)
```

- Usage Example

```Bash
./http_model_metadata --http_port 8000 --http_address localhost
{"name":"dummy","versions":["1"],"platform":"OpenVINO","inputs":[{"name":"b","datatype":"FP32","shape":[1,10]}],"outputs":[{"name":"a","datatype":"FP32","shape":[1,10]}]}
```
### Run the Client to perform inference <a name="http-model-infer"></a>

- Command

```Bash
./http_infer_dummy --help
Sends requests via KServe REST API.
Usage:
  http_infer_dummy [OPTION...]

  -h, --help                    Show this help message and exit
      --http_address HTTP_ADDRESS
                                Specify url to REST service.  (default: 
                                localhost)
      --http_port PORT          Specify port to REST service.  (default: 
                                8000)
      --input_name INPUT_NAME   Specify input tensor name.  (default: b)
      --output_name OUTPUT_NAME
                                Specify input tensor name.  (default: a)
      --model_name MODEL_NAME   Define model name, must be same as is in 
                                service.  (default: dummy)
      --model_version MODEL_VERSION
                                Define model version.
      --timeout TIMEOUT         Request timeout. (default: 0)
```

- Usage Example

```Bash
./http_infer_dummy --http_port 8000
0 => 1
1 => 2
2 => 3
3 => 4
4 => 5
5 => 6
6 => 7
7 => 8
8 => 9
9 => 10
======Client Statistics======
Number of requests: 1
Total processing time: 2.18683 ms
Latency: 2.18683 ms
Requests per second: 457.283
```

## Examples with Resnet Model

### Download the Pretrained Model
Download the model files and store them in the `models` directory
```Bash
mkdir -p models/resnet/1
curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml -o models/resnet/1/resnet50-binary-0001.bin -o models/resnet/1/resnet50-binary-0001.xml
```

### Start the Model Server Container with Resnet Model
```Bash
docker run --rm -d -v $(pwd)/models:/models -p 9000:9000 -p 8000:8000 openvino/model_server:latest --model_name resnet --model_path /models/resnet --port 9000 --rest_port 8000 --layout NHWC:NCHW --plugin_config '{"PERFORMANCE_HINT":"LATENCY"}'
```

Once you finish above steps, you are ready to run the samples.

### Run the Client to perform inference using gRPC API <a name="grpc-model-infer-resnet"></a>
```Bash
./grpc_infer_resnet --help
Sends requests via KServe gRPC API.
Usage:
  grpc_infer_resnet [OPTION...]

  -h, --help                    Show this help message and exit
      --images_list IMAGES      Path to a file with a list of labeled 
                                images. 
      --labels_list LABELS      Path to a file with a list of labels. 
      --grpc_address GRPC_ADDRESS
                                Specify url to grpc service.  (default: 
                                localhost)
      --grpc_port PORT          Specify port to grpc service.  (default: 
                                9000)
      --input_name INPUT_NAME   Specify input tensor name.  (default: 0)
      --output_name OUTPUT_NAME
                                Specify input tensor name.  (default: 1463)
      --model_name MODEL_NAME   Define model name, must be same as is in 
                                service.  (default: resnet)
      --model_version MODEL_VERSION
                                Define model version.
      --timeout TIMEOUT         Request timeout. (default: 0)
```

- Usage Example

```Bash
./grpc_infer_resnet --images_list ../../../common/resnet_input_images.txt --labels_list ../../../common/resnet_labels.txt --grpc_port 9000  
../../../../demos/common/static/images/airliner.jpeg classified as 404 airliner 
../../../../demos/common/static/images/arctic-fox.jpeg classified as 279 Arctic fox, white fox, Alopex lagopus 
../../../../demos/common/static/images/bee.jpeg classified as 309 bee 
../../../../demos/common/static/images/golden_retriever.jpeg classified as 207 golden retriever 
../../../../demos/common/static/images/gorilla.jpeg classified as 366 gorilla, Gorilla gorilla 
../../../../demos/common/static/images/magnetic_compass.jpeg classified as 635 magnetic compass 
../../../../demos/common/static/images/peacock.jpeg classified as 84 peacock 
../../../../demos/common/static/images/pelican.jpeg classified as 144 pelican 
../../../../demos/common/static/images/snail.jpeg classified as 113 snail 
../../../../demos/common/static/images/zebra.jpeg classified as 340 zebra 
Accuracy 100%
======Client Statistics======
Number of requests: 10
Total processing time: 96.651 ms
Latency: 9.6651 ms
Requests per second: 103.465
```

### Run the Client to perform asynchronous inference using gRPC API <a name="grpc-model-async-infer-resnet"></a>
```Bash
./grpc_async_infer_resnet --help
Sends requests via KServe gRPC API.
Usage:
  grpc_async_infer_resnet [OPTION...]

  -h, --help                    Show this help message and exit
      --images_list IMAGES      Path to a file with a list of labeled 
                                images. 
      --labels_list LABELS      Path to a file with a list of labels. 
      --grpc_address GRPC_ADDRESS
                                Specify url to grpc service.  (default: 
                                localhost)
      --grpc_port PORT          Specify port to grpc service.  (default: 
                                9000)
      --input_name INPUT_NAME   Specify input tensor name.  (default: 0)
      --output_name OUTPUT_NAME
                                Specify input tensor name.  (default: 1463)
      --model_name MODEL_NAME   Define model name, must be same as is in 
                                service.  (default: resnet)
      --model_version MODEL_VERSION
                                Define model version.
      --timeout TIMEOUT         Request timeout. (default: 0)
```

- Usage Example

```Bash
./grpc_async_infer_resnet --images_list ../../../common/resnet_input_images.txt --labels_list ../../../common/resnet_labels.txt --grpc_port 9000
../../../../demos/common/static/images/airliner.jpeg classified as 404 airliner 
../../../../demos/common/static/images/bee.jpeg classified as 309 bee 
../../../../demos/common/static/images/gorilla.jpeg classified as 366 gorilla, Gorilla gorilla 
../../../../demos/common/static/images/magnetic_compass.jpeg classified as 635 magnetic compass 
../../../../demos/common/static/images/peacock.jpeg classified as 84 peacock 
../../../../demos/common/static/images/snail.jpeg classified as 113 snail 
../../../../demos/common/static/images/arctic-fox.jpeg classified as 279 Arctic fox, white fox, Alopex lagopus 
../../../../demos/common/static/images/golden_retriever.jpeg classified as 207 golden retriever 
../../../../demos/common/static/images/zebra.jpeg classified as 340 zebra 
../../../../demos/common/static/images/pelican.jpeg classified as 144 pelican 
Accuracy 100%
======Client Statistics======
Number of requests: 10
Total processing time: 31 ms
Latency: 28.9219 ms
Requests per second: 34.5759
```

### Run the Client to perform inference using REST API <a name="http-model-infer-resnet"></a>
```Bash
./http_infer_resnet --help
Sends requests via KServe REST API.
Usage:
  http_infer_resnet [OPTION...]

  -h, --help                    Show this help message and exit
      --images_list IMAGES      Path to a file with a list of labeled 
                                images. 
      --labels_list LABELS      Path to a file with a list of labels. 
      --http_address HTTP_ADDRESS
                                Specify url to REST service.  (default: 
                                localhost)
      --http_port PORT          Specify port to REST service.  (default: 
                                9000)
      --input_name INPUT_NAME   Specify input tensor name.  (default: 0)
      --output_name OUTPUT_NAME
                                Specify input tensor name.  (default: 1463)
      --model_name MODEL_NAME   Define model name, must be same as is in 
                                service.  (default: resnet)
      --model_version MODEL_VERSION
                                Define model version.
      --timeout TIMEOUT         Request timeout. (default: 0)

```

- Usage Example

```Bash
 ./http_infer_resnet --images_list ../../../common/resnet_input_images.txt --labels_list ../../../common/resnet_labels.txt --http_port 8000
../../../../demos/common/static/images/airliner.jpeg classified as 404 airliner 
../../../../demos/common/static/images/zebra.jpeg classified as 340 zebra 
../../../../demos/common/static/images/arctic-fox.jpeg classified as 279 Arctic fox, white fox, Alopex lagopus 
../../../../demos/common/static/images/bee.jpeg classified as 309 bee 
../../../../demos/common/static/images/golden_retriever.jpeg classified as 207 golden retriever 
../../../../demos/common/static/images/gorilla.jpeg classified as 366 gorilla, Gorilla gorilla 
../../../../demos/common/static/images/magnetic_compass.jpeg classified as 635 magnetic compass 
../../../../demos/common/static/images/peacock.jpeg classified as 84 peacock 
../../../../demos/common/static/images/pelican.jpeg classified as 144 pelican 
../../../../demos/common/static/images/snail.jpeg classified as 113 snail 
Accuracy 100%
======Client Statistics======
Number of requests: 10
Total processing time: 115.804 ms
Latency: 11.5804 ms
Requests per second: 86.3526
```

### Run the Client to perform asynchronous inference using REST API <a name="grpc-model-async-infer-resnet"></a>
```Bash
./http_async_infer_resnet --help
Sends requests via KServe REST API.
Usage:
  http_async_infer_resnet [OPTION...]

  -h, --help                    Show this help message and exit
      --images_list IMAGES      Path to a file with a list of labeled 
                                images. 
      --labels_list LABELS      Path to a file with a list of labels. 
      --http_address HTTP_ADDRESS
                                Specify url to REST service.  (default: 
                                localhost)
      --http_port PORT          Specify port to REST service.  (default: 
                                8000)
      --input_name INPUT_NAME   Specify input tensor name.  (default: 0)
      --output_name OUTPUT_NAME
                                Specify input tensor name.  (default: 1463)
      --model_name MODEL_NAME   Define model name, must be same as is in 
                                service.  (default: resnet)
      --model_version MODEL_VERSION
                                Define model version.
      --timeout TIMEOUT         Request timeout. (default: 0)
```

- Usage Example

```Bash
./http_async_infer_resnet --images_list ../../../common/resnet_input_images.txt --labels_list ../../../common/resnet_labels.txt --http_port 8000
../../../../demos/common/static/images/airliner.jpeg classified as 404 airliner 
../../../../demos/common/static/images/zebra.jpeg classified as 340 zebra 
../../../../demos/common/static/images/arctic-fox.jpeg classified as 279 Arctic fox, white fox, Alopex lagopus 
../../../../demos/common/static/images/bee.jpeg classified as 309 bee 
../../../../demos/common/static/images/golden_retriever.jpeg classified as 207 golden retriever 
../../../../demos/common/static/images/gorilla.jpeg classified as 366 gorilla, Gorilla gorilla 
../../../../demos/common/static/images/magnetic_compass.jpeg classified as 635 magnetic compass 
../../../../demos/common/static/images/peacock.jpeg classified as 84 peacock 
../../../../demos/common/static/images/pelican.jpeg classified as 144 pelican 
../../../../demos/common/static/images/snail.jpeg classified as 113 snail 
Accuracy 100%
======Client Statistics======
Number of requests: 10
Total processing time: 178 ms
Latency: 42.5398 ms
Requests per second: 23.5074
```