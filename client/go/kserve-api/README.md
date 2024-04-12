# KServe API usage samples

OpenVINO Model Server introduced support for [KServe API](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2), including [Triton](https://github.com/triton-inference-server)'s raw format extension.

This guide shows how to interact with KServe API endpoints over gRPC. It covers following topics:
- [GRPC API Examples](#grpc-examples)
  - [grpc_server_live](#run-the-client-to-get-server-liveness)
  - [grpc_server_ready](#run-the-client-to-get-server-readiness)
  - [grpc_server_metadata](#run-the-client-to-get-server-metadata)
  - [grpc_model_ready](#run-the-client-to-get-model-readiness)
  - [grpc_model_metadata](#run-the-client-to-get-metadata)
  - [grpc_infer_dummy](#run-the-client-to-perform-inference)
  - [grpc_infer_resnet](#examples-with-resnet-model)

## Before you run the samples

### Clone OpenVINO&trade; Model Server GitHub repository and go to the top directory.
```Bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
```

### Start the Model Server Container with Dummy Model
```Bash
docker run --rm -d -v $(pwd)/src/test/dummy:/models -p 9000:9000 openvino/model_server:latest --model_name dummy --model_path /models --port 9000
```

### Build client library and samples
```Bash
cd client/go/kserve-api
bash build.sh
cd build
```

## GRPC Examples

## GRPC Examples with Dummy Model

This section demonstrates inference on a simple model, which increments each provided value.

### Run the Client to get server liveness

- Command

```Bash
./grpc_server_live --help
Usage of ./grpc_server_live:
  -u string
        Inference Server URL.  (default "localhost:9000")
```

- Usage Example

```Bash
./grpc_server_live -u localhost:9000
Server Live: true
```

### Run the Client to get server readiness

- Command

```Bash
./grpc_server_ready --help
Usage of ./grpc_server_ready:
  -u string
        Inference Server URL.  (default "localhost:9000")
```

- Usage Example

```Bash
./grpc_server_ready -u localhost:9000
Server Ready: true
```

### Run the Client to get server metadata

- Command

```Bash
./grpc_server_metadata --help
Usage of ./grpc_server_metadata:
  -u string
        Inference Server URL.  (default "localhost:9000")
```

- Usage Example

```Bash
./grpc_server_metadata -u localhost:9000
name:"OpenVINO Model Server" version:"2022.3.8fb11b33"
```

### Run the Client to get model readiness

- Command

```Bash
./grpc_model_ready --help
Usage of ./grpc_model_ready:
  -n string
        Name of model being served.  (default "dummy")
  -u string
        Inference Server URL.  (default "localhost:9000")
  -v string
        Version of model.
```

- Usage Example

```Bash
./grpc_model_ready -u localhost:9000
Model Ready: true
```

### Run the Client to get metadata

- Command

```Bash
./grpc_model_metadata --help
Usage of ./grpc_model_metadata:
  -n string
        Name of model being served.  (default "dummy")
  -u string
        Inference Server URL.  (default "localhost:9000")
  -v string
        Version of model.
```

- Usage Example

```Bash
./grpc_model_metadata -u localhost:9000
name:"dummy" versions:"1" platform:"OpenVINO" inputs:{name:"b" datatype:"FP32" shape:1 shape:10} outputs:{name:"a" datatype:"FP32" shape:1 shape:10}
```

### Run the Client to perform inference

- Command

```Bash
./grpc_infer_dummy --help
Usage of ./grpc_infer_dummy:
  -n string
        Name of model being served.  (default "dummy")
  -u string
        Inference Server URL.  (default "localhost:9000")
  -v string
        Version of model.
```

- Usage Example

```Bash
./grpc_infer_dummy -u localhost:9000

Checking Inference Outputs
--------------------------
0.000000 => 1.000000
1.000000 => 2.000000
2.000000 => 3.000000
3.000000 => 4.000000
4.000000 => 5.000000
5.000000 => 6.000000
6.000000 => 7.000000
7.000000 => 8.000000
8.000000 => 9.000000
9.000000 => 10.000000
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
docker run --rm -d -v $(pwd)/models:/models -p 9000:9000 openvino/model_server:latest --model_name resnet --model_path /models/resnet --port 9000 --layout NHWC:NCHW --plugin_config '{"PERFORMANCE_HINT":"LATENCY"}'
```

Once you finish above steps, you are ready to run the samples.

### Run the Client to perform inference using gRPC API
```Bash
./grpc_infer_resnet --help
Usage of ./grpc_infer_resnet:
  -i string
        Path to a file with a list of labeled images.
  -l string
        Path to a file with a list of labels.
  -n string
        Name of model being served.  (default "resnet")
  -u string
        Inference Server URL.  (default "localhost:9000")
  -v string
        Version of model.
```

- Usage Example

```Bash
./grpc_infer_resnet -i ../../../common/resnet_input_images.txt -l ../../../common/resnet_labels.txt -u localhost:9000
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
```