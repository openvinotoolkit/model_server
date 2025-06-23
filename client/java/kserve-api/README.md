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

### Prerequisite

[OpenJDK Runtime Environment](https://openjdk.org/) version 11 or newer
[Apache Maven](https://maven.apache.org/) version 3.6.3 or newer

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
cd client/java/kserve-api
mvn install
```

> **NOTE:** In proxy environment, you may need to set `http_proxy` and `https_proxy` environment variables before running the above command. To do that with mvn, run:
```Bash
mvn install \
  -Dhttp.proxyHost=<proxy_host> \
  -Dhttp.proxyPort=<proxy_port> \
  -Dhttps.proxyHost=<proxy_host> \
  -Dhttps.proxyPort=<proxy_port>
```

## GRPC Examples


## GRPC Examples with Dummy Model

This section demonstrates inference on a simple model, which increments each provided value.

### Run the Client to get server liveness

- Command

```Bash
java -cp target/grpc-client.jar clients.grpc_server_live --help
usage: grpc_server_live [OPTION]...
 -a,--grpc_address <GRPC_ADDRESS>   Specify url to grpc service.
 -h,--help                          Show this help message and exit
 -p,--grpc_port <GRPC_PORT>         Specify port to grpc service.
```

- Usage Example

```Bash
java -cp target/grpc-client.jar clients.grpc_server_live --grpc_port 9000 --grpc_address localhost
Server Live: true
```

### Run the Client to get server readiness

- Command

```Bash
java -cp target/grpc-client.jar clients.grpc_server_ready --help
usage: grpc_server_ready [OPTION]...
 -a,--grpc_address <GRPC_ADDRESS>   Specify url to grpc service.
 -h,--help                          Show this help message and exit
 -p,--grpc_port <GRPC_PORT>         Specify port to grpc service.
```

- Usage Example

```Bash
java -cp target/grpc-client.jar clients.grpc_server_ready --grpc_port 9000 --grpc_address localhost
Server Ready: true
```

### Run the Client to get server metadata

- Command

```Bash
java -cp target/grpc-client.jar clients.grpc_server_metadata --help
usage: grpc_server_metadata [OPTION]...
 -a,--grpc_address <GRPC_ADDRESS>   Specify url to grpc service.
 -h,--help                          Show this help message and exit
 -p,--grpc_port <GRPC_PORT>         Specify port to grpc service.
```

- Usage Example

```Bash
java -cp target/grpc-client.jar clients.grpc_server_metadata --grpc_port 9000 --grpc_address localhost
name: "OpenVINO Model Server"
version: "2022.3.befa4df9"
```

### Run the Client to get model readiness

- Command

```Bash
java -cp target/grpc-client.jar clients.grpc_model_ready --help
usage: grpc_model_ready [OPTION]...
 -a,--grpc_address <GRPC_ADDRESS>     Specify url to grpc service.
 -h,--help                            Show this help message and exit
 -n,--model_name <MODEL_NAME>         Define model name, must be same as
                                      is in service
 -p,--grpc_port <GRPC_PORT>           Specify port to grpc service.
 -v,--model_version <MODEL_VERSION>   Define model version.
```

- Usage Example

```Bash
java -cp target/grpc-client.jar clients.grpc_model_ready --grpc_port 9000 --grpc_address localhost --model_name dummy
Model Ready: true
```

### Run the Client to get metadata

- Command

```Bash
java -cp target/grpc-client.jar clients.grpc_model_metadata --help
usage: grpc_model_metadata [OPTION]...
 -a,--grpc_address <GRPC_ADDRESS>     Specify url to grpc service.
 -h,--help                            Show this help message and exit
 -n,--model_name <MODEL_NAME>         Define model name, must be same as
                                      is in service
 -p,--grpc_port <GRPC_PORT>           Specify port to grpc service.
 -v,--model_version <MODEL_VERSION>   Define model version.
```

- Usage Example

```Bash
java -cp target/grpc-client.jar clients.grpc_model_metadata --grpc_port 9000 --grpc_address localhost --model_name dummy
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

### Run the Client to perform inference
```Bash
java -cp target/grpc-client.jar clients.grpc_infer_dummy --help
usage: grpc_infer_dummy [OPTION]...
 -a,--grpc_address <GRPC_ADDRESS>     Specify url to grpc service.
 -h,--help                            Show this help message and exit
 -i,--input_name <INPUT_NAME>         Specify input tensor name.
 -n,--model_name <MODEL_NAME>         Define model name, must be same as
                                      is in service
 -o,--output_name <OUTPUT_NAME>       Specify output tensor name.
 -p,--grpc_port <GRPC_PORT>           Specify port to grpc service.
 -v,--model_version <MODEL_VERSION>   Define model version.
```

- Usage Example

```Bash
java -cp target/grpc-client.jar clients.grpc_infer_dummy --grpc_port 9000 --grpc_address localhost
0.0 => 1.0
1.0 => 2.0
2.0 => 3.0
3.0 => 4.0
4.0 => 5.0
5.0 => 6.0
6.0 => 7.0
7.0 => 8.0
8.0 => 9.0
9.0 => 10.0
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
java -cp target/grpc-client.jar clients.grpc_infer_resnet --help
usage: grpc_infer_resnet [OPTION]...
 -a,--grpc_address <GRPC_ADDRESS>     Specify url to grpc service.
 -h,--help                            Show this help message and exit
 -i,--input_name <INPUT_NAME>         Specify input tensor name.
 -imgs,--image_list <IMAGES>          Path to a file with a list of
                                      labeled images.
 -lbs,--labels <LABELS>               Path to a file with a list of
                                      labels.
 -n,--model_name <MODEL_NAME>         Define model name, must be same as
                                      is in service
 -p,--grpc_port <GRPC_PORT>           Specify port to grpc service.
 -v,--model_version <MODEL_VERSION>   Define model version.
```

- Usage Example

```Bash
java -cp target/grpc-client.jar clients.grpc_infer_resnet -imgs ./resnet_input_images.txt -lbs ../../common/resnet_labels.txt --grpc_port 9000
../../../demos/common/static/images/airliner.jpeg classified as 404 airliner
../../../demos/common/static/images/arctic-fox.jpeg classified as 279 Arctic fox, white fox, Alopex lagopus
../../../demos/common/static/images/bee.jpeg classified as 309 bee
../../../demos/common/static/images/golden_retriever.jpeg classified as 207 golden retriever
../../../demos/common/static/images/gorilla.jpeg classified as 366 gorilla, Gorilla gorilla
../../../demos/common/static/images/magnetic_compass.jpeg classified as 635 magnetic compass
../../../demos/common/static/images/peacock.jpeg classified as 84 peacock
../../../demos/common/static/images/pelican.jpeg classified as 144 pelican
../../../demos/common/static/images/snail.jpeg classified as 113 snail
../../../demos/common/static/images/zebra.jpeg classified as 340 zebra
```
