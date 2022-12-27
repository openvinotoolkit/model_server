# KServe API usage samples

OpenVINO Model Server introduced support for [KServe API](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2), including [Triton](https://github.com/triton-inference-server)'s raw format externsion.

This guide shows how to interact with KServe API endpoints on gRPC. It covers following topics:
- <a href="#grpc-api">GRPC API Examples </a>
  - <a href="#grpc-server-live">grpc_server_live</a>
  - <a href="#grpc-server-ready">grpc_server_ready</a>
  - <a href="#grpc-server-metadata">grpc_server_metadata</a>
  - <a href="#grpc-model-ready">grpc_model_ready</a>
  - <a href="#grpc-model-metadata">grpc_model_metadata</a>
  - <a href="#grpc-model-infer">grpc_infer_dummy</a>

## Before you run the samples

### Prerequisits

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

## GRPC Examples <a name="grpc-api"></a>


## GRPC Examples with Dummy Model

This section demonstrates inference on a simple model, which increments each provided value. 

### Run the Client to get server liveness <a name="grpc-server-live"></a>

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

### Run the Client to get server readiness <a name="grpc-server-ready"></a>

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
Server Live: true
```

### Run the Client to get server metadata <a name="grpc-server-metadata"></a>

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
Name: OpenVINO Model Server
Version: 2022.2.c290da85
```

### Run the Client to get model readiness <a name="grpc-model-ready"></a>

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

### Run the Client to get metadata <a name="grpc-model-metadata"></a>

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
java -cp target/grpc-client.jar clients.grpc_model_ready --grpc_port 9000 --grpc_address localhost --model_name dummy
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