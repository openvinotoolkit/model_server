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
docker run --rm -d -v $(pwd)/src/test/dummy:/models -p 9000:9000 -p 8000:8000 openvino/model_server:latest --model_name dummy --model_path /models --port 9000 
```

### Build client library and samples
```Bash
cd client/java/kserve-api
mvn install
```


### Run the Client to perform inference
```Bash
java -cp target/grpc-client.jar clients.grpc_infer_dummy --help
usage: grpc_infer_dummy
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