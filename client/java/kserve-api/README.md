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
mvn compile
```


### Run the Client to perform inference

```Bash
mvn exec:java -Dexec.mainClass=clients.grpc_infer_dummy -Dexec.args="localhost 9111" # TEMP
```