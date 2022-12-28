# KServe API usage samples

OpenVINO Model Server introduced support for [KServe API](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2), including [Triton](https://github.com/triton-inference-server)'s raw format externsion.

This guide shows how to interact with KServe API endpoints over gRPC. It covers following topics:
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
docker run --rm -d -v $(pwd)/src/test/dummy:/models -p 9000:9000 openvino/model_server:latest --model_name dummy --model_path /models --port 9000 
```

### Build client library and samples
```Bash
cd client/go/kserve-api
bash build.sh
cd build
```

## GRPC Examples <a name="grpc-api"></a>


## GRPC Examples with Dummy Model

This section demonstrates inference on a simple model, which increments each provided value. 

### Run the Client to get server liveness <a name="grpc-model-infer"></a>

- Command

```Bash
./grpc_infer_dummy --help
Usage of ./grpc_infer_dummy:
  -m string
        Name of model being served.  (default "dummy")
  -u string
        Inference Server URL.  (default "localhost:9000")
  -x string
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