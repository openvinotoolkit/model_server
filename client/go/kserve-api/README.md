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

### Run the Client to get server liveness <a name="grpc-server-live"></a>

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

### Run the Client to get server readiness <a name="grpc-server-ready"></a>

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

### Run the Client to get server metadata <a name="grpc-server-metadata"></a>

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

### Run the Client to get model readiness <a name="grpc-model-ready"></a>

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

### Run the Client to get metadata <a name="grpc-model-metadata"></a>

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