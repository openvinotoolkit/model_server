# KServe API usage samples

OpenVINO Model Server 2022.2 release introduced support for [KServe API](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2).

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
cd client/cpp/kserve-api
cmake . && make
```

## GRPC Examples <a name="grpc-api"></a>


## GRPC Examples with Dummy Model

This section demonstrates inference on a simple model, which increments each provided value. 


Once you finish above steps, you are ready to run the samples.

### Run the Client to get server liveness <a name="grpc-server-live"></a>

- Command

```Bash
./grpc_server_live --help
Sends requests via KServe gRPC API to check if server is alive.
Usage:
  grpc_server_live [OPTION...]

  -h, --help              Show this help message and exit
      --grpc_address arg  Specify url to grpc service.  (default: 
                          localhost)
      --grpc_port arg     Specify port to grpc service.  (default: 9000)
      --timeout arg       Request timeout. (default: 0)
```

- Usage Example 

```Bash
./grpc_server_live --grpc_port 9000 --grpc_address localhost
Server Live: True
```

### Run the Client to perform inference

```Bash
./grpc_infer_dummy --help
Sends requests via KServe gRPC API.
Usage:
  grpc_infer_dummy [OPTION...]

  -h, --help              Show this help message and exit
      --grpc_address arg  Specify url to grpc service.  (default: 
                          localhost)
      --grpc_port arg     Specify port to grpc service.  (default: 9000)
      --input_name arg    Specify input tensor name.  (default: b)
      --output_name arg   Specify input tensor name.  (default: a)
      --model_name arg    Define model name, must be same as is in service. 
                           (default: dummy)
      --model_version     Define model version.
      --timeout arg       Request timeout. (default: 0)
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