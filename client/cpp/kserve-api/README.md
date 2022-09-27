# KServe API usage samples

OpenVINO Model Server 2022.2 release introduced support for [KServe API](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2).

## Before you run the samples

### Clone OpenVINO&trade; Model Server GitHub repository and go to the directory with the samples.
```Bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/client/cpp/kserve-api
```

### Build client library and samples
```Bash
cmake . && make
```

## GRPC Examples with Dummy Model

### Start the Model Server Container with Dummy Model
```Bash
docker run --rm -d -v $(pwd)/models:/models -p 9000:9000 openvino/model_server:latest --model_name dummy --model_path /models/dummy --port 9000 
```

Once you finish above steps, you are ready to run the samples.

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
completed_request_count 1
cumulative_total_request_time_ns 5207356
cumulative_send_time_ns 46654
cumulative_receive_time_ns 5312
```