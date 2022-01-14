# Image Classification Demo with OVMS

This demo provides 3 clients:
- _classification_client_sync_ - simple client using synchronous gRPC API, testing accurracy of classification models
- _classification_client_async_benchmark_ - client using asynchronous gRPC API, testing accurracy and performance with real image data
- _synthetic_client_async_benchmark_ - client using asynchronous gRPC API, testing performance with synthetic data, stripped out of OpenCV dependency

To build examplary clients, run `make` command in this directory. It will build docker image named `ovms_cpp_clients` with all dependencies.
The example clients image also contains test images required for accurracy measurements. It is also possible to use custom images.

## Prepare classification model

Start OVMS with resnet50-binary model:
```bash
curl -L --create-dir https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin -o resnet50-binary/1/model.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml -o resnet50-binary/1/model.xml
```

# Client requesting prediction synchronously

The client sends requests synchronously and displays latency for each request.
You can specify number of iterations and layout: `nchw`, `nhwc` or `binary`.
Each request contains image in selected format.
The client also tests server responses for accurracy.

## Prepare the server
```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd)/resnet50-binary:/model -p 9001:9001 openvino/model_server:latest \
--model_path /model --model_name resnet --port 9001 --layout NHWC
```

## Start the client:
```bash
docker run --rm --network host -e "no_proxy=localhost" ovms_cpp_clients ./classification_client_sync --grpc_port=9001 --iterations=10 --layout="binary"

call predict ok
call predict time: 24ms
outputs size is 1
call predict ok
call predict time: 23ms
outputs size is 1
call predict ok
call predict time: 23ms
outputs size is 1
...
Overall accuracy: 90%
Total time divided by number of requests: 25ms
```

# Clients requesting predictions asynchronously

The client sends requests asynchronously to mimic parallel clients scenario.
There are plenty of parameters to configure those clients.

| name | description | default | available with synthetic data |
| --- | --- | --- | --- |
| grpc_address | url to grpc service | localhost | yes |
| grpc_port | port to grpc service | 9000 | yes |
| model_name | model name to request | resnet | yes |
| input_name | input tensor name with image | 0 | no, deduced automatically |
| output_name | output tensor name with classification result | 1463 | no |
| iterations | number of requests to be send by each producer thread | 10 | yes |
| batch_size | batch size of each iteration | 1 | no, deduced automatically |
| images_list | path to a file with a list of labeled images | input_images.txt | no |
| layout | binary, nhwc or nchw | nchw | no, deduced automatically |
| producers | number of threads asynchronously scheduling prediction | 1 | yes |
| consumers | number of threads receiving responses | 8 | yes |
| max_parallel_requests | maximum number of parallel inference requests; 0=no limit | 100 | yes |
| benchmark_mode | 1 removes pre/post-processing and logging; 0 enables accurracy measurement | 0 | no |

## Async client with real image data

### Prepare the server
```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd)/resnet50-binary:/model -p 9001:9001 openvino/model_server:latest \
--model_path /model --model_name resnet --port 9001 --layout NCHW
```

### Start the client:
```bash
docker run --rm --network host -e "no_proxy=localhost" ovms_cpp_clients ./classification_client_async_benchmark --grpc_port=9001 --layout="nchw" --iterations=2000 --batch_size=1 --max_parallel_requests=100 --consumers=8 --producers=1 --benchmark_mode=1

Address: localhost:9001
Model name: resnet
Images list path: input_images.txt

Running the workload...
========================
        Summary
========================
Benchmark mode: True
Accuracy: N/A
Total time: 1976ms
Total iterations: 2000
Layout: nchw
Batch size: 1
Producer threads: 1
Consumer threads: 8
Max parallel requests: 100
Avg FPS: 1012.15
```

## Async client with synthetic data

This client is simplified to test performance of any model/pipeline by requesting `GetModelMetadata` endpoint and using such information to prepare synthetic inputs with matching shape and precision. It also does not need OpenCV as dependency.

> NOTE: It is required that endpoint does not use dynamic shape.

### Prepare the server
```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd)/resnet50-binary:/model -p 9001:9001 openvino/model_server:latest \
--model_path /model --model_name resnet --port 9001 --layout NCHW
```

### Start the client:
```bash
docker run --rm --network host -e "no_proxy=localhost" ovms_cpp_clients ./synthetic_client_async_benchmark --grpc_port=9001 --iterations=2000 --max_parallel_requests=100 --consumers=8 --producers=1

Address: localhost:11337
Model name: resnet
Synthetic inputs:
        0: (1,3,224,224); DT_FLOAT

Running the workload...
========================
        Summary
========================
Total time: 1933ms
Total iterations: 2000
Producer threads: 1
Consumer threads: 8
Max parallel requests: 100
Avg FPS: 1034.66
```
