# Example clients in C++

To build examplary clients, run `make` command in this directory. It will build docker image named `ovms_cpp_clients` with all dependencies.

There are 2 clients:
- _resnet_client_sync_ - simple client for sending synchronous requests
- _resnet_client_benchmark_ - client for sending requests asynchronously, for benchmarking purposes

The example clients image also contains test images for accurracy measurements.

## Prepare the model

Start OVMS with resnet50-binary model:
```bash
curl -L --create-dir https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin -o resnet50-binary/1/model.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml -o resnet50-binary/1/model.xml
```


## ResNet synchronous client

The client sends requests synchronously and displays latency for each request.
You can specify number of iterations and layout: `nchw`, `nhwc` or `binary`.
Each request contains image in selected format.
The client also tests server responses for accurracy.

### Prepare the server
```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd)/resnet50-binary:/model -p 9001:9001 openvino/model_server:latest \
--model_path /model --model_name resnet --port 9001 --layout NHWC
```

### Start the client:
```bash
docker run --rm --network host -e "no_proxy=localhost" ovms_cpp_clients ./resnet_client_sync --grpc_port=9001 --iterations=10 --layout="binary"

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

## ResNet asynchronous client for benchmarking

The client sends requests asynchronously to mimic parallel clients scenario.
There are plenty of parameters to configure the client.

| name | description | default |
| --- | --- | --- |
| grpc_address | url to grpc service | localhost |
| grpc_port | port to grpc service | 9000 |
| model_name | model name to request | resnet |
| input_name | input tensor name with image | 0 |
| output_name | output tensor name with classification result | 1463 |
| iterations | number of requests to be send by each producer thread | 10 |
| batch_size | batch size of each iteration | 1 |
| images_list | path to a file with a list of labeled images | input_images.txt |
| layout | binary, nhwc or nchw | nchw |
| producers | number of threads asynchronously scheduling prediction | 1 |
| consumers | number of threads receiving responses | 8 |
| max_parallel_requests | maximum number of parallel inference requests; 0=no limit | 100 |
| benchmark_mode | 1 removes pre/post-processing and logging; 0 enables accurracy measurement | 0 |

### Prepare the server
```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd)/resnet50-binary:/model -p 9001:9001 openvino/model_server:latest \
--model_path /model --model_name resnet --port 9001 --layout NCHW --batch_size 4
```

### Start the client:
```bash
docker run --rm --network host -e "no_proxy=localhost" ovms_cpp_clients ./resnet_client_benchmark --grpc_port=9001 --layout="nchw" --iterations=2000 --batch_size=4 --max_parallel_requests=100 --consumers=8 --producers=1 --benchmark_mode=1

Address: localhost
Port: 9001
Images list path: input_images.txt
Layout: nchw
Starting the workload
========================
        Summary
========================
Benchmark mode: True
Accuracy: N/A
Total time: 7925ms
Total iterations: 2000
Layout: nchw
Batch size: 4
Producer threads: 1
Consumer threads: 8
Max parallel requests: 100
Avg FPS: 1009.46
```
