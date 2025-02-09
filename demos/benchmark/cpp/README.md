# Benchmark Client (C++) {#ovms_demo_benchmark_app_cpp}

This demo provides a benchmark client that uses asynchronous gRPC API and tests performance with synthetic data (stripped out of OpenCV dependency).

To build the client, run `make` command in this directory. It will build docker image named `ovms_cpp_benchmark` with all dependencies.
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/benchmark/cpp
make
```
The application can be used with any model or pipeline served in OVMS, by requesting `GetModelMetadata` endpoint and using such information to prepare synthetic inputs with matching shape and precision.

> **Note**: In this directory you can only see the code specific to the benchmark client. The code shared with other C++ demos as well as all building utilities are placed in the [common C++ directory](https://github.com/openvinotoolkit/model_server/tree/main/demos/common/cpp).

> **Note**: It is required that endpoint does not use dynamic shape.

## Usage example

### Prepare the model
Start OVMS with resnet50-binary model:
```bash
curl -L --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin -o resnet50-binary/1/model.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml -o resnet50-binary/1/model.xml
chmod -R 755 resnet50-binary
```

### Prepare the server
```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd)/resnet50-binary:/model -p 9001:9001 openvino/model_server:latest \
--model_path /model --model_name resnet --port 9001 --layout NCHW
```

### Start the client:
```bash
docker run --rm --network host -e "no_proxy=localhost" ovms_cpp_benchmark ./synthetic_client_async_benchmark --grpc_port=9001 --iterations=2000 --max_parallel_requests=100 --consumers=8 --producers=1

Address: localhost:9001
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
