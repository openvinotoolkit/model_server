# C API inference demo (C/C++) {#ovms_demo_capi_inference_demo}

This demo demonstrate how to use C API from the OpenVINO Model Server to create C and C++ application.
Building the application is executed inside the docker container to illustrate end to end usage flow.
Check C API full documentation [here](../../docs/model_server_c_api.md).

## Prepare demo image
Enter the directory with the example and build the demo docker image with all dependencies and examples that will be named `openvino/model_server-capi`.
The example image also contains dummy model and config.json required for the applications.
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/c_api_minimal_app
make
```

The make command downloads the `ovms.tar.gz` package from web and executes the c_api_minimal_app/capi_files/demos/MakefileCapi to build the applications.
And executes the /ovms/bin/demo1 application and /ovms/bin/demo2_c application in the image environment.

You can find the source code for the example applications in the ovms repository path src/main_capi.c and src/main_capi.cpp.
or make modifications in the built image:
```bash
docker run -it openvino/model_server-capi:latest
cat main_capi.c
cat main_capi.cpp
```

Afterwards rebuild and run the modified examples using the MakefileCapi rules:
```bash
make -f MakefileCapi c
make -f MakefileCapi cpp
```

It will link the main_capi.cpp binary with the `libovms_shared.so` library from /ovms/lib and use the headers from /ovms/include directory:
```bash
g++ main_capi.cpp -I/ovms/include -L/ovms/lib -lovms_shared -o main_capi
./main_capi
```

The example output is:
```bash
[2022-12-19 11:39:41.428][14][serving][info][modelinstance.cpp:797] Loaded model dummy; version: 1; batch size: 30; No of InferRequests: 12
[2022-12-19 11:39:41.428][14][serving][debug][modelversionstatus.cpp:88] setAvailable: dummy - 1 (previous state: LOADING) -> error: OK
[2022-12-19 11:39:41.428][14][serving][info][modelversionstatus.cpp:113] STATUS CHANGE: Version 1 of model dummy status change. New status: ( "state": "AVAILABLE", "error_code": "OK" )
[2022-12-19 11:39:41.428][14][serving][info][model.cpp:88] Updating default version for model: dummy, from: 0
[2022-12-19 11:39:41.428][14][serving][info][model.cpp:98] Updated default version for model: dummy, to: 1
[2022-12-19 11:39:41.428][14][modelmanager][info][modelmanager.cpp:478] Configuration file doesn't have custom node libraries property.
[2022-12-19 11:39:41.428][14][modelmanager][info][modelmanager.cpp:495] Configuration file doesn't have pipelines property.
[2022-12-19 11:39:41.428][14][serving][info][servablemanagermodule.cpp:44] ServableManagerModule started
Server ready for inference
[2022-12-19 11:39:41.428][14][serving][debug][capi.cpp:606] Processing C-API inference request for servable: dummy; version: 1
[2022-12-19 11:39:41.428][14][serving][debug][modelmanager.cpp:1350] Requesting model: dummy; version: 1.
[2022-12-19 11:39:41.428][14][serving][debug][modelinstance.cpp:1013] Model: dummy, version: 1 already loaded
[2022-12-19 11:39:41.428][14][serving][debug][modelinstance.cpp:1187] Getting infer req duration in model dummy, version 1, nireq 0: 0.002 ms
[2022-12-19 11:39:41.428][478][modelmanager][info][modelmanager.cpp:900] Started model manager thread
[2022-12-19 11:39:41.428][14][serving][debug][modelinstance.cpp:1195] Preprocessing duration in model dummy, version 1, nireq 0: 0.000 ms
[2022-12-19 11:39:41.428][14][serving][debug][modelinstance.cpp:1205] Deserialization duration in model dummy, version 1, nireq 0: 0.019 ms
[2022-12-19 11:39:41.428][479][modelmanager][info][modelmanager.cpp:920] Started cleaner thread
[2022-12-19 11:39:41.429][14][serving][debug][modelinstance.cpp:1213] Prediction duration in model dummy, version 1, nireq 0: 0.369 ms
[2022-12-19 11:39:41.429][14][serving][debug][modelinstance.cpp:1222] Serialization duration in model dummy, version 1, nireq 0: 0.011 ms
[2022-12-19 11:39:41.429][14][serving][debug][modelinstance.cpp:1230] Postprocessing duration in model dummy, version 1, nireq 0: 0.000 ms
[2022-12-19 11:39:41.429][14][serving][debug][capi.cpp:650] Total C-API req processing time: 0.474 ms
output is correct
No more job to be done, will shut down
[2022-12-19 11:39:41.429][14][serving][info][grpcservermodule.cpp:177] GRPCServerModule shutting down
[2022-12-19 11:39:41.429][14][serving][info][grpcservermodule.cpp:180] Shutdown gRPC server
[2022-12-19 11:39:41.429][14][serving][info][grpcservermodule.cpp:184] GRPCServerModule shutdown
[2022-12-19 11:39:41.429][14][serving][info][httpservermodule.cpp:54] HTTPServerModule shutting down
[evhttp_server.cc : 320] NET_LOG: event_base_loopexit() exits with value 0
[evhttp_server.cc : 253] NET_LOG: event_base_dispatch() exits with value 1
[2022-12-19 11:39:41.604][14][serving][info][httpservermodule.cpp:59] Shutdown HTTP server
[2022-12-19 11:39:41.604][14][serving][info][servablemanagermodule.cpp:54] ServableManagerModule shutting down
[2022-12-19 11:39:41.604][478][modelmanager][info][modelmanager.cpp:916] Stopped model manager thread
[2022-12-19 11:39:41.604][479][modelmanager][info][modelmanager.cpp:930] Stopped cleaner thread
[2022-12-19 11:39:41.604][14][serving][info][modelmanager.cpp:985] Shutdown model manager
[2022-12-19 11:39:41.604][14][serving][info][modelmanager.cpp:993] Shutdown cleaner thread
[2022-12-19 11:39:41.604][14][serving][info][servablemanagermodule.cpp:57] ServableManagerModule shutdown
```

It is also possible to use custom model but that requires copying it to the built image and adjust the configs and example applications accordingly.
To make the changes permanent in the resulting demo image you can modify the specific dockerfiles in c_api_minimal_app/capi_files/ directory 
Dockerfile.ubuntu
Dockerfile.redhat

And run the demo make with os specific arguments:
```bash
make BASE_OS=redhat
```

## Build libovms_shared.so
Alternative to getting `ovms.tar.gz` package from web you can build it yourself from sources. To build the capi docker image, you must first build the `ovms.tar.gz` package with the `libovms_shared.so` library and `ovms.h` header. 
run `make` command in ovms git main directory.
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make
```

And then execute the alternative make target:
```bash
cd demos/c_api_minimal_app
make all_docker
```

# Capi Benchmark

This guide shows how to perform benchmark tests using OpenVINO Model Server C-API.
It starts the model server and initiates a load from multithreaded clients sending sequential requests with synthetic data.
It can demonstrate expected throughput and latency for a specific model, configuration and C-API calls.


Clone OpenVINO™ Model Server GitHub repository and go to the top directory.
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/c_api_minimal_app
```
Build the tool
```bash
make
```

- Command
```bash
docker run openvino/model_server-capi:latest -c /ovms/bin/capi_benchmark
OpenVINO Model Server
Usage:
  /ovms/bin/capi_benchmark [OPTION...]

  -h, --help                    Show this help message and exit
      --log_level LOG_LEVEL     serving log level - one of TRACE, DEBUG,
                                INFO, WARNING, ERROR (default: ERROR)
      --config_path CONFIG_PATH
                                Config file path for OVMS to read (default:
                                /ovms/src/test/configs/config_benchmark.json)
      --niter NITER             number of inferences to conduct (default:
                                1000)
      --nstreams NSTREAMS       number of execution streams to be performed
                                simultaneously (suggested for best
                                throughput is NUM_STREAMS of a ovms model
                                config) (default: 1)
      --servable_name MODEL_NAME
                                Model name to sent request to
      --servable_version MODEL_VERSION
                                workload threads per ireq, if not set
                                version will be set by default model
                                version policy (default: 0)
      --mode MODE               Workload mode. Possible values:
                                INFERENCE_ONLY, RESET_BUFFER, RESET_REQUEST
                                (default: INFERENCE_ONLY)
      --seed SEED               Random values generator seed.
```

## Available modes

- INFERENCE_ONLY (default), measures only the time it takes to perform inferences using C-API
- RESET_BUFFER, include time it takes to recreate inference request data buffer
- RESET_REQUEST, include time it takes to recreate whole inference request 

Perform the measurement using sample model, one can specify different model using `config_path` option and specifying desired config file.
```bash
docker run openvino/model_server-capi:latest -c '/ovms/bin/capi_benchmark --servable_name dummy --nstreams 12 --config_path /ovms/demos/config_benchmark.json'
Mode requested: INFERENCE_ONLY
Server ready for inference
Benchmark starting workload
FPS: 235128
Average latency : 0.053ms
main() exit
```

