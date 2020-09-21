## OpenVINO Model Server 2021.1 Release Notes

The OpenVINO Model Server, release 2021.1 RC2, is available as a binary with Dockerfile for pre-production use only. Please follow the README instructions to build the image. Transitioning from Python-based version (2020.4) to C++ implementation (2021.1) should be mostly transparent. There are no changes required on the client side. 

## Key Features and Enhancements 

* Much higher scalability in a single service instance. You can now utilize the full capacity of available hardware. Expect
linear scalability when introducing additional resources while avoiding any bottleneck on the frontend.
* Shorter latency between the client and the server. This is especially noticeable with 
high performance accelerators or CPUs.
* Reduced footprint. By switching to C++ and reducing dependencies, the Docker image is reduced to ~400MB uncompressed.
* Binary package and executable. Deploy the service by unpacking a binary package -- making it easy to use on bare-metal or inside a Docker container.
* Support for online model updates. The server monitors configuration file changes and reloads models as needed without
restarting the service.
* Support for dynamic input streams gives the option of dynamically changing the number of streams to achieve optimal latency. 
* Support for dynamic batch size and shape enables variable batch sizes and shapes without restarting the service.  

## Known Issues

This preview drop is pre-alpha quality and not intended for production use. 

1.Using the serving api with heavy load of over 24k requests can sometimes produce gRPC response errors.
This issue is under investigation and will be fixed in official release.

2.When ovms server is started with config.json file the "nireq" config setting from the file is ignored.
The workaround is to use the "NIREQ" environment variable to set the nireq value for all models.

## Changes in 2021.1 
Moving from 2020.4 to 2021.1 introduces a few changes and optimizations which primarily impact the server deployment and configuration process. These changes are documented below. 

### Docker Container Entrypoint 

To simplify deployment with containers, the Docker entrypoint now requires only parameters specific to Model Server: 

Old command:
```bash
docker run -d -v $(pwd)/model:/models/face-detection/1 -e LOG_LEVEL=DEBUG -p 9000:9000 openvino/ubuntu18_model_server 
/ie-serving-py/start_server.sh ie_serving model --model_path /models/face-detection --model_name face-detection --port 9000  --shape auto 
```
New command:  
```bash
docker run -d -v $(pwd)/model:/models/face-detection/1 -p 9000:9000 openvino/model_server \
--model_path /models/face-detection --model_name face-detection --port 9000  --shape auto --log_level DEBUG
```

### Simplified Command Line Parameters  

Subcommands for `model` and `config` are no longer used. Single-model mode or multi-model mode of serving
will be determined based on whether `--config_path` or `--model_name` is defined. `--config_path` or `--model_name` are exclusive.

### Log Level and Log File Path 

Instead of environment variables `LOG_LEVEL` and `LOG_PATH`, log level and path are now defined in command line parameters to simplify configuration. 

###  grpc_workers Parameter Meaning

In the Python implementation (2020.4 and below) this parameter defined the number of frontend threads. In the C++ implementation (2021.1 and above) this defines the number of internal gRPC server objects to increase the maximum bandwidth capacity. Consider tuning
if you expect multiple clients sending requests in parallel to the server. 

### Model data type conversion

In the Python implementation (2020.4 and below) the user request input data type was automatically converted to provided model data type. In some cases such conversion impacted the performance of inference request. In the C++ implementation (2021.1 and above) the user input data type must be the same as the model input data type.
