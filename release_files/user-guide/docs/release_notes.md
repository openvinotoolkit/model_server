## OpenVINO Model Server C++ version release notes

Transition from python based version to C++ implementation should be mostly transparent.  No changes on the clients
side are expected. 
After migration to new implementation, the following advantages will be available to you:

* Much higher scalability in a single service instance. You can now utilize whole capacity of available hardware. Expect
linear scalability with additional resources without the bottleneck on the frontend side.
* Shorter latency and quicker communication between the client and the server. It will be especially noticeable with 
high performance accelerators or CPU, where every millisecond matters.
* Smaller docker image. Without python and other dropped dependencies, the docker image is greatly reduced to ~400MB 
in uncompressed format.
* Binary package and executable. It is much easier to deploy OVMS by just unpacking the binary package. It is easy to use
on baremetal and as docker container.
* Configuration file online updates. OVMS will monitor configuration file changes and reload the models as needed without
the service restart.

## Known issues

This preview drop in a pre-alpha quality product.

1.Using the serving api with heavy load of over 24k requests can sometimes produce gRPC response errors.
This issue is under investigation and will be fixed in official release.

2.When ovms server is started with config.json file the "nireq" config setting from the file is ignored.
The workaround is to use the "NIREQ" environment variable to set the nireq value for all models.

## Changes versus Python OVMS version 
While the C++ implementation was recreated from scratch, there were introduced a few changes and optimizations
which will affect mostly the deployment and configuration process.


### Docker container entrypoint 

Instead of starting docker container with long command and parameters, docker entrypoint is set to make required only the parameters of OVMS: 

```bash
docker run -d -v $(pwd)/model:/models/face-detection/1 -e LOG_LEVEL=DEBUG -p 9000:9000 openvino/ubuntu18_model_server 
/ie-serving-py/start_server.sh ie_serving model --model_path /models/face-detection --model_name face-detection --port 9000  --shape auto 
```
vs 
```bash
docker run -d -v $(pwd)/model:/models/face-detection/1 -p 9000:9000 openvino/model_server \
--model_path /models/face-detection --model_name face-detection --port 9000  --shape auto --log_level DEBUG
```

### OVMS command line parameters simplification. 

There is not no need to use `model` and `config` subcommands. The single model mode and multi model mode of the serving
is determined based on the added parameters. `--config_path` and `--model_name` are exclusive.

### Log level and log file path is configurable via command line

Instead of environment variables `LOG_LEVEL` and `LOG_PATH`, logging parameters are defined in command line parameters.
It will unify the configuration and make it easier to document all configuration options in the serving help output.

###  grpc_workers parameter meaning

In python implementation this parameter was setting the number of frontend threads. In C++ implementation
it is setting the number of internal gRPC server objects to increase maximum bandwidth capacity. Consider tuning
it if you expect multiple clients sending requests in parallel.



