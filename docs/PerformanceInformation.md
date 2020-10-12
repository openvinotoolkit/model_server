# OpenVINO&trade; Model Server Performance Information

## Introduction

This document gives an overview of various parameters that can be configured to achieve maximum performance efficiency. 

## Input data

While sending the input data for inference execution, try to adjust the numerical data type to reduce the message size.


For example: To reduce the network bandwidth usage following can be tried-
- Send the image representation as uint8 instead of float data. 
- For REST API calls, it might help to reduce the numbers precisions in the json message with a command similar to `np.round(imgs.astype(np.float),decimals=2)`. 

## Multiple model server instances

OpenVINO Model Server can be scaled vertically by adding more resources or horizontally by adding more instances 
of the service on multiple hosts. 

While hosting multiple instances of OVMS on a single host, it is optimal to ensure CPU affinity for the containers. It can be arranged
via [CPU manager for Kuburnetes](https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-policies/).

An equivalent in the docker, would be starting the containers with the option `--cpuset-cpus`.

In case of using CPU plugin to run the inference, it might be also beneficial to tune the configuration parameters like:

| Parameters      | Description |
| :---        |    :----   |
| CPU_THREADS_NUM       | Specifies the number of threads that CPU plugin should use for inference.     |
| CPU_BIND_THREAD   |   Binds inference threads to CPU cores.      |
| CPU_THROUGHPUT_STREAMS | Specifies number of CPU "execution" streams for the throughput mode |


> **NOTE:** For additional information about all parameters read [OpenVINO supported plugins](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_CPU.html).

- Example
1. While passing the plugin configuration, omit the `KEY_` phase. 
2. Following docker command will set `KEY_CPU_THROUGHPUT_STREAMS` parameter to a value `KEY_CPU_THROUGHPUT_NUMA`:

```
docker run --rm -d --cpuset-cpus 0,1,2,3 -v <model_path>:/opt/model -p 9001:9001 openvino/model_server:latest\
--model_path /opt/model --model_name my_model --port 9001 --nireq 4 \
--plugin_config '{"CPU_THROUGHPUT_STREAMS": "2", "CPU_THREADS_NUM": "4"}'

```

## Multi worker configuration

OpenVINO Model Server in C++ implementation is using scalable multithreaded gRPC and REST interface, however in some hardware configuration it might become a bottleneck for high performance backend with OpenVINO.

- To increase the throughput, a parameter `--grps_workers` is introduced which increases the number of gRPC server instances. This way the OVMS can achieve bandwidth utilization over 30Gb/s if there is sufficient network link.

- Another parameter impacting the performance is `nireq`. It defines the size of the model queue for inference execution.
It should be at least as big as the number of assigned OpenVINO streams or expected parallel clients (grpc_wokers >= nireq).


### Plugin configuration

Depending on the plugin employed to run the inference operation, you can tune the execution behaviour with a set of parameters.
> **NOTE:** For additional information, read [supported configuration parameters for CPU Plugin](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_CPU.html).

Model's plugin configuration is a dictionary of param:value pairs passed to OpenVINO Plugin on network load. It can be set with `plugin_config` parameter. 

Following docker command sets a parameter `KEY_CPU_THROUGHPUT_STREAMS` to a value `32`. It will be efficient configuration when the number of parallel connections exceeds 32:

```
docker run --rm -d -v <model_path>:/opt/model -p 9001:9001 openvino/model_server:latest \
--model_path /opt/model --model_name my_model --port 9001 --grpc_workers 8  --nireq 32 \
--plugin_config '{"CPU_THROUGHPUT_STREAMS": "32"}'

```
> **NOTE:** Depending on the target device, there are different sets of plugin configuration and tuning options. Learn more about list of supported plugins [here](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_Supported_Devices.html).

