# Performance tuning {#ovms_docs_performance_tuning}

## Introduction

This document gives an overview of various parameters that can be configured to achieve maximum performance efficiency. 

## Performance Hints
The `PERFORMANCE_HINT` plugin config property enables you to specify a performance mode for the plugin to be more efficient for particular use cases.

#### THROUGHPUT
This mode prioritizes high throughput, balancing between latency and power. It is best suited for tasks involving multiple jobs, like inference of video feeds or large numbers of images.

To enable Performance Hints for your application, use the following command:
@sphinxdirective

.. tab:: CPU  

   .. code-block:: sh

        docker run --rm -d -v <model_path>:/opt/model -p 9001:9001 openvino/model_server:latest \
            --model_path /opt/model --model_name my_model --port 9001 \
            --plugin_config '{"PERFORMANCE_HINT": "THROUGHTPUT"}' \
            --target_device CPU

.. tab:: GPU

   .. code-block:: sh  
   
        docker run --rm -d --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
            -v <model_path>:/opt/model -p 9001:9001 openvino/model_server:latest \
            --model_path /opt/model --model_name my_model --port 9001 \
            --plugin_config '{"PERFORMANCE_HINT": "THROUGHTPUT"}' \
            --target_device GPU

@endsphinxdirective

#### LATENCY
This mode prioritizes low latency, providing short response time for each inference job. It performs best for tasks where inference is required for a single input image, like a medical analysis of an ultrasound scan image. It also fits the tasks of real-time or nearly real-time applications, such as an industrial robot's response to actions in its environment or obstacle avoidance for autonomous vehicles.
Note that currently the `PERFORMANCE_HINT` property is supported by CPU and GPU devices only. [More information](https://docs.openvino.ai/2022.1/openvino_docs_IE_DG_supported_plugins_AUTO.html#performance-hints).

To enable Performance Hints for your application, use the following command:
@sphinxdirective

.. tab:: CPU  

   .. code-block:: sh

        docker run --rm -d -v <model_path>:/opt/model -p 9001:9001 openvino/model_server:latest \
            --model_path /opt/model --model_name my_model --port 9001 \
            --plugin_config '{"PERFORMANCE_HINT": "LATENCY"}' \
            --target_device CPU

.. tab:: GPU

   .. code-block:: sh  
   
        docker run --rm -d --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
            -v <model_path>:/opt/model -p 9001:9001 openvino/model_server:latest \
            --model_path /opt/model --model_name my_model --port 9001 \
            --plugin_config '{"PERFORMANCE_HINT": "LATENCY"}' \
            --target_device GPU

@endsphinxdirective

> **NOTE**: CPU_THROUGHPUT_STREAMS and PERFORMANCE_HINT should not be used together.

## Adjusting the number of streams in CPU and GPU target devices

OpenVINO&trade; Model Server can be tuned to a single client use case or a high concurrency. It is done via setting the number of
execution streams. They split the available resources to perform parallel execution of multiple requests.
It is particularly efficient for models which cannot effectively consume all CPU cores or for CPUs with high number of cores.

By default, for `CPU` target device, OpenVINO Model Server sets the value CPU_THROUGHPUT_AUTO and GPU_THROUGHTPUT_AUTO for `GPU` target device. It calculates the number of streams based on number of available CPUs. It gives a compromise between the single client scenario and the high concurrency.

If this default configuration is not suitable, adjust it with the `CPU_THROUGHPUT_STREAMS` parameter defined as part 
of the device plugin configuration. 

In a scenario where the number of parallel connections is close to 1, set the following parameter:

`--plugin_config '{"CPU_THROUGHPUT_STREAMS": "1"}'`

When the number of concurrent requests is higher, increase the number of streams. Make sure, however, that the number of streams is lower than the average volume of concurrent inference operations. Otherwise, the server might not be fully utilized.
Number of streams should not exceed the number of CPU cores.

For example, with ~50 clients sending the requests to the server with 48 cores, set the number of streams to 24:

`--plugin_config '{"CPU_THROUGHPUT_STREAMS": "24"}'`

## Input data in REST API calls

While using REST API, you can adjust the data format to optimize the communication and deserialization from json format. 

While sending the input data for inference execution, try to adjust the numerical data type to reduce the message size.

- reduce the numbers precisions in the json message with a command similar to `np.round(imgs.astype(np.float),decimals=2)`. 
- use [binary data format](binary_input.md) encoded with base64 - sending compressed data will greatly reduce the traffic and speed up the communication.
- with binary input format it is the most efficient to send the images with the resolution of the configured model. It will avoid image resizing on the server to fit the model.

## Scalability

OpenVINO Model Server can be scaled vertically by adding more resources or horizontally by adding more instances of the service on multiple hosts. 

While hosting multiple instances of OVMS with constrained CPU resources, it is optimal to ensure CPU affinity for the containers. 
It can be arranged via [CPU manager for Kubernetes](https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-policies/).

An equivalent in the docker, would be starting the containers with the option `--cpuset-cpus` instead of `--cpus`.

In case of using CPU plugin to run the inference, it might be also beneficial to tune the configuration parameters like:

| Parameters      | Description |
| :---        |    :----   |
| CPU_THREADS_NUM       | Specifies the number of threads that CPU plugin should use for inference.     |
| CPU_BIND_THREAD   |   Binds inference threads to CPU cores.      |
| CPU_THROUGHPUT_STREAMS | Specifies number of CPU "execution" streams for the throughput mode |


> **NOTE:** For additional information about all parameters read [OpenVINO supported plugins](https://docs.openvino.ai/2022.1/namespaceInferenceEngine_1_1PluginConfigParams.html?#detailed-documentation).

- Example:
1. While passing the plugin configuration, omit the `KEY_` phase. 
2. Following docker command will set `KEY_CPU_THROUGHPUT_STREAMS` parameter to a value `KEY_CPU_THROUGHPUT_NUMA`:

```
docker run --rm -d --cpuset-cpus 0,1,2,3 -v <model_path>:/opt/model -p 9001:9001 openvino/model_server:latest\
--model_path /opt/model --model_name my_model --port 9001 \
--plugin_config '{"CPU_THROUGHPUT_STREAMS": "1"}'

```

## CPU Power Management Settings
To save power, the OS can decrease the CPU frequency and increase a volatility of the latency values. Similarly the Intel® Turbo Boost Technology may also affect the stability of results. For best reproducibility, consider locking the frequency to the processor base frequency (refer to the https://ark.intel.com/ for your specific CPU). For example, in Linux setting the relevant values for the /sys/devices/system/cpu/cpu* entries does the trick. [Read more](https://docs.openvino.ai/2022.1/openvino_docs_optimization_guide_dldt_optimization_guide.html). High-level commands like cpupower also exists:
```
$ cpupower frequency-set --min 3.1GHz
```

## Tuning Model Server configuration parameters           

OpenVINO Model Server in C++ implementation is using scalable multithreaded gRPC and REST interface, however in some hardware configuration it might become a bottleneck for high performance backend with OpenVINO.

- To increase the throughput, a parameter `--grpc_workers` is introduced which increases the number of gRPC server instances. In most cases the default value of `1` will be sufficient.
  In case of particularly heavy load and many parallel connections, higher value might increase the transfer rate.

- Another parameter impacting the performance is `nireq`. It defines the size of the model queue for inference execution.
It should be at least as big as the number of assigned OpenVINO streams or expected parallel clients (grpc_wokers >= nireq).
  
- Parameter `file_system_poll_wait_seconds` defines how often the model server will be checking if new model version gets created in the model repository. 
The default value is 1 second which ensures prompt response to creating new model version. In some cases, it might be recommended to reduce the polling frequency
  or even disable it. For example, with cloud storage, it could cause a cost for API calls to the storage cloud provider. Detecting new versions 
  can be disabled with a value `0`.


## Plugin configuration

Depending on the device employed to run the inference operation, you can tune the execution behavior with a set of parameters. Each device is handled by its OpenVINO plugin.

> **NOTE**: For additional information, read [supported configuration parameters for all plugins](https://docs.openvino.ai/2022.1/namespaceInferenceEngine_1_1PluginConfigParams.html?#detailed-documentation).

Model's plugin configuration is a dictionary of param:value pairs passed to OpenVINO Plugin on network load. It can be set with `plugin_config` parameter. 

Following docker command sets a parameter `KEY_CPU_THROUGHPUT_STREAMS` to a value `32` and `KEY_CPU_BIND_THREAD` to `NUMA`.

```
docker run --rm -d -v <model_path>:/opt/model -p 9001:9001 openvino/model_server:latest \
--model_path /opt/model --model_name my_model --port 9001 --grpc_workers 8  --nireq 32 \
--plugin_config '{"CPU_THROUGHPUT_STREAMS": "32", "CPU_BIND_THREAD": "NUMA"}'
```
