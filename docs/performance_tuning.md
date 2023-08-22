# Performance tuning {#ovms_docs_performance_tuning}

## Introduction

This document gives an overview of various parameters that can be configured to achieve maximum performance efficiency. 

## Example model

Download ResNet50 model

```bash
mkdir models
docker run -u $(id -u):$(id -g) -v ${PWD}/models:/models openvino/ubuntu20_dev:latest omz_downloader --name resnet-50-tf --output_dir /models
docker run -u $(id -u):$(id -g) -v ${PWD}/models:/models:rw openvino/ubuntu20_dev:latest omz_converter --name resnet-50-tf --download_dir /models --output_dir /models --precisions FP32
mv ${PWD}/models/public/resnet-50-tf/FP32 ${PWD}/models/public/resnet-50-tf/1
```

## Performance Hints
The `PERFORMANCE_HINT` plugin config property enables you to specify a performance mode for the plugin to be more efficient for particular use cases.

#### THROUGHPUT
This mode prioritizes high throughput, balancing between latency and power. It is best suited for tasks involving multiple jobs, like inference of video feeds or large numbers of images.

To enable Performance Hints for your application, use the following command:

CPU

   ```bash
         docker run --rm -d -v ${PWD}/models/public/resnet-50-tf:/opt/model -p 9001:9001 openvino/model_server:latest \
               --model_path /opt/model --model_name resnet --port 9001 \
               --plugin_config '{"PERFORMANCE_HINT": "THROUGHPUT"}' \
               --target_device CPU
   ```

GPU

@sphinxdirective
.. code-block:: sh

         docker run --rm -d --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
               -v ${PWD}/models/public/resnet-50-tf:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
               --model_path /opt/model --model_name resnet --port 9001 \
               --plugin_config '{"PERFORMANCE_HINT": "THROUGHPUT"}' \
               --target_device GPU

@endsphinxdirective

#### LATENCY
This mode prioritizes low latency, providing short response time for each inference job. It performs best for tasks where inference is required for a single input image, like a medical analysis of an ultrasound scan image. It also fits the tasks of real-time or nearly real-time applications, such as an industrial robot's response to actions in its environment or obstacle avoidance for autonomous vehicles.
Note that currently the `PERFORMANCE_HINT` property is supported by CPU and GPU devices only. [More information](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_Performance_Hints.html#performance-hints-how-it-works).

To enable Performance Hints for your application, use the following command:

CPU

   ```bash
         docker run --rm -d -v ${PWD}/models/public/resnet-50-tf:/opt/model -p 9001:9001 openvino/model_server:latest \
               --model_path /opt/model --model_name resnet --port 9001 \
               --plugin_config '{"PERFORMANCE_HINT": "LATENCY"}' \
               --target_device CPU
   ```

GPU
   
   ```bash
         docker run --rm -d --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
               -v ${PWD}/models/public/resnet-50-tf:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
               --model_path /opt/model --model_name resnet --port 9001 \
               --plugin_config '{"PERFORMANCE_HINT": "LATENCY"}' \
               --target_device GPU
   ```

> **NOTE**: NUM_STREAMS and PERFORMANCE_HINT should not be used together.

## Adjusting the number of streams in CPU and GPU target devices

OpenVINO&trade; Model Server can be tuned to a single client use case or a high concurrency. It is done via setting the number of
execution streams. They split the available resources to perform parallel execution of multiple requests.
It is particularly efficient for models which cannot effectively consume all CPU cores or for CPUs with high number of cores.

By default, number of streams is optimized for execution with minimal latency with low concurrency. The number of execution streams will be equal to the number of CPU sockets or GPU cards.
If that default configuration is not suitable, adjust it with the `NUM_STREAMS` parameter defined as part 
of the device plugin configuration or set the performance hint to `THROUGHPUT`. 

In a scenario with a single connections/client, set the following parameter:

`--plugin_config '{"NUM_STREAMS": "1"}'`

When the number of concurrent requests is high, increase the number of streams. Make sure, however, that the number of streams is lower than the average volume of concurrent inference operations. Otherwise, the server might not be fully utilized.

Number of streams should not exceed the number of cores.

For example, with ~50 clients sending the requests to the server with 48 cores, set the number of streams to 24:

`--plugin_config '{"NUM_STREAMS": "24"}'`

## Input data in REST API calls

While using REST API, you can adjust the data format to optimize the communication and deserialization from json format. Here are some tips to effectively use REST interface when working with OpenVINO Model Server:

- use [binary data format](binary_input.md) when possible(for TFS API binary data format is support ony for JPEG/PNG inputs, for KFS API there are no such limitations ) - binary data representation is smaller in terms of request size and easier to process on the server side. 
- when working with images, consider sending JPEG/PNG directly - compressed data will greatly reduce the traffic and speed up the communication.
- with JPEG/PNG it is the most efficient to send the images with the resolution of the configured model. It will avoid image resizing on the server to fit the model.
- if you decide to send data inside JSON object, try to adjust the numerical data type to reduce the message size i.e. reduce the numbers precisions in the json message with a command similar to `np.round(imgs.astype(np.float),decimals=2)`. 

## Scalability

OpenVINO Model Server can be scaled vertically by adding more resources or horizontally by adding more instances of the service on multiple hosts. 

While hosting multiple instances of OVMS with constrained CPU resources, it is optimal to ensure CPU affinity for the containers. 
It can be arranged via [CPU manager for Kubernetes](https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-policies/).

An equivalent in the docker, would be starting the containers with the option `--cpuset-cpus` instead of `--cpus`.

In case of using CPU plugin to run the inference, it might be also beneficial to tune the configuration parameters like:

| Parameters      | Description |
| :---        |    :----   |
| INFERENCE_NUM_THREADS       | Specifies the number of threads that CPU plugin should use for inference.     |
| AFFINITY   |   Binds inference threads to CPU cores.      |
| NUM_STREAMS | Specifies number of execution streams for the throughput mode |


> **NOTE:** For additional information about all parameters read about [OpenVINO device properties](https://docs.openvino.ai/2023.0/groupov_runtime_cpp_prop_api.html?#detailed-documentation).

- Example:
Following docker command will set `NUM_STREAMS` parameter to a value `1`:

```bash
docker run --rm -d --cpuset-cpus 0,1,2,3 -v ${PWD}/models/public/resnet-50-tf:/opt/model -p 9001:9001 openvino/model_server:latest \
--model_path /opt/model --model_name resnet --port 9001 \
--plugin_config '{"NUM_STREAMS": "1"}'

```

> **NOTE:** Deployment of the OpenVINO Model Server including the autoscaling capability can be automated in Kubernetes and OpenShift using the operator. [Read more about](https://github.com/openvinotoolkit/operator/blob/main/docs/autoscaling.md)

## CPU Power Management Settings
To save power, the OS can decrease the CPU frequency and increase a volatility of the latency values. Similarly the Intel® Turbo Boost Technology may also affect the stability of results. For best reproducibility, consider locking the frequency to the processor base frequency (refer to the https://ark.intel.com/ for your specific CPU). For example, in Linux setting the relevant values for the /sys/devices/system/cpu/cpu* entries does the trick. High-level commands like cpupower also exists:
```
$ cpupower frequency-set --min 3.1GHz
```

## Tuning Model Server configuration parameters           

OpenVINO Model Server in C++ implementation is using scalable multithreaded gRPC and REST interface, however in some hardware configuration it might become a bottleneck for high performance backend with OpenVINO.

- To increase the throughput, a parameter `--grpc_workers` is introduced which increases the number of gRPC server instances. In most cases the default value of `1` will be sufficient.
  In case of particularly heavy load and many parallel connections, higher value might increase the transfer rate.

- Another parameter impacting the performance is `nireq`. It defines the size of the model queue for inference execution.
It should be at least as big as the number of assigned OpenVINO streams or expected parallel clients (grpc_workers >= nireq).
  
- Parameter `file_system_poll_wait_seconds` defines how often the model server will be checking if new model version gets created in the model repository. 
The default value is 1 second which ensures prompt response to creating new model version. In some cases, it might be recommended to reduce the polling frequency
  or even disable it. For example, with cloud storage, it could cause a cost for API calls to the storage cloud provider. Detecting new versions 
  can be disabled with a value `0`.

- Collecting metrics has negligible performance overhead when used with models of average size and complexity. However, when used with lightweight, fast models, the metric incrementation can consume noticeable proportion of CPU time compared to actual inference. Take it into account while enabled metrics for such models.

- Log level `DEBUG` produces significant amount of logs. Usually the impact of generating logs on overall performance is negligible, but for very high throughput use cases consider using `--log_level INFO` which is also the default setting.

## Plugin configuration

Depending on the device employed to run the inference operation, you can tune the execution behavior with a set of parameters. Each device is handled by its OpenVINO plugin.

> **NOTE**: For additional information, read [supported configuration parameters for all plugins](https://docs.openvino.ai/2023.0/groupov_runtime_cpp_prop_api.html?#detailed-documentation).

Model's plugin configuration is a dictionary of param:value pairs passed to OpenVINO Plugin on network load. It can be set with `plugin_config` parameter. 

Following docker command sets a parameter `NUM_STREAMS` to a value `32` and `AFFINITY` to `NUMA`.

```bash
docker run --rm -d -v ${PWD}/models/public/resnet-50-tf:/opt/model -p 9001:9001 openvino/model_server:latest \
--model_path /opt/model --model_name resnet --port 9001 --grpc_workers 8  --nireq 32 \
--plugin_config '{"NUM_STREAMS": "32", "AFFINITY": "NUMA"}'
```

## Analyzing performance issues

Recommended steps to investigate achievable performance and discover bottlenecks:
1. [Launch OV benchmark app](https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html?highlight=benchmark)

      **Note:** It is useful to drop plugin configuration from benchmark app using `-dump_config` and then use the same plugin configuration in model loaded into OVMS

      **Note:** When launching benchmark app use `-inference_only=false`. Otherwise OV avoids setting input tensor of inference each time which is not comparable flow to OVMS.
2. [Launch OVMS benchmark client](https://docs.openvino.ai/2023.0/ovms_demo_benchmark_client.html) on the same machine as OVMS
3. [Launch OVMS benchmark client](https://docs.openvino.ai/2023.0/ovms_demo_benchmark_client.html) from remote machine
4. Measure achievable network bandwidth with tools such as [iperf](https://github.com/esnet/iperf)
