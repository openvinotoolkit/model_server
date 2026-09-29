# Performance tuning {#ovms_docs_performance_tuning}

## Introduction

This document gives an overview of various parameters that can be configured to achieve maximum performance efficiency.


## Text generation using LLM calculator with continuous batching

There are several important considerations for tuning LLM serving via the OpenAI API endpoint:

- Choose a cache size that matches the model, context length and expected concurrency. A practical starting point is dynamic resizing with value `0`, then observing cache usage in server logs under normal load.
      Dynamic allocation can minimize RAM usage when demand is low, but with unrestricted concurrency and long contexts it can still grow enough to cause out-of-memory issues.
      Static cache sizing is safer when you need hard memory limits for KV cache usage. It can improve prompt reuse under steady load, while requests that do not fit may be preempted.

- The `--max_num_batched_tokens` parameter influences how prompts are divided and grouped in the prefill phase.
      The default value `256` is efficient for short contexts and high concurrency.
      Increasing this value to `4096`, `8192`, or even the model context limit can improve first-token latency.

- On multi-socket systems, text generation is typically scoped to a single NUMA node.
      Disabling virtual NUMA nodes can improve locality in some deployments (for example, one NUMA node per CPU socket), but this is workload- and platform-dependent.
      Always benchmark with and without vNUMA before standardizing this setting.

- Models with linear attention also use `--cache_interval_multiplier`, which affects linear attention cache behavior.
      The default value is adaptive and generally optimized for memory usage.
      Lower values can improve prefix-caching efficiency by using smaller chunks, but also increase memory consumption.
      Allowed range: `8-256`.

- Set `--enable_prefix_cache true` (default) in the graph configuration to reuse KV cache for sequential requests with repeated prompt tokens (for example, chat history). This avoids duplicated prompt evaluation.

- Use lower precision via model quantization and KV cache precision settings to improve throughput and reduce memory usage.

- `--rest_workers` can limit the number of concurrent requests processed by the model server.
      By default it is set to the number of vCPU cores, which is usually correct.
      You can increase it for stress benchmarking with very high client counts, or reduce it to prevent server overload (some clients will wait for a connection).

Check also a guide about [handling long context requests](../demos/continuous_batching/long_context/README.md).

## Classic model tuning

The following sections focus on classic model serving scenarios (for example, image classification and detection models) and are not specific to LLM continuous batching.

### Performance Hints
The `PERFORMANCE_HINT` plugin config property enables you to specify a performance mode for the plugin to be more efficient for particular use cases.

#### THROUGHPUT
This mode prioritizes high throughput, balancing between latency and power. It is best suited for tasks involving multiple jobs, like inference of video feeds or large numbers of images.

To enable Performance Hints for your application, use the following command:

CPU

```text
docker run --rm -d -v ${PWD}/models/public/resnet-50-tf:/opt/model -p 9001:9001 openvino/model_server:latest \
      --model_path /opt/model --model_name resnet --port 9001 \
      --plugin_config "{\"PERFORMANCE_HINT\": \"THROUGHPUT\"}" \
      --target_device CPU
```

GPU

```text
docker run --rm -d --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
      -v ${PWD}/models/public/resnet-50-tf:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
      --model_path /opt/model --model_name resnet --port 9001 \
      --plugin_config "{\"PERFORMANCE_HINT\": \"THROUGHPUT\"}" \
      --target_device GPU
```

#### LATENCY
This mode prioritizes low latency, providing short response time for each inference job. It performs best for tasks where inference is required for a single input image, like a medical analysis of an ultrasound scan image. It also fits the tasks of real-time or nearly real-time applications, such as an industrial robot's response to actions in its environment or obstacle avoidance for autonomous vehicles.
Note that currently the `PERFORMANCE_HINT` property is supported by CPU and GPU devices only. [More information](https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/high-level-performance-hints.html#performance-hints-how-it-works).

To enable Performance Hints for your application, use the following command:

CPU

```text
docker run --rm -d -v ${PWD}/models/public/resnet-50-tf:/opt/model -p 9001:9001 openvino/model_server:latest \
      --model_path /opt/model --model_name resnet --port 9001 \
      --plugin_config "{\"PERFORMANCE_HINT\": \"LATENCY\"}" \
      --target_device CPU
```

GPU

```text
docker run --rm -d --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
      -v ${PWD}/models/public/resnet-50-tf:/opt/model -p 9001:9001 openvino/model_server:latest-gpu \
      --model_path /opt/model --model_name resnet --port 9001 \
      --plugin_config "{\"PERFORMANCE_HINT\": \"LATENCY\"}" \
      --target_device GPU
```

> **NOTE**: NUM_STREAMS and PERFORMANCE_HINT should not be used together.

### Adjusting the number of streams in CPU and GPU target devices

OpenVINO&trade; Model Server can be tuned to a single client use case or a high concurrency. It is done via setting the number of
execution streams. They split the available resources to perform parallel execution of multiple requests.
It is particularly efficient for models which cannot effectively consume all CPU cores or for CPUs with high number of cores.

By default, number of streams is optimized for execution with minimal latency with low concurrency. The number of execution streams will be equal to the number of CPU sockets or GPU cards.
If that default configuration is not suitable, adjust it with the `NUM_STREAMS` parameter defined as part
of the device plugin configuration or set the performance hint to `THROUGHPUT`.

In a scenario with a single connection/client, set the following parameter:

`--plugin_config "{\"NUM_STREAMS\": \"1\"}"`

When the number of concurrent requests is high, increase the number of streams. Make sure, however, that the number of streams is lower than the average volume of concurrent inference operations. Otherwise, the server might not be fully utilized.

Number of streams should not exceed the number of cores.

For example, with ~50 clients sending the requests to the server with 48 cores, set the number of streams to 24:

`--plugin_config "{\"NUM_STREAMS\": \"24\"}"`

### Disabling CPU pinning

By default, OpenVINO Model Server enables CPU thread pinning for better performance.
You can switch it off with plugin config.
Disabling thread pinning can be beneficial in complex applications with multiple workloads running in parallel.

`--plugin_config "{\"ENABLE_CPU_PINNING\": false}"`

### Input data in REST API calls

While using REST API, you can adjust the data format to optimize communication and JSON deserialization. Here are some tips to use the REST interface efficiently with OpenVINO Model Server:

- use [binary data format](binary_input.md) when possible - binary data representation is smaller in terms of request size and easier to process on the server side.
- when working with images, consider sending JPEG/PNG directly - compressed data will greatly reduce the traffic and speed up the communication.
- with JPEG/PNG it is the most efficient to send the images with the resolution of the configured model. It will avoid image resizing on the server to fit the model.
- if you decide to send data inside JSON object, try to adjust the numeric precision to reduce message size. For example: `np.round(imgs.astype(np.float32), decimals=2)`.

## Scalability

OpenVINO Model Server can be scaled vertically by adding more resources or horizontally by adding more instances of the service on multiple hosts.

While hosting multiple instances of OVMS with constrained CPU resources, it is optimal to ensure CPU affinity for the containers.
It can be arranged via [CPU manager for Kubernetes](https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-policies/).

An equivalent in the docker, would be starting the containers with the option `--cpuset-cpus` instead of `--cpus`.

In case of using CPU plugin to run the inference, it might be also beneficial to tune the configuration parameters like:

| Parameters      | Description |
| :---        |    :----   |
| INFERENCE_NUM_THREADS       | Specifies the number of threads that CPU plugin should use for inference.     |
| NUM_STREAMS | Specifies number of execution streams for the throughput mode. |
| ENABLE_CPU_PINNING | This property allows CPU threads pinning during inference. |


> **NOTE:** For additional information about all parameters read about [OpenVINO device properties](https://docs.openvino.ai/2026/api/c_cpp_api/group__ov__runtime__cpp__prop__api.html).

- Example:
Following docker command will set `NUM_STREAMS` parameter to a value `1`:

```text
docker run --rm -d --cpuset-cpus 0,1,2,3 -v ${PWD}/models/public/resnet-50-tf:/opt/model -p 9001:9001 openvino/model_server:latest \
--model_path /opt/model --model_name resnet --port 9001 \
--plugin_config "{\"NUM_STREAMS\": \"1\"}"

```

> **NOTE:** OpenVINO Model Server automatically detects the number of CPU cores allocated to a container or Kubernetes pod.
> Based on that detection, selected parameters are adjusted to optimize resource usage and performance under CPU constraints.
> This adaptive behavior applies to `INFERENCE_NUM_THREADS`, `NUM_STREAMS`, and `ENABLE_CPU_PINNING`.
> OVMS also detects the allowed number of open files and adjusts the number of REST workers accordingly, because workers contribute to file descriptor usage.


## CPU Power Management Settings
To save power, the OS can decrease CPU frequency and increase latency variability. Similarly, Intel® Turbo Boost Technology may also affect result stability. For best reproducibility, consider locking frequency to the processor base frequency (refer to https://ark.intel.com/ for your specific CPU). For example, on Linux, setting relevant values for `/sys/devices/system/cpu/cpu*` entries does the trick. High-level commands like `cpupower` also exist:
```
$ cpupower frequency-set --min 3.1GHz
```

## Network Configuration for Optimal Performance

By default, OVMS endpoints are bound to all IPv4 addresses. On some systems, `localhost` resolves to an IPv6 address first, which can add client-side fallback time before switching to IPv4. This can result in extra latency.
It can be mitigated by using `http://127.0.0.1` as the API URL on the client side.

To optimize network connection performance:

Alternatively, IPv6 can be enabled in the model server using `--grpc_bind_address` and `--rest_bind_address`.
For example:
```
--grpc_bind_address 127.0.0.1,::1 --rest_bind_address 127.0.0.1,::1
```
or
```
--grpc_bind_address 0.0.0.0,:: --rest_bind_address 0.0.0.0,::
```

## Tuning Model Server configuration parameters

OpenVINO Model Server is implemented in C++ with scalable multithreaded gRPC and REST interfaces. However, in some hardware configurations the serving layer can become a bottleneck for a high-performance OpenVINO backend.

- To increase throughput, the `--grpc_workers` parameter increases the number of gRPC server instances. In most cases, the default value `1` is sufficient.
  In case of particularly heavy load and many parallel connections, higher value might increase the transfer rate.

- Another parameter impacting performance is `nireq`. It defines the size of the model queue for inference execution.
It should be at least as large as the number of assigned OpenVINO streams. For high-load scenarios, increase it toward the expected level of parallel clients (`nireq >= NUM_STREAMS`).

- Parameter `file_system_poll_wait_seconds` defines how often the model server will be checking if new model version gets created in the model repository.
The default value is 1 second which ensures prompt response to creating new model version. In some cases, it might be recommended to reduce the polling frequency
  or even disable it. For example, with cloud storage, it could cause a cost for API calls to the storage cloud provider. Detecting new versions
  can be disabled with a value `0`.

- Collecting metrics has negligible overhead for models of average size and complexity. However, for lightweight and very fast models, metrics incrementation can consume a noticeable share of CPU time compared to actual inference. Take this into account when enabling metrics for such models.

- Log level `DEBUG` produces significant amount of logs. Usually the impact of generating logs on overall performance is negligible, but for very high throughput use cases consider using `--log_level INFO` which is also the default setting.


## Analyzing performance issues

Recommended steps to investigate achievable performance and discover bottlenecks:
1. [Launch OV benchmark app](https://docs.openvino.ai/2026/get-started/learn-openvino/openvino-samples/benchmark-tool.html)

      **Note:** It is useful to extract plugin configuration from benchmark app with `-dump_config`, then apply the same plugin configuration to the model loaded in OVMS.

      **Note:** When launching benchmark app, use `-inference_only=false`. Otherwise, OpenVINO avoids setting the input tensor for each inference call, which is not comparable to OVMS request flow.
2. [Launch OVMS benchmark client](../demos/benchmark/README.md) on the same machine as OVMS
3. [Launch OVMS benchmark client](../demos/benchmark/README.md) from remote machine
4. Measure achievable network bandwidth with tools such as [iperf](https://github.com/esnet/iperf)

## Analyzing accuracy issues

Please note that the target devices GPU and CPU with AMX feature usually change the default model execution precision from FP32 to BF16.
It is recommended to compare accuracy results versus OpenVINO benchmark app.

It is possible to enforce a specific runtime precision by using a plugin config parameter `INFERENCE_PRECISION_HINT`. For example:
 `--plugin_config "{\"INFERENCE_PRECISION_HINT\": \"f32\"}"`.

