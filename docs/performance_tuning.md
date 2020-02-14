# Performance tuning

## Input data

When you send the input data for inference execution, try to adjust the numerical data type to reduce the message size.
For example you might consider sending the image representation as uint8 instead of float data. For REST API calls,
it might help to reduce the numbers precisions in the json message with a command similar to 
`np.round(imgs.astype(np.float),decimals=2)`. It will reduce the network bandwidth usage. 

## Multiple model server instances

OpenVINO Model Server can be scaled horizontally by adding more instances of the service and including any type
of the load-balancing. Beside scaling the inference across multiple nodes, it is also advantageous for the throughout 
results, to multiply the instances on a single machine. 

It could be easily accomplished in Kubernetes by setting a deployment configuration with many replicas. In case 
of latency optimization, create the replicas with high CPU allocation. When throughput is most important, add
more replicas with smaller, restricted CPU resources on the Kubernetes level.

Below are listed exemplary environment settings in 2 scenarios.

**Optimization for latency** - 1 docker container consuming all 80vCPU on the node:

**Optimization for throughput** - 20 docker containers on the node consuming 4vCPU each:

While hosting multiple instances of OVMS, it is optimal to ensure CPU affinity for the containers. It can be arranged
via [CPU manager for Kuburnetes](https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-policies/)
An equivalent in the docker, would be starting the containers with the option `--cpuset-cpus`.

In case of using CPU plugin to run the inference, it might be also beneficial to tune the configuration parameters like:
* KEY_CPU_THREADS_NUM
* KEY_CPU_BIND_THREAD
* KEY_CPU_THROUGHPUT_STREAMS

Read about available parameters on [OpenVINO supported plugins](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_CPU.html).

While passing the plugin configuration, omit the `KEY_` phase. Example docker command, setting `KEY_CPU_THROUGHPUT_STREAMS` parameter
 to a value `KEY_CPU_THROUGHPUT_NUMA`:

```
docker run --rm -d --cpuset-cpus 0,1,2,3 -v <model_path>:/opt/model -p 9001:9001 ie-serving-py:latest /ie-serving-py/start_server.sh ie_serving model \
--model_path /opt/model --model_name my_model --port 9001 --grpc_workers 8  --nireq 1 \
--plugin_config "{\"CPU_THROUGHPUT_STREAMS\": \"CPU_THROUGHPUT_NUMA\",\"CPU_THREADS_NUM\": \"4\"}"
```

## Multi worker configuration

Starting from version 2019 R3, it's possible to run OpenVINO Model Server with multiple server workers. It enables concurrent processing of
multiple clients requests. It employs OpenVINO asynchronous inference execution mode. 
Consider running model server with multiple server workers, if you're interested in throughput optimization without  
creating multiple OVMS instances. You can expect the best results especially, when running the workloads on the accelerators like HDDL.

Number of servers workers can be set on server startup with parameters `grpc_workers` and `rest_workers`.
To really make use of multiple server workers, you should also set number of inference parallel requests
for model with `nireq` parameter. This parameter defines how many inference operations can run in parallel.

You should have at least as many server workers as inference requests set in `nireq` parameter in your models 
(**grpc_wokers >= nireq** and **rest_workers >= nireq**).

**Note**: When using CPU plugin, the vertical scalability (adding mode cores to the OVMS instance) is a bit less efficient 
then horizontal (adding more smaller OVMS instances). Add more instances if you need to maximize the throughput and you
don't have restrictions of the number of instances and their RAM usage.

### HDDL accelerators

A single worker on OpenVINO Model Server could use only a single VPU (Vision Processing Unit) from HDDL accelerators at a time.
Multiple workers can increase the throughput. With parallel handling of the requests, it is possible to utilize all available VPUs.
We recommend having even 4x inference requests per VPU for maximum results. 
For example if you have one HDDL accelerator with 8x VPUs you can start model server in docker container with:

```docker run --rm -d --device /dev/ion:/dev/ion -v /var/tmp:/var/tmp -v <model_path>:/opt/model -p 9001:9001 ie-serving-py:latest \
/ie-serving-py/start_server.sh ie_serving model --model_path /opt/model --model_name my_model --port 9001 \
--grpc_workers 64  --nireq 32 --target_device HDDL
```

**Note** HDDL plugin is not available in the public docker image on [dockerhub](https://hub.docker.com/r/intelaipg/openvino-model-server/)
It needs to be built using url to [full OpenVINO binary package](docker_container.md#building-the-image).


### Plugin configuration

Depending on the plugin employed to run the inference operation, you can tune the execution behaviour with a set of parameters.
They are listed on [supported configuration parameters for CPU Plugin](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_CPU.html).

Model's plugin configuration is a dictionary of param:value pairs passed to OpenVINO Plugin on network load.
You can set it with `plugin_config` parameter. 

Example docker command, setting a parameter `KEY_CPU_THROUGHPUT_STREAMS` to a value `KEY_CPU_THROUGHPUT_AUTO`:

```
docker run --rm -d -v <model_path>:/opt/model -p 9001:9001 ie-serving-py:latest /ie-serving-py/start_server.sh ie_serving model \
--model_path /opt/model --model_name my_model --port 9001 --grpc_workers 64  --nireq 32 \
--plugin_config "{\"CPU_THROUGHPUT_STREAMS\": \"CPU_THROUGHPUT_AUTO\"}"
```

