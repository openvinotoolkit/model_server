# Performance tuning

## Input data

When you send the input data for inference execution try to adjust the numerical data type to reduce the message size.
For example you might consider sending the image representation as uint8 instead of float data. For REST API calls,
it might help to reduce the numbers precisions in the json message with a command similar to 
`np.round(imgs.astype(np.float),decimals=2)`. It will reduce the network bandwidth usage. 

## Multiple model server instances

Usually, there is no need to tune any environment variables according to the allocated resources. In some cases 
it might be however beneficial to adjust the threading parameters to fit the allocated resources in optimal way.
This is especially relevant in configuration when multiple services are being used on a single node. Another situation is 
in horizontal scalability in Kubernetes when the throughput can be increased by employing big volume of small containers.

Below are listed exemplary environment settings in 2 scenarios.

**Optimization for latency** - 1 container consuming all 80vCPU on the node:
```
OMP_NUM_THREADS=40
KMP_SETTINGS=1
KMP_AFFINITY=granularity=fine,verbose,compact,1,0
KMP_BLOCKTIME=1
```
**Optimization for throughput** - 20 containers on the node consuming 4vCPU each:
```
OMP_NUM_THREADS=4
KMP_SETTINGS=1
KMP_AFFINITY=granularity=fine,verbose,compact,1,0
KMP_BLOCKTIME=1
```

*Note* With the version *2019 R2*, OpenVINO Model Server is using [TBB](https://software.intel.com/en-us/tbb) threading instead
 of [OMP](https://www.openmp.org/).
 
It gives better performance especially in scenarios with shared CPU and enabled multiple AI models. It also do not require
tuning the environment variables, which was required with OMP. The model server with TBB will be more flexible and
easier to deploy in environments with dynamic load and resource allocation.

## Multiple worker configuration

It's possible to run OpenVINO Model Server with multiple server workers to enable concurrent processing of
multiple clients requests. In combination with asynchronous inference mode it allows model server to perform multiple inferences in parallel. 
Consider running model server with multiple server workers, especially if you're interested in throughput optimization and you don't want to 
create multiple OVMS instances.

Number of servers workers can be set on server startup with parameters `grpc_workers` and `rest_workers`.
To really make use of multiple server workers, you should also set number of [infer requests](https://docs.openvinotoolkit.org/latest/classie__api_1_1InferRequest.html) 
for model with `nireq` parameter. Infer requests are handlers for inferences, so their amount is the maximum number of inferences that can be performed in parallel on the model (model version to be more specific).

Keep in mind that if there are less server workers than infer request in a model, 
the maximum number of inferences performed in parallel will be limited to the amount of server workers. 
That's why you should always have at least as many server workers as infer requests in your model (**grpc_wokers/rest_workers >= nireq**).

### HDDL accelerators

Model server with multiple workers was proved to greatly increase VPUs utilization and overall throughput compared to single worker, synchronous version.
We recommend having 4x infer requests per VPU. For example if you have one HDDL accelerator with 8x VPUs you can start model server in docker container with:

`docker run --rm -d --device /dev/ion:/dev/ion -v /var/tmp:/var/tmp -v <model_path>:/opt/model -p 9001:9001 ie-serving-py:latest /ie-serving-py/start_server.sh ie_serving model --model_path /opt/model --model_name my_model --port 9001 --grpc_workers 64  --nireq 32 --target_device HDDL`

### CPU 

To perform multiple inferences in parallel on CPU you should additionally provide model's network configuration.
Model's network configuration is a map of param:value pairs passed to OpenVINO Plugin on network load.
You can set it with `network_config` parameter. Check out [supported configuration parameters for CPU Plugin](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_CPU.html).

Example docker command with setting `KEY_CPU_THROUGHPUT_STREAMS` to `KEY_CPU_THROUGHPUT_AUTO`:

`docker run --rm -d -v <model_path>:/opt/model -p 9001:9001 ie-serving-py:latest /ie-serving-py/start_server.sh ie_serving model --model_path /opt/model --model_name my_model --port 9001 --grpc_workers 64  --nireq 32 --network_config "{\"CPU_THROUGHPUT_STREAMS\": \"CPU_THROUGHPUT_AUTO\"}"`

You might need to tune model server with `grpc_workers`/`rest_workers`, `nireq` and `network_config` parameters to achieve desired performance for your setup.
If you run model server on machine with many cores (lets say 80 virtual cores), you might find it hard to utilize them. 
It can be caused by limited multi threading performance in Python. 
In case you experience this issue consider having more OVMS instances and dividing resources between them.