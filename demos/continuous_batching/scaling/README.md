# Scaling on a dual CPU socket server {#ovms_demos_continuous_batching_scaling}

> **Note**: This demo uses Docker and has been tested only on Linux hosts

Text generation in OpenVINO Model Server on Xeon CPU is the most efficient when its container is bound to a single NUMA node. 
That ensures the fastest memory access and avoids intra socket communication.

The example below demonstrate how serving can be scaled on dual socket Intel(R) Xeon(R) 6972P servers to multiply the throughout. 
It deploys 6 instances of the model server allocated to different NUMA nodes on 2 CPU sockets. The client calls are distributed using an Nginx proxy as a load balancer.

![drawing](./loadbalancing.png)

## Start the Model Server instances

Let's assume we have two CPU sockets server with two NUMA nodes. 
```bash
lscpu | grep NUMA
NUMA node(s):                         6
NUMA node0 CPU(s):                    0-31,192-223
NUMA node1 CPU(s):                    32-63,224-255
NUMA node2 CPU(s):                    64-95,256-287
NUMA node3 CPU(s):                    96-127,288-319
NUMA node4 CPU(s):                    128-159,320-351
NUMA node5 CPU(s):                    160-191,352-383
```

Export the model:
```bash
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/requirements.txt
mkdir models
python export_model.py text_generation --source_model meta-llama/Meta-Llama-3-8B-Instruct --model Meta-Llama-3-8B-Instruct_FP16 --weight-format fp16 --model_repository_path models
```

```bash
docker run --cpuset-cpus $(lscpu | grep node0 | cut -d: -f2)  -d --rm -p 8003:8003 -v $(pwd)/models/Meta-Llama-3-8B-Instruct_FP16:/model:ro openvino/model_server:latest --rest_port 8003 --model_name meta-llama/Meta-Llama-3-8B-Instruct --model_path /model
docker run --cpuset-cpus $(lscpu | grep node1 | cut -d: -f2)  -d --rm -p 8004:8004 -v $(pwd)/models/Meta-Llama-3-8B-Instruct_FP16:/model:ro openvino/model_server:latest --rest_port 8004 --model_name meta-llama/Meta-Llama-3-8B-Instruct --model_path /model
docker run --cpuset-cpus $(lscpu | grep node2 | cut -d: -f2)  -d --rm -p 8005:8005 -v $(pwd)/models/Meta-Llama-3-8B-Instruct_FP16:/model:ro openvino/model_server:latest --rest_port 8005 --model_name meta-llama/Meta-Llama-3-8B-Instruct --model_path /model
docker run --cpuset-cpus $(lscpu | grep node3 | cut -d: -f2)  -d --rm -p 8006:8006 -v $(pwd)/models/Meta-Llama-3-8B-Instruct_FP16:/model:ro openvino/model_server:latest --rest_port 8006 --model_name meta-llama/Meta-Llama-3-8B-Instruct --model_path /model
docker run --cpuset-cpus $(lscpu | grep node4 | cut -d: -f2)  -d --rm -p 8007:8007 -v $(pwd)/models/Meta-Llama-3-8B-Instruct_FP16:/model:ro openvino/model_server:latest --rest_port 8007 --model_name meta-llama/Meta-Llama-3-8B-Instruct --model_path /model
docker run --cpuset-cpus $(lscpu | grep node5 | cut -d: -f2)  -d --rm -p 8008:8008 -v $(pwd)/models/Meta-Llama-3-8B-Instruct_FP16:/model:ro openvino/model_server:latest --rest_port 8008 --model_name meta-llama/Meta-Llama-3-8B-Instruct --model_path /model
```
Confirm in logs if the containers loaded the models successfully.

## Start Nginx load balancer

The configuration below is a basic example distributing the clients between two started instances.
```
events {
    worker_connections 10000;
}
stream {
    upstream ovms-cluster {
        least_conn;
        server localhost:8003;
        server localhost:8004;
        server localhost:8005;
        server localhost:8006;
        server localhost:8007;
        server localhost:8008;
    }
    server {
        listen 80;
        proxy_pass ovms-cluster;
    }
}

```
Start the Nginx container with: 
```bash
docker run -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf:ro -d --net=host -p 80:80 nginx
```

## Testing the scalability

Start benchmarking script like in [demo](../README.md), pointing to the load balancer port and host.
```bash
python benchmark_serving.py --host localhost --port 80 --endpoint /v3/chat/completions --backend openai-chat --model meta-llama/Meta-Llama-3-8B-Instruct --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 6000 --request-rate 20
Initial test run completed. Starting main benchmark run...
Traffic request rate: 20

============ Serving Benchmark Result ============
Successful requests:                     6000
Benchmark duration (s):                  475.21
Total input tokens:                      1321177
Total generated tokens:                  1101195
Request throughput (req/s):              12.61
Output token throughput (tok/s):         2317.29
Total Token throughput (tok/s):          5097.49
---------------Time to First Token----------------
Mean TTFT (ms):                          34457.47
Median TTFT (ms):                        30524.49
P99 TTFT (ms):                           86403.04
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          223.55
Median TPOT (ms):                        238.49
P99 TPOT (ms):                           261.74
```

# Scaling on a multi GPU host

Similarity to multi NUMA node servers, scalability can be achieved on multi GPU systems. The commands below were executed on a host with dual Battlemage B580 GPU cards.

## Start the Model Server instances

Let's assume we have two CPU sockets server with two NUMA nodes. 
```bash
ls -1 /dev/dri/
by-path
card0
card1
renderD128
renderD129
```

Export the model:
```bash
python export_model.py text_generation --source_model meta-llama/Meta-Llama-3-8B-Instruct --model Meta-Llama-3-8B-Instruct_INT4 --weight-format int4 --model_repository_path models --target_device GPU --cache 2
```

```bash
docker run --device /dev/dri/renderD128 -d --rm -p 8003:8003 -u 0 -v $(pwd)/models/Meta-Llama-3-8B-Instruct_INT4:/model:ro openvino/model_server:latest --rest_port 8003 --model_name meta-llama/Meta-Llama-3-8B-Instruct --model_path /model
docker run --device /dev/dri/renderD129 -d --rm -p 8004:8004 -u 0 -v $(pwd)/models/Meta-Llama-3-8B-Instruct_INT4:/model:ro openvino/model_server:latest --rest_port 8004 --model_name meta-llama/Meta-Llama-3-8B-Instruct --model_path /model
```
Confirm in logs if the containers loaded the models successfully.

## Start Nginx load balancer

The configuration below is a basic example distributing the clients between two started instances.
```
events {
    worker_connections 10000;
}
stream {
    upstream ovms-cluster {
        least_conn;
        server localhost:8003;
        server localhost:8004;
    }
    server {
        listen 80;
        proxy_pass ovms-cluster;
    }
}

```
Start the Nginx container with: 
```bash
docker run -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf:ro -d --net=host -p 80:80 nginx
```

## Testing the scalability

Start benchmarking script like in [demo](../README.md), pointing to the load balancer port and host.
```bash
python benchmark_serving.py --host localhost --port 80 --endpoint /v3/chat/completions --backend openai-chat --model meta-llama/Meta-Llama-3-8B-Instruct --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 2000 --request-rate 20
Initial test run completed. Starting main benchmark run...
Traffic request rate: 20

============ Serving Benchmark Result ============
Successful requests:                     2000
Benchmark duration (s):                  203.92
Total input tokens:                      430392
Total generated tokens:                  369419
Request throughput (req/s):              9.88
Output token throughput (tok/s):         1963.46
Total Token throughput (tok/s):          4020.90
---------------Time to First Token----------------
Mean TTFT (ms):                          27419.09
Median TTFT (ms):                        27605.88
P99 TTFT (ms):                           64154.36
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          97.58
Median TPOT (ms):                        104.11
P99 TPOT (ms):                           114.94
```


# Scaling in Kubernetes

TBD
