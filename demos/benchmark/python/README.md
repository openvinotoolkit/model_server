# Benchmark Client for OVMS {#ovms_demo_benchmark_app}

## Introduction

The benchmark client introduced in this directory is written in Python 3. It is recommended to use the benchmark client as a
docker container. Prior to transmission, the client downloads metadata from the server, which contains a list of available models,
their versions as well as accepted input and output shapes. Then it generates tensors containing random data with shapes matched
to the models served by the service. Both the length of the dataset and the workload duration can be specified independently. The
synthetic data created is then served in a loop iterating over the dataset until the workload length is satisfied. As the main role
of the client is performance measurement all aspects unrelated to throughput and/or latency are ignored. This means the client does
not validate the received responses nor does it estimate accuracy as these activities would negatively affect the measured performance
metrics on the client side.

In addition to the standard data format, the client also supports stateful models (recognizing dependencies between consecutive
inference requests) as well as binary input for select file formats (PNG and JPEG). Both channel types, insecure and certificate
secured, are supported. Secrets/certificates have to be mounted on a separated volume as well as their path has to be specified by
command line. The secure connection can be used, for example, to benchmark the Nginx OVMS plugin, which can be build from public 
source with the built-in Nginx reverse proxy load balancer.

A single docker container can run many parallel clients in separate processes. Measured metrics (especially throughput, latency,
and counters) are collected from all client processes and then combined upon which they can be printed in JSON format/syntax for
the entire parallel workload. If the docker container is run in the deamon mode the final logs can be shown using the `docker logs`
command. Results can also be exported to a Mongo database. In order to do this the appropriate identification metadata has to
be specified in the command line.

## OVMS Deployment

Let's prepare and start OVMS before building and running the benchmark client:
```
docker run -p 30001:30001 -p 30002:30002 -d -v ${PWD}/workspace:/workspace openvino/model_server --model_path \
                     /workspace/resnet50-tf-fp32 --model_name resnet50-tf-fp32 --port 30001 --rest_port 30002
```
where a model directory looks like that:
```
workspace
└── resnet50-tf-fp32
    └── 1
        ├── resnet50-tf-fp32.bin
        └── resnet50-tf-fp32.xml
```

## Build Client Docker Image

To build the docker image and tag it as `benchmark_client` run:
```
docker build . -t benchmark_client
```


## Selected Commands

To check available option use `-h`, `--help` switches:
```
  docker run benchmark_client --help

  [-h] [-i ID] [-c CONCURRENCY] [-a SERVER_ADDRESS]
  [-p GRPC_PORT] [-r REST_PORT] [-l] [-b [BS [BS ...]]]
  [-s [SHAPE [SHAPE ...]]] [-d [DATA [DATA ...]]] [-j]
  [-m MODEL_NAME] [-k DATASET_LENGTH] [-v MODEL_VERSION]
  [-n STEPS_NUMBER] [-t DURATION] [-u WARMUP] [-w WINDOW]
  [-e ERROR_LIMIT] [-x ERROR_EXPOSITION] [--max_value MAX_VALUE]
  [--min_value MIN_VALUE] [--step_timeout STEP_TIMEOUT]
  [--metadata_timeout METADATA_TIMEOUT] [-y DB_CONFIG]
  [--print_all] [--certs_dir CERTS_DIR] [-q STATEFUL_LENGTH]
  [--stateful_id STATEFUL_ID] [--stateful_hop STATEFUL_HOP]
  [--nv_triton] [--sync_interval SYNC_INTERVAL]
  [--quantile_list [QUANTILE_LIST [QUANTILE_LIST ...]]]
  [--hist_factor HIST_FACTOR] [--hist_base HIST_BASE]
  [--internal_version]


This is a benchmarking client (version 1.17) which uses TF API over the gRPC
internet protocol to communicate with serving services (like OVMS, TFS, etc.).
```

The version can be checked by using `--internal_version` switch as follows:
```
  docker run benchmark_client --internal_version

  1.17
```

The client is able to download the metadata of the served models. If you are
unsure which models and versions are served and what status they have, you can
list this information by specifying the `--list_models` switch (also a short
form `-l` is available):
```
  docker run benchmark_client -a 10.91.242.153 -r 30002 --list_models
  
  XI worker: try to send request to endpoint: http://10.91.242.153:30002/v1/config
  XI worker: received status code is 200.
  XI worker: found models and their status:
  XI worker:  model: resnet50-tf-fp32, version: 1 - AVAILABLE
  XI worker:  model: resnet50-tf-int8, version: 1 - AVAILABLE
```
Names, model shape, as well as information about data types of both inputs and
outputs can also be downloaded for all available models using the same listing
switches and adding `-m <model-name>` and `-v <model-version>` to the command
line. The option `-i` is only to add a prefix to the standard output with a name
of an application instance. For example:
```
  docker run benchmark_client -a 10.91.242.153 -r 30002 -l -m resnet50-tf-fp32 -p 30001 -i id
  
  XI id: try to send request to endpoint: http://10.91.242.153:30002/v1/config
  XI id: received status code is 200.
  XI id: found models and their status:
  XI id:  model: resnet50-tf-fp32, version: 1 - AVAILABLE
  XI id: request for metadata of model resnet50-tf-fp32...
  XI id: Metadata for model resnet50-tf-fp32 is downloaded...
  XI id: set version of model resnet50-tf-fp32: 1
  XI id: inputs:
  XI id:  input_name_1:
  XI id:   name: input_name_1
  XI id:   dtype: DT_FLOAT
  XI id:   tensorShape: {'dim': [{'size': '1'}, {'size': '3'}, {'size': '224'}, {'size': '224'}]}
  XI id: outputs:
  XI id:  output_name_1:
  XI id:   name: output_name_1
  XI id:   dtype: DT_FLOAT
  XI id:   tensorShape: {'dim': [{'size': '1'}, {'size': '1001'}]}
```
Be sure the model name specified is identical to the model name shown when using
the `--list_models` parameter. A model version is not required but it can be added
when multiple versions are available for a specific model name.

The introduced benchmark client supports generation of requests with multiple and
different batch sizes in a single workload. The switches `-b`, `--bs` can be used
to specify this parameter. For example, in order to set multiple batch sizes --
let's say 1, 2, 4, 2 -- the following command can be called:
```
docker run benchmark_client -a 10.91.242.153 -p 30001 -m resnet50-tf-fp32 -b 1-2-4-2 -n 8
```
This phrase means that 4 batches with random data will be generated. They will
include respectively 1, 2, 4, and 2 images -- 9 in total. The client will send
each batch 2 times (in the following order: 1, 2, 4, 2, 1, 2, 4, 2) because 8
iterations are required.
