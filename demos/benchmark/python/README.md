# Benchmark Client (Python) {#ovms_demo_benchmark_app}

## Introduction

The benchmark client introduced in this directory is written in Python 3. Benchmark client uses TFServing API and KServe API to communicate with model servers. It is recommended to use the benchmark client as a docker container. Prior to transmission, the client downloads metadata from the server, which contains a list of available models, their versions as well as accepted input and output shapes. Then it generates tensors containing random data with shapes matched to the models served by the service. Both the length of the dataset and the workload duration can be specified independently. The synthetic data created is then served in a loop iterating over the dataset until the workload length is satisfied. As the main role of the client is performance measurement all aspects unrelated to throughput and/or latency are ignored. This means the client does not validate the received responses nor does it estimate accuracy as these activities would negatively affect the measured performance metrics on the client side.

In addition to the standard data format, the client also supports stateful models (recognizing dependencies between consecutive
inference requests) as well as binary input for select file formats (PNG and JPEG).

![urandom generated input image](readme-img-urandom.png) ![xrandom generated input image](readme-img-xrandom.png)


Furthermore the client supports multiple precisions: `FP16`, `FP32`, `FP64`, `INT8`, `INT16`, `INT32`, `INT64`, `UINT8`, `UINT16`, `UINT32`, `UINT64`. Both channel types, insecure and certificate secured, are supported. Secrets/certificates have to be mounted on a separated volume as well as their path has to be specified by command line. The secure connection can be used, for example, to benchmark the Nginx OVMS plugin, which can be build from public source with the built-in Nginx reverse proxy load balancer.

A single docker container can run many parallel clients in separate processes. Measured metrics (especially throughput, latency,
and counters) are collected from all client processes and then combined upon which they can be printed in JSON format/syntax for
the entire parallel workload. If the docker container is run in the deamon mode the final logs can be shown using the `docker logs`
command. Results can also be exported to a Mongo database. In order to do this the appropriate identification metadata has to
be specified in the command line.

## OVMS Deployment

First of all, download a model and create an appropriate directory tree. For example, for some resnet 50 model from Intel's Open Model Zoo:

```bash
mkdir -p workspace/resnet50-binary-0001/1
cd workspace/resnet50-binary-0001/1
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin
cd ../../..
```

Let's start OVMS before building and running the benchmark client as follows:
```bash
docker run -p 30001:30001 -p 30002:30002 -d -v ${PWD}/workspace:/workspace openvino/model_server --model_path \
                     /workspace/resnet50-binary-0001 --model_name resnet50-binary-0001 --port 30001 --rest_port 30002
```
where a model directory looks like that:
```bash
workspace
└── resnet50-binary-0001
    └── 1
        ├── resnet50-binary-0001.bin
        └── resnet50-binary-0001.xml
```

## Build Client Docker Image

To build the docker image and tag it as `benchmark_client` run:
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/benchmark/python
docker build . -t benchmark_client
```

## Selected Commands

To check available options use `-h`, `--help` switches:
```bash
  docker run benchmark_client --help

usage: main.py [-h] [-i ID] [-c CONCURRENCY] [-a SERVER_ADDRESS]
               [-p GRPC_PORT] [-r REST_PORT] [-l] [-b [BS [BS ...]]]
               [-s [SHAPE [SHAPE ...]]] [-d [DATA [DATA ...]]] [-j]
               [-m MODEL_NAME] [-k DATASET_LENGTH] [-v MODEL_VERSION]
               [-n STEPS_NUMBER] [-t DURATION] [-u WARMUP] [-w WINDOW]
               [-e ERROR_LIMIT] [-x ERROR_EXPOSITION]
               [--max_throughput MAX_THROUGHPUT] [--max_value MAX_VALUE]
               [--min_value MIN_VALUE] [--xrand XRAND] [--dump_png]
               [--step_timeout STEP_TIMEOUT]
               [--metadata_timeout METADATA_TIMEOUT] [-Y DB_ENDPOINT]
               [-y [DB_METADATA [DB_METADATA ...]]] [--print_all]
               [--print_time] [--report_warmup] [--certs_dir CERTS_DIR]
               [-q STATEFUL_LENGTH] [--stateful_id STATEFUL_ID]
               [--stateful_hop STATEFUL_HOP] [--sync_interval SYNC_INTERVAL]
               [--quantile_list [QUANTILE_LIST [QUANTILE_LIST ...]]]
               [--hist_factor HIST_FACTOR] [--hist_base HIST_BASE]
               [--internal_version] [--unbuffered] [--api {TFS,KFS,REST}]

This is benchmarking client which uses TFS/KFS API to communicate with
OVMS/TFS/KFS-based-services.
```

The version can be checked by using `--internal_version` switch as follows:
```bash
  docker run benchmark_client --internal_version

  2.6
```

The client is able to download the metadata of the served models. If you are
unsure which models and versions are served and what status they have, you can
list this information by specifying the `--list_models` switch (also a short
form `-l` is available):
```bash
docker run --network host benchmark_client -a localhost -r 30002 --list_models

Client 2.6
NO_PROXY=localhost no_proxy=localhost python3 /ovms_benchmark_client/main.py -a localhost -r 30002 --list_models
XI worker: try to send request to endpoint: http://localhost:30002/v1/config
XI worker: received status code is 200.
XI worker: found models and their status:
XI worker:  model: resnet50-binary-0001, version: 1 - AVAILABLE

```
Names, model shape, as well as information about data types of both inputs and
outputs can also be downloaded for all available models using the same listing
switches and adding `-m <model-name>` and `-v <model-version>` to the command
line. The option `-i` is used only to add a prefix to the standard output with a name
of an application instance. For example:
```bash
docker run --network host benchmark_client -a localhost -r 30002 -l -m resnet50-binary-0001 -p 30001 -i id

Client 2.6
NO_PROXY=localhost no_proxy=localhost python3 /ovms_benchmark_client/main.py -a localhost -r 30002 -l -m resnet50-binary-0001 -p 30001 -i id
XW id: Finished execution. If you want to run inference remove --list_models.
XI id: try to send request to endpoint: http://localhost:30002/v1/config
XI id: received status code is 200.
XI id: found models and their status:
XI id:  model: resnet50-binary-0001, version: 1 - AVAILABLE
XI id: request for metadata of model resnet50-binary-0001...
XI id: Metadata for model resnet50-binary-0001 is downloaded...
XI id: set version of model resnet50-binary-0001: 1
XI id: inputs:
XI id:  0:
XI id:   name: 0
XI id:   dtype: DT_FLOAT
XI id:   tensorShape: {'dim': [{'size': '1'}, {'size': '3'}, {'size': '224'}, {'size': '224'}]}
XI id: outputs:
XI id:  1463:
XI id:   name: 1463
XI id:   dtype: DT_FLOAT
XI id:   tensorShape: {'dim': [{'size': '1'}, {'size': '1000'}]}

```
Be sure the model name specified is identical to the model name shown when using
the `--list_models` parameter. A model version is not required but it can be added
when multiple versions are available for a specific model name.

The introduced benchmark client supports generation of requests with multiple and
different batch sizes in a single workload. The switches `-b`, `--bs` can be used
to specify this parameter.

The workload can be generated only if its length is specified by iteration number
`-n`, `--steps_number` or duration length `-t`, `--duration`. To see report also on warmup time window use `--report_warmup` switch. Example for 8 requests
will be generated as follows (remember to add `--print_all` to show metrics in stdout):
```bash
docker run --network host benchmark_client -a localhost -r 30002 -m resnet50-binary-0001 -p 30001 -n 8 --report_warmup --print_all

Client 2.6
NO_PROXY=localhost no_proxy=localhost python3 /ovms_benchmark_client/main.py -a localhost -r 30002 -m resnet50-binary-0001 -p 30001 -n 8 --report_warmup --print_all
XI worker: request for metadata of model resnet50-binary-0001...
XI worker: Metadata for model resnet50-binary-0001 is downloaded...
XI worker: set version of model resnet50-binary-0001: 1
XI worker: inputs:
XI worker:  0:
XI worker:   name: 0
XI worker:   dtype: DT_FLOAT
XI worker:   tensorShape: {'dim': [{'size': '1'}, {'size': '3'}, {'size': '224'}, {'size': '224'}]}
XI worker: outputs:
XI worker:  1463:
XI worker:   name: 1463
XI worker:   dtype: DT_FLOAT
XI worker:   tensorShape: {'dim': [{'size': '1'}, {'size': '1000'}]}
XI worker: new random range: 0.0, 255.0
XI worker: batchsize sequence: [1]
XI worker: dataset length (0): 1
XI worker: --> dim: 1
XI worker: --> dim: 3
XI worker: --> dim: 224
XI worker: --> dim: 224
XI worker: Generated data shape: (1, 3, 224, 224)
XI worker: start workload...
XI worker: stop warmup: 374943.047975389
XI worker: stop window: inf
XI worker: Workload started!
XI worker: Warmup normally stopped: 374943.075028319
XI worker: Window normally start: 374943.07504102
XI worker: Window stopped: 374943.354446821
XI worker: total_duration: 0.3065074360347353
XI worker: total_batches: 8
XI worker: total_frames: 8
XI worker: start_timestamp: 374943.047975189
XI worker: stop_timestamp: 374943.354482625
XI worker: pass_batches: 8
XI worker: fail_batches: 0
XI worker: pass_frames: 8
XI worker: fail_frames: 0
XI worker: first_latency: 0.027021727000828832
XI worker: pass_max_latency: 0.0449919409584254
XI worker: fail_max_latency: 0.0
XI worker: brutto_batch_rate: 26.100508697261724
XI worker: brutto_frame_rate: 26.100508697261724
XI worker: netto_batch_rate: 26.127558971388062
XI worker: netto_frame_rate: 26.127558971388062
XI worker: frame_passrate: 1.0
XI worker: batch_passrate: 1.0
XI worker: mean_latency: 0.0382737630061456
XI worker: mean_latency2: 0.0015027467953827884
XI worker: stdev_latency: 0.006153524252994284
XI worker: cv_latency: 0.16077656780197483
XI worker: pass_mean_latency: 0.0382737630061456
XI worker: pass_mean_latency2: 0.0015027467953827884
XI worker: pass_stdev_latency: 0.006153524252994284
XI worker: pass_cv_latency: 0.16077656780197483
XI worker: fail_mean_latency: 0.0
XI worker: fail_mean_latency2: 0.0
XI worker: fail_stdev_latency: 0.0
XI worker: fail_cv_latency: 0.0
XI worker: window_total_duration: 0.27940580097492784
XI worker: window_total_batches: 8
XI worker: window_total_frames: 8
XI worker: window_start_timestamp: 374943.07504102
XI worker: window_stop_timestamp: 374943.354446821
XI worker: window_pass_batches: 8
XI worker: window_fail_batches: 0
XI worker: window_pass_frames: 8
XI worker: window_fail_frames: 0
XI worker: window_first_latency: 0.027021727000828832
XI worker: window_pass_max_latency: 0.0449919409584254
XI worker: window_fail_max_latency: 0.0
XI worker: window_brutto_batch_rate: 28.632190069374655
XI worker: window_brutto_frame_rate: 28.632190069374655
XI worker: window_netto_batch_rate: 26.127558971388062
XI worker: window_netto_frame_rate: 26.127558971388062
XI worker: window_frame_passrate: 1.0
XI worker: window_batch_passrate: 1.0
XI worker: window_mean_latency: 0.0382737630061456
XI worker: window_mean_latency2: 0.0015027467953827884
XI worker: window_stdev_latency: 0.006153524252994284
XI worker: window_cv_latency: 0.16077656780197483
XI worker: window_pass_mean_latency: 0.0382737630061456
XI worker: window_pass_mean_latency2: 0.0015027467953827884
XI worker: window_pass_stdev_latency: 0.006153524252994284
XI worker: window_pass_cv_latency: 0.16077656780197483
XI worker: window_fail_mean_latency: 0.0
XI worker: window_fail_mean_latency2: 0.0
XI worker: window_fail_stdev_latency: 0.0
XI worker: window_fail_cv_latency: 0.0
XI worker: window_hist_latency_4: 2
XI worker: window_hist_latency_9: 1
XI worker: window_hist_latency_8: 5
XI worker: warmup_total_duration: 0.02705443004379049
XI worker: warmup_total_batches: 0
XI worker: warmup_total_frames: 0
XI worker: warmup_start_timestamp: 374943.047973889
XI worker: warmup_stop_timestamp: 374943.075028319
XI worker: warmup_pass_batches: 0
XI worker: warmup_fail_batches: 0
XI worker: warmup_pass_frames: 0
XI worker: warmup_fail_frames: 0
XI worker: warmup_first_latency: inf
XI worker: warmup_pass_max_latency: 0.0
XI worker: warmup_fail_max_latency: 0.0
XI worker: warmup_brutto_batch_rate: 0.0
XI worker: warmup_brutto_frame_rate: 0.0
XI worker: warmup_netto_batch_rate: 0.0
XI worker: warmup_netto_frame_rate: 0.0
XI worker: warmup_frame_passrate: 0.0
XI worker: warmup_batch_passrate: 0.0
XI worker: warmup_mean_latency: 0.0
XI worker: warmup_mean_latency2: 0.0
XI worker: warmup_stdev_latency: 0.0
XI worker: warmup_cv_latency: 0.0
XI worker: warmup_pass_mean_latency: 0.0
XI worker: warmup_pass_mean_latency2: 0.0
XI worker: warmup_pass_stdev_latency: 0.0
XI worker: warmup_pass_cv_latency: 0.0
XI worker: warmup_fail_mean_latency: 0.0
XI worker: warmup_fail_mean_latency2: 0.0
XI worker: warmup_fail_stdev_latency: 0.0
XI worker: warmup_fail_cv_latency: 0.0
```
Many other client options together with benchmarking examples are presented in
[an additional PDF document](https://github.com/openvinotoolkit/model_server/blob/releases/2023/0/docs/python-benchmarking-client-16feb.pdf). 