# Benchmark Client (Python) {#ovms_demo_benchmark_app}

## Introduction

The benchmark client introduced in this directory is written in Python 3. Benchmark client uses TFServing API and KServe API to communicate with model servers. It is recommended to use the benchmark client as a docker container. Prior to transmission, the client downloads metadata from the server, which contains a list of available models, their versions as well as accepted input and output shapes. Then it generates tensors containing random data with shapes matched to the models served by the service. Both the length of the dataset and the workload duration can be specified independently. The synthetic data created is then served in a loop iterating over the dataset until the workload length is satisfied. As the main role of the client is performance measurement all aspects unrelated to throughput and/or latency are ignored. This means the client does not validate the received responses nor does it estimate accuracy as these activities would negatively affect the measured performance metrics on the client side.

In addition to the standard data format, the client also supports stateful models (recognizing dependencies between consecutive
inference requests) as well as binary input for selected file formats (PNG and JPEG).

![urandom generated input image](readme-img-urandom.png) ![xrandom generated input image](readme-img-xrandom.png)


Furthermore the client supports multiple precisions: `FP16`, `FP32`, `FP64`, `INT8`, `INT16`, `INT32`, `INT64`, `UINT8`, `UINT16`, `UINT32`, `UINT64`. Both channel types, insecure and certificate secured, are supported. Secrets/certificates have to be mounted on a separated volume as well as their path has to be specified by command line. The secure connection can be used, for example, to benchmark the Nginx OVMS plugin, which can be build from public source with the built-in Nginx reverse proxy load balancer.

A single docker container can run many parallel clients in separate processes. Measured metrics (especially throughput, latency,
and counters) are collected from all client processes and then combined upon which they can be printed in JSON format/syntax for
the entire parallel workload. If the docker container is run in the daemon mode the final logs can be shown using the `docker logs`
command. Results can also be exported to a Mongo database. In order to do this the appropriate identification metadata has to
be specified in the command line.

Since 2.7 update, these Benchmark Client measurement options were introduced: language models testing with in-built string data input and support for testing `MediaPipe` graphs in the OVMS. For each of them, there is a need to specify the input data method. Data method `-d string` creates sample text data.

Benchmarking of OVMS integrated with MediaPipe is possible for KServe API via gRPC protocol. In this case there is also a necessity to feed client with a pre-prepared `numpy` file consisting of a numpy array.

There is also an option to test dynamic models using `shape` parameter.

Last two use cases are furthermore described in this document.

## Build Benchmark Client Docker Image

To build the docker image and tag it as `benchmark_client` run:
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/benchmark/python
docker build . -t benchmark_client
```

## OVMS Deployment

First of all, download a model and create an appropriate directory tree. For example, for the resnet50 model:

```bash
mkdir -p ${HOME}/models
curl -L https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.bin -o ${HOME}/models/resnet50.bin
curl -L https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.xml -o ${HOME}/models/resnet50.xml
```

Let's start OVMS before building and running the benchmark client as follows (more deployment options described in [docs](../../../docs/home.md)):
```bash
docker run --rm -d -u $(id -u) -v ${HOME}/models:/models -p 9000:9000 -p 8000:8000 \
  openvino/model_server:latest \
  --model_name resnet --model_path /models/resnet50.xml \
  --mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" --layout "NHWC:NCHW" \
  --port 9000 --rest_port 8000
```

## Selected Commands

To check available options use `-h`, `--help` switches:
```bash
  docker run benchmark_client --help

Client 2.7
NO_PROXY=localhost no_proxy=localhost python3 /ovms_benchmark_client/main.py --help
usage: main.py [-h] [-i ID] [-c CONCURRENCY] [-a SERVER_ADDRESS]
               [-p GRPC_PORT] [-r REST_PORT] [-l] [-b [BS ...]]
               [-s [SHAPE ...]] [-d [DATA ...]] [-j] [-m MODEL_NAME]
               [-k DATASET_LENGTH] [-v MODEL_VERSION] [-n STEPS_NUMBER]
               [-t DURATION] [-u WARMUP] [-w WINDOW] [-e ERROR_LIMIT]
               [-x ERROR_EXPOSITION] [--max_throughput MAX_THROUGHPUT]
               [--max_value MAX_VALUE] [--min_value MIN_VALUE] [--xrand XRAND]
               [--dump_png] [--step_timeout STEP_TIMEOUT]
               [--metadata_timeout METADATA_TIMEOUT] [-Y DB_ENDPOINT]
               [-y [DB_METADATA ...]] [--print_all] [-ps] [--print_time]
               [--report_warmup] [--certs_dir CERTS_DIR] [-q STATEFUL_LENGTH]
               [--stateful_id STATEFUL_ID] [--stateful_hop STATEFUL_HOP]
               [--sync_interval SYNC_INTERVAL]
               [--quantile_list [QUANTILE_LIST ...]]
               [--hist_factor HIST_FACTOR] [--hist_base HIST_BASE]
               [--internal_version] [--unbuffered] [--api {KFS,REST}]

    This is benchmarking client which uses KFS/REST API to communicate with OVMS/KFS/REST-based-services.
```

The version can be checked by using `--internal_version` switch as follows:
```bash
  docker run benchmark_client --internal_version

  Client 2.7
  NO_PROXY=localhost no_proxy=localhost python3 /ovms_benchmark_client/main.py --internal_version
  2.7
```

The client is able to download the metadata of the served models. If you are
unsure which models and versions are served and what status they have, you can
list this information by specifying the `--list_models` switch (also a short
form `-l` is available):
```bash
docker run --network host benchmark_client -a localhost -r 8000 --list_models

Client 2.7
NO_PROXY=localhost no_proxy=localhost python3 /ovms_benchmark_client/main.py -a localhost -r 8000 --list_models
          XW worker: Finished execution. If you want to run inference remove --list_models.
          XI worker: try to send request to endpoint: http://localhost:8000/v1/config
          XI worker: received status code is 200.
          XI worker: found models and their status:
          XI worker:  model: resnet, version: 1 - AVAILABLE
```
## Sample benchmarks

Names, model shape, as well as information about data types of both inputs and
outputs can also be downloaded for all available models using the same listing
switches and adding `-m <model-name>` and `-v <model-version>` to the command
line. The option `-i` is used only to add a prefix to the standard output with a name
of an application instance. For example:
```bash
docker run --network host benchmark_client -a localhost -r 8000 -l -m resnet -p 9000 -i id

Client 2.7
NO_PROXY=localhost no_proxy=localhost python3 /ovms_benchmark_client/main.py -a localhost -r 8000 -l -m resnet -p 9000 -i id
          XW id: Finished execution. If you want to run inference remove --list_models.
          XI id: try to send request to endpoint: http://localhost:8000/v1/config
          XI id: received status code is 200.
          XI id: found models and their status:
          XI id:  model: resnet, version: 1 - AVAILABLE
          XI id: request for metadata of model resnet...
          XI id: Metadata for model resnet is downloaded...
          XI id: set version of model resnet: 1
          XI id: name: "resnet"
              versions: "1"
              platform: "OpenVINO"
              inputs {
                name: "image"
                datatype: "FP32"
                shape: 1
                shape: 224
                shape: 224
                shape: 3
              }
              outputs {
                name: "output"
                datatype: "FP32"
                shape: 1
                shape: 1000
              }
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
x

Client 2.7
NO_PROXY=localhost no_proxy=localhost python3 /ovms_benchmark_client/main.py -a localhost -r 8000 -m resnet -p 9000 -n 8 --report_warmup --print_all
         XI worker: request for metadata of model resnet...
          XI worker: Metadata for model resnet is downloaded...
          XI worker: set version of model resnet: 1
          XI worker: name: "resnet"
              versions: "1"
              platform: "OpenVINO"
              inputs {
                name: "image"
                datatype: "FP32"
                shape: 1
                shape: 224
                shape: 224
                shape: 3
              }
              outputs {
                name: "output"
                datatype: "FP32"
                shape: 1
                shape: 1000
              }
          XI worker: new random range: 0.0, 255.0
          XI worker: batchsize sequence: [1]
          XI worker: dataset length (image): 1
          XI worker: --> dim: 1
          XI worker: --> dim: 224
          XI worker: --> dim: 224
          XI worker: --> dim: 3
          XI worker: Generated data shape: (1, 224, 224, 3)
          XI worker: start workload...
          XI worker: stop warmup: 6308546.650373343
          XI worker: stop window: inf
          XI worker: Workload started!
          XI worker: Warmup normally stopped: 6308546.655597869
          XI worker: Window normally start: 6308546.655628495
          XI worker: Window stopped: 6308546.677919754
          XI worker: total_duration: 0.027586651034653187
          XI worker: total_batches: 8
          XI worker: total_frames: 8
          XI worker: start_timestamp: 6308546.65037293
          XI worker: stop_timestamp: 6308546.677959581
          XI worker: pass_batches: 8
          XI worker: fail_batches: 0
          XI worker: pass_frames: 8
          XI worker: fail_frames: 0
          XI worker: first_latency: 0.00516059435904026
          XI worker: pass_max_latency: 0.00516059435904026
          XI worker: fail_max_latency: 0.0
          XI worker: brutto_batch_rate: 289.9953310733781
          XI worker: brutto_frame_rate: 289.9953310733781
          XI worker: netto_batch_rate: 293.82200804483256
          XI worker: netto_frame_rate: 293.82200804483256
          XI worker: frame_passrate: 1.0
          XI worker: batch_passrate: 1.0
          XI worker: mean_latency: 0.003403421025723219
          XI worker: mean_latency2: 1.2081388128051991e-05
          XI worker: stdev_latency: 0.0007057715279869992
          XI worker: cv_latency: 0.20737120757400987
          XI worker: pass_mean_latency: 0.003403421025723219
          XI worker: pass_mean_latency2: 1.2081388128051991e-05
          XI worker: pass_stdev_latency: 0.0007057715279869992
          XI worker: pass_cv_latency: 0.20737120757400987
          XI worker: fail_mean_latency: 0.0
          XI worker: fail_mean_latency2: 0.0
          XI worker: fail_stdev_latency: 0.0
          XI worker: fail_cv_latency: 0.0
          XI worker: window_total_duration: 0.02229125890880823
          XI worker: window_total_batches: 8
          XI worker: window_total_frames: 8
          XI worker: window_start_timestamp: 6308546.655628495
          XI worker: window_stop_timestamp: 6308546.677919754
          XI worker: window_pass_batches: 8
          XI worker: window_fail_batches: 0
          XI worker: window_pass_frames: 8
          XI worker: window_fail_frames: 0
          XI worker: window_first_latency: 0.00516059435904026
          XI worker: window_pass_max_latency: 0.00516059435904026
          XI worker: window_fail_max_latency: 0.0
          XI worker: window_brutto_batch_rate: 358.88506937752436
          XI worker: window_brutto_frame_rate: 358.88506937752436
          XI worker: window_brutto_batch_rate: 358.88506937752436
          XI worker: window_brutto_frame_rate: 358.88506937752436
          XI worker: window_netto_batch_rate: 293.82200804483256
          XI worker: window_netto_frame_rate: 293.82200804483256
          XI worker: window_frame_passrate: 1.0
          XI worker: window_batch_passrate: 1.0
          XI worker: window_mean_latency: 0.003403421025723219
          XI worker: window_mean_latency2: 1.2081388128051991e-05
          XI worker: window_stdev_latency: 0.0007057715279869992
          XI worker: window_cv_latency: 0.20737120757400987
          XI worker: window_pass_mean_latency: 0.003403421025723219
          XI worker: window_pass_mean_latency2: 1.2081388128051991e-05
          XI worker: window_pass_stdev_latency: 0.0007057715279869992
          XI worker: window_pass_cv_latency: 0.20737120757400987
          XI worker: window_fail_mean_latency: 0.0
          XI worker: window_fail_mean_latency2: 0.0
          XI worker: window_fail_stdev_latency: 0.0
          XI worker: window_fail_cv_latency: 0.0
          XI worker: window_hist_latency_0: 8
          XI worker: warmup_total_duration: 0.0052260542288422585
          XI worker: warmup_total_batches: 0
          XI worker: warmup_total_frames: 0
          XI worker: warmup_start_timestamp: 6308546.650371815
          XI worker: warmup_stop_timestamp: 6308546.655597869
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

## Dynamic models benchmarking

In order to test dynamic models, `-s` (shape) parameter needs to be specified. You can use dynamic model, although some static models also can have dynamic shape specified. First download the model to workspace.
```bash
mkdir -p workspace/face-detection-retail-0005/1
cd workspace/face-detection-retail-0005/1
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-detection-retail-0005/FP32/face-detection-retail-0005.bin
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-detection-retail-0005/FP32/face-detection-retail-0005.xml
cd ../../..
```
Next start OVMS having dynamic input shape specified.
```bash
docker run -u $(id -u) -p 9000:9000 -p 8000:8000 -d -v ${PWD}/workspace:/workspace openvino/model_server --model_path /workspace/face-detection-retail-0005 --model_name face-detection-retail-0005 --shape "(-1,3,-1,-1)" --port 9000 --rest_port 8000
```
To generate request with specified shape, it is necessary to set input shape explicitly. It is done by specifying `-s` or `--shape` parameter, followed by desired numbers.
```bash
docker run --network host benchmark_client -a localhost -r 8000 -m face-detection-retail-0005 -p 9000 -n 8 -s 1 3 300 300 --print_all
Client 2.7
NO_PROXY=localhost no_proxy=localhost python3 /ovms_benchmark_client/main.py -a localhost -r 8000 -m face-detection-retail-0005 -p 9000 -n 8 -s 1 3 300 300 --print_all
          XI worker: request for metadata of model face-detection-retail-0005...
          XI worker: Metadata for model face-detection-retail-0005 is downloaded...
          XI worker: set version of model face-detection-retail-0005: 1
          XI worker: inputs:
          XI worker:  input.1:
          XI worker:   name: input.1
          XI worker:   dtype: DT_FLOAT
          XI worker:   tensorShape: {'dim': [{'size': '-1'}, {'size': '3'}, {'size': '-1'}, {'size': '-1'}]}
          XI worker: outputs:
          XI worker:  527:
          XI worker:   name: 527
          XI worker:   dtype: DT_FLOAT
          XI worker:   tensorShape: {'dim': [{'size': '1'}, {'size': '1'}, {'size': '-1'}, {'size': '7'}]}
          XI worker: new random range: 0.0, 255.0
          XI worker: batchsize sequence: [1]
          XI worker: dataset length (input.1): 1
          XI worker: --> dim: 1
          XI worker: --> dim: 3
          XI worker: --> dim: 300
          XI worker: --> dim: 300
          XI worker: Generated data shape: (1, 3, 300, 300)
          XI worker: start workload...
...
```

## Summarize benchmarking results

Summary of the benchmark results can be viewed with command option ```-ps```.
```
docker run --network host benchmark_client -a localhost -r 8000 -m face-detection-retail-0005 -p 9000 -s 2 3 300 300 -t 20 -u 2 -w 10 -ps
```

Sample output log with results summary:

```
Client 2.7
NO_PROXY=localhost no_proxy=localhost python3 /ovms_benchmark_client/main.py -a localhost -r 8000 -m face-detection-retail-0005 -p 9000 -s 2 3 300 300 -t 20 -u 2 -w 10 -ps
          XI worker: start workload...

### Benchmark Parameters ###
 Model: face-detection-retail-0005
 Input shape: ['2', '3', '300', '300']
 Request concurrency: 1
 Test Duration (s): Total (t): 20.00 | Warmup (u): 2.00 | Window (w): 10.00

### Benchmark Summary ###
 ## General Metrics ##
 Duration(s): Total: 20.01 | Window: 10.01
 Batches: Total: 1781 | Window: 891

 ## Latency Metrics (ms) ##
 Mean: 11.20 | stdev: 0.74 | p50: 12.78 | p90: 15.26 | p95: 15.56

 ## Throughput Metrics (fps) ##
 Frame Rate (FPS): Brutto: 89.01 | Netto: 89.24
 Batch Rate (batches/s): Brutto: 89.01 | Netto: 89.24
```
## MediaPipe benchmarking

Reuse the resnet50 model downloaded earlier in the [OVMS Deployment](#ovms-deployment) step (`${HOME}/models`), then start OVMS container with `config.json` including mediapipe servable. OVMS should be built with MediaPipe enabled.
```bash
docker run -u $(id -u) -p 9000:9000 -p 8000:8000 -d \
  -v ${HOME}/models:/workspace \
  -v ${PWD}/sample_data:/workspace/sample_data \
  openvino/model_server --port 9000 --rest_port 8000 --config_path /workspace/sample_data/config.json
```
Requests for benchmarking are prepared basing on array from a numpy file. This data file is fed to Benchmark Client by specifying switch `-d <data-file>.npy`. Note that we can use numpy data in the same manner also for single models and pipelines if KServe API is set. You can create sample data with `Python3`, specifying array shape and precision. Generated .npy file should be saved to sample_data directory for this example.
```bash
python -c 'import numpy as np ; \
arr = np.ones((1,3,224,224),dtype=np.float32); \
np.save("sample_data/resnet50.npy", arr)'
```
Having MediaPipe graph file and servable specified in config.json, we call it by its name instead of the model name: `-m <mediapipe-servable-name>`. It is necessary to set `--api KFS` since the Mediapipe graphs are exposed only via KServe API.
```bash
docker run -v ${PWD}/sample_data:/workspace/sample_data --network host benchmark_client -a localhost -r 8000 -m resnet_mediapipe -p 9000 -n 8 --api KFS -d /workspace/sample_data/resnet50.npy --report_warmup --print_all
```

Many other client options together with benchmarking examples are presented in
[an additional PDF document](https://github.com/openvinotoolkit/model_server/blob/releases/2026/4/docs/python-benchmarking-client-16feb.pdf).
