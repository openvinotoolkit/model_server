# Benchmark KServe Endpoints {#ovms_demo_benchmark_kserve}

## Introduction

The KServe V2 inference API standardizes model metadata and inference requests over
gRPC and HTTP/REST. OpenVINO Model Server (OVMS) exposes this API for classic deep
learning models, including image classification, object detection, and other
tensor-based workloads. This demo benchmarks ResNet-50 image classification; it
does not cover generative models served through the OpenAI-compatible API.

[benchmark.py](benchmark.py) uses the Triton Python client to benchmark both REST
and gRPC on Linux and Windows. It only requests KServe model metadata and
inference; it does not depend on Triton-specific model configuration endpoints.
It supports multiple inputs, dynamic shapes, and binary files such as JPEG images.

The guide also covers NVIDIA perf_analyzer. Both tools can benchmark gRPC;
perf_analyzer REST requires an OVMS build with the model-configuration extension.

## Prerequisites

This deployment follows the [classic model quickstart](../../../README.md#serve-a-classic-model-with-kserve-api),
adding a REST endpoint alongside gRPC. Linux requires Docker. On Windows, install
the [OVMS binary package](../../../docs/deploying_server_baremetal.md) and make
`ovms.exe` available in your command prompt. Run these commands from a writable
directory, and keep the downloaded files there while the server is running.

Download the model on either OS:

```console
curl -fL https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.bin -o resnet50.bin
curl -fL https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.xml -o resnet50.xml
```

If the ports are occupied, choose unused ports and update the client URLs below.
For Docker, change the host side of each port mapping, for example
`-p 19000:9000 -p 18000:8000`.

The server's `PERFORMANCE_HINT` selects an OpenVINO execution strategy:
`LATENCY` for one client, or `THROUGHPUT` for concurrent clients. These are server
settings, not benchmark-client flags. They can improve performance for the selected
workload, but gains depend on hardware and load. Do not combine these hints with
an explicit `NUM_STREAMS` setting. See [performance tuning](../../../docs/performance_tuning.md).

## Use python script benchmark.py

### Installation

Use Python 3.12. Download the client files into the same writable directory as
the model files:

```console
curl -fL https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/benchmark/kserve/benchmark.py -o benchmark.py
curl -fL https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/benchmark/kserve/requirements.txt -o requirements.txt
```

On Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use a native command prompt; WSL is not required:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

Print help on either OS:

```console
python benchmark.py --help
```

### Parameters

| Parameter | Meaning | Default |
|---|---|---|
| `--model`, `-m` | Served model name | Required |
| `--model-version` | Specific model version | Server-selected version |
| `--endpoint` | `rest` or `grpc` | `grpc` |
| `--host` | Hostname or IP address, without a URL scheme or path | `localhost` |
| `--port` | Endpoint port | REST: 8000; gRPC: 9000 |
| `--concurrency` | Maximum number of synchronous requests in flight | 1 |
| `--requests`, `--request-count` | Total requests across all workers, not per worker | 5000 |
| `--timeout` | Client request timeout in seconds | 60 |
| `--shape NAME:DIMS` | Full input shape, including batch dimension; repeat for multiple inputs | Model metadata |
| `--input-file NAME=PATH` | Raw binary file sent as BYTES, or numeric `.npy` array; repeat for multiple inputs | Zero-filled numeric tensors |
| `--json` | Emit a machine-readable JSON report | Human-readable report |

### Single-Client Latency

#### Start OVMS in Latency Mode

Run only one demo server at a time. On Linux, start it from the directory containing
the downloaded model:

```bash
docker run --rm -d --name ovms-benchmark-python \
	--user "$(id -u):$(id -g)" -v "${PWD}:/models:ro" \
	-p 9000:9000 -p 8000:8000 \
	openvino/model_server:latest \
	--model_name resnet --model_path /models/resnet50.xml \
	--mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" \
	--layout "NHWC:NCHW" --port 9000 --rest_port 8000 \
	--plugin_config '{"PERFORMANCE_HINT":"LATENCY"}'
```

On Windows, start OVMS in a separate command prompt:

```bat
ovms.exe --model_name resnet --model_path resnet50.xml --mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" --layout "NHWC:NCHW" --port 9000 --rest_port 8000 --plugin_config "{\"PERFORMANCE_HINT\":\"LATENCY\"}"
```

Check readiness before running the client commands from this demo directory. Repeat
until the readiness request returns HTTP 200; metadata shows the input shape:

```console
curl -f -i http://localhost:8000/v2/models/resnet/ready
curl -f http://localhost:8000/v2/models/resnet
```

#### Run the Latency Benchmark

Use concurrency 1 to keep one request in flight. Batch size is 1. The input is an
NHWC FP32 image tensor with shape `[1,224,224,3]`, matching the layout configured
above. The script discovers fixed shapes and datatypes from metadata and reuses
zero-filled numeric tensors when no input files are provided. No image is needed
for these tensor-only examples.

The latency and throughput outputs below were captured with the multiprocessing
script and the corresponding performance hint on Linux. Host ports 19020/18020
were used instead of 9000/8000; only the port mappings and client ports changed.
These shared-machine measurements are not reference performance numbers.

gRPC:

```console
python benchmark.py --model resnet --endpoint grpc --host localhost --port 9000 --concurrency 1 --requests 5000
```

Output:

```text
Number of requests: 5000
Successful requests: 5000
Failed requests: 0
Success rate: 100.00%
Latency mean (average): 3.384 ms
Latency median (p50): 3.332 ms
Latency p90: 3.758 ms
Latency p99: 4.200 ms
Latency stddev: 0.462 ms
Throughput: 295.300 successful requests/s
Elapsed time: 16.932 s
Max concurrency: 1 (requested 1)
```

REST:

```console
python benchmark.py --model resnet --endpoint rest --host localhost --port 8000 --concurrency 1 --requests 5000
```

Output:

```text
Number of requests: 5000
Successful requests: 5000
Failed requests: 0
Success rate: 100.00%
Latency mean (average): 4.037 ms
Latency median (p50): 4.023 ms
Latency p90: 4.227 ms
Latency p99: 4.778 ms
Latency stddev: 1.530 ms
Throughput: 247.601 successful requests/s
Elapsed time: 20.194 s
Max concurrency: 1 (requested 1)
```

### Throughput with Concurrency 20

#### Start OVMS in Throughput Mode

Stop the latency server before switching modes. On Linux:

```bash
docker stop ovms-benchmark-python
```

If starting directly in throughput mode, skip the stop command. Start OVMS from
the directory containing the model:

```bash
docker run --rm -d --name ovms-benchmark-python \
	--user "$(id -u):$(id -g)" -v "${PWD}:/models:ro" \
	-p 9000:9000 -p 8000:8000 \
	openvino/model_server:latest \
	--model_name resnet --model_path /models/resnet50.xml \
	--mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" \
	--layout "NHWC:NCHW" --port 9000 --rest_port 8000 \
	--plugin_config '{"PERFORMANCE_HINT":"THROUGHPUT"}'
```

On Windows, press Ctrl+C in the latency server prompt, then run:

```bat
ovms.exe --model_name resnet --model_path resnet50.xml --mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" --layout "NHWC:NCHW" --port 9000 --rest_port 8000 --plugin_config "{\"PERFORMANCE_HINT\":\"THROUGHPUT\"}"
```

Repeat the readiness check before benchmarking:

```console
curl -f -i http://localhost:8000/v2/models/resnet/ready
```

#### Run the Throughput Benchmark

Use concurrency 20 to keep up to 20 requests in flight. This is concurrency, not a
batch of 20 images; each request still contains one image.

gRPC:

```console
python benchmark.py --model resnet --endpoint grpc --host localhost --port 9000 --concurrency 20 --requests 5000
```

Output:

```text
Number of requests: 5000
Successful requests: 5000
Failed requests: 0
Success rate: 100.00%
Latency mean (average): 10.192 ms
Latency median (p50): 9.733 ms
Latency p90: 11.180 ms
Latency p99: 18.302 ms
Latency stddev: 2.033 ms
Throughput: 1903.932 successful requests/s
Elapsed time: 2.626 s
Max concurrency: 20 (requested 20)
```

REST:

```console
python benchmark.py --model resnet --endpoint rest --host localhost --port 8000 --concurrency 20 --requests 5000
```

Output:

```text
Number of requests: 5000
Successful requests: 5000
Failed requests: 0
Success rate: 100.00%
Latency mean (average): 9.426 ms
Latency median (p50): 8.520 ms
Latency p90: 13.412 ms
Latency p99: 17.836 ms
Latency stddev: 2.453 ms
Throughput: 1403.959 successful requests/s
Elapsed time: 3.561 s
Max concurrency: 20 (requested 20)
```

Each worker is a separate spawned Python process with its own client, tensors,
and GIL. The number of processes is the smaller of concurrency and request count.
No live Triton client is shared with or inherited by a child process. The spawn
start method is used on both Linux and Windows.

Metadata lookup, file reads, input transfer to children, and tensor preparation
happen before timing. A barrier starts the measurement after all workers are
ready; it stops at the last completed request. Latency samples and request
intervals are returned once per worker after its loop, not sent over IPC for every
request. Peak concurrency is calculated from overlapping intervals after the run,
so there is no shared counter lock in the inference loop. Result transfer,
aggregation, and process shutdown are excluded from benchmark wall time.

Initial connection establishment and first-inference effects can still affect
measured latency; there is no automatic warmup. Process startup adds to total
command runtime, and each process holds its own interpreter and input data. Allow
enough client RAM and CPU resources when increasing concurrency. When calling the
benchmark from another Python program, protect the entry point with
`if __name__ == "__main__":`, as the script itself does.

### Dynamic Shapes and Multiple Inputs

When metadata contains dynamic dimensions such as `-1`, supply a concrete shape.
The following illustrates a model named `dynamic_model` with one dynamic input
named `image`; deploy a corresponding model before using it:

```console
python benchmark.py --model dynamic_model --endpoint grpc --shape image:1,224,224,3 --requests 5000
```

Shapes must match static dimensions in metadata and contain only positive sizes.
For a scalar input, use `--shape NAME:scalar`. With a single input, the name can be
omitted: `--shape 1,224,224,3`.

For models with multiple inputs, repeat the named options. For example, a model
named `text_model` with two dynamic INT64 inputs could be benchmarked with:

```console
python benchmark.py --model text_model --endpoint rest --shape input_ids:1,128 --shape attention_mask:1,128 --input-file input_ids=input_ids.npy --input-file attention_mask=attention_mask.npy --concurrency 20
```

Numeric `.npy` files must have the model's datatype and the selected shape. Their
shape is used automatically if `--shape` is omitted. Inputs without a file use
zeros. These example datasets and additional models are not bundled with the demo.

### BYTES Inputs and JPEG Files

For a BYTES input, provide a raw file with `--input-file`. Its complete contents
are used as one tensor element, without decoding or base64 conversion by the
script. If the selected shape has several elements, the same payload is repeated
in each element. Dynamic BYTES inputs also require a concrete `--shape`.

OVMS also supports encoded-image BYTES requests for compatible numeric image
inputs. For ResNet, a raw JPEG file selects BYTES with shape `[1]`, even though
metadata describes an FP32 image tensor. OVMS performs decoding and preprocessing
server-side, so this measures a different workload from tensor-only inference.

Download an image:

```console
curl -fL https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/common/static/images/bee.jpeg -o image.jpeg
```

Benchmark it over gRPC:

```console
python benchmark.py --model resnet --endpoint grpc --input-file image=image.jpeg --concurrency 1 --requests 5000
```

Historical JPEG output from the earlier threaded client with default server
settings, not the performance-hint deployments (the image was read from the
repository during validation):

```text
Number of requests: 5000
Successful requests: 5000
Failed requests: 0
Success rate: 100.00%
Latency mean (average): 3.568 ms
Latency median (p50): 3.725 ms
Latency p90: 3.889 ms
Latency p99: 4.003 ms
Latency stddev: 0.470 ms
Throughput: 279.449 successful requests/s
Elapsed time: 17.892 s
Max concurrency: 1 (requested 1)
```

Or over REST:

```console
python benchmark.py --model resnet --endpoint rest --input-file image=image.jpeg --concurrency 20 --requests 5000
```

Output:

```text
Number of requests: 5000
Successful requests: 5000
Failed requests: 0
Success rate: 100.00%
Latency mean (average): 54.979 ms
Latency median (p50): 54.676 ms
Latency p90: 55.627 ms
Latency p99: 56.494 ms
Latency stddev: 14.089 ms
Throughput: 362.903 successful requests/s
Elapsed time: 13.778 s
Max concurrency: 20 (requested 20)
```

For a single input, `--input-file image.jpeg` is equivalent. For multiple BYTES
inputs, repeat `--input-file NAME=PATH` for each one. Quote the entire argument if
a path contains spaces, for example `--input-file "image=C:\test images\image.jpeg"`.
Raw files select BYTES; use `.npy` for numeric tensors. Encoded-file support depends
on the server/model, so an arbitrary binary file is not necessarily a valid image.

### Metrics

| Metric | Definition |
|---|---|
| Number of requests | Completed attempts, including successes and failures |
| Success rate | Successful requests divided by all attempts, as a percentage |
| Latency mean (average) | Arithmetic mean of successful-request client latency, in milliseconds |
| Latency median (p50), p90, p99 | Successful-request latency percentiles, in milliseconds |
| Latency stddev | Population standard deviation of successful-request latency, in milliseconds |
| Throughput | Successful requests divided by benchmark wall time, in requests/second |
| Max concurrency | Observed peak client calls in flight, not the configured server capacity |

Mean and average are the same metric. Latency includes the Triton client call,
transport, and server processing, not just model execution. Failed requests are
counted but excluded from latency statistics; if all fail, latency is `N/A`
(`null` in JSON) and throughput is zero. A few error messages are retained for
diagnosis. There are no automatic retries. Exit codes are 0 for all requests
succeeding, 1 for any failed inference, and 2 for setup or input errors.

Use `--json` for structured output. If fewer requests than workers are requested,
the worker count is capped at the request count. Repeated inputs are useful for
load generation, not accuracy evaluation. Keep model, device, shapes, payloads,
and client/server placement unchanged when comparing runs. The Python client and
shared-host CPU contention can limit throughput; use a separate load-generator
machine for representative server measurements.

### Multiprocessing Throughput

The process implementation was compared with the same request loop executed in a
thread pool. Both loops reused prepared tensors, recorded request intervals, and
avoided shared counter locks, isolating the effect of running separate Python
interpreters. The server used ResNet with 20 OpenVINO streams, configured by adding
`--plugin_config "{\"NUM_STREAMS\":\"20\"}"` to the OVMS startup command. Both clients
used concurrency 20, 5,000 requests, batch size 1, and zero-filled FP32 inputs.

Two sequential rounds were run, with client order reversed in the second round.
All runs completed 5,000/5,000 requests with observed peak concurrency 20:

| Endpoint | Round | Threads (requests/s) | Processes (requests/s) |
|---|---|---|---|
| gRPC | 1 | 1142.963 | 2772.955 |
| gRPC | 2 | 1080.752 | 2998.014 |
| REST | 1 | 836.398 | 2907.600 |
| REST | 2 | 895.188 | 2140.443 |

The two-round average improved by approximately 2.60x for gRPC and 2.92x for REST.
This supports using processes to parallelize serialization across separate GILs,
but shared-host variability remains significant; the results are not performance
guarantees.

A separate multiprocessing gRPC run on the same tuned server produced:

```console
python benchmark.py --model resnet --endpoint grpc --port 19020 --concurrency 20 --requests 5000
```

```text
Number of requests: 5000
Successful requests: 5000
Failed requests: 0
Success rate: 100.00%
Latency mean (average): 6.616 ms
Latency median (p50): 6.297 ms
Latency p90: 8.146 ms
Latency p99: 11.370 ms
Latency stddev: 1.622 ms
Throughput: 2941.964 successful requests/s
Elapsed time: 1.700 s
Max concurrency: 20 (requested 20)
```

Host ports 19020/18020 were used for this separate tuned instance. The deployment
at the top of this guide does not set `NUM_STREAMS=20` by default.

Before moving to the next tool, stop the Python demo server on Linux:

```bash
docker stop ovms-benchmark-python
```

For a Windows binary deployment, press Ctrl+C in the server command prompt.

## Use perf_analyser

### Installation

The executable is named `perf_analyzer`. On Linux, use the virtual environment
created above, or create one with `python3.12 -m venv .venv` and activate it with
`source .venv/bin/activate`. Install the tool:

```bash
python -m pip install "perf-analyzer==2.60.0"
```

The PyPI package contains a Linux executable, not a native Windows executable.
On Windows, run the client in WSL using the Linux installation commands. A native
Windows OVMS deployment can still serve the model; replace `localhost` with the
Windows host address if required by WSL networking. For a native Windows
benchmark client, use the Python script in the preceding section.

Print the available options:

```bash
perf_analyzer
```

In version 2.60.0, invoking the tool without arguments prints usage; `-h` and
`--help` are not supported.

### Single-Client Latency

#### Start OVMS in Latency Mode

Stop any Python demo server first. On Linux, start OVMS from the directory
containing the downloaded model:

```bash
docker run --rm -d --name ovms-benchmark-perf \
	--user "$(id -u):$(id -g)" -v "${PWD}:/models:ro" \
	-p 9000:9000 -p 8000:8000 \
	openvino/model_server:latest \
	--model_name resnet --model_path /models/resnet50.xml \
	--mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" \
	--layout "NHWC:NCHW" --port 9000 --rest_port 8000 \
	--plugin_config '{"PERFORMANCE_HINT":"LATENCY"}'
```

Or start OVMS on Windows in a separate command prompt:

```bat
ovms.exe --model_name resnet --model_path resnet50.xml --mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" --layout "NHWC:NCHW" --port 9000 --rest_port 8000 --plugin_config "{\"PERFORMANCE_HINT\":\"LATENCY\"}"
```

Wait for readiness before running perf_analyzer:

```console
curl -f -i http://localhost:8000/v2/models/resnet/ready
```

#### Run the Latency Benchmark

Use the same model input as the Python examples: synchronous gRPC, batch size 1,
FP32 shape `[1,224,224,3]`, zero-filled inputs, and 5,000 total requests.
`--input-data zero` is important because perf_analyzer otherwise generates random
input. Concurrency 1 targets single-client latency:

```bash
perf_analyzer -m resnet --concurrency-range 1 --sync --service-kind triton \
	-u localhost:9000 -i gRPC -b 1 --shape image:1,224,224,3 \
	--input-data zero --request-count 5000
```

REST alternative, using the same latency-mode server. This requires the
[Model Configuration API](../../../docs/model_server_rest_api_kfs.md#model-configuration-api):

```bash
perf_analyzer -m resnet --concurrency-range 1 --sync --service-kind triton \
	-u localhost:8000 -i http -b 1 --shape image:1,224,224,3 \
	--input-data zero --request-count 5000
```

gRPC output excerpt with `PERFORMANCE_HINT=LATENCY` (host port 19020 in validation):

```text
Request concurrency: 1
	Client:
		Request count: 5000
		Throughput: 277.75 infer/sec
		Avg latency: 3153 usec (standard deviation 501 usec)
		p50 latency: 3118 usec
		p90 latency: 3686 usec
		p95 latency: 3772 usec
		p99 latency: 4040 usec
```

### Throughput with Concurrency 20

#### Start OVMS in Throughput Mode

Stop the latency server on Linux:

```bash
docker stop ovms-benchmark-perf
```

Skip the stop command if starting directly in throughput mode. Start OVMS from the
directory containing the model:

```bash
docker run --rm -d --name ovms-benchmark-perf \
	--user "$(id -u):$(id -g)" -v "${PWD}:/models:ro" \
	-p 9000:9000 -p 8000:8000 \
	openvino/model_server:latest \
	--model_name resnet --model_path /models/resnet50.xml \
	--mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" \
	--layout "NHWC:NCHW" --port 9000 --rest_port 8000 \
	--plugin_config '{"PERFORMANCE_HINT":"THROUGHPUT"}'
```

On Windows, press Ctrl+C in the latency server prompt, then run:

```bat
ovms.exe --model_name resnet --model_path resnet50.xml --mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" --layout "NHWC:NCHW" --port 9000 --rest_port 8000 --plugin_config "{\"PERFORMANCE_HINT\":\"THROUGHPUT\"}"
```

Repeat the readiness check:

```console
curl -f -i http://localhost:8000/v2/models/resnet/ready
```

#### Run the Throughput Benchmark

Use concurrency 20 to keep up to 20 batch-one requests in flight:

```bash
perf_analyzer -m resnet --concurrency-range 20 --sync --service-kind triton \
	-u localhost:9000 -i gRPC -b 1 --shape image:1,224,224,3 \
	--input-data zero --request-count 5000
```

REST alternative, using the same throughput-mode server and an OVMS build with
the Model Configuration API:

```bash
perf_analyzer -m resnet --concurrency-range 20 --sync --service-kind triton \
	-u localhost:8000 -i http -b 1 --shape image:1,224,224,3 \
	--input-data zero --request-count 5000
```

gRPC output excerpt with `PERFORMANCE_HINT=THROUGHPUT` (host port 19020 in validation):

```text
Request concurrency: 20
	Client:
		Request count: 5000
		Throughput: 1666.3 infer/sec
		Avg latency: 9892 usec (standard deviation 1946 usec)
		p50 latency: 9391 usec
		p90 latency: 11039 usec
		p95 latency: 15464 usec
		p99 latency: 17940 usec
```

### Historical gRPC Comparison

The following comparison predates multiprocessing and performance-hint deployment
examples. It uses the earlier threaded Python implementation and default server
settings, not the new `LATENCY`/`THROUGHPUT` configurations. All latencies are in ms:

| Concurrency | Client | Requests | Throughput (requests/s) | Mean (ms) | p90 (ms) | p99 (ms) | Stddev (ms) |
|---|---|---|---|---|---|---|---|
| 1 | Python | 5000 | 290.992 | 3.425 | 3.600 | 4.276 | 0.318 |
| 1 | perf_analyzer | 5000 | 294.090 | 2.995 | 3.536 | 3.662 | 0.449 |
| 20 | Python | 5000 | 577.816 | 34.472 | 35.536 | 39.154 | 10.379 |
| 20 | perf_analyzer | 5000 | 555.520 | 32.633 | 33.113 | 33.616 | 1.100 |

Throughput differed by about 1.1% at concurrency 1 and 4.0% at concurrency 20,
relative to perf_analyzer. Mean latency differed by about 14.4% and 5.6%,
respectively. Python had higher p99 and standard deviation at concurrency 20.
Different client implementations, timing boundaries, connection setup, scheduling,
and shared-system load can affect these results. These are single sequential runs
without an explicit warmup; they are a sanity check, not evidence that either
client is faster or that their latency distributions are equivalent. For a rigorous
comparison, repeat and alternate client runs on an otherwise idle system.

### REST with perf_analyzer

OVMS builds that include the [Model Configuration API](../../../docs/model_server_rest_api_kfs.md#model-configuration-api)
can also be benchmarked directly with perf_analyzer over REST. Older builds may
stop with `failed to get model config: Invalid request URL`; use an updated OVMS
build or the Python client above in that case. Install the Linux tool as described
in [Installation](#installation-1). REST commands are included in each mode above;
use the `LATENCY` deployment for concurrency 1 and the `THROUGHPUT` deployment for
concurrency 20.

The REST commands were previously verified against an OVMS binary built from this checkout, using
`--plugin_config '{"NUM_STREAMS":"20"}'` and REST at `172.17.0.3:18200` instead of
`localhost:8000`. The returned `Request count`, `Throughput`, `Avg latency`, and
`p99 latency` values were:

| Concurrency | Request count | Throughput (infer/sec) | Avg latency (usec) | p99 latency (usec) |
|---|---|---|---|---|
| 1 | 5000 | 238.075 | 3972 | 5749 |
| 20 | 5000 | 1666.44 | 5570 | 10530 |

The default-version and explicit-version config URLs both returned valid model
configurations. These shared-machine runs confirm compatibility, not reference
performance or a controlled comparison with the earlier Python measurements.

### Validation Notes

Both performance-hint deployments were tested on Linux using this demo's virtual
environment and `openvino/model_server:latest`, with host ports 19020/18020.
The Python gRPC/REST commands and perf_analyzer gRPC commands each completed
5,000 requests in both modes. OVMS selected 1 stream for `LATENCY` and 112 streams
for `THROUGHPUT` on the test host; the selected counts are hardware-dependent.
The REST perf_analyzer runs in the preceding section were tested separately with
an updated OVMS binary and explicit `NUM_STREAMS=20`, not these hint deployments.
Windows deployment and WSL execution were not tested.

Earlier Python input-validation runs used this demo's `.venv`, tritonclient 2.60.0, and
`openvino/model_server:latest` (reporting OVMS 2026.5.0.f55a2d2bd). Host ports
19000/18000 were used because the default ports were occupied.

| Input | Protocol | Concurrency | Result |
|---|---|---|---|
| FP32 tensor | gRPC | 1 and 20 | 5000/5000 successful requests per run |
| FP32 tensor | REST | 1 and 20 | 5000/5000 successful requests per run |
| JPEG/BYTES | gRPC | 1 | 5000/5000 successful requests |
| JPEG/BYTES | REST | 20 | 5000/5000 successful requests |

Run the focused tests on either OS:

```console
python -m unittest -v test_benchmark
```

Tests cover dynamic/static shapes, multiple inputs, numeric files, BYTES payloads,
failure accounting, latency statistics, spawned worker execution, initialization
failure, and interval-based peak concurrency. The process implementation was also
verified on a 20-stream server with 5,000 tensor requests per REST/gRPC run,
100 JPEG/BYTES requests per protocol at concurrency 4, and an all-failing inference
run that preserved the JSON report and exit code 1. All dependencies
were also resolved as Windows Python 3.12 wheels, but execution on Windows was not
tested.

Stop the Linux demo server when finished:

```bash
docker stop ovms-benchmark-perf
```

For a Windows binary deployment, press Ctrl+C in the server command prompt.
