# OpenVINO&trade; Model Server Developer Guide

## Introduction

This guide describes developing, building, testing, and debugging OVMS on a Linux host. You can work locally or use VS Code on a Windows laptop connected through Remote - SSH and Dev Containers. Run Docker and Make commands on the Linux host, and run Bazel and GDB inside the build container.

Unless stated otherwise, run host commands from the repository root and build-container commands from `/ovms`. Model downloads use paths relative to these directories.

- [Prerequisites](#prerequisites)
- [Build Environment](#build-environment)
- [Remote Development With VS Code](#remote-development-with-vs-code)
- [Build and Test](#build-and-test)
- [Debugging](#debugging)
- [Functional and Performance Tests](#functional-and-performance-tests)
- [Optional Configuration](#optional-configuration)

## Prerequisites

The Linux host needs Git, Make, Docker, and permission to run Docker commands. Keep the repository on the Linux host, not on the Windows laptop. Image builds and model downloads require network access; configure proxy variables if your network requires them.

For remote development, install VS Code with the Remote - SSH and Dev Containers extensions on the Windows laptop. Connect to the Linux host using Remote - SSH before running the host commands below.

Python 3.12 is required on the Linux host only for model-preparation scripts and Python-based functional/client tests. Ensure `python3` on `PATH` points to that installation. `make venv` creates the host test environment using that interpreter; Python development inside the builder uses the container's environment.

## Build Environment

### Prepare the Builder Image

Run on the Linux host:

```text
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make ovms_builder_image RUN_TESTS=0
```

This creates `openvino/model_server-build:latest` using Ubuntu 24 by default, without running unit tests. Reuse an existing compatible builder image when available. Unlike `make docker_build`, this target does not also package OVMS and create release images. Other operating systems and image names are covered in [Optional Configuration](#optional-configuration).

When reusing an older builder, ensure it provides the Bazel version required by the repository's `.bazelversion`. If Bazel reports that the required version is missing, install that version inside the builder or rebuild the builder image before compiling OVMS. Install client Python dependencies separately in the environment used to run the client; the builder does not necessarily include them.

### Start the Build Container

Run on the Linux host, from the repository root:

```text
docker run -dit --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -p 9178:9178 -p 9179:9179 -v "${PWD}":/ovms -w /ovms \
    --entrypoint bash openvino/model_server-build:latest
```

The repository is mounted at `/ovms`. Docker assigns the container name automatically. The debugging permissions allow GDB to inspect processes; use this container for development, not production. The published ports expose gRPC on `9178` and REST on `9179` when OVMS is started with those ports.

On a shared host, choose unused host ports and update client endpoints accordingly. Publish ports when creating the container; an existing container without these mappings must be recreated to add them. Do not start another server on the same host ports while this container is running.

For a terminal-only workflow, find the container name and open its shell. Run on the Linux host, replacing `<container_name>` with the Docker-generated name:

```text
docker ps
docker exec -it <container_name> bash
```

## Remote Development With VS Code

In the Remote - SSH window connected to the Linux host, run **Dev Containers: Attach to Running Container...** from the Command Palette. Select the container using `openvino/model_server-build:latest` and open `/ovms`.

Edit features from the Windows laptop while running Bazel builds, tests, and GDB in the attached container's terminal. Edits persist in the Linux host repository through the bind mount; no Windows source synchronization is needed. Docker and Make commands still run in a separate Linux-host terminal, not in the attached container.

Reuse the container across development sessions. If it is stopped, run on the Linux host, replacing `<container_name>` with its name from `docker ps -a`:

```text
docker ps -a
docker start <container_name>
```

Attach VS Code again and open `/ovms`.

## Build and Test

### Compile OVMS

Run inside the build container, from `/ovms`:

```text
bazel build //src:ovms
```

For changes requiring MediaPipe, add `--config=mp_on`. For Red Hat containers, add `--//:distro=redhat` to every Bazel build and test command. Ubuntu is the default distro. See the repository Bazel configuration for other feature flags.

### Prepare Test Models

Some tests require downloaded models. Prepare only the models needed by your tests before running them; model preparation can take significant time. Run on the Linux host, from the repository root:

```text
make prepare_models
```

To prepare only the LLM test models, run on the Linux host:

```text
./prepare_llm_models.sh ./src/test/llm_testing
```

The mounted repository makes these models available inside the build container.

### Run Unit Tests

During development, start with a focused fixture. Run inside the build container:

```text
bazel test --test_summary=detailed --test_output=all \
    --test_filter='ModelVersionStatus.*' //src:ovms_test
```

`bazel test` builds and runs the target. `--test_filter` selects the fixture or test, `--test_summary=detailed` reports failures, and `--test_output=all` includes test output. See the [Bazel command-line reference](https://bazel.build/reference/command-line-reference).

After relevant focused tests pass, run the broader unit-test workflow on the Linux host:

```text
make run_unit_tests
```

This uses the builder image. GPU-specific tests and host driver requirements are optional and described under [GPU Access and Tests](#gpu-access-and-tests).

### Build and Test Python Integration with OVMS

Python no longer has a separate on/off build configuration. Building `//src:ovms` links the MediaPipe runtime and host-side Python calculators into the executable and includes `libovmspython`, `libpython_calculators`, and `pyovms` as runtime data dependencies. Use `--config=mp_on` for changes that require MediaPipe.

Run inside the build container:

```text
bazel build //src:ovms
bazel test \
    //src:python_runtime_library_test \
    //src/python:pythoninterpretermodule_lifecycle_test \
    //src/python:pythoninterpretermodule_backend_failure_test \
    //src/python:pythoninterpretermodule_borrowed_interpreter_test
```

For changes to the Python binding itself, run inside the build container:

```text
bazel test //src/python/binding:test_python_binding
```

`python_runtime_library_test` checks runtime library symbols and `pyovms` importability. The interpreter module tests cover lifecycle, initialization failure, and borrowed-interpreter behavior. `./run_unit_tests.sh`, run inside the build container, executes the standard suite, including `//src:ovms_test` and these runtime tests. Use focused tests while iterating and the standard suite before submitting broader runtime changes.

## Debugging

### Debug OVMS With GDB

Use the running build container prepared above. Run inside the build container, from `/ovms`:

```text
bazel build //src:ovms -c dbg
gdb --args ./bazel-bin/src/ovms --config_path <config_path> --port 9178 --rest_port 9179
```

Replace `<config_path>` with your model configuration. Its model paths must be accessible inside the container, for example under `/ovms/models`. `bazel-bin` points to the executable in Bazel's output directory. Set breakpoints in GDB, then enter `run` to start OVMS.

For a standalone ResNet example, download the model and start GDB inside the build container, from `/ovms`:

```text
mkdir -p models/resnet/1
curl -fL https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.bin -o models/resnet/1/resnet50.bin
curl -fL https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.xml -o models/resnet/1/resnet50.xml
gdb --args ./bazel-bin/src/ovms --model_name resnet --model_path /ovms/models/resnet \
    --mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" --layout "NHWC:NCHW" \
    --port 9178 --rest_port 9179
```

To build the builder's OVMS artifacts in debug mode as well, run on the Linux host:

```text
make ovms_builder_image RUN_TESTS=0 BAZEL_BUILD_TYPE=dbg
docker run -dit --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -p 9178:9178 -p 9179:9179 -v "${PWD}":/ovms -w /ovms \
    --entrypoint bash openvino/model_server-dbg-build:latest
```

Use this container instead of the standard builder, not alongside it on the same ports. `BAZEL_BUILD_TYPE=dbg` adds `-dbg` to the image base name. To build applicable source-built dependencies in debug mode, also use `CMAKE_BUILD_TYPE=Debug`; this does not change prebuilt dependencies.

### Connect an External Client Container

Start OVMS under GDB as above and enter `run`. From a separate terminal on the Linux host, use `--network=host` for the client container. This makes the published OVMS endpoints available at `localhost:9178` for gRPC and `http://localhost:9179` for REST.

Run on the Linux host to check readiness:

```text
docker run --rm --network=host curlimages/curl:latest -i http://localhost:9179/v2/health/ready
```

Expect HTTP `200 OK` once OVMS is ready and the model state is `AVAILABLE` in the server log. For an existing build container without published ports, share its network namespace instead. Run on the Linux host, replacing `<container_name>` with the running build container's name:

```text
docker run --rm --network=container:<container_name> curlimages/curl:latest --noproxy '*' -i http://localhost:9179/v2/health/ready
```

With this network setting, `localhost` refers to the build container, so no host port mapping is required. Use `localhost:9178` for a gRPC client sharing the same network namespace.

Then run your inference client container with the same network setting and the matching endpoint. Requests can wait or time out while OVMS is paused at a breakpoint. Host networking here refers to the Linux server, not the Windows laptop.

### Debug Unit Tests

Build the test binary and launch a focused test inside the build container:

```text
bazel build //src:ovms_test -c dbg
gdb --args ./bazel-bin/src/ovms_test --gtest_filter='OvmsConfigTest.emptyInput'
```

For tests that fork, enter `set follow-fork-mode child` at the GDB prompt before `run`. To trace OpenVINO calls, add `--define OV_TRACE=1` when building OVMS or its tests.

### Profile With Minitrace

Prepare the ResNet model using the download commands in [Debug OVMS With GDB](#debug-ovms-with-gdb).

For iterative development, enable minitrace and start OVMS inside the build container:

```text
bazel build --config=linux --copt="-DMTR_ENABLED" //src:ovms
./bazel-bin/src/ovms --model_name resnet --model_path /ovms/models/resnet \
	--mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" --layout "NHWC:NCHW" \
	--trace_path /ovms/trace.json --port 9178
```

Stop OVMS gracefully so it writes `trace.json`. Open the file with Chrome's `chrome://tracing` tool.

For tracing with a release image instead, build on the Linux host:

```text
make docker_build MINITRACE=ON
```

The model and trace directories must exist on the host. Ensure the container user can write to the trace directory; avoid making the entire repository world-writable. Run on the Linux host, using different ports if the build container is still running:

```text
mkdir -p traces
docker run -it -v "${PWD}":/workspace:rw -p 9180:9178 \
    openvino/model_server:latest --model_name resnet --model_path /workspace/models/resnet \
	--mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" --layout "NHWC:NCHW" \
    --trace_path /workspace/traces/trace.json --port 9178
```

Stop OVMS gracefully to persist `traces/trace.json`, then open it with the same tracing tool.

### Profiling Macros

| Macro | Description | Example |
|---|---|---|
| OVMS_PROFILER_FUNCTION | Trace the current function. | `OVMS_PROFILER_FUNCTION();` |
| OVMS_PROFILER_SCOPE | Trace a named scope. | `OVMS_PROFILER_SCOPE("My Code Scope");` |
| OVMS_PROFILER_SYNC_BEGIN | Begin a synchronous event. | `OVMS_PROFILER_SYNC_BEGIN("My Event");` |
| OVMS_PROFILER_SYNC_END | End the matching synchronous event. | `OVMS_PROFILER_SYNC_END("My Event");` |
| OVMS_PROFILER_ASYNC_BEGIN | Begin an asynchronous event with an identifier. | `OVMS_PROFILER_ASYNC_BEGIN("My Event", unique_id);` |
| OVMS_PROFILER_ASYNC_END | End the event with the same name and identifier. | `OVMS_PROFILER_ASYNC_END("My Event", unique_id);` |

See [profiler.hpp](../src/profiler.hpp) for details.

## Functional and Performance Tests

These workflows run on the Linux host and are separate from incremental Bazel development. Prepare the required release image using [Full Release Builds](#full-release-builds) before running tests against it.

### Host Python Environment

With Python 3.12 available as `python3`, run on the Linux host:

```text
make venv
source .venv/bin/activate
```

Use this environment for host Python clients and tests.

### Run Functional Tests

Run on the Linux host:

```text
make test_functional
```

To select one test and an image, run on the Linux host:

```text
make test_functional TEST_PATH=tests/functional/test_embeddings.py::TestEmbeddings::test_on_commit_embeddings_endpoints TT_OVMS_IMAGE_NAME=openvino/model_server:latest
```

| Variable | Purpose |
|---|---|
| `OVMS_CPP_DOCKER_IMAGE` | Image base name. |
| `OVMS_CPP_IMAGE_TAG` | Image tag. |
| `TT_OVMS_IMAGE_NAME` | Complete image name for the test environment. |
| `TT_LOGGING_LEVEL` | Test logging level. |
| `TT_LOGGING_LEVEL_OVMS` | OVMS logging level. |
| `BUILD_LOGS` | Artifact directory. |
| `START_CONTAINER_COMMAND` | Test-harness container startup command. |

Set configuration variables in the host environment before running tests. For example, run on the Linux host:

```text
export TT_OVMS_IMAGE_NAME="openvino/model_server:latest"
```

Alternatively, set them in the test harness's `user_config.py` using Python's `os.environ`. Keep local configuration separate from shared test defaults.

### Debug Functional Tests

The preferred starting point is [Debug OVMS With GDB](#debug-ovms-with-gdb) with an external client reproducing the failing request. A functional test may launch its own OVMS container with different models and ports; match that test's configuration rather than assuming the development container is used automatically.

For the advanced startup-delay workflow, temporarily add the following Dockerfile instruction at the end of the `build` stage in `Dockerfile.ubuntu`, before the next `FROM` instruction. Do not append it to the end of the multi-stage Dockerfile, which changes the release stage instead:

```dockerfile
ENTRYPOINT ["/bin/bash", "-c", "sleep 3600; echo Server started on port; sleep 100000"]
```

Run on the Linux host to rebuild and launch the selected test:

```text
make ovms_builder_image RUN_TESTS=0 BAZEL_BUILD_TYPE=dbg
make test_functional TEST_PATH=tests/functional/test_embeddings.py::TestEmbeddings::test_on_commit_embeddings_endpoints TT_OVMS_IMAGE_NAME=openvino/model_server-dbg-build:latest
```

This workaround requires a test harness that honors the image entrypoint and waits for the startup message. Ensure its container startup configuration grants `SYS_PTRACE` and permits debugging with `seccomp=unconfined`. Remove the temporary Dockerfile change after debugging.

In a second Linux-host terminal, find and enter the test-created container, replacing `<container_name>` with its name:

```text
docker ps
docker exec -it <container_name> bash
```

Run inside that test container. Replace the example model and ports with those expected by the selected test, then enter `run` at the GDB prompt:

```text
cd /ovms
gdb --args ./bazel-bin/src/ovms --model_name age_gender --model_path /opt/ml/age_gender --port 9000 --rest_port 5500 --log_level TRACE
```

In another terminal inside the same test container, release the startup delay. On Ubuntu, run:

```text
apt-get update
apt-get install -y psmisc
killall sleep
```

On Red Hat, use the corresponding package manager instead:

```text
dnf install -y psmisc
killall sleep
```

Package installation requires root in the container. The startup message lets the waiting test continue. Control GDB in the second terminal and observe results in the original test terminal.

### Run an Inference Client

For the standalone release-image example, download the [ResNet50 model](https://huggingface.co/OpenVINO/resnet50-int8-ov) on the Linux host, from the repository root:

```text
mkdir -p models/resnet/1
curl -fL https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.bin -o models/resnet/1/resnet50.bin
curl -fL https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.xml -o models/resnet/1/resnet50.xml
```

Run on the Linux host to start a release container. This example uses host port `9180` to avoid the build container's `9178`:

```text
docker run -d -u "$(id -u)" -v "${PWD}/models":/models -p 9180:9178 \
	openvino/model_server:latest --model_name resnet --model_path /models/resnet \
	--mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" --layout "NHWC:NCHW" --port 9178
```

Run the client on the Linux host in the activated Python environment. Change to the samples directory because the script imports helper modules using paths relative to its working directory. Return to the repository root afterward:

```text
cd client/python/kserve-api/samples
pip3 install -r requirements.txt
python grpc_infer_resnet.py \
    --images_numpy_path ../../imgs_nhwc.npy --labels_numpy_path ../../lbs.npy \
	--iterations 1000 --model_name resnet --batchsize 1 \
	--input_name image --output_name output --transpose_input False --grpc_port 9180
cd ../../../..
```

The client reads NHWC image arrays and labels and sends batches to `resnet`. Stop and remove the release container when finished using its Docker-generated name.

In a CPU smoke test with freshly built OVMS and OpenVINO 2026.5.0, all 1,000 requests succeeded and classification accuracy was 90% on the repeated ten-image sample set. The first image was recognized as `airliner` (class `404`), matching its label. These are observed results, not guaranteed accuracy or performance values. When testing OVMS inside the build container instead of the release container above, use port `9178` and install the client requirements in the Python environment used inside that container.

| Argument | Purpose |
|---|---|
| `--images_numpy_path` | Path to the NHWC input image array. |
| `--labels_numpy_path` | Path to the expected labels. |
| `--iterations 1000` | Number of inference iterations. |
| `--batchsize 1` | Number of images per request. |
| `--input_name image` | Model input name. |
| `--output_name output` | Model output name. |
| `--transpose_input False` | Keep NHWC input to match the server's layout configuration. |

### Run Performance Tests

Use the [Python benchmark client](../demos/benchmark/python/README.md) for latency and throughput measurements. The legacy `tests/performance` scripts and their Make targets are no longer available. Build the client image on the Linux host, from the repository root:

```text
docker build -t benchmark_client demos/benchmark/python
```

With the release container from [Run an Inference Client](#run-an-inference-client) running, execute the benchmark on the Linux host:

```text
docker run --rm --network=host benchmark_client -a localhost -p 9180 -m resnet --api GRPC -b 1 -c 1 -t 30
```

The benchmark reports latency and throughput without validating classification accuracy. Increase `-c` for concurrency measurements and adjust `-p` to the published host port. Performance measurements should use release builds, not GDB or minitrace builds. For a request-reproduction workload rather than a performance measurement, point the client at the build container's `9178` instead.

### Test a Packaged Binary

On the Linux host, unpack the release archive and inspect its shared-library dependencies. Replace `<os_name>` with the package directory:

```text
tar -xvzf dist/<os_name>/ovms.tar.gz -C dist/<os_name>/
ldd dist/<os_name>/ovms/bin/ovms
export OVMS_BINARY_PATH="${PWD}/dist/<os_name>/ovms/bin/ovms"
```

Dependencies should resolve to the packaged `ovms/lib` directory where applicable. If needed, configure the library search path on the Linux host before running binary-mode tests:

```text
export LD_LIBRARY_PATH="${PWD}/dist/<os_name>/ovms/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

Configure the test harness to use binary mode and `OVMS_BINARY_PATH` instead of a Docker image. These values can also be set through `os.environ` in its local Python configuration.

## Optional Configuration

### Operating Systems and Proxies

Supported `BASE_OS` values include `ubuntu24` (default), `ubuntu22`, `ubuntu26`, and `redhat`. Use a matching OpenVINO package when overriding the distro. See [Building From Source](./build_from_source.md) for package and build options.

Run on the Linux host, replacing `<URL>` with the matching package URL:

```text
make ovms_builder_image RUN_TESTS=0 BASE_OS=ubuntu26 DLDT_PACKAGE_URL=<URL>
```

For Red Hat, run on the Linux host:

```text
make ovms_builder_image RUN_TESTS=0 BASE_OS=redhat
```

Use `--//:distro=redhat` for Bazel commands inside that container. Keep Ubuntu and Red Hat images distinguishable on shared hosts using custom image names. Forward required `http_proxy`, `https_proxy`, and `no_proxy` environment variables when starting containers that need network access.

### GPU Access and Tests

CPU development does not require GPU drivers or a GPU-enabled container. For GPU or VA API tests, first prepare the Linux host using the [Intel GPU installation guide](https://dgpu-docs.intel.com/driver/installation.html#ubuntu). The required packages depend on the host OS and GPU; follow that guide rather than treating one driver package list as universal.

Create the build container with access to the required devices, for example by adding `--device=/dev/dri` to the startup command. Ensure the container user has permission to access those devices and prepare the GPU test models.

Run on the Linux host:

```text
make prepare_models RUN_GPU_TESTS=1
make run_unit_tests RUN_GPU_TESTS=1
```

For a focused GPU test, add `--test_env RUN_GPU_TESTS=1` to the Bazel test command inside the GPU-enabled build container. Tests requiring an Intel Data Center GPU need suitable hardware. VA API unit tests are skipped for Red Hat builds.

### Full Release Builds

Build packaged artifacts and release images only when needed for deployment or container-based tests. Run on the Linux host:

```text
make docker_build RUN_TESTS=0
```

This includes the builder, package, and release-image stages. The default images include `openvino/model_server:latest`, the GPU image/tag, the NGINX MTLS image, and `openvino/model_server-build:latest`. See [Building From Source](./build_from_source.md) for detailed options. Use `RUN_TESTS=1` when tests should run as part of the image build.

### Custom Images and Cache

On shared hosts, use a unique image base name and unused ports. Run on the Linux host, replacing the placeholders:

```text
make docker_build OVMS_CPP_DOCKER_IMAGE=<unique_image_name>
make test_functional OVMS_CPP_DOCKER_IMAGE=<unique_image_name>
```

The builder image is named `<unique_image_name>-build:latest`; debug builds use `<unique_image_name>-dbg-build:latest`. Use the matching image when starting containers.

To rebuild without Docker cache, run on the Linux host:

```text
make ovms_builder_image RUN_TESTS=0 NO_DOCKER_CACHE=true
```

### Copilot Skills

To add a repository skill for GitHub Copilot in VS Code, create `.github/skills/<skill-name>/SKILL.md` with YAML frontmatter containing a unique `name` and a concise `description`, followed by the workflow instructions. Copilot discovers repository skills when the workspace is open. Describe the workflow in chat so the matching skill can be selected.