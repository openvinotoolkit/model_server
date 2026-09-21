# OpenVINO&trade; Model Server Developer Guide

## Introduction

This document gives information and steps to run and debug tests. It gives information about following points :

- [OpenVINO™ Model Server Developer Guide](#openvino-model-server-developer-guide)
	- [Introduction](#introduction)
	- [Set up the Development Environment](#set-up-the-development-environment)
	- [Prepare Environment to Use the Tests](#prepare-environment-to-use-the-tests)
		- [Step 1: Compile source code](#step-1-compile-source-code)
		- [Step 2: Install software](#step-2-install-software)
	- [Run the Tests](#run-the-tests)
	- [Run unit tests](#run-unit-tests)
	- [Debugging](#debugging)
		- [Option 1. Use OpenVINO Model Server build image.](#option-1-use-openvino-model-server-build-image)
		- [Option 2. Build OVMS image with minitrace enabled](#option-2-build-ovms-image-with-minitrace-enabled)
		- [Profiling macros](#profiling-macros)

## Set up the Development Environment

The tests in this guide are written in Python. Therefore, to complete the functional tests, Python 3.8 must be installed.

In-case of problems, see [Debugging](#debugging).

## Prepare Environment to Use the Tests

   ```bash
   git clone https://github.com/openvinotoolkit/model_server.git
   cd model_server
   ```

### Step 1: Compile source code
1. Build the development `openvino/model_server-build` Docker* image

   ```bash
   make docker_build
   ```
   or
   ```
   make docker_build DLDT_PACKAGE_URL=<URL>
   ```

   > **Note**: URL to OpenVINO Toolkit package can be received after registration on [OpenVINO&trade; Toolkit website](https://software.intel.com/en-us/openvino-toolkit/choose-download)

   `docker_build` target by default builds multiple docker images:
   - `openvino/model_server:latest` - smallest release image containing only necessary files to run model server on CPU
   - `openvino/model_server:latest-gpu` - release image containing support for Intel GPU and CPU
   - `openvino/model_server:latest-nginx-mtls` - release image containing exemplary NGINX MTLS configuration
   - `openvino/model_server-build:latest` - image with builder environment containing all the tools to build OVMS

   > **Note**: docker_build target accepts the same set of parameters as release_image target described here: [build_from_source.md](./build_from_source.md)

2. Download test LLM models
   ```bash
   ./prepare_llm_models.sh ./src/test/llm_testing
   ```

3. Mount the source code in the Docker container :
	```bash
	docker run -it -v ${PWD}:/ovms --entrypoint bash -p 9178:9178 openvino/model_server-build:latest
	```

4. In the docker container context compile the source code via (choose distro `ubuntu` or `redhat` depending on the image type):
	```bash
	bazel build --//:distro=ubuntu --config=mp_on_py_on //src:ovms
> **NOTE**: There are several options that would disable specific parts of OVMS. For details check ovms bazel build files.
	```

5. From the container, run a single unit test (choose distro `ubuntu` or `redhat` depending on the image type):
	```bash
	bazel test --//:distro=ubuntu --config=mp_on_py_on --test_summary=detailed --test_output=all --test_filter='ModelVersionStatus.*' //src:ovms_test
	```

| Argument      | Description |
| :---        |    :----   |
| `test`       | builds and runs the specified test target       |
| `--test_summary=detailed`   |   the output includes failure information       |
| `--test_output=all` | log all tests stdout at the end |
| `--test_filter='ModelVersionStatus.*'` | limits the tests run to the indicated test  |
| `//src:ovms_test` | the test source |
> **NOTE**: For more information, see the [bazel command-line reference](https://docs.bazel.build/versions/master/command-line-reference.html)
> **NOTE**: If container has access to Intel GPU device and test models, add `--test_env RUN_GPU_TESTS=1` to run GPU unit tests.


6. Select one of these options to change the target image name or network port to be used in tests. It might be helpful on a shared development host:

	* With a Docker cache :

	```
	OVMS_CPP_DOCKER_IMAGE=<replace_with_unique_image_name> make docker_build
    OVMS_CPP_DOCKER_IMAGE=<replace_with_unique_image_name> make test_functional
	```

	* Without a Docker cache :

	```
	make docker_build NO_DOCKER_CACHE=true
	```


### Step 2: Install software

1. Install Python release 3.8.

> **NOTE**: Python is only necessary to complete the functional tests in this guide.

2. Install the `virtualenv` package :

	```
	pip3 install virtualenv
	```

Now the tests can be run.

## Run the Tests

Use the tests below depending on the requirement.

Click the test that needs to be run:

<details><summary>Run test inference</summary>

1. Download an exemplary model [ResNet50 model](https://huggingface.co/OpenVINO/resnet50-int8-ov) :

```bash
mkdir -p ${HOME}/models
curl -L https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.bin -o ${HOME}/models/resnet50.bin
curl -L https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.xml -o ${HOME}/models/resnet50.xml
```

2. Start OVMS docker container with downloaded model

```bash
docker run -d --name server-test -u $(id -u) -v ${HOME}/models:/models -p 9178:9178 \
openvino/model_server:latest --model_name resnet --model_path /models/resnet50.xml \
--mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" --layout "NHWC:NCHW" --port 9178
```

3. The grpc client connects to the OpenVINO Model Server service that is running on port 9178.

	```bash
	make venv
	source .venv/bin/activate
	cd client/python/kserve-api/samples/
	pip3 install -r requirements.txt
	python grpc_infer_resnet.py --grpc_port 9178 --images_numpy_path ../../imgs_nhwc.npy --labels_numpy_path ../../lbs.npy --input_name image --output_name output --model_name resnet --transpose_input False
	```

Where:

| Argument Used     | Description |
| :---        |    :----   |
| `images_numpy_path tests/performance/imgs.npy`  | The path to a numpy array. `imgs.npy` is the numpy array with a batch of input data.|
| `labels_numpy_path tests/performance/labels.npy`| Includes a numpy array  named labels.npy. This array has image classification results       |
| `iteration 1000` | Run the data 1000 times |
| `batchsize 1` | Batch size to be used in the inference request |
| `report_every 10` | Number of iterations followed by results summary report|
| `input_name image` | Name of the deployed model input called "image" |
| `output_name output` | Name of the deployed model output called "output"|

</details>

<details><summary>Run functional tests</summary>

The functional tests are written in Python. Therefore, to complete the tests in this section, Python 3.6 - 3.8 must be installed.
> **NOTE**: In-case of additional problems, see the [debugging section](#debugging).

1. Run command

```bash
make test_functional
```

- Configuration options are :

| Variable    | Description |
| :---        |    :----   |
| `IMAGE`  | Docker image name for the tests.|
| `TEST_DIR_CACHE`| Location from which models and test data are downloaded.|
| `TEST_DIR` | Location to which models and test data are copied during tests.|
| `TEST_DIR_CLEANUP` | Set to `True` to remove the directory under `TEST_DIR` after the tests.|
| `LOG_LEVEL` | The log level.|
| `BUILD_LOGS` | Path to save artifacts.|
| `START_CONTAINER_COMMAND` | The command to start the OpenVINO Model Storage container.|
| `CONTAINER_LOG_LINE` | The log line in the container that confirms the container started properly.|

2. Add any configuration variables to the command line in this format :

```bash
export IMAGE="openvino/model_server:latest"
```

3. To make command repetition easier, create and store the configuration options in a file named `user_config.py`. Put this file in the main project directory.

- Example:

```python
os.environ["IMAGE"] = "openvino/model_server"
```
</details>

<details><summary>Run tests on an OpenVINO Model Server binary file</summary>

1. To run tests on an OpenVINO Model Server binary file, use export to specify the following variable in `user_config.py` or in the environment.
Replace `"/home/<example_path>/dist/<os_name>/ovms/bin/ovms"` with the path to your binary file:

```bash
tar -xvzf dist/<os_name>/ovms.tar.gz -C dist/<os_name>/
```

```python
os.environ["OVMS_BINARY_PATH"] = "'${PWD}'/dist/<os_name>/ovms/bin/ovms"
```

```bash
export OVMS_BINARY_PATH="'${PWD}'/dist/<os_name>/ovms/bin/ovms"
```

2. The following command executed in the of OpenVINO Model Server binary file should return paths to the unpacked `lib` directory included in `ovms.tar.gz` (`ovms/bin/./../lib`).
```bash
ldd dist/<os_name>/ovms/bin/ovms
```

3. Otherwise use export to specify the following variable in `user_config.py` file or in the environment :

```python
os.environ["LD_LIBRARY_PATH"] = "'${PWD}'/dist/<os_name>/ovms/lib"
```

```bash
export LD_LIBRARY_PATH="'${PWD}'/dist/<os_name>/ovms/lib"
```

</details>

> **NOTE**: For additional problems, see the [debugging section](#debugging).


## Run unit tests

Executing the unit tests require building the ovms build image. All unit tests are expected to be started in a container using model server build image. Some unit tests require test models to be pulled and attached to the container.
The following commands create the build image and start the unit tests:

```
make ovms_builder_image
make run_unit_tests
```

To run unit tests, verifying integration with Intel GPUs, add `RUN_GPU_TESTS=1` parameter:
```
make run_unit_tests RUN_GPU_TESTS=1
```

> NOTE: It is required to follow [this guide](https://dgpu-docs.intel.com/driver/installation.html#ubuntu) to prepare host machine to work with VA API (Ubuntu).

On bare metal, run (just once):
```
sudo apt install -y \
    linux-headers-$(uname -r) \
    linux-modules-extra-$(uname -r) \
    flex bison \
    intel-fw-gpu intel-i915-dkms xpu-smi
sudo reboot
```

> NOTE: It is required to execute unit tests on machine with Intel Data Center GPU.

> NOTE: For RedHat base OS unit tests, which require VA API, are skipped.

## Checking code coverage of unit tests

To check code coverage of unit tests, execute the following command to create build image and run unit tests with code coverage enabled:

```
make ovms_builder_image BASE_OS=ubuntu24 CHECK_COVERAGE=1 RUN_TESTS=1 MEDIAPIPE_DISABLE=0 PYTHON_DISABLE=0 OV_USE_BINARY=1 OVMS_CPP_DOCKER_IMAGE=ovms_coverage
```

Then run `get_coverage` target to extract report from the container:

```
make get_coverage OVMS_CPP_DOCKER_IMAGE=ovms_coverage
```

It should create report in `genhtml` directory. Open `index.html` file in this directory to check the code coverage report.

## Debugging

Debugging options are available. Click on the required option :


<details><summary>Use gdb to debug in Docker</summary>

1. Build a project in a debug mode :
	```bash
	make docker_build BAZEL_BUILD_TYPE=dbg
	```

	> **NOTE**: You can build also the debug version of the major dependencies like OpenVINO Runtime using extra flag `CMAKE_BUILD_TYPE=Debug`.

2. Run the container :
	```bash
	docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v ${PWD}:/ovms -p 9178:9178 --entrypoint bash openvino/model_server-build:latest
	```
3.	Prepare resnet50 model for OVMS in /models catalog and recompile the OpenVINO Model Server in docker container with debug symbols using command:
	```bash
	mkdir -p /models/1 && curl -L https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.bin -o /models/1/resnet50.bin && curl -L https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.xml -o /models/1/resnet50.xml
	```
	```bash
	bazel build --config=mp_on_py_on //src:ovms -c dbg
	```
	```bash
	gdb --args ./bazel-bin/src/ovms --model_name resnet --model_path /models \
	--mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" --layout "NHWC:NCHW" --port 9178
	```
    > **NOTE**: For best results, use the makefile parameter `BAZEL_BUILD_TYPE=dbg` to build the dependencies in debug mode as shown above


- For unit test debugging, run command :
	```bash
	gdb --args ./bazel-bin/src/./ovms_test --gtest_filter='OvmsConfigTest.emptyInput'
	```

- For forking tests debugging, enable fork follow mode by running command:
	```
	# (in gdb cli) set follow-fork-mode child
	```
- For tracing what OpenVINO calls are used underneath you can use `--define OV_TRACE=1` option when building ovms with bazel or its tests.
</details>

<details><summary>Use minitrace to display flame graph</summary>

Download the model files and store them in the `models` directory
```bash
mkdir -p models/resnet/1
curl -L https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.bin -o models/resnet/1/resnet50.bin
curl -L https://huggingface.co/OpenVINO/resnet50-int8-ov/resolve/main/resnet50.xml -o models/resnet/1/resnet50.xml
```

### Option 1. Use OpenVINO Model Server build image.
This is convenient way during development in case it is needed to add new or remove already existing traces.

1. Build OVMS build image locally.
```bash
make docker_build
```

2. Start the container.
```bash
docker run -it -v ${PWD}:/ovms --entrypoint bash -p 9178:9178 openvino/model_server-build:latest
```

3. Build OVMS with minitrace enabled.
```bash
bazel build --config=linux --copt="-DMTR_ENABLED" //src:ovms
```

4. Run OVMS with `--trace_path` specifying where to save flame graph JSON file.
```bash
bazel-bin/src/ovms --model_name resnet --model_path models/resnet \
--mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" --layout "NHWC:NCHW" --trace_path trace.json --port 9178
```

5. During app exit, the trace info will be saved into `trace.json`.

6. Use Chrome web browser `chrome://tracing` tool to display the graph.

### Option 2. Build OVMS image with minitrace enabled
This is convenient when final image has to be used on different machine and no changes to existing traces do not need to be modified for debugging.

1. Build OVMS with minitrace enabled locally.
```bash
make docker_build MINITRACE=ON
```

2. Run OVMS with minitrace enabled and `--trace_path` to specify where to save trace JSON file. Since the file is flushed and saved at container shutdown, mount the host directory with write access to persist the file after container stops.
```bash
mkdir traces
chmod -R 777 traces

docker run -it -v ${PWD}:/workspace:rw -p 9178:9178 openvino/model_server --model_name resnet --model_path /workspace/models/resnet \
--mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" --layout "NHWC:NCHW" --trace_path /workspace/traces/trace.json --port 9178
```

3. During app exit, the trace info will be saved into `${PWD}/traces/trace.json`.

4. Use Chrome web browser `chrome://tracing` tool to display the graph, similarly to Option 1.

### Profiling macros
| Macro | Description | Example Usage |
|---|---|---|
| OVMS_PROFILE_FUNCTION | Add this macro at the very beginning of a function. This will automatically add function name to trace marker. | `OVMS_PROFILER_FUNCTION();`  |
| OVMS_PROFILE_SCOPE | Add this macro at the beginning of a code scope and add marker name. This will automatically add ending marker at the end of code scope.  | `OVMS_PROFILER_SCOPE("My Code Scope Marker");`  |
| OVMS_PROFILE_SYNC_BEGIN | For custom start and end markers, use this macro to mark beginning of synchronous event. Remember to use the same marker name for beginning and end. | `OVMS_PROFILER_SYNC_BEGIN("My Synchronous Event");` |
| OVMS_PROFILE_SYNC_END | For custom start and end markers, use this macro to mark ending of synchronous event. Remember to use the same marker name for beginning and end. | `OVMS_PROFILER_SYNC_END("My Synchronous Event");` |
| OVMS_PROFILE_ASYNC_BEGIN | For custom start and end markers, use this macro to mark beginning of asynchronous event. Remember to use the same marker name and id for beginning and end. Asynchronous markers need an identifier to correctly match events. | `OVMS_PROFILER_ASYNC_BEGIN("My Asynchronous Event", unique_id);` |
| OVMS_PROFILE_ASYNC_END | For custom start and end markers, use this macro to mark end of asynchronous event. Remember to use the same marker name and id for beginning and end. Asynchronous markers need an identifier to correctly match events. | `OVMS_PROFILER_ASYNC_END("My Asynchronous Event", unique_id);` |

More information can be found in [profiler.hpp](../src/profiler.hpp) file.

</details>

<details><summary>Debug functional tests</summary>

Use OpenVINO Model Server build image because it installs the necessary tools.

1. Add the ENTRYPOINT line in Dockerfile.ubuntu:
	```bash
	echo 'ENTRYPOINT ["/bin/bash", "-c", "sleep 3600; echo Server started on port; sleep 100000"]' >> Dockerfile.ubuntu
	```

2. Build the project in debug mode :
	```bash
	make docker_build BAZEL_BUILD_TYPE=dbg
	```

3. Open a terminal.

4. Run a test in this terminal. Change `TEST_PATH` to point to the test you want to debug:
	```bash
	make test_functional TEST_PATH=tests/functional/test_batching.py::TestBatchModelInference::test_run_inference_rest IMAGE=openvino/model_server-build:latest
	```

5. Open a second terminal.

6. In this terminal identify the ID/hash of a running Docker container:
	```bash
	docker ps
	```

7. Use the ID to execute a new bash shell into this container and start gdb. Make sure the parameters you pass to the OpenVINO Model Server match the parameters in the test code :
	```bash
	docker exec -ti HASH bash
	```
	In docker container:
	```bash
	cd /ovms/bazel-bin/src/ ; gdb --args ./ovms  --model_name age_gender --model_path /opt/ml/age_gender --port 9000 --rest_port 5500 --log_level TRACE
	```

8. Open a third terminal.

9. In this terminal use the Docker container ID/hash to stop the sleep process that is preventing the tests from starting. These tests are waiting for stdout text "Server started on port" :
	```bash
	docker exec -ti HASH bash
	```
	In docker container:
	```bash
	yum install psmisc; killall sleep
	```

10. Return to the first terminal to debug the test execution.

</details>
