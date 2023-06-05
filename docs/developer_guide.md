# OpenVINO&trade; Model Server Developer Guide 

## Introduction

This document gives information and steps to run and debug tests. It gives information about following points :

1. <a href="#set-env">Set up the Development Environment</a>
2. <a href="#test-prep">Prepare environment to use the tests</a>
3. <a href="#test-run">Run the tests of your choice</a>
	* Inference test
	* Functional tests
	* Performance tests
	* Tests on a Python image
	* Tests on an OpenVINO Model Server binary file
4. <a href="#debug">Learn to debug</a>
	* How to use `gdb` to debug in Docker
	* How to debug functional tests

## Set up the Development Environment <a name="set-env"></a>

The tests in this guide are written in Python. Therefore, to complete the functional tests, Python 3.8 must be installed. 

In-case of problems, see <a href="#debug">Debugging</a>.

## Prepare Environment to Use the Tests <a name="test-prep"></a>

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
2. Mount the source code in the Docker container :
	```bash
	docker run -it -v ${PWD}:/ovms --entrypoint bash -p 9178:9178 openvino/model_server-build:latest 
	```

3. In the docker container context compile the source code via :
	```bash
	bazel build //src:ovms
	```

4. From the container, run a single unit test :
	```bash
	bazel test --test_summary=detailed --test_output=all --test_filter='ModelVersionStatus.*' //src:ovms_test
	```

| Argument      | Description |
| :---        |    :----   |
| `test`       | builds and runs the specified test target       |
| `--test_summary=detailed`   |   the output includes failure information       |
| `--test_output=all` | log all tests |
| `--test_filter='ModelVersionStatus.*'` | limits the tests run to the indicated test  | 
| `//src:ovms_test` | the test source |
> **NOTE**: For more information, see the [bazel command-line reference](https://docs.bazel.build/versions/master/command-line-reference.html)


	
5. Select one of these options to change the target image name or network port to be used in tests. It might be helpful on a shared development host:

	* With a Docker cache :
	
	```
	OVMS_CPP_DOCKER_IMAGE=<replace_with_unique_image_name> make docker_build
    OVMS_CPP_DOCKER_IMAGE=<replace_with_unique_image_name> make test_functional
    OVMS_CPP_CONTAINER_PORT=<unique_network_port> make test_perf
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

## Run the Tests <a name="test-run"></a>

Use the tests below depending on the requirement. 

Click the test that needs to be run:

<details><summary>Run test inference</summary>

1. Download an exemplary model [ResNet50-binary model](https://docs.openvinotoolkit.org/2022.1/omz_models_intel_resnet50_binary_0001_description_resnet50_binary_0001.html) :

	```bash
	source tests/performance/download_model.sh
	```

	The script stores the model in the user home folder. 

2. Start OVMS docker container with downloaded model

```bash
docker run -d --name server-test -v ~/resnet50-binary:/models/resnet50-binary -p 9178:9178 \
openvino/model_server:latest --model_name resnet-binary --model_path /models/resnet50-binary --port 9178
```

3. The grpc client connects to the OpenVINO Model Server service that is running on port 9178.

	```bash
	make venv
	source .venv/bin/activate
	pip3 install -r demos/common/python/requirements.txt
	python3 tests/performance/grpc_latency.py --images_numpy_path tests/performance/imgs.npy --labels_numpy_path tests/performance/labels.npy \
	--iteration 1000 --model_name resnet-binary --batchsize 1 --report_every 100 --input_name 0 --output_name 1463 --grpc_port 9178
	```

Where:

| Argument Used     | Description |
| :---        |    :----   |
| `images_numpy_path tests/performance/imgs.npy`  | The path to a numpy array. `imgs.npy` is the numpy array with a batch of input data.|
| `labels_numpy_path tests/performance/labels.npy`| Includes a numpy array  named labels.npy. This array has image classification results       |
| `iteration 1000` | Run the data 1000 times |
| `batchsize 1` | Batch size to be used in the inference request | 
| `report_every 10` | Number of iterations followed by results summary report|
| `input_name 0` | Name of the deployed model input called "0" | 
| `output_name 1463` | Name of the deployed model output called "1463"|

</details>

<details><summary>Run functional tests</summary>

The functional tests are written in Python. Therefore, to complete the tests in this section, Python 3.6 - 3.8 must be installed. 
> **NOTE**: In-case of additional problems, see the <a href="#debug">debugging section</a>.

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

<details><summary>Run performance tests</summary>

Automated tests are configured to use the ResNet50 model.    

1. Execute command to run latency test 
```bash
make test_perf
```
- Output
```bash
Running latency test
[--] Starting iterations
[--] Iteration   100/ 1000; Current latency: 10.52ms; Average latency: 11.35ms
[--] Iteration   200/ 1000; Current latency: 10.99ms; Average latency: 11.03ms
[--] Iteration   300/ 1000; Current latency: 9.60ms; Average latency: 11.02ms
[--] Iteration   400/ 1000; Current latency: 10.20ms; Average latency: 10.93ms
[--] Iteration   500/ 1000; Current latency: 10.45ms; Average latency: 10.84ms
[--] Iteration   600/ 1000; Current latency: 10.70ms; Average latency: 10.82ms
[--] Iteration   700/ 1000; Current latency: 9.47ms; Average latency: 10.88ms
[--] Iteration   800/ 1000; Current latency: 10.70ms; Average latency: 10.83ms
[--] Iteration   900/ 1000; Current latency: 11.09ms; Average latency: 10.85ms
[--] Iterations:  1000; Final average latency: 10.86ms; Classification accuracy: 100.0%
``` 

2. Execute command to run throughput test 
```bash
make test_throughput
```
- Output

```bash
Running throughput test
[25] Starting iterations
[23] Starting iterations
...
[11] Starting iterations
[24] Iterations:   500; Final average latency: 20.50ms; Classification accuracy: 100.0%
[25] Iterations:   500; Final average latency: 20.81ms; Classification accuracy: 100.0%
[6 ] Iterations:   500; Final average latency: 20.80ms; Classification accuracy: 100.0%
[26] Iterations:   500; Final average latency: 20.80ms; Classification accuracy: 100.0%
...
[11] Iterations:   500; Final average latency: 20.84ms; Classification accuracy: 100.0%

real	0m13.397s
user	1m22.277s
sys	0m39.333s
1076 FPS
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

> **NOTE**: For additional problems, see the <a href="#debug">debugging section</a>. 

## Debugging <a name="debug"></a>

Debugging options are available. Click on the required option :


<details><summary>Use gdb to debug in Docker</summary>

1. Build a project in a debug mode :
	```bash
	make docker_build BAZEL_BUILD_TYPE=dbg
	```

2. Run the container :
	```bash
	docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v ${PWD}:/ovms -p 9178:9178 --entrypoint bash openvino/model_server-build:latest
	```
3.	Prepare resnet50 model for OVMS in /models catalog and recompile the OpenVINO Model Server in docker container with debug symbols using command:
	```bash
	mkdir -p /models/1 && wget -P /models/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin && wget -P /models/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml
	```
	```bash
	bazel build //src:ovms -c dbg
	```
	```bash
	gdb --args ./bazel-bin/src/ovms --model_name resnet --model_path /models
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
</details>

<details><summary>Use minitrace to display flame graph</summary>

Download the model files and store them in the `models` directory
```bash
mkdir -p models/resnet/1
curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml -o models/resnet/1/resnet50-binary-0001.bin -o models/resnet/1/resnet50-binary-0001.xml
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
bazel build --copt="-DMTR_ENABLED" //src:ovms
```

4. Run OVMS with `--trace_path` specifying where to save flame graph JSON file.
```bash
bazel-bin/src/ovms --model_name resnet --model_path models/resnet --trace_path trace.json
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
docker run -it -v ${PWD}:/workspace:rw -p 9178:9178 openvino/model_server --model_name resnet --model_path /workspace/models/resnet --trace_path /workspace/trace.json 
```

3. During app exit, the trace info will be saved into `${PWD}/trace.json`.

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
