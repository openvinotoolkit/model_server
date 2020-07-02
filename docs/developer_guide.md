## Development environment BKM

Mount the source code inside the docker container
```bash
docker run -it -v ${PWD}:/ovms --entrypoint bash -p 9178:9178 ovms-build:latest 
```

Compile code using command:

```bash
bazel build //src:ovms
```

Run single unit test with flag "--test_filter", e.g.:

```bash
bazel test --test_summary=detailed --test_output=all --test_filter='ModelVersionStatus.*' //src:ovms_test
```

Run build on shared machine with extra makefile flags:

```bash
OVMS_CPP_DOCKER_IMAGE=rr_ovms OVMS_CPP_CONTAINTER_PORT=9278 make docker_build
```

Build using a non-default OS docker image:

```bash
make BASE_OS=clearlinux OVMS_CPP_DOCKER_IMAGE=my-ovms-clearlinux-image
```

Build, without using a docker cache:

```bash
make NO_DOCKER_CACHE=true
```

### Debugging in docker (using `gdb`)  
Run container:
```
$ docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v ${PWD}:/ovms -p 9178:9178 -e "http_proxy=$http_proxy" -e "https_proxy=$https_proxy" --entrypoint bash ovms-build:latest
```
In container install and run `gdb`, recompile ovms with debug symbols:
```
[root@72dc3b874772 ovms]# yum -y install gdb
[root@72dc3b874772 ovms]# bazel build //src:ovms -c dbg
[root@72dc3b874772 ovms]# gdb --args ./bazel-bin/src/ovms --model_name resnet --model_path /model
```


## OVMS testing

### Python prerequisites

To use Makefile targets which operate on Python it is required to have:
* Python >= 3.6
* `virtualenv` package installed system-wide:
```
pip3 install virtualenv
```

### Testing inference with an arbitrary model

You can download an exemplary model using script `tests/performance/download_model.sh`. It is ResNet50 quantized to INT8 precision.
The script stores the model in the user home folder. You can use any other model from OpenVINO model zoo.

When the docker container is started like in the example above, use and adjust the following grpc client.
It connects to previously started OVMS service running on a default port 9178.

```bash
make venv
source .venv/bin/activate
python3 tests/performance/grpc_latency.py --images_numpy_path tests/performance/imgs.npy --labels_numpy_path tests/performance/labels.npy \
 --iteration 1000 --batchsize 1 --report_every 100 --input_name data
```

`images_numpy_path` parameter should include numpy array with a batch of input data.

`labels_numpy_path` includes a numpy array with image classification results for the test dataset to measure accuracy.


### Running functional tests

```bash
make test_functional
``` 
Default tests configuration can be changed by usage of environment variables. 
To store them in a file, create `user_config.py` in the main directory of the project.
The following variables are available for customization:

`IMAGE` - docker image name which should be used to run tests.

`TEST_DIR` -  location where models and test data should be downloaded.

`LOG_LEVEL` - set log level.

`BUILD_LOGS` - path to dir where artifacts should be stored.

`START_CONTAINER_COMMAND` - command to start ovms container.

`CONTAINER_LOG_LINE` - log line to check in container to confirm that it has started properly.

Example usage:

```bash
os.environ["IMAGE"] = "ie-serving-py:latest"
```

There's also an option to specify these variables in environment via command line by using export, e.g.:
```bash
export IMAGE="ie-serving-py:latest"
```

### Running basic performance tests

Automated tests are configure to use ResNet50 model quantized to INT8 precision.    

```bash
make test_perf
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

```bash
make test_throughput
Running throughput test
[25] Starting iterations
[23] Starting iterations
.....
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

### Running tests on Python image

WARNING: at this point not all tests will pass. Further changes are needed to achieve that.

To run tests (test_batching, test_mapping, test_single_model) on Python image specify following variables in user_config.py or in environment by using export:
```
os.environ["START_CONTAINER_COMMAND"] = "/ie-serving-py/start_server.sh ie_serving model "
os.environ["CONTAINER_LOG_LINE"] = "server listens on port"
```

To run tests (test_model_version_policy, test_model_versions_handling, test_multi_models) on Python image specify following variables in user_config.py or in environment by using export:
```
os.environ["START_CONTAINER_COMMAND"] = "/ie-serving-py/start_server.sh ie_serving config "
os.environ["CONTAINER_LOG_LINE"] = "server listens on port"
```

### Running tests on OVMS binary file

To run tests on OVMS binary file specify following variable in user_config.py or in environment by using export, its value should be replaced by actual path to binary file:

```
os.environ["OVMS_BINARY_PATH"] = "/home/example_path/ovms/bin/ovms"
```

The following command executed in location of OVMS binary file should return paths to "lib" directory included in ovms.tar.gz file (ovms/bin/./../lib).
```
ldd ./ovms
```
Otherwise specify following variable in user_config.py or in environment by using export:
```
os.environ["LD_LIBRARY_PATH"] = ""
```
