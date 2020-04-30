# OpenVINO&trade; Model Server

OpenVINO&trade; Model Server is a scalable, high-performance solution for serving machine learning models optimized for Intel&reg; architectures. 
The server provides an inference service via gRPC endpoint or REST API -- making it easy to deploy new algorithms and AI experiments using the same 
architecture as [TensorFlow Serving](https://github.com/tensorflow/serving) for any models trained in a framework that is supported 
by [OpenVINO](https://software.intel.com/en-us/openvino-toolkit). 

The server implements gRPC interface and REST API framework with data serialization and deserialization using TensorFlow Serving API,
 and OpenVINO&trade; as the inference execution provider. Model repositories may reside on a locally accessible file system (e.g. NFS),
  Google Cloud Storage (GCS), Amazon S3 or MinIO.
  
OVMS is now implemented in C++ and provides much higher scalability compared to its predecessor in Python version.
You can take advantage of all the power of Xeon CPU capabilities or AI accelerators and expose it over the network interface.


## Building

Build the docker image using command:
 
```bash
~/ovms-c$ make docker_build
```
It will generate the image, tagged as `ovms:latest`.


## Running the serving component as a docker container:
Docker container is using the serving application as the entrypoint, so you just need to pass its parameters in the docker command:
```bash
docker run -d -v /models/model_folder:/opt/ml:ro -p 9178:9178 ovms --model_name <model_name> --model_path /opt/ml --port 9178
```
All parameters are documented below:
```bash
OpenVINO Model Server
Usage:
  ./bazel-bin/src/ovms [OPTION...]

  -h, --help  show this help message and exit

 config options:
      --config_path CONFIG_PATH
                                absolute path to json configuration file
      --port PORT               gRPC server port (default: 9178)
      --rest_port REST_PORT     REST server port, the REST server will not be
                                started if rest_port is blank or set to 0
                                (default: 0)
      --grpc_workers GRPC_WORKERS
                                number of gRPC servers. Recommended to be >=
                                NIREQ. Default value calculated at runtime:
                                NIREQ + 2
      --rest_workers REST_WORKERS
                                number of workers in REST server - has no
                                effect if rest_port is not set (default: 24)

 model options:
      --model_name MODEL_NAME   name of the model
      --model_path MODEL_PATH   absolute path to model, as in tf serving
      --batch_size BATCH_SIZE   sets models batchsize, int value or auto.
                                This parameter will be ignored if shape is set
                                (default: 0)
      --shape SHAPE             sets models shape (model must support
                                reshaping). If set, batch_size parameter is ignored
      --model_version_policy MODEL_VERSION_POLICY
                                model version policy
      --nireq NIREQ             Number of parallel inference request
                                executions for model. Recommended to be >=
                                CPU_THROUGHPUT_STREAMS. Default value calculated at
                                runtime: CPU cores / 8
      --target_device TARGET_DEVICE
                                Target device to run the inference (default:
                                CPU)
      --plugin_config PLUGIN_CONFIG
                                a dictionary of plugin configuration keys and
                                their values
```
## Testing

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

When the docker container is started like in the example above, use and adjust the following grpc client:

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

## Server Logging
OpenVINO™ model server accepts 3 logging levels:

* ERROR: Logs information about inference processing errors and server initialization issues.
* INFO: Presents information about server startup procedure.
* DEBUG: Stores information about client requests.

The default setting is INFO, which can be altered by setting environment variable LOG_LEVEL.

The captured logs will be displayed on the model server console. While using docker containers or kubernetes the logs can be examined using `docker logs` or `kubectl logs` commands respectively.

It is also possible to save the logs to a local file system by configuring an environment variable LOG_PATH with the absolute path pointing to a log file. Please see example below for usage details.

```bash
docker run --name ie-serving --rm -d -v /models/:/opt/ml:ro -p 9001:9001 --env LOG_LEVEL=DEBUG --env LOG_PATH=/var/log/ie_serving.log \
 cpp-experiments:latest --config_path /opt/ml/config.json --port 9001
 
docker logs ie-serving 
```

## Developer guide

Mount the source code inside the docker container
```bash
docker run -it -v ${PWD}:/ovms --entrypoint bash -p 9178:9178 ovms:latest 
```

Compile code using command:

```bash
bazel build //src:ovms
```