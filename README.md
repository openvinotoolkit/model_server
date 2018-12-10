# OpenVINO&trade; model server

Inference model server implementation with gRPC interface, compatible with TensorFlow serving API and OpenVINO&trade; as the execution backend.


## Project overview

“OpenVINO&trade; model server” is a flexible, high-performance inference serving component for artificial intelligence models.  
The software makes it easy to deploy new algorithms and AI experiments, 
while keeping the same server architecture and APIs like in [TensorFlow Serving](https://github.com/tensorflow/serving). 

It provides out-of-the-box integration with models supported by [OpenVINO&trade;](https://software.intel.com/en-us/openvino-toolkit)
and allows frameworks such as [AWS Sagemaker](https://github.com/aws/sagemaker-tensorflow-containers) to serve AI models with OpenVINO&trade;. 

It is implemented as a python service using gRPC library interface; data serialization and deserialization 
using TensorFlow; and OpenVINO&trade; for inference execution. It acts as an integrator and a bridge exposing CPU optimized 
inference engine over gRPC network interface. 

Review the [Architecture concept](docs/architecture.md) document for more details.

## Getting it up and running

[Using a docker container](docs/docker_container.md)

[Landing on bare metal or virtual machine](docs/host.md)


## Advanced configuration

[Custom CPU extensions](docs/cpu_extension.md)

Using FPGA and Movidius (TBD)


## API documentation

OpenVINO&trade; model server API is documented in proto buffer files  in [tensorflow_serving_api](ie_serving/tensorflow_serving_api/prediction_service.proto).
**Note:** The implementations for *Predict* and *GetModelMetadata* function calls are currently available. These are the most generic function calls and should address most of the usage scenarios.

[predict Function Spec](ie_serving/tensorflow_serving_api/predict.proto) has two message definitions: *PredictRequest* and  *PredictResponse*.  
* *PredictRequest* specifies information about the model spec, a map of input data serialized via [TensorProto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto) to a string format and an optional output filter.
* *PredictResponse* includes a map of outputs serialized by [TensorProto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto) and information about the used model spec.
 

[get_model_metadata Function spec](ie_serving/tensorflow_serving_api/get_model_metadata.proto) has three message definitions: *SignatureDefMap*, *GetModelMetadataRequest*, *GetModelMetadataResponse*. A function call GetModelMetadata accepts model spec information as input and returns Signature Definition content in the format similar to TensorFlow Serving.

Refer to the example client code to learn how to use this API and submit the requests using gRPC interface.

## Usage examples

[Kubernetes](example_k8s)

[Sagemaker](example_sagemaker)

[Client Code Example](example_client)


## Benchmarking results

[Report for resnet models](docs/benchmark.md)

## References

[OpenVINO&trade;](https://software.intel.com/en-us/openvino-toolkit)

[TensorFlow Serving](https://github.com/tensorflow/serving)

[gRPC](https://grpc.io/)


## Troubleshooting
### Server logging

OpenVINO&trade; model server accepts 3 logging levels:

* ERROR: Logs information about inference processing errors and server initialization issues.
* INFO: Presents information about server startup procedure.
* DEBUG: Stores information about client requests.

The default setting is **INFO**, which can be altered by setting environment variable `LOG_LEVEL`.

The captured logs will be displayed on the model server console. While using docker containers or kubernetes the logs
can be examined using `docker logs` or `kubectl logs` commands respectively.

It is also possible to save the logs to a local file system by configuring an environment variable `LOG_PATH` with the absolute path pointing to a log file. Please see example below for usage details.

```
docker run --name ie-serving --rm -d -v /models/:/opt/ml:ro -p 9001:9001 --env LOG_LEVEL=DEBUG --env LOG_PATH=/var/log/ie_serving.log \
 ie-serving-py:latest /ie-serving-py/start_server.sh ie_serving config --config_path /opt/ml/config.json --port 9001
 
docker logs ie-serving 

```  


### Model import issues
OpenVINO&trade; model server will fail to start when any of the defined model cannot be loaded successfully. The root cause of
the failure can be determined based on the collected logs on the console or in the log file.

The following problem might occur during model server initialization and model loading:
* Missing model files in the location specified in the configuration file.
* Missing version sub-folders in the model folder.
* Model files require custom CPU extension.

### Client request issues
When the model server starts successfully and all the models are imported, there could be a couple of reasons for errors 
in the request handling. 
The information about the failure reason is passed to the gRPC client in the response. It is also logged on the 
model server in the DEBUG mode.

The possible issues could be:
* Incorrect shape of the input data.
* Incorrect input key name which does not match the tensor name or set input key name in `tensors_mapping.json`.
* Incorrectly serialized data on the client side.

### Resource allocation
RAM consumption might depend on the size and volume of the models configured for serving. It should be measured experimentally, however it can be estimated that each model will consume RAM size equal to the size of the model weights file (.bin file).
Every version of the model creates a separate inference engine object, so it is recommended to mount only the desired model versions.

OpenVINO&trade; model server consumes all available CPU resources unless they are restricted by operating system, docker or 
kubernetes capabilities.

Usually, there is no need to tune any environment variables according to the allocated resources.

### Usage monitoring
It is possible to track the usage of the models including processing time while DEBUG mode is enabled.
With this setting model server logs will store information about all the incoming requests.
You can parse the logs to analyze: volume of requests, processing statistics and most used models.

## Known limitations and plans

* Currently, only *Predict* and *GetModelMetadata* calls are implemented using Tensorflow Serving API. 
*Classify*, *Regress* and *MultiInference* are planned to be added.
* *GetModelMetadata* is reporting incorrect shape of model outputs. A fix is pending in OpenVINO&trade; API for reporting outputs shape.
* In current version, models need to be mounted or present on the local file system. 
There are plans to add support for s3, ftp, sftp and http storage.
* Currently, model versions are detected at server start time. Adding new versions requires service restart or
 starting new docker container. There are plans to add online detection of new model versions and config file changes.
* Output_filter is not effective in the Predict call. All outputs defined in the model are returned to the clients. 

## Contribution

### Contribution rules

All contributed code must be compatible with the [Apache 2](https://www.apache.org/licenses/LICENSE-2.0) license.

All changes needs to have passed style, unit and functional tests.

All new features need to be covered by tests.

### Building

`make docker_build_bin` to build new docker image using OpenVINO binary distribution

`make docker_build_src` to build new docker image using OpenVINO source code

### Testing

`make style` to run linter tests

`make unit` to execute unit tests (it requires OpenVINO installation followed by `make install`)

`make test` to execute functional tests (it requires building the docker image in advance). Running 
the tests require also preconfigured env variables `GOOGLE_APPLICATION_CREDENTIALS`, `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY` and `AWS_REGION` with permissions to access models used in tests.
To run tests limited to models to locally downloaded models use command:

`make test_local_only`


## Contact

Submit Github issue to ask question, submit a request or report a bug.


