# OpenVINO&trade; Model Server
OpenVINO&trade; Model Server is a scalable, high-performance solution for serving machine learning models optimized for Intel&reg; architectures. The server provides an inference service via gRPC enpoint or REST API -- making it easy to deploy new algorithms and AI experiments using the same architecture as [TensorFlow Serving](https://github.com/tensorflow/serving) for any models trained in a framework that is supported by [OpenVINO](https://software.intel.com/en-us/openvino-toolkit). 

The server is implemented as a python service using the gRPC interface library or falcon REST API framework with data serialization and deserialization using TensorFlow, and OpenVINO&trade; as the inference execution provider. Model repositories may reside on a locally accessible file system (e.g. NFS), Google Cloud Storage (GCS), Amazon S3 or MinIO.

Review the [Architecture concept](docs/architecture.md) document for more details.

A few key features: 
- Support for multiple frameworks. Serve models trained in popular formats such as Caffe*, TensorFlow*, MXNet* and ONNX*.
- Deploy new [model versions](https://github.com/IntelAI/OpenVINO-model-server/blob/master/docs/docker_container.md#model-version-policy) without changing client code.
- Support for AI accelerators including [Intel Movidius Myriad VPUs](https://www.intel.ai/intel-movidius-myriad-vpus/#gs.xrw7cj). The server can be enabled both on [Bare Metal Hosts](docs/host.md#using-hddl-accelerators) or in
[Docker containers](docs/docker_container.md#starting-docker-container-with-hddl).
- [Kubernetes deployments](deploy). The server can be deployed in a Kubernetes cluster allowing the inference service to scale horizontally and ensure high availability.  
- [Sagemaker integration](example_sagemaker). The server supports using AWS SageMaker containers for serving inferece execution.  
- Supports [multi-worker configuration](https://github.com/IntelAI/OpenVINO-model-server/blob/master/docs/performance_tuning.md#multi-worker-configuration) and [parallel inference execution](https://github.com/IntelAI/OpenVINO-model-server/blob/master/docs/performance_tuning.md#multiple-model-server-instances).
- [Model reshaping](https://github.com/IntelAI/OpenVINO-model-server/blob/master/docs/docker_container.md#model-reshaping). The server supports reshaphing models in runtime. 

## Getting Up and Running

[Using a docker container](docs/docker_container.md)

[Landing on bare metal or virtual machine](docs/host.md)


## Advanced Configuration

[Custom layer extensions](docs/cpu_extension.md)

[Performance tuning](docs/performance_tuning.md)

Using FPGA (TBD)


## gRPC API Documentation

OpenVINO&trade; Model Server gRPC API is documented in the proto buffer files in [tensorflow_serving_api](https://github.com/tensorflow/serving/tree/r1.14/tensorflow_serving/apis). **Note:** The implementations for *Predict*, *GetModelMetadata* and *GetModelStatus* function calls are currently available. 
These are the most generic function calls and should address most of the usage scenarios.

[predict function spec](https://github.com/tensorflow/serving/blob/r1.14/tensorflow_serving/apis/predict.proto) has two message definitions: *PredictRequest* and  *PredictResponse*.  
* *PredictRequest* specifies information about the model spec, a map of input data serialized via 
[TensorProto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto) to a string format.
* *PredictResponse* includes a map of outputs serialized by 
[TensorProto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto) and information about the used model spec.
 
[get_model_metadata function spec](https://github.com/tensorflow/serving/blob/r1.14/tensorflow_serving/apis/get_model_metadata.proto) has three message definitions:
 *SignatureDefMap*, *GetModelMetadataRequest*, *GetModelMetadataResponse*. 
 A function call GetModelMetadata accepts model spec information as input and returns Signature Definition content in the format similar to TensorFlow Serving.

[get model status function spec](https://github.com/tensorflow/serving/blob/r1.14/tensorflow_serving/apis/get_model_status.proto) can be used to report
all exposed versions including their state in their lifecycle. 

Refer to the [example client code](example_client) to learn how to use this API and submit the requests using the gRPC interface.

Using the gRPC interface is recommended for optimal performace due to its faster implementation of input data deserialization. gRPC achieves lower latency, especially with larger input messages like images. 

## RESTful API Documentation 

OpenVINO&trade; Model Server RESTful API follows the documentation from [tensorflow serving rest api](https://www.tensorflow.org/tfx/serving/api_rest).

Both row and column format of the requests are implemented. 
**Note:** Just like with gRPC, only the implementations for *Predict*, *GetModelMetadata* and *GetModelStatus* function calls are currently available. 

Only the numerical data types are supported. 

Review the exemplary clients below to find out more how to connect and run inference requests.

REST API is recommended when the primary goal is in reducing the number of client side python dependencies and simpler application code.

## Usage Examples
- Using *Predict* function over [gRPC](example_client/#submitting-grpc-requests-based-on-a-dataset-from-numpy-files) 
and [RESTful API](example_client/#rest-api-client-to-predict-function) with numpy data input
- [Using *GetModelMetadata* function  over gRPC and RESTful API](example_client/#getting-info-about-served-models)
- [Using *GetModelStatus* function  over gRPC and RESTful API](example_client/#getting-model-serving-status)
- [Example script submitting jpeg images for image classification](example_client/#submitting-grpc-requests-based-on-a-dataset-from-a-list-of-jpeg-files)
- [Deploy with Kubernetes](deploy)
- [Jupyter notebook - REST API client for age-gender classification](example_client/REST_age_gender.ipynb)

## References

[OpenVINO&trade;](https://software.intel.com/en-us/openvino-toolkit)

[TensorFlow Serving](https://github.com/tensorflow/serving)

[gRPC](https://grpc.io/)

[RESTful API](https://restfulapi.net/)

[Inference at scale in Kubernetes](https://www.intel.ai/inference-at-scale-in-kubernetes)

[OpenVINO Model Server boosts AI](https://www.intel.ai/openvino-model-server-boosts-ai-inference-operations/)

## Troubleshooting
### Server Logging

OpenVINO&trade; model server accepts 3 logging levels:

* ERROR: Logs information about inference processing errors and server initialization issues.
* INFO: Presents information about server startup procedure.
* DEBUG: Stores information about client requests.

The default setting is **INFO**, which can be altered by setting environment variable `LOG_LEVEL`.

The captured logs will be displayed on the model server console. While using docker containers or kubernetes the logs
can be examined using `docker logs` or `kubectl logs` commands respectively.

It is also possible to save the logs to a local file system by configuring an environment variable `LOG_PATH` with the absolute path pointing to a log file. 
Please see example below for usage details.

```
docker run --name ie-serving --rm -d -v /models/:/opt/ml:ro -p 9001:9001 --env LOG_LEVEL=DEBUG --env LOG_PATH=/var/log/ie_serving.log \
 ie-serving-py:latest /ie-serving-py/start_server.sh ie_serving config --config_path /opt/ml/config.json --port 9001
 
docker logs ie-serving 

```  


### Model Import Issues
OpenVINO&trade; Model Server loads all defined models versions according 
to set [version policy](docs/docker_container.md#model-version-policy). 
A model version is represented by a numerical directory in a model path, 
containing OpenVINO model files with .bin and .xml extensions.

Below are examples of incorrect structure:
```bash
models/
├── model1
│   ├── 1
│   │   ├── ir_model.bin
│   │   └── ir_model.xml
│   └── 2
│       ├── somefile.bin
│       └── anotherfile.txt
└── model2
    ├── ir_model.bin
    ├── ir_model.xml
    └── mapping_config.json
```

In above scenario, server will detect only version `1` of `model1`.
Directory `2` does not contain valid OpenVINO model files, so it won't 
be detected as a valid model version. 
For `model2`, there are correct files, but they are not in a numerical directory. 
The server will not detect any version in `model2`.

When new model version is detected, the server loads the model files 
and starts serving new model version. This operation might fail for the following reasons:
- there is a problem with accessing model files (i. e. due to network connectivity issues
to the  remote storage or insufficient permissions)
- model files are malformed and can not be imported by the Inference Engine
- model requires custom CPU extension

In all those situations, the root cause is reported in the server logs or in the response from a call
to GetModelStatus function. 

Detected but not loaded model version will not be served and will report status
`LOADING` with error message: `Error occurred while loading version`.
When model files become accessible or fixed, server will try to 
load them again on the next [version update](docs/docker_container.md#updating-model-versions) 
attempt.

At startup, the server will enable gRPC and REST API endpoint, after all configured models and detected model versions
are loaded successfully (in AVAILABLE state).

The server will fail to start if it can not list the content of configured model paths.


### Client Request Issues
When the model server starts successfully and all the models are imported, there could be a couple of reasons for errors 
in the request handling. 
The information about the failure reason is passed to the gRPC client in the response. It is also logged on the 
model server in the DEBUG mode.

The possible issues could be:
* Incorrect shape of the input data.
* Incorrect input key name which does not match the tensor name or set input key name in `mapping_config.json`.
* Incorrectly serialized data on the client side.

### Resource Allocation
RAM consumption might depend on the size and volume of the models configured for serving. It should be measured experimentally, 
however it can be estimated that each model will consume RAM size equal to the size of the model weights file (.bin file).
Every version of the model creates a separate inference engine object, so it is recommended to mount only the desired model versions.

OpenVINO&trade; model server consumes all available CPU resources unless they are restricted by operating system, docker or 
kubernetes capabilities.

### Usage Monitoring
It is possible to track the usage of the models including processing time while DEBUG mode is enabled.
With this setting model server logs will store information about all the incoming requests.
You can parse the logs to analyze: volume of requests, processing statistics and most used models.

## Inference Results Serialization

Model server employs configurable serialization function. 

The default implementation starting from 2020.1 version is
 [_prepare_output_with_make_tensor_proto](ie_serving/server/predict_utils.py).
It employs TensorFlow function [make_tensor_proto](https://www.tensorflow.org/api_docs/python/tf/make_tensor_proto). 
For most of the models it returns TensorProto response with inference results serialized to string via a numpy.toString call. 
This method achieves low latency, especially for models with big size of the output.

Prior 2020.1 version, serialization was using function [_prepare_output_as_AppendArrayToTensorProto](ie_serving/server/predict_utils.py).
Contrary to make_tensor_proto, it returns the inference results as TensorProto object containing a list of numerical elements.


In both cases, the results can be deserialized on the client side with [make_ndarray](https://www.tensorflow.org/api_docs/python/tf/make_ndarray).
If you're using tensorflow's `make_ndarray` to read output
 in your client application, then the transition between those methods is transparent.
 
Add environment variable `SERIALIZATION_FUNCTION=_prepare_output_as_AppendArrayToTensorProto` to enforce the usage 
of legacy serialization method.
 
## Known Limitations and Plans

* Currently, *Predict*, *GetModelMetadata* and *GetModelStatus* calls are implemented using Tensorflow Serving API. 
*Classify*, *Regress* and *MultiInference* are planned to be added.
* Output_filter is not effective in the Predict call. All outputs defined in the model are returned to the clients. 


## Contribution

### Contribution Rules

All contributed code must be compatible with the [Apache 2](https://www.apache.org/licenses/LICENSE-2.0) license.

All changes needs to have passed style, unit and functional tests.

All new features need to be covered by tests.

### Building
Docker image with OpenVINO Model Server can be built with several options: 
- `make docker_build_bin dldt_package_url=<url>` - using Intel Distribution of OpenVINO binary package (ubuntu base image)
- `make docker_build_apt_ubuntu` - using OpenVINO apt packages with ubuntu base image
- `make docker_build_ov_base` - using public image of OpenVINO runtime base image
- `make docker_build_clearlinux` - using clearlinux base image with DLDT package 

*Note:* Images based on ubuntu include OpenVINO 2020.1. <br>
In clearlinux based image, it is 2019.3 - to be upgraded later soon.

### Testing

`make style` to run linter tests

`make unit` to execute unit tests (it requires OpenVINO installation followed by `make install`)
Alternatively unit tests can be executed in a container by running the script `./tests/scripts/unit-tests.sh`

`make test` to execute full set of functional tests (it requires [building the docker image](README.md#building) in advance). 


## Contact

Submit Github issue to ask question, request a feature or report a bug.


---
\* Other names and brands may be claimed as the property of others.
