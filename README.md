# OpenVINO&trade; Model Server

OpenVINO&trade; Model Server (OVMS) is a scalable, high-performance solution for serving machine learning models optimized for Intel&reg; architectures. 
The server provides an inference service via gRPC or REST API - making it easy to deploy new algorithms and AI experiments using the same 
architecture as [TensorFlow Serving](https://github.com/tensorflow/serving) for any models trained in a framework that is supported 
by [OpenVINO](https://software.intel.com/en-us/openvino-toolkit). 

The server implements gRPC and REST API framework with data serialization and deserialization using TensorFlow Serving API,
 and OpenVINO&trade; as the inference execution provider. Model repositories may reside on a locally accessible file system (e.g. NFS),
  Google Cloud Storage (GCS), Amazon S3, Minio or Azure Blob Storage.
  
OVMS is now implemented in C++ and provides much higher scalability compared to its predecessor in Python version.
You can take advantage of all the power of Xeon CPU capabilities or AI accelerators and expose it over the network interface.
Read [release notes](https://github.com/openvinotoolkit/model_server/releases) to find out what's new in C++ version.

Review the [Architecture concept](docs/architecture.md) document for more details.

A few key features: 
- Support for multiple frameworks. Serve models trained in popular formats such as Caffe*, TensorFlow*, MXNet* and ONNX*.
- Online deployment of new [model versions](docs/model_version_policy.md).
- Support for AI accelerators including [Intel Movidius Myriad VPUs](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_VPU.html), 
[GPU](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_CL_DNN.html) and [HDDL](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_HDDL.html). 
- The server can be enabled both on [Docker containers](docs/docker_container.md).
- [Kubernetes deployments](deploy). The server can be deployed in a Kubernetes cluster allowing the inference service to scale horizontally and ensure high availability.  
- [Model reshaping](docs/shape_and_batch_size.md). The server supports reshaping models in runtime. 
- [Model ensemble](docs/ensemble_scheduler.md) (preview). Connect multiple models to deploy complex processing solutions and reduce overhead of sending data back and forth.

**Note:** OVMS has been tested on CentOS* and Ubuntu*. Publically released docker images are based on CentOS.

## Build OpenVINO Model Server
Build the docker image using command:
```bash
make docker_build
```
called from the root directory of this github repository.

It will generate the images, tagged as:
* `openvino/model_server:latest` - with CPU, NCS and HDDL support
* `openvino/model_server:latest-gpu` - with CPU, NCS, HDDL and iGPU support

as well as a release package (.tar.gz, with ovms binary and necessary libraries), in a ./dist directory.

The release package is compatible with linux machines on which `glibc` version is greater than or equal to the build image version 2.17.
For debugging, an image with a suffix `-build` is also generated (i.e. `openvino/model_server-build:latest`).

*Note:* Images include OpenVINO 2021.1 release. <br>


## Run OpenVINO Model Server

A demonstration how to use OpenVINO Model Server can be found in [a quick start guide](docs/ovms_quickstart.md).

More detailed guides to using Model Server in various scenarios can be found here:

* [Models repository configuration](docs/models_repository.md)

* [Using a docker container](docs/docker_container.md)

* [Performance tuning](docs/performance_tuning.md)

* [Model Ensemble Scheduler](docs/ensemble_scheduler.md)


## API documentation

### GRPC 

Learn more about [GRPC API](docs/model_server_grpc_api.md)

Refer to the [GRPC example client code](example_client/README.md#grpc-api-client-examples) to learn how to use and submit the requests using the gRPC interface.

### REST

Learn more about [REST API](docs/model_server_rest_api.md)

Refer to the [REST API example client code](./example_client/README.md#rest-api-client-examples) to learn how to use REST API 

## Testing

Learn more about tests in the [developer guide](docs/developer_guide.md)


## Known Limitations

* Currently, `Predict`, `GetModelMetadata` and `GetModelStatus` calls are implemented using Tensorflow Serving API. 
* `Classify`, `Regress` and `MultiInference` are not included.
* Output_filter is not effective in the Predict call. All outputs defined in the model are returned to the clients. 


## OpenVINO Model Server Contribution Policy

* All contributed code must be compatible with the [Apache 2](https://www.apache.org/licenses/LICENSE-2.0) license.

* All changes needs to have pass linter, unit and functional tests.

* All new features need to be covered by tests.

Follow a [contributor guide](docs/contributing.md) and a [developer guide](docs/developer_guide.md).


## References

* [OpenVINO&trade;](https://software.intel.com/en-us/openvino-toolkit)

* [TensorFlow Serving](https://github.com/tensorflow/serving)

* [gRPC](https://grpc.io/)

* [RESTful API](https://restfulapi.net/)

* [Inference at scale in Kubernetes](https://www.intel.ai/inference-at-scale-in-kubernetes)


## Contact

Submit Github issue to ask question, request a feature or report a bug.


---
\* Other names and brands may be claimed as the property of others.



