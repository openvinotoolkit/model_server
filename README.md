# OpenVINO&trade; Model Server

OpenVINO&trade; Model Server is a scalable, high-performance solution for serving machine learning models optimized for Intel&reg; architectures. 
The server provides an inference service via gRPC endpoint or REST API -- making it easy to deploy new algorithms and AI experiments using the same 
architecture as [TensorFlow Serving](https://github.com/tensorflow/serving) for any models trained in a framework that is supported 
by [OpenVINO](https://software.intel.com/en-us/openvino-toolkit). 

The server implements gRPC interface and REST API framework with data serialization and deserialization using TensorFlow Serving API,
 and OpenVINO&trade; as the inference execution provider. Model repositories may reside on a locally accessible file system (e.g. NFS),
  Google Cloud Storage (GCS), Amazon S3 or Minio.
  
OVMS is now implemented in C++ and provides much higher scalability compared to its predecessor in Python version.
You can take advantage of all the power of Xeon CPU capabilities or AI accelerators and expose it over the network interface.
Read [release notes](docs/release_notes.md) to find out about changes from the Python implementation.

Review the [Architecture concept](docs/architecture.md) document for more details.

A few key features: 
- Support for multiple frameworks. Serve models trained in popular formats such as Caffe*, TensorFlow*, MXNet* and ONNX*.
- Deploy new [model versions](https://github.com/IntelAI/OpenVINO-model-server/blob/master/docs/docker_container.md#model-version-policy) without changing client code.
- Support for AI accelerators including [Intel Movidius Myriad VPUs](https://www.intel.ai/intel-movidius-myriad-vpus/#gs.xrw7cj), 
[GPU](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_CL_DNN.html) and [HDDL](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_HDDL.html). 
- The server can be enabled both on [Bare Metal Hosts](docs/host.md#using-hddl-accelerators) or in
[Docker containers](docs/docker_container.md#starting-docker-container-with-hddl).
- [Kubernetes deployments](deploy). The server can be deployed in a Kubernetes cluster allowing the inference service to scale horizontally and ensure high availability.  
- [Model reshaping](docs/docker_container.md#model-reshaping). The server supports reshaphing models in runtime. 

## Building
Build the docker image using command:
```bash
~/ovms-c$ make docker_build
```
It will generate the image, tagged as `ovms:latest`, as well as a release package (.tar.gz, with ovms binary and necessary libraries), in a ./dist directory.
The release package should work on a any linux machine with glibc >= one used by the build image.
For debugging, an image with a suffix -build is also generated (i.e. ovms-build:latest).

*Note:* Images include OpenVINO 2020.4 release. <br>


## Running the Server

Start using OpenVINO Model Server in 5 Minutes or less:

```bash

# Download model into a separate directory
curl --create-dirs https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.xml https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.bin -o model/face-detection-retail-0004.xml -o model/face-detection-retail-0004.bin

# Start the container serving gRPC on port 9000
docker run -d -v $(pwd)/model:/models/face-detection/1 -p 9000:9000 ger-registry-pre.caas.intel.com/ovms/model_server:latest --model_path /models/face-detection --model_name face-detection --port 9000 --log_level DEBUG

# Download the example client script
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/client_utils.py -o client_utils.py https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/face_detection.py -o face_detection.py  https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/client_requirements.txt -o client_requirements.txt

# Download an image to be analyzed
curl --create-dirs https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/images/people/people1.jpeg -o images/people1.jpeg

# Install client dependencies
pip install -r client_requirements.txt

# Create a folder for results
mkdir results

# Run inference and store results in the newly created folder
python face_detection.py --batch_size 1 --width 600 --height 400 --input_images_dir images --output_dir results
```
A more detailed description of the steps above can be found [here](docs/ovms_quickstart.md).

More complete guides to using Model Server in various scenarios can be found here:

* [Models repository configuration](docs/models_repository.md)

* [Using a docker container](docs/docker_container.md)

* [Landing on bare metal or virtual machine](docs/host.md)

* [Performance tuning](docs/performance_tuning.md)

* [Model Ensemble Scheduler](docs/ensemble_scheduler.md)


## gRPC API Documentation

OpenVINO&trade; Model Server gRPC API is documented in the proto buffer files in [tensorflow_serving_api](https://github.com/tensorflow/serving/tree/r2.2/tensorflow_serving/apis). **Note:** The implementations for *Predict*, *GetModelMetadata* and *GetModelStatus* function calls are currently available. 
These are the most generic function calls and should address most of the usage scenarios.

[predict function spec](https://github.com/tensorflow/serving/blob/r2.2/tensorflow_serving/apis/predict.proto) has two message definitions: *PredictRequest* and  *PredictResponse*.  
* *PredictRequest* specifies information about the model spec, a map of input data serialized via 
[TensorProto](https://github.com/tensorflow/tensorflow/blob/r2.2/tensorflow/core/framework/tensor.proto) to a string format.
* *PredictResponse* includes a map of outputs serialized by 
[TensorProto](https://github.com/tensorflow/tensorflow/blob/r2.2/tensorflow/core/framework/tensor.proto) and information about the used model spec.
 
[get_model_metadata function spec](https://github.com/tensorflow/serving/blob/r2.2/tensorflow_serving/apis/get_model_metadata.proto) has three message definitions:
 *SignatureDefMap*, *GetModelMetadataRequest*, *GetModelMetadataResponse*. 
 A function call GetModelMetadata accepts model spec information as input and returns Signature Definition content in the format similar to TensorFlow Serving.

[get model status function spec](https://github.com/tensorflow/serving/blob/r2.2/tensorflow_serving/apis/get_model_status.proto) can be used to report
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

## Testing

Learn more about tests in the [developer guide](docs/developer_guide.md)


## Known Limitations and Plans

* Currently, *Predict*, *GetModelMetadata* and *GetModelStatus* calls are implemented using Tensorflow Serving API. 
* Classify*, *Regress* and *MultiInference* are not included.
* Output_filter is not effective in the Predict call. All outputs defined in the model are returned to the clients. 


## Contribution

### Contribution Rules

All contributed code must be compatible with the [Apache 2](https://www.apache.org/licenses/LICENSE-2.0) license.

All changes needs to have passed style, unit and functional tests.

All new features need to be covered by tests.

Follow a [contributor guide](docs/contributing.md) and a [developer guide](docs/developer_guide.md)


## References

[OpenVINO&trade;](https://software.intel.com/en-us/openvino-toolkit)

[TensorFlow Serving](https://github.com/tensorflow/serving)

[gRPC](https://grpc.io/)

[RESTful API](https://restfulapi.net/)

[Inference at scale in Kubernetes](https://www.intel.ai/inference-at-scale-in-kubernetes)

[OpenVINO Model Server boosts AI](https://www.intel.ai/openvino-model-server-boosts-ai-inference-operations/)


## Contact

Submit Github issue to ask question, request a feature or report a bug.


---
\* Other names and brands may be claimed as the property of others.



