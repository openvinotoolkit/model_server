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
- Deploy new model versions without changing client code.

More complete guides to using Model Server in various scenarios can be found here:

* [Models repository configuration](docs/models_repository.md)

## Advanced Configuration

[Performance tuning](docs/performance_tuning.md)

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
and with numpy data input

## Known Limitations and Plans

* Currently, *Predict*, *GetModelMetadata* and *GetModelStatus* calls are implemented using Tensorflow Serving API. 
* Classify*, *Regress* and *MultiInference* are not included.
* Output_filter is not effective in the Predict call. All outputs defined in the model are returned to the clients. 


## References

[OpenVINO&trade;](https://software.intel.com/en-us/openvino-toolkit)

[TensorFlow Serving](https://github.com/tensorflow/serving)

[gRPC](https://grpc.io/)

[RESTful API](https://restfulapi.net/)

[Inference at scale in Kubernetes](https://www.intel.ai/inference-at-scale-in-kubernetes)

[OpenVINO Model Server boosts AI](https://www.intel.ai/openvino-model-server-boosts-ai-inference-operations/)


---
\* Other names and brands may be claimed as the property of others.
