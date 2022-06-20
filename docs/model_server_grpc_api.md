# gRPC API {#ovms_docs_grpc_api}

## Introduction 
This document gives information about OpenVINO&trade; Model Server gRPC API. It is documented in the proto buffer files in [tensorflow_serving_api](https://github.com/tensorflow/serving/tree/r1.14/tensorflow_serving/apis). 
Using the gRPC interface is recommended for optimal performance due to its faster implementation of input data deserialization. gRPC achieves lower latency, especially with larger input messages like images. 

This document covers following API's endpoints coming from Tensorflow Serving gRPC API:
* <a href="#model-status">Model Status API</a>
* <a href="#model-metadata">Model Metadata API </a>
* <a href="#predict">Predict API </a>

> **NOTE**: The implementations for *Predict*, *GetModelMetadata* and *GetModelStatus* function calls are currently available. 
These are the most generic function calls and should address most of the usage scenarios.

Additionally in 2022.2 release of OpenVINO Model Server there is preview support for KServe gRPC API which is documented in [KServe](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md) repository.
This includes following endpoints:

* <a href="#kfs-model-metadata">Kserve Model Metadata API </a>
* <a href="#kfs-inference">KServe Inference API</a>

> **NOTE**: Only the implementations for *Inference* and *ModelMetadata* function calls are currently available as preview with the remaining API calls coming in next release.


## Model Status API <a name="model-status"></a>

- Description

Gets information about the status of served models including Model Version

 [Get Model Status proto](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/get_model_status.proto) defines three message definitions used while calling Status endpoint: *GetModelStatusRequest*, *ModelVersionStatus*, *GetModelStatusResponse* that are used to report all exposed versions including their state in their lifecycle.

 Read more about [Get Model Status API usage](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/client/python/tensorflow-serving-api/samples/README.md#model-status-api).


## Model Metadata API <a name="model-metadata"></a>

Gets information about the served models. A function called GetModelMetadata accepts model spec information as input and returns Signature Definition content in a format similar to TensorFlow Serving.
 
[Get Model Metadata proto](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/get_model_metadata.proto) has three message definitions: *SignatureDefMap*, *GetModelMetadataRequest*, *GetModelMetadataResponse*. 

Read more about [Get Model Metadata API usage](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/client/python/tensorflow-serving-api/samples/README.md#model-metadata-api).


## Predict API <a name="predict"></a>

Endpoint for running an inference with loaded models or [DAGs](./dag_scheduler.md).

[Predict proto](https://github.com/tensorflow/serving/blob/r1.14/tensorflow_serving/apis/predict.proto) has two message definitions: *PredictRequest* and  *PredictResponse*.
 * *PredictRequest* specifies information about the model spec, a map of input data serialized via 
[TensorProto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto) to a string format.
 * *PredictResponse* includes a map of outputs serialized by
[TensorProto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto) and information about the used model spec.

Read more about [Predict API usage](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/client/python/tensorflow-serving-api/samples/README.md#predict-api)

Check [how binary data is handled in OpenVINO Model Server](./binary_input.md)


## KServe Model Metadata API <a name="kfs-model-metadata"></a>
Gets information about the served models. Model name and model version are accepted as parameters.

Check KServe documentation for more [details](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#model-metadata-1).

Example of getting model metadata with KServe API is available [here](TODO).

## KServe Inference API <a name="kfs-inference"></a>
Run inference with requested model or [DAGs](./dag_scheduler.md).
Check KServe documentation for more [details](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference-1) about API.

Example of inference with KServe API is available [here](TODO).

Read supported about supported scope in 2022.2 release [TODO](TODO).


## See Also

- [Example client code](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/client/python/tensorflow-serving-api/samples/README.md) shows how to use GRPC API and REST API.
- [TensorFlow Serving](https://github.com/tensorflow/serving)
- [gRPC](https://grpc.io/)




 




