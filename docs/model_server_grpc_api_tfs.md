# TensorFlow Serving compatible gRPC API {#ovms_docs_grpc_api_tfs}

## Introduction
This document gives information about OpenVINO&trade; Model Server gRPC API compatible with TensorFlow Serving. It is documented in the proto buffer files in [tensorflow_serving_api](https://github.com/tensorflow/serving/tree/r2.9/tensorflow_serving/apis).
Using the gRPC interface is recommended for optimal performance due to its faster implementation of input data deserialization. gRPC achieves lower latency, especially with larger input messages like images.

This document covers following API's endpoints coming from Tensorflow Serving gRPC API:
* [Model Status API](#model-status-api)
* [Model Metadata API ](#model-metadata-api)
* [Predict API](#predict-api)

> **NOTE**: The implementations for *Predict*, *GetModelMetadata* and *GetModelStatus* function calls are currently available.
These are the most generic function calls and should address most of the usage scenarios.

## Model Status API

Gets information about the status of served models including Model Version

 [Get Model Status proto](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/get_model_status.proto) defines three message definitions used while calling Status endpoint: *GetModelStatusRequest*, *ModelVersionStatus*, *GetModelStatusResponse* that are used to report all exposed versions including their state in their lifecycle.

 Read more about [Get Model Status API usage](https://github.com/openvinotoolkit/model_server/blob/releases/2025/3/client/python/tensorflow-serving-api/samples/README.md#model-status-api).


## Model Metadata API

Gets information about the served models. A function called GetModelMetadata accepts model spec information as input and returns Signature Definition content in a format similar to TensorFlow Serving.

[Get Model Metadata proto](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/get_model_metadata.proto) has three message definitions: *SignatureDefMap*, *GetModelMetadataRequest*, *GetModelMetadataResponse*.

Read more about [Get Model Metadata API usage](https://github.com/openvinotoolkit/model_server/blob/releases/2025/3/client/python/tensorflow-serving-api/samples/README.md#model-metadata-api).


## Predict API

Endpoint for running an inference with loaded models or [DAGs](./dag_scheduler.md).

[Predict proto](https://github.com/tensorflow/serving/blob/r2.9/tensorflow_serving/apis/predict.proto) has two message definitions: *PredictRequest* and  *PredictResponse*.
 * *PredictRequest* specifies information about the model spec, a map of input data serialized via
[TensorProto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto) to a string format.
 * *PredictResponse* includes a map of outputs serialized by
[TensorProto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto) and information about the used model spec.

Read more about [Predict API usage](https://github.com/openvinotoolkit/model_server/blob/releases/2025/3/client/python/tensorflow-serving-api/samples/README.md#predict-api)

Also, using `string_val` field it is possible to send binary encoded images that would be preprocessed by OVMS using opencv and converted to OpenVINO-friendly format. For more information check [how binary data is handled in OpenVINO Model Server](./binary_input_tfs.md)

## See Also

- [Example client code](https://github.com/openvinotoolkit/model_server/blob/releases/2025/3/client/python/tensorflow-serving-api/samples/README.md) shows how to use GRPC API and REST API.
- [TensorFlow Serving](https://github.com/tensorflow/serving)
- [gRPC](https://grpc.io/)

