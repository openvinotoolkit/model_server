# OpenVINO&trade; Model Server gRPC API Documentation

## Introduction 
This documents gives information about OpenVINO&trade; Model Server gRPC API. It is documented in the proto buffer files in [tensorflow_serving_api](https://github.com/tensorflow/serving/tree/r1.14/tensorflow_serving/apis). 
Using the gRPC interface is recommended for optimal performance due to its faster implementation of input data deserialization. gRPC achieves lower latency, especially with larger input messages like images. 

This document covers following API:
* <a href="#model-status">Model Status API</a>
* <a href="#model-metadata">Model MetaData API </a>
* <a href="#predict">Predict API </a>


> **Note:** The implementations for *Predict*, *GetModelMetadata* and *GetModelStatus* function calls are currently available. 
These are the most generic function calls and should address most of the usage scenarios.



## Model Status API <a name="model-status"></a>

- Description

Gets information about the status of served models including Model Version

 [Get Model Status proto](https://github.com/tensorflow/serving/blob/r1.14/tensorflow_serving/apis/get_model_status.proto) defines three message definitions used while calling Status endpoint: *GetModelStatusRequest*, *ModelVersionStatus*, *GetModelStatusResponse* that are used to report all exposed versions including their state in their lifecycle.

 Read more about *Get Model Status API* usage [here](./example_client.md#model-status-api)       

## Model MetaData API <a name="model-metadata"></a>

- Description

Gets information about the served models. A function call GetModelMetadata accepts model spec information as input and returns Signature Definition content in the format similar to TensorFlow Serving.
 
[Get Model Metadata proto](https://github.com/tensorflow/serving/blob/r1.14/tensorflow_serving/apis/get_model_metadata.proto) has three message definitions: *SignatureDefMap*, *GetModelMetadataRequest*, *GetModelMetadataResponse*. 

Read more about *Get Model Metadata API* usage [here](./example_client.md#model-metadata-api)       


## Predict API <a name="predict"></a>

- Description

Sends requests via TFS gRPC API using images in numpy format. It displays performance statistics and optionally the model accuracy.

 [Predict proto](https://github.com/tensorflow/serving/blob/r1.14/tensorflow_serving/apis/predict.proto) has two message definitions: *PredictRequest* and  *PredictResponse*.  
 * *PredictRequest* specifies information about the model spec, a map of input data serialized via 
[TensorProto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto) to a string format.
 * *PredictResponse* includes a map of outputs serialized by 
[TensorProto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto) and information about the used model spec.

There are two ways in which gRPC request can be submitted for Predict API:
1. Submitting gRPC requests based on a dataset from numpy files
2. Submitting gRPC requests based on a dataset from a list of jpeg files

Read more about *Predict API* usage [here](./example_client.md#predict-api)       

## See Also

- [Example client code](https://github.com/openvinotoolkit/model_server/tree/main/example_client) shows how to use these API and submit the requests using the gRPC interface.
- [TensorFlow Serving](https://github.com/tensorflow/serving)
- [gRPC](https://grpc.io/)




 




