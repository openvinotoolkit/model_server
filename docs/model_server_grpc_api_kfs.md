# KServe compatible gRPC API {#ovms_docs_grpc_api_kfs}

## Introduction 
This document gives information about OpenVINO&trade; Model Server gRPC API compatible with KServe. It is documented in [KServe](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md) repository. 
Using the gRPC interface is recommended for optimal performance due to its faster implementation of input data deserialization. gRPC achieves lower latency, especially with larger input messages like images. 

The API includes following endpoints:
* <a href="#kfs-server-live">Server Live API </a>
* <a href="#kfs-server-ready">Server Ready API </a>
* <a href="#kfs-server-metadata">Server Metadata API </a>
* <a href="#kfs-model-ready">Model Ready API </a>
* <a href="#kfs-model-metadata">Model Metadata API </a>
* <a href="#kfs-model-infer"> Inference API </a>

> **NOTE**: Examples of using each of above endpoints can be found in [KServe samples](https://github.com/openvinotoolkit/model_server/tree/develop/client/python/kserve-api/samples/README.md).


## Server Live API <a name="kfs-server-live"></a>
Gets information about server liveness. Server is alive when communication channel can be established successfully.

Check KServe documentation for more [details](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#server-live-1).

## Server Ready API <a name="kfs-server-ready"></a>
Gets information about server readiness. Server is ready when initial configuration has been loaded. Server gets into ready state only once and remains in that state for the rest of its lifetime regardless the outcome of the initial loading phase. If some of the models have not been loaded successfully, server still becomes ready when the loading procedure finishes. 

Check KServe documentation for more [details](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#server-ready-1).

## Server Metadata API <a name="kfs-server-metadata"></a>
Gets information about the server itself. 

Check KServe documentation for more [details](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#server-metadata-1).

## Model Ready API <a name="kfs-model-ready"></a>
Gets information about readiness of the specific model. Model is ready when it's fully capable to run inference. 

Check KServe documentation for more [details](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#model-ready-1).

## Model Metadata API <a name="kfs-model-metadata"></a>
Gets information about the specific model.

Check KServe documentation for more [details](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#model-metadata-1).

## Inference API <a name="kfs-model-infer"></a>
Run inference with requested model or [DAG](./dag_scheduler.md).

Check KServe documentation for more [details](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference-1).

> **NOTE**: Inference supports putting tensor buffers either in `ModelInferRequest`'s [InferTensorContents](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/grpc_predict_v2.proto#L155) and [raw_input_contents](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/grpc_predict_v2.proto#L202). There is no support for BF16 data type and there is no support for using FP16 in `InferTensorContents`. In case of sending images files or strings BYTES data type should be used and data should be put in `InferTensorContents`'s `bytes_contents` or `raw_input_contents`.

Also, using `BYTES` datatype it is possible to send to model or pipeline, that have 4 (or 5 in case of [demultiplexing](demultiplexing.md)) shape dimensions, binary encoded images that would be preprocessed by OVMS using opencv and converted to OpenVINO-friendly format. For more information check [how binary data is handled in OpenVINO Model Server](./binary_input_kfs.md)

## See Also

- [Example client code](https://github.com/openvinotoolkit/model_server/tree/develop/client/python/kserve-api/samples/README.md) shows how to use GRPC API and REST API.
- [KServe API](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2)
- [gRPC](https://grpc.io/)

