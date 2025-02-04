# KServe compatible gRPC API {#ovms_docs_grpc_api_kfs}

## Introduction
This document gives information about OpenVINO&trade; Model Server gRPC API compatible with KServe. It is documented in [KServe](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md) repository.
Using the gRPC interface is recommended for optimal performance due to its faster implementation of input data deserialization. gRPC achieves lower latency, especially with larger input messages like images.

The API includes following endpoints:
* [Server Live API](#server-live-api)
* [Server Ready API](#server-ready-api)
* [Server Metadata API](#server-metadata-api)
* [Model Ready API](#model-ready-api)
* [Model Metadata API](#model-metadata-api)
* [Inference API](#inference-api)
* [Streaming Inference API](#streaming-inference-api-extension)

> **NOTE**: Examples of using each of above endpoints can be found in [KServe samples](https://github.com/openvinotoolkit/model_server/tree/releases/2025/0/client/python/kserve-api/samples/README.md).


## Server Live API
Gets information about server liveness. Server is alive when communication channel can be established successfully.

Check KServe documentation for more [details](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#server-live-1).

## Server Ready API
Gets information about server readiness. Server is ready when initial configuration has been loaded. Server gets into ready state only once and remains in that state for the rest of its lifetime regardless the outcome of the initial loading phase. If some of the models have not been loaded successfully, server still becomes ready when the loading procedure finishes.

Check KServe documentation for more [details](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#server-ready-1).

## Server Metadata API
Gets information about the server itself.

Check KServe documentation for more [details](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#server-metadata-1).

## Model Ready API
Gets information about readiness of the specific model. Model is ready when it's fully capable to run inference.

Check KServe documentation for more [details](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#model-ready-1).

## Model Metadata API
Gets information about the specific model.

Check KServe documentation for more [details](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#model-metadata-1).

## Inference API
Run inference with requested model, [DAG](./dag_scheduler.md) or [MediaPipe Graph](./mediapipe.md).

Check KServe documentation for more [details](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference-1).

> **NOTE**: Inference supports putting tensor buffers either in `ModelInferRequest`'s [InferTensorContents](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/grpc_predict_v2.proto#L155) and [raw_input_contents](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/grpc_predict_v2.proto#L202). There is no support for BF16 data type and there is no support for using FP16 in `InferTensorContents`. In case of sending images files or strings BYTES data type should be used and data should be put in `InferTensorContents`'s `bytes_contents` or `raw_input_contents`.

Also, using `BYTES` datatype it is possible to send to model or pipeline, that have 4 (or 5 in case of [demultiplexing](demultiplexing.md)) shape dimensions, binary encoded images that would be preprocessed by OVMS using opencv and converted to OpenVINO-friendly format. For more information check [how binary data is handled in OpenVINO Model Server](./binary_input_kfs.md)

## Streaming Inference API (extension)
Run streaming inference with [MediaPipe Graph](./mediapipe.md).

Check documentation for more [details](./streaming_endpoints.md).

## See Also

- [Example client code](https://github.com/openvinotoolkit/model_server/tree/releases/2025/0/client/python/kserve-api/samples/README.md) shows how to use GRPC API and REST API.
- [KServe API](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2)
- [gRPC](https://grpc.io/)

