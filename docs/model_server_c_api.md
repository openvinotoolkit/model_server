# OpenVINO Model Server C-API {#ovms_docs_c_api}

## Introduction

This document describes OpenVINO Model Server (OVMS) C API that allows OpenVINO Model Server to be linked into C/C++ applications. With exceptions listed at the end of this document, all capabilities of OpenVINO Model Server are included in the shared library.

With OpenVINO Model Server 2023.1 release C-API is no longer in preview state and is now public. This version contains few breaking changes. Following function names changed - `*Get*` was removed from their name:
- `OVMS_StatusGetCode`
- `OVMS_StatusGetDetails`
- `OVMS_InferenceResponseGetOutputCount`
- `OVMS_InferenceResponseGetOutput`
- `OVMS_InferenceResponseGetParameterCount`
- `OVMS_InferenceResponseGetParameter`
- `OVMS_ServableMetadataGetInputCount`
- `OVMS_ServableMetadataGetOutputCount`
- `OVMS_ServableMetadataGetInput`
- `OVMS_ServableMetadataGetOutput`
- `OVMS_ServableMetadataGetInfo`

## API Description

Server functionalities are encapsulated in shared library built from OpenVINO Model Server source. To include OpenVINO Model Server you need to link this library with your application and use C API defined in [header file](https://github.com/openvinotoolkit/model_server/blob/releases/2025/1/src/ovms.h).


Calling a method to start the model serving in your application initiates the OpenVINO Model Server as a separate thread. Then you can schedule inference both directly from app using C API and gRPC/HTTP endpoints.

API is versioned according to [SemVer 2.0](https://semver.org/). Calling `OVMS_ApiVersion` it is possible to get `major` and `minor` version number.
- major - incremented when new, backward incompatible changes are introduced to the API itself (API call removal, name change, parameter change)
- minor - incremented when API is modified but backward compatible (new API call added)

There is no patch version number. Underlying functionality changes not related to API itself are tracked via OpenVINO Model Server version. OpenVINO Model Server and OpenVINO versions can be tracked via logs or `ServerMetadata` request (via KServe API).

### Server configuration and start

To start OpenVINO Model Server you need to create `OVMS_Server` object using `OVMS_ServerNew`, with set of `OVMS_ServerSettings` and `OVMS_ModelsSettings` that describe how the server should be configured. Once the server is started using `OVMS_ServerStartFromConfigurationFile` you can schedule the inferences using `OVMS_Inference`. To stop server, you must call `OVMS_ServerDelete`. While the server is alive you can schedule both in process inferences as well as use gRPC API to schedule inferences from remote machine. Optionally you can also enable HTTP service. One can also query metadata using `OVMS_ServerMetadata`. Example how to use OpenVINO Model Server with C/C++ application is [here](../demos/c_api_minimal_app/README.md).

### Error handling
Most of OpenVINO Model Server C API functions return `OVMS_Status` object pointer indicating the success or failure. Success is indicated by nullptr (NULL). Failure is indicated by returning `OVMS_Status` object. The status code can be extracted using `OVMS_StatusCode` function and the details of error can be retrieved using `OVMS_StatusDetails` function.

The ownership of `OVMS_Status` is passed to the caller of the function. You must delete the object using `OVMS_StatusDelete`.

### Inference

To execute inference using C API you must follow steps described below.

#### Prepare inference request
Create an inference request using `OVMS_InferenceRequestNew` specifying which servable name and optionally version to use. Then specify input tensors with `OVMS_InferenceRequestAddInput` and set the tensor data using `OVMS_InferenceRequestInputSetData`. Optionally you can also set one or all outputs with `OVMS_InferenceRequestAddOutput` and `OVMS_InferenceRequestOutputSetData`. For asynchronous inference you also have to set callback with `OVMS_InferenceRequestSetCompletionCallback`.

#### Using OpenVINO Remote Tensor
With OpenVINO Model Server C-API you could also leverage the OpenVINO remote tensors support. Check original documentation [here](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device/remote-tensor-api-gpu-plugin.html). In order to use OpenCL buffers you need to first create `cl::Buffer` and then use its pointer in setting input with `OVMS_InferenceRequestInputSetData` or output with `OVMS_InferenceRequestOutputSetData` and buffer type `OVMS_BUFFERTYPE_OPENCL`. In case of VA surfaces you need to create appropriate VA surfaces and then use the same calls with buffer type `OVMS_BUFFERTYPE_VASURFACE_Y` and `OVMS_BUFFERTYPE_VASURFACE_UV`.

#### Invoke inference
Execute inference with OpenVINO Model Server using `OVMS_Inference` synchronous call. During inference execution you must not modify `OVMS_InferenceRequest` and bound memory buffers.

#### Process inference response
If the inference was successful, you receive `OVMS_InferenceRequest` object. After processing the response, you must free the response memory by calling `OVMS_InferenceResponseDelete`.

To process response, first you must check for inference error. If no error occurred, you must iterate over response outputs and parameters using `OVMS_InferenceResponseOutputCount` and `OVMS_InferenceResponseParameterCount`. Then you must extract details describing each output and parameter using `OVMS_InferenceResponseOutput` and `OVMS_InferenceResponseParameter`. Example how to use OpenVINO Model Server with C/C++ application is [here](../demos/c_api_minimal_app/README.md). While in example app you have only single thread scheduling inference request you can execute multiple inferences simultaneously using different threads.

**Note**: After inference execution is finished you can reuse the same `OVMS_InferenceRequest` by using `OVMS_InferenceRequestInputRemoveData`, and then setting different tensor data with `OVMS_InferenceRequestSetData`.

#### Server liveness and readiness
To check if OpenVINO Model Server is alive and will respond to requests you can use `OVMS_ServerLive`. Note that live status doesn't guarantee the model readiness. Check the readiness with `OVMS_ServerReady' call to show if initial configuration loading has finished including loading all correctly configured models.

#### Servable readiness
To check if servable is ready for inference and metadata requests use `OVMS_GetServableState` specifying name and optionally version.

#### Servable metadata
Execute `OVMS_GetServableMetadata` call to get information about servable inputs, outputs. If the request was successful you receive `OVMS_ServableMetadata` object. To get information about every input/output you must use first check for number of inputs/outputs with `OVMS_ServableMetadataInputCount`/`OVMS_ServableMetadataOutputCount`, and then use `OVMS_ServableMetadataInput` and `OVMS_ServableMetadataOuput` calls to extract details about each input/output. After retrieving required data you must release response object with `OVMS_ServableMetadataDelete`.

#### Server Metadata
To check server metadata use `OVMS_ServerMetadata` call. It will create new object of type `OVMS_Metadata` that you need to later release with `OVMS_StringFree`. It will contain information about version of OpenVINO and version of Model Server. To serialize `OVMS_ServerMetadata` to string JSON you can use `OVMS_SerializeMetadataToString` function. This allocates char table that needs to be released later as well with `OVMS_StringFree`.

## Limitations
* Launching server in single model mode is not supported. You must use configuration file.
* There is no direct support for jpeg/png encoded input format through C API.
* There is no metrics endpoint exposed through C API.
* Inference scheduled through C API does not have metrics `ovms_requests_success`,`ovms_requests_fail` and `ovms_request_time_us` counted.
* You cannot turn gRPC endpoint off, REST API endpoint is optional.
* There is no support for stateful models.
* There is no support for mediapipe graphs.
* Currently this interface is not enabled on Windows server version
