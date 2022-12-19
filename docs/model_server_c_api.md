# OpenVINO Model Server C-API (preview feature) {#ovms_docs_c_api}

## Introduction

This document describes OpenVINO Model Server (OVMS) C API that allows OVMS to be linked into C/C++ applications. With exceptions listed below, all capabilities of OVMS are included in the shared library.

## Preview limitations
* Launching server in single model mode is not supported. You must use configuration file.
* DAG pipelines cannot be used directly through C API. DAG inferences can be scheduled only by gRPC/HTTP endpoints.
* There is no support for native file format (jpg/png) through C API.
* There is no server live, server ready, model ready, model metadata, metrics endpoints exposed through C API.
* Inference scheduled through C API are not accounted for in metrics.
* You cannot turn gRPC service off, HTTP service is off by default but can be enabled.
* There is no API for asynchronous inference.
* There is no support for stateful models.

## API Description

Server functionalities are encapsulated in shared library built from OVMS source. To include OVMS you need to link this library with your application and use C API defined in [header file](../src/ovms.h). To run OVMS you need to spawn process that will keep OVMS alive. Then you can schedule inference both directly from app using C API and gRPC/HTTP endpoints.

### Server configuration

To start OVMS you need to create `OVMS_Server` object using `OVMS_ServerNew`, with set of `OVMS_ServerSettings` and `OVMS_ModelsSettings` that describe how the server should be configured. Once the server is started using `OVMS_ServerStartFromConfigurationFile` you can schedule the inferences using `OVMS_Inference`. Example how to use OVMS with C/C++ application is [here](../demos/c_api_minimal_app/README.md).

### Error handling
Most of OVMS C API functions return `OVMS_Status` object pointer indicating the success or failure. Success is indicated by nullptr (NULL). Failure is indicated by returning `OVMS_Status` object. The status code can be extracted using `OVMS_StatusGetCode` function and the details of error can be retrieved using `OVMS_StatusGetDetails` function.

The ownership of `OVMS_Status` is passed to the caller of the function. You must delete the object using `OVMS_StatusDelete`.

### Inference

To execute inference using C API you must follow steps described below.

#### Prepare inference request
Create an inference request using `OVMS_InferenceRequestNew` specifying which servable name and optionally version to use. Then specify input tensors with `OVMS_InferenceRequestAddInput` and set the tensor data using `OVMS_InferenceRequestSetData`.

#### Invoke inference
Execute inference with OVMS using `OVMS_Inference` synchronous call. During inference execution you must not modify `OVMS_InferenceRequest` and bound memory buffers.

#### Process inference response
If the inference was successful, you receive `OVMS_InferenceRequest` object. After processing the response, you must free the response memory by calling `OVMS_InferenceResponseDelete`.

To process response, first you must check for inference error. If no error occurred, you must iterate over response outputs and parameters using `OVMS_InferenceResponseGetOutputCount` and `OVMS_InferenceResponseGetParameterCount`. Then you must extract details describing each output and parameter using `OVMS_InferenceResponseGetOutput` and `OVMS_InferenceResponseGetParameter`. Example how to use OVMS with C/C++ application is [here](../demos/c_api_minimal_app/README.md). While in example app you have only single thread scheduling inference request you can execute multiple inferences simultaneously using different threads.

**Note**: After inference execution is finished you can reuse the same `OVMS_InferenceRequest` by using `OVMS_InferenceRequestInputRemoveData` and then setting different tensor data with `OVMS_InferenceRequestSetData`.
