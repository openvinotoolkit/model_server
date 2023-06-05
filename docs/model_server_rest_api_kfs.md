# KServe compatible RESTful API {#ovms_docs_rest_api_kfs}

## Introduction
In addition with [gRPC APIs](./model_server_grpc_api_kfs.md) OpenVINO&trade; model server also supports RESTful APIs which follows the documentation from [KServe REST API](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#httprest). REST API is recommended when the primary goal is in reducing the number of client-side python dependencies and simpler application code.

This document covers the following API:
* <a href="#kfs-server-live">Server Live API </a>
* <a href="#kfs-server-ready">Server Ready API </a>
* <a href="#kfs-server-metadata">Server Metadata API </a>
* <a href="#kfs-model-ready">Model Ready API </a>
* <a href="#kfs-model-metadata">Model Metadata API </a>
* <a href="#kfs-model-infer"> Inference API </a>

## Server Live API <a name="kfs-server-live"></a>
**Description**

Get information about server liveness.

**URL**

```
GET http://${REST_URL}:${REST_PORT}/v2/health/live
```

**Response format**

The information about server liveness is provided in the response status code. If server is alive, status code is 200. Otherwise it's 4xx. Response does not have any content in the body.

**Usage Example**
```
$ curl -i http://localhost:5000/v2/health/live

HTTP/1.1 200 OK
Content-Type: application/json
Date: Tue, 09 Aug 2022 09:20:24 GMT
Content-Length: 2
```

See also [code samples](https://github.com/openvinotoolkit/model_server/tree/develop/client/python/kserve-api/samples) for getting server liveness with KServe API on HTTP Server Live endpoint.

## Server Ready API <a name="kfs-server-ready"></a>
**Description**

Get information about server readiness.

**URL**

```
GET http://${REST_URL}:${REST_PORT}/v2/health/ready
```

**Response format**

The information about server readiness is provided in the response status code. If server is ready, status code is 200. Otherwise it's 4xx. Response does not have any content in the body.

**Usage Example**
```
$ curl -i http://localhost:5000/v2/health/ready

HTTP/1.1 200 OK
Content-Type: application/json
Date: Tue, 09 Aug 2022 09:22:14 GMT
Content-Length: 2
```

See also [code samples](https://github.com/openvinotoolkit/model_server/tree/develop/client/python/kserve-api/samples) for getting server readiness with KServe API on HTTP Server Ready endpoint.

## Server Metadata API <a name="kfs-server-metadata"></a>
**Description**

Get information about the server.

**URL**

```
GET http://${REST_URL}:${REST_PORT}/v2
```

**Response format**

If successful:

```JSON
{
  "name" : $string,
  "version" : $string,
  "extensions" : [ $string, ... ]
}
```

Else:
```JSON
{
  "error": $string
}
```

**Usage Example**
```
$ curl http://localhost:5000/v2
{"name":"OpenVINO Model Server","version":"2022.2.0.fd742507"}
```

For detailed description of the response contents see [KServe API docs](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#server-metadata).

See also [code samples](https://github.com/openvinotoolkit/model_server/tree/develop/client/python/kserve-api/samples) for getting server metadata with KServe API on HTTP Server Metadata endpoint.

## Model Ready API <a name="kfs-model-ready"></a>
**Description**

Get information about model readiness.

**URL**

```
GET http://${REST_URL}:${REST_PORT}/v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/ready
```

**Response format**

The information about model readiness is provided in the response status code. If model is ready for inference, status code is 200. Otherwise it's 4xx. Response does not have any content in the body.

**Usage Example**
```
$ curl -i http://localhost:5000/v2/models/resnet/ready

HTTP/1.1 200 OK
Content-Type: application/json
Date: Tue, 09 Aug 2022 09:25:31 GMT
Content-Length: 2
```

See also [code samples](https://github.com/openvinotoolkit/model_server/tree/develop/client/python/kserve-api/samples) for getting model readiness with KServe API on HTTP Model Ready endpoint.



## Model Metadata API <a name="model-metadata"></a>
**Description**

Get information about the model.

**URL**
```
GET http://${REST_URL}:${REST_PORT}/v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]
```

> **Note** :Including ${MODEL_VERSION} is optional. If omitted the model metadata for the latest version is returned in the response. ???

**Response format**

If successful:
```JSON
{
  "name" : $string,
  "versions" : [ $string, ... ] #optional,
  "platform" : $string,
  "inputs" : [ $metadata_tensor, ... ],
  "outputs" : [ $metadata_tensor, ... ]
}
```

where:
```JSON
$metadata_tensor =
{
  "name" : $string,
  "datatype" : $string,
  "shape" : [ $number, ... ]
}
```

Else:
```JSON
{
  "error": $string
}
```


**Usage example**
```
$ curl http://localhost:8000/v2/models/resnet
{"name":"resnet","versions":["1"],"platform":"OpenVINO","inputs":[{"name":"0","datatype":"FP32","shape":[1,224,224,3]}],"outputs":[{"name":"1463","datatype":"FP32","shape":[1,1000]}]}
```

For detailed description of the response contents see [KServe API docs](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#model-metadata).

See also [code samples](https://github.com/openvinotoolkit/model_server/tree/develop/client/python/kserve-api/samples) for running getting model metadata with KServe API on HTTP Model Metadata endpoint.

## Inference API <a name="kfs-model-infer"></a>
**Description**

Endpoint for running an inference with loaded models or [DAGs](./dag_scheduler.md).

**URL**
```
POST http://${REST_URL}:${REST_PORT}/v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/infer
```

**Request Body Format**
```JSON
{
  "id" : $string #optional,
  "parameters" : $parameters #optional,
  "inputs" : [ $request_input, ... ],
  "outputs" : [ $request_output, ... ] #optional
}
``` 

where:
```JSON
$request_input =
{
  "name" : $string,
  "shape" : [ $number, ... ],
  "datatype"  : $string,
  "parameters" : $parameters #optional,
  "data" : $tensor_data
}

$request_output =
{
  "name" : $string,
  "parameters" : $parameters #optional,
}
```

> Note: In `tensor_data` elements may be presented in their multi-dimensional representation, or as a flattened one-dimensional representation. Before inference execution tensor data is flattened, and only elements count in `tensor_data` is validated.

Besides numerical values, it is possible to pass encoded images using Binary Data extension:

As a JPEG / PNG encoded images - in this case binary encoded data is loaded by OVMS using OpenCV which then converts it to OpenVINO-friendly data format for inference. Input is treated as encoded image when datatype is `BYTES` and model or pipeline have 4 (or 5 in case of [demultiplexing](demultiplexing.md)) shape dimensions. Every batch the BYTES input needs to be preceded by 4 bytes, litte endian, that contains its size. 

```JSON
Content-Type: application/octet-stream
Inference-Header-Content-Length: <xx>
Content-Length: <xx+9472>
{
"model_name" : "my_model",
"inputs" : [
   {
      "name" : "model_input",
      "shape" : [ 1 ],
      "datatype" : "BYTES"
   }
]
}
<0x00250000 (9472 as four bytes little endian)><9472 bytes of data for model_input tensor>
```


As a raw data - it means it wont be preprocessed by OVMS. To send raw data using Binary Data extension use other data types than `BYTES`.

```JSON
Content-Type: application/octet-stream
Inference-Header-Content-Length: <xx>
Content-Length: <xx+(3 x 1080000)>
{
"model_name" : "my_model",
"inputs" : [
   {
      "name" : "model_input",
      "shape" : [ 3, 300, 300, 3 ],
      "datatype" : "FP32"
   },

]
}
<3240000 bytes of the whole data batch for model_input tensor>
```

*sending strings inside binary extension also require preceding every batch by 4 bytes, litte endian, that contains its size.

Check [how binary data is handled in OpenVINO Model Server](./binary_input.md) for more informations.


**Response Format**

If successful:

```JSON
{
  "model_name" : $string,
  "model_version" : $string #optional,
  "id" : $string,
  "parameters" : $parameters #optional,
  "outputs" : [ $response_output, ... ]
}
```

where:
```JSON
$response_output =
{
  "name" : $string,
  "shape" : [ $number, ... ],
  "datatype"  : $string,
  "parameters" : $parameters #optional,
  "data" : $tensor_data
}
```

Else:

```JSON
{
  "error": <error message string>
}
```

Outputs of response can be send in binary format using Binary Data extension. To force a output to be sent in binary format you need to use "binary_data" : true parameter in request JSON. For example:
```JSON
{
  "model_name" : "mymodel",
  "inputs" : [...],
  "outputs" : [
    {
      "name" : "output0",
      "parameters" : {
        "binary_data" : true
      }
    }
  ]
}
```

Assuming the output datatype is FP32 and shape is [ 2, 2 ] response to this request would be:

```JSON
HTTP/1.1 200 OK
Content-Type: application/octet-stream
Inference-Header-Content-Length: <yy>
Content-Length: <yy+16>
{
  "outputs" : [
    {
      "name" : "output0",
      "shape" : [ 2, 2 ],
      "datatype"  : "FP32",
      "parameters" : {
        "binary_data_size" : 16
      }
    }
  ]
}
<16 bytes of data for output0 tensor>
```

For detailed description of request and response contents see [KServe API docs](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference).

> Note: More efficient way of running inference via REST is sending data in a binary format outside of the JSON object, by using [binary data extension](./binary_input_kfs.md). 

See also [code samples](https://github.com/openvinotoolkit/model_server/tree/develop/client/python/kserve-api/samples) for running inference with KServe API on HTTP Inference endpoint.
