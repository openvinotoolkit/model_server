# Inference Server for Azure Media Services 

OpenVINO Inference Server for AMS is an AI Extension to Live Video Analytics on IoT Edge.
It enables easy delegation of inference operations to OpenVINO backend in media analytics pipelines.

The integration model is depicted below:
![archtecture](AI_extension.png)

OpenVINO Inference Server is running as a docker container and exposes a LVA REST API interface for the 
pipeline applications. This interface supports a range of model categories and return json response 
including model metadata like attribute, labels or classes names.

Beside LVA REST API, the Inference Server expose also the complete OpenVINO Model Server REST and gRPC API,
which could be used with arbitrary OpenVINO model. 

## LVA REST API

HTTP contract is defined as follows:
* OpenVINO Inference Server acts as the HTTP server
* LVA acts as the HTTP client


| POST        | http://hostname/<model_name> |
| ------------- |-------------|
| Accept      | application/json, */* |
| Authorization     | None |
| Content-Type | image/jpeg <br> image/png <br>  image/bmp |
|User-Agent|Azure Media Services|
|Body |Image bytes, binary encoded in one of the supported content types |

Example:
```bash
POST http://localhost:5000/vehicle-detection HTTP/1.1
Host: localhost:5000
x-ms-client-request-id: d6050cd4-c9f2-42d3-9adc-53ba7e440f17
Content-Type: image/bmp
Content-Length: 519222

(Image Binary Content)

```

Response:

|Response||
| ------------- |-------------|
| Status code | 200 OK - Inference results found <br>204 No Content - No content found by the AI <br> 400 Bad Request - Not expected <br> 500 Internal Server Error - Not expected <br> 503 Server Busy - AMS will back-off based on “Retry-After” header or based on a default amount of time in case header not preset.|
| Content-Type     | application/json|
| Content-Length | Body length, in bytes |
| Body | JSON object with single “inferences” property. |



## Supported models categories


### Object detection


### Classification models


## Launching and configuration of OpenVINO Inference Server


## Building docker image


## Testing