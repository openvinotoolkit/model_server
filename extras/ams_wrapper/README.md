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

*Note:* Depending on the model configuration, input image resolution needs to match the model expected size or
it will be resized automatically. 

Response:

|Response||
| ------------- |-------------|
| Status code | 200 OK - Inference results found <br>204 No Content - No content found by the AI <br> 400 Bad Request - Not expected <br> 500 Internal Server Error - Not expected <br> 503 Server Busy - AMS will back-off based on “Retry-After” header or based on a default amount of time in case header not preset.|
| Content-Type     | application/json|
| Content-Length | Body length, in bytes |
| Body | JSON object with single “inferences” property. |



## Supported models categories

Models configured in Inference Server need to belong to one of defined categories. The category
defines what kind of data is in the model response and what is its format. Read the categories
characteristics below to find out about the requirements. Each model needs to have associated
config file, which describe included, classes, attributes and labels. The config can also specify
input data like resolution or pre-processing parameters.


### Object detection

In this category, models should return response in the shape `[1, 1, N, 7]` where N is the number of detected bounding boxes.
For each detection, the description has the 
format: [image_id, label, conf, x_min, y_min, x_max, y_max], where:
- image_id - ID of the image in the batch
- label - predicted class ID
- conf - confidence for the predicted class
- (x_min, y_min) - coordinates of the top left bounding box corner
- (x_max, y_max) - coordinates of the bottom right bounding box corner.

There are many models, which meets this criteria, in the [OpenVINO Model Zoo](https://docs.openvinotoolkit.org/2019_R1/_docs_Pre_Trained_Models.html).
2 exemplary modes are include in the Inference Server docker image:
* vehicle-detection - [vehicle-detection-adas-binary-0001](https://github.com/opencv/open_model_zoo/tree/master/models/intel/vehicle-detection-adas-binary-0001)
* face-detection - [face-detection-adas-0001](https://github.com/opencv/open_model_zoo/tree/master/models/intel/face-detection-adas-binary-0001)

Each model should include also a configuration file in json format. Example of such
configuration file is [here](ams_models/vehicle_detection_adas_model.json)

### Classification models

Classification models, in this category, give the results in softmax layer. This include a set
of probabilities in classes defined for the model.
Each output of the model should have the shape `[1, C , ...]`. First dimension represent the batch size,
which should be set to 1. `C` represent all classes defined in the model. Remaining dimensions 
are ignored (if present, first index is used).

Examples of such models are available in the [OpenVINO Model Zoo](https://docs.openvinotoolkit.org/2019_R1/_docs_Pre_Trained_Models.html).
2 of such models are copied into the Inference Server docker image:
* vehicle-attributes-recognition - [vehicle-attributes-recognition-barrier-0039](https://github.com/opencv/open_model_zoo/tree/master/models/intel/vehicle-attributes-recognition-barrier-0039)
* emotions-recognition - [emotions-recognition-retail-0003](https://github.com/opencv/open_model_zoo/tree/master/models/intel/emotions-recognition-retail-0003) 

## Launching and configuration of OpenVINO Inference Server

OpenVINO Inference Server for AMS includes two components which require proper configuration.
* OpenVINO Model Server - serves all models and executes inference operation
* LVA REST API wrapper - translates LVA API, run pre and post processing operations, communicates with OVMS via localhost and gRPC interface.

OpenVINO Model server requires file `/opt/ams_models/ovms_config.json` which is by default configured
to use [4 exemplary models](../ams_models/ovms_config.json).

LVA REST API wrapper requires model configuration files describing all enabled models.
Models config files should be present in `/opt/ams_models/` folder and their name should have the name 
matched with the model name configured in `ovms_config.json`.

Model files in OpenVINO Intermediate Representation format should be stored in the folders structure
like defined on [OVMS documentation](../../docs/docker_container.md#preparing-the-models).
They can be mounted locally or hosted on supported cloud storage.

Configuration files for OVMS and LVA REST API wrapper can be mounted into to container
from local filesystem or as Kubernetes configmap records.

By default, the Inference Server is started, serving exemplary models, using predefined configuration.


## Performance tuning


## Building docker image


## Testing