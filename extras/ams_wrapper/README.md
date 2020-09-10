# Inference Server for Azure Media Services 

OpenVINO™ Model Server is an AI Extension used on IoT Edge devices together with [Live Video Analytics (LVA)](http://aka.ms/lva). 
It enables easy delegation of inference requests to OpenVINO in media analytics pipelines. 

A pre-built image of this server is available via the Azure Marketplace [here](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/intel_corporation.ovms?tab=Overview).

The integration model is depicted below:
![archtecture](AI_extension.png)

OpenVINO Model Server is running as a docker container and exposes an REST API interface for 
the pipeline applications that conforms to the HTTP extension [contract](https://docs.microsoft.com/en-us/azure/media-services/live-video-analytics-edge/analyze-with-ai-your-choice-how-to#media-graph-http-extension-contract-definitions) specified for LVA. This interface supports a range of model categories and returns JSON response 
including model metadata like attribute, labels or classes names. 

Beside the REST API, the server also exposes the complete OpenVINO Model Server REST and gRPC API,
 which could be used with an arbitrary OpenVINO model. 

## REST API 

The HTTP contract is defined as follows:
* OpenVINO Model Server acts as the HTTP server 
* LVA acts as the HTTP client


| POST        | http://hostname:port/<endpoint_name> |
| ------------- |-------------|
| Accept      | application/json, */* |
| Authorization     | None |
| Content-Type | image/jpeg <br> image/png <br>  image/bmp |
|User-Agent|Azure Media Services|
|Body |Image bytes, binary encoded in one of the supported content types |

The `endpoint_name` defines the model you want to apply to the image. The allowed values are:
* vehicleDetection
* vehicleClassification
* personVehicleBikeDetection
* faceDetection

See below for more details about these models.

Example:

```bash
POST http://localhost:5000/vehicleDetection HTTP/1.1
Host: localhost:5000
x-ms-client-request-id: d6050cd4-c9f2-42d3-9adc-53ba7e440f17
Content-Type: image/bmp
Content-Length: 519222

(Image Binary Content)

```

*Note:* Depending on the model configuration, input image resolution needs to match the model's expected size or
it will be resized automatically. 

Response:

|Response||
| ------------- |-------------|
| Status code | 200 OK - Inference results found <br>204 No Content - No content found by the AI <br> 400 Bad Request - Not expected <br> 500 Internal Server Error - Not expected <br> 503 Server Busy - AMS will back-off based on “Retry-After” header or based on a default amount of time in case header not preset.|
| Content-Type     | application/json|
| Content-Length | Body length, in bytes |
| Body | JSON object with single “inferences” property. |



## Supported models categories

Models configured in OpenVINO Model Server for LVA need to belong to one of defined categories. 
The category defines what kind of data is in the model response and in what format. 
See the category characteristics below to learn more about their requirements. 
Each model needs to have an associated config file, which describes the included classes, 
attributes and labels. The config can also specify input data – like resolution or pre-processing parameters. 


### Object detection

In this category, models should return response in the shape `[1, 1, N, 7]` where N is the number of detected bounding boxes.
For each detection, the description has the 
format: [image_id, label, conf, x_min, y_min, x_max, y_max], where:
- image_id - ID of the image in the batch
- label - predicted class ID
- conf - confidence for the predicted class
- (x_min, y_min) - coordinates of the top left bounding box corner
- (x_max, y_max) - coordinates of the bottom right bounding box corner.

There are several Object Detection models available in the [OpenVINO Model Zoo](https://docs.openvinotoolkit.org/2020.2/_models_intel_index.html). 
The Inference Server Docker image comes pre-built with the following Object Detection models: 

* vehicle-detection - [vehicle-detection-adas-binary-0001](https://github.com/opencv/open_model_zoo/tree/master/models/intel/vehicle-detection-adas-binary-0001)
* face-detection - [face-detection-adas-0001](https://github.com/opencv/open_model_zoo/tree/master/models/intel/face-detection-adas-binary-0001)
* person-vehicle-bike-detection-crossroad - [person-vehicle-bike-detection-crossroad-0078](http://•%09https:/github.com/opencv/open_model_zoo/tree/master/models/intel/person-vehicle-bike-detection-crossroad-0078)

Each model should also include a configuration file in json format. Example of such
configuration file is [here](https://github.com/openvinotoolkit/model_server/blob/master/extras/ams_models/vehicle_detection_adas_model.json)

Below is a sample of the model output:
```json
{
    "inferences": [
      {
        "type": "entity",
        "subtype": "vehicleDetection",
        "entity": {
          "tag": {
            "value": "vehicle",
            "confidence": 0.15077224373817444
          },
          "box": {
            "l": 0,
            "t": 0.25023195147514343,
            "w": 0.853949785232544,
            "h": 0.7095241844654083
          }
        }
      }
    ]
 }
```

### Classification models

Classification models, in this category, give the results in softmax layer. This includes a set
of probabilities in classes defined for the model.
Each output of the model should have the shape `[1, C , ...]`. First dimension represents the batch size,
which should be set to 1. `C` represents all classes defined in the model. Remaining dimensions 
are ignored (if present, first index is used).

Examples of such models are available in the [OpenVINO Model Zoo](https://docs.openvinotoolkit.org/2020.2/_models_intel_index.html).
The following classification model is included in the OVMS docker image for LVA: 
* vehicle-attributes-recognition - [vehicle-attributes-recognition-barrier-0039](https://github.com/opencv/open_model_zoo/tree/master/models/intel/vehicle-attributes-recognition-barrier-0039)

Model includes a configuration file in json format. See an example configuration file [here](../ams_models/vehicle_attributes_model.json)

Below is a sample of such model:
```json
{
    "inferences": [
      {
        "type": "classification",
        "subtype": "color",
        "classification": {
          "tag": {
            "value": "gray",
            "confidence": 0.6284224987030029
          }
        }
      },
      {
        "type": "classification",
        "subtype": "type",
        "classification": {
          "tag": {
            "value": "truck",
            "confidence": 0.9994522929191589
          }
        }
      }
    ]
  }
```

## Deployment and configuration of OpenVINO Inference Server for LVA    

OpenVINO Inference Server for LVA includes two components which require proper configuration.
* OpenVINO Model Server - serves all models and executes inference operation
* LVA REST API wrapper - translates LVA API, run pre and post processing operations, communicates with OVMS via localhost and gRPC interface.

OpenVINO Model server requires file `/opt/ams_models/ovms_config.json` which is by default configured
to use [4 exemplary models](../ams_models/ovms_config.json).

LVA REST API wrapper requires model configuration files describing all enabled models.
Models config files should be present in `/opt/ams_models/` folder. Model configuration files should have _model.json suffix.
They include the mapping between ovms models from `ovms_config.json` and LVA REST API endpoint name.

Model files in OpenVINO Intermediate Representation format should be stored in the folders structure
like defined on [OVMS documentation](../../docs/docker_container.md#preparing-the-models).

Note: OVMS for LVA was tested for the models included in the docker image. 
It should be possible to swap the models if they have correspondent format and content of the inputs and outputs. 
It is, however, not officially supported. 

## Target device selection for models: 

By default, the OpenVINO Model Server for LVA is started, serving included models, using CPU as the target device executing the inference requests. 
Besides CPU, OVMS for LVA is currently supporting 
[Myriad Plugin](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_MYRIAD.html), 
[GPU Plugin](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_CL_DNN.html)
and [Heterogeneous Plugin](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_HETERO.html). 

It is possible to set the target device for each individual model by setting additional environment variable 
OVMS_MODEL_DEVICES, or passing –ovms_model_devices parameter to OVMS for LVA, with a semicolon separated list of models with assigned target devices. 
Below is an example: 
OVMS_MODEL_DEVICES='vehicle_detection_adas=CPU;vehicle_attributes=MYRIAD' 

Note: Depending on the device, additional docker parameters might be required to mount required resources in the container.
There should be also installed appropriate drivers on the host operating system.
Refer to the plugin documentation and examples from [OVMS documentation](../../docs/docker_container.md#starting-docker-container-with-ncs).

## Starting docker container

# OpenVINO Inference Server for LVA parameters
 

OpenVINO Model Server for LVA is started vi a command `/ams_wrapper/start_ams.py`. It accepts the following parameters:

```

  --ams_port AMS_PORT   Port for AMS Service to listen on (default: 5000)

  --ovms_port OVMS_PORT

                        Port for OVMS to listen on (default: 9000)

  --workers WORKERS     AMS service workers (default: 20)

  --grpc_workers GRPC_WORKERS

                        OVMS service workers (default: 10)

  --ovms_model_devices OVMS_MODEL_DEVICES

                        Colon delimited list of model devices, in following

                        format: '<model_1_name>=<device_name>;<model_2_name>=<

                        device_name>'

```
Docker container can be initiated via a command:
```bash
docker run <image_name> /ams_wrapper/start_ams.py
```

## Performance tuning
OVMS component is configured for most generic manner to work well in low and high spec resources. 
In case there is a need to adjust the performance parameters, it can be done by replacing ovms_config.json 
file in a custom docker image or mounting it from a local storage. In that case follow the recommendation 
for [OVMS](../../docs/performance_tuning.md) and [OpenVINO plugins](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_Supported_Devices.html).  

## Building docker image

The image in Azure Marketplance is built using the make target:

`make docker_build_ams_clearlinux` – clearlinux base image with Opensource version of OpenVINO 


It is also possible to build the image on centos base using the command below. It requires the OpenVINO toolkit binary
package url which can be received after user [registration](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download/linux.html)

`make docker_build_ams DLDT_PACKAGE_URL= <url to binary OV package>` - centos base image with a binary OV package 


## Testing

Run all functional tests 

`make ams_test` 


Test response from the deployed component:
 

`curl -X POST -H "Content-Type: image/jpg" --data-binary @test_image.jpg localhost:5000/vehicleDetection | jq `


## Models configuration

Model configuration json file has the following fields:

```
endpoint: name for the endpoint under which model will be exposed [REQUIRED]
model_type: type of the model, currently supported types are: detection, classification_attributes
 [REQUIRED]
inputs: list of model's input configurations, refer to input configuration schema below [REQUIRED]
outputs: list of model's outputs configurations, refer to output configuration schema below [REQUIRED]
```

Model's input configuration schema:

```
input_name: name of the input [REQUIRED]
channels: number of channels in input image, default: 3 [OPTIONAL]
target_height: image's height expected by model, images with different height will be resized to match target height [OPTIONAL]
target_width: image's width expected by model, images with different width will be resized to match target width [OPTIONAL]
color_format: color format expected by model, supported formats: RGB, BGR, default: BGR [OPTIONAL]
scale: scaling factor for image pixel values, each pixel value will be multiplied by this value if specified [OPTIONAL]
standardization: If set to true, images will be linearly scaled to have mean 0 and variance 1, valid values: true/false, default: false [OPTIONAL]
input_format: image input format, supported formats: NHWC, NCHW, default: NCHW [OPTIONAL]
```
Model's output configuration schema:
```
output_name: name of the output [REQUIRED]
value_index_mapping: dictionary containing description of fields in model's inference output [OPTIONAL]
classes: dictionary containing mapping of classes to values returned by the model [OPTIONAL]
confidence_threshold: if set, only inference results with confidence higher or equal to confidence_threshold will be returned [OPTIONAL]
top_k_results: if set, only top k predicitons, sorted by confidence, will be returned (where k is the value of this parameter) [OPTIONAL]
is_softmax: defines if this output should be considered as Softmax, valid values: true/false, default: true [OPTIONAL]
value_multiplier:must be set if is_softmax is set to False, inference results for this specific output will be multiplied by this value [OPTIONAL]
for example if age value is returned in the inference output (0.2456777) we need to multiply it by 100 to get the human readable age format. 
is_softmax should be then set to False and value_multiplier must be set to 100.

```

Model configuration example:

```json
{
    "endpoint": "vehicleDetection",
    "model_type": "detection",
    "inputs": [
        {
          "input_name": "data",
          "input_format": "NCHW",
          "color_format": "BGR",
          "target_height": 384,
          "target_width": 672,
          "channels": 3,
          "standardization": false
        }
      ],
    "outputs": [
        {
            "output_name": "detection_out",
            "classes": {
                "background": 0.0,
                "vehicle": 1.0
            },
            "value_index_mapping": {
                "image_id": 0,
                "value": 1,
                "confidence": 2,
                "x_min": 3,
                "y_min": 4,
                "x_max": 5,
                "y_max": 6
            }
        }
    ],
    "ovms_mapping": {
        "model_name": "vehicle_detection_adas",
        "model_version": 0
    }
}
```
