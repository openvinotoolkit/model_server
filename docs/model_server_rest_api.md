# OpenVINO&trade; Model Server RESTful API Documentation

## Introduction
In addition with [gRPC APIs](./model_server_grpc_api.md) OpenVINO&trade; model server also supports RESTful APIs which follows the documentation from [tensorflow serving REST API](https://www.tensorflow.org/tfx/serving/api_rest). Both row and column format of the request are implemented in these APIs. REST API is recommended when the primary goal is in reducing the number of client side python dependencies and simpler application code.

> **Note** : Only numerical data type is supported.

This document covers following API:
* <a href="#model-status">Model Status API</a>
* <a href="#model-metadata">Model MetaData API </a>
* <a href="#predict">Predict API </a>
* <a href="#config-reload">Config Reload API </a>
* <a href="#config-status">Config Status API </a>


> **Note** : The implementations for Predict, GetModelMetadata and GetModelStatus function calls are currently available. These are the most generic function calls and should address most of the usage scenarios.

## Model Status API <a name="model-status"></a>
* Description

Get information about the status of served models

* URL 

```Bash
GET http://${REST_URL}:${REST_PORT}/v1/models/${MODEL_NAME}/versions/${MODEL_VERSION}
```
> **Note** : Including /versions/${MODEL_VERSION} is optional. If omitted status for all versions is returned in the response.

* Response format

If successful, returns a JSON of following format :
```Bash
{
  'model_version_status':[
    {
      'version': <model version>|<string>,
      'state': <model state>|<string>,
      'status': {
        'error_code': <error code>|<string>,
        'error_message': <error message>|<string>
      }
    }
  ]
}
```

* Usage Example
```Bash
$ curl http://localhost:8001/v1/models/person-detection/versions/1
{
  'model_version_status':[
    {
      'version': '1', 
      'state': 'AVAILABLE', 
      'status': {
        'error_code': 'OK', 
        'error_message': ''
      }
    }
  ]
}
```
Read more about *Get Model Status API* usage [here](./../example_client/README.md#model-status-api-1)

## Model Metadata API <a name="model-metadata"></a>
* Description 

Get the metadata of a model in the model server.

* URL 
```Bash
GET http://${REST_URL}:${REST_PORT}/v1/models/${MODEL_NAME}/versions/${MODEL_VERSION}/metadata
```
> **Note** :Including ${MODEL_VERSION} is optional. If omitted the model metadata for the latest version is returned in the response.

* Response format

If successful, returns a JSON representation of [GetModelMetadataResponse](https://github.com/tensorflow/serving/blob/5369880e9143aa00d586ee536c12b04e945a977c/tensorflow_serving/apis/get_model_metadata.proto#L23) protobuf.

* Usage example
```Bash
$ curl http://localhost:8001/v1/models/person-detection/versions/1/metadata
{
  "modelSpec": {
    "name": "person-detection",
    "version": "1"
  },
  "metadata": {
    "signature_def": {
      "@type": "type.googleapis.com/tensorflow.serving.SignatureDefMap",
      "signatureDef": {
        "serving_default": {
          "inputs": {
            "data": {
              "name": "data_2:0",
              "dtype": "DT_FLOAT",
              "tensorShape": {
                "dim": [
                  {
                    "size": "1"
                  },
                  {
                    "size": "3"
                  },
                  {
                    "size": "400"
                  },
                  {
                    "size": "600"
                  }
                ]
              }
            }
          },
          "outputs": {
            "detection_out": {
              "name": "detection_out_2:0",
              "dtype": "DT_FLOAT",
              "tensorShape": {
                "dim": [
                  {
                    "size": "1"
                  },
                  {
                    "size": "1"
                  },
                  {
                    "size": "200"
                  },
                  {
                    "size": "7"
                  }
                ]
              }
            }
          },
          "methodName": "tensorflow/serving/predict"
        }
      }
    }
  }
}
```
Read more about *Get Model Metadata API* usage [here](./../example_client/README.md#model-metadata-api-1)

## Predict API <a name="predict"></a>
* Description

Sends requests via TensorFlow Serving RESTful API using images in numpy array or binary format. It displays performance statistics and optionally the model accuracy.

* URL
```
POST http://${REST_URL}:${REST_PORT}/v1/models/${MODEL_NAME}/versions/${MODEL_VERSION}:predict
```
* Request Header
```
{
  // (Optional) Serving signature to use.
  // If unspecifed default serving signature is used.
  "signature_name": <string>,

  // Input Tensors in row ("instances") or columnar ("inputs") format.
  // A request can have either of them but NOT both.
  "instances": <value>|<(nested)list>|<list-of-objects>
  "inputs": <value>|<(nested)list>|<object>
}
``` 
> **Note**
Read [How to specify input tensors in row format](https://www.tensorflow.org/tfx/serving/api_rest#specifying_input_tensors_in_row_format) and [How to specify input tensors in column format](https://www.tensorflow.org/tfx/serving/api_rest#specifying_input_tensors_in_column_format) for more details.

* Response

A request in [row format](https://www.tensorflow.org/tfx/serving/api_rest#specifying_input_tensors_in_row_format) has response formatted as follows :
```
{
  "predictions": <value>|<(nested)list>|<list-of-objects>
}
```
A request in [column format](https://www.tensorflow.org/tfx/serving/api_rest#specifying_input_tensors_in_column_format) has response formatted as follows :
```
{
  "outputs": <value>|<(nested)list>|<object>
}
```

Beside numerical values, it is possible to pass binary inputs. They must be Base64 encoded in passed in `b64` key like below:
```
{
  "instances": [
    {
      "image": { "b64": "aW1hZ2UgYnl0ZXM=" },
    },
    {
      "image": { "b64": "YXdlc29tZSBpbWFnZSBieXRlcw==" },
    }
  ]
}
```
Check [how binary data is handled in OpenVINO Model Server](binary_input_ouput.md)

Read more about *Predict API* usage examples [here](./../example_client/README.md#predict-api-1)

## Config Reload API <a name="config-reload"></a>
* Description  

Sends requests via RESTful API to trigger config reloading and gets models and [DAGs](./dag_scheduler.md) statuses as a response.This endpoint can be used with disabled automatic config reload to ensure changes in configuration are applied in a specific time and also to get confirmation about reload operation status. Typically this option is to be used when OVMS is started with a parameter `--file_system_poll_wait_seconds 0`.
Reload operation does not pass new configuration to OVMS server. The configuration file changes needs to be applied by the OVMS administrator. The REST API call just initiate applying the configuration file which is already present.

* URL  
```
POST http://${REST_URL}:${REST_PORT}/v1/config/reload
```
* FLOW  

Flow after receiving request:
1) If config file was changed - reload config.
2) If any model version directory was changed or new version was added - reload this model.
3) If any model that is part of a DAG was changed or new version was added - reload this pipeline.
4) In case there are no errors in the reload operation, the response includes the status of all models and DAGs, otherwise error message is returned.

* Request  
To trigger reload, HTTP POST request with empty body should be sent on given URL. Example `curl` command:

```Bash
curl --request POST http://${REST_URL}:${REST_PORT}/v1/config/reload
```
* Response  
In case of config reload success, response contains JSON with aggregation of getModelStatus responses for all models and DAGs after reload is finished, along with operation status: 
```JSON
{ 
"<model name>": 
{ 
  "model_version_status": [     
  { 
     "version": <model version>|<string>,
     "state": <model state>|<string>, 
     "status":
{ 
  "error_code": <error code>|<string>, 
  "error_message": <error message>|<string>       
} 
  }, 
  ...  
] 
}, 
... 
} 
```

In case of any failure during execution: 
 
```JSON
{ 
  "error": <error message>|<string> 
} 
```
When operation succeeds HTTP response status code is
  - `201` when config(config file or model version) was reloaded 
  - `200` when reload was not required, already applied or OVMS was started in single model mode
When operation fails another status code is returned.

Possible messages returned on error:

- obtaining config file change time failed (file is not exisiting or cannot be accessed):
```JSON
{
  "error": "Config file not found or cannot open."
}
```
- config file was changed and config reloading failed (file content is not a valid JSON, any of model or DAG config is incorrect):
```JSON
{
  "error": "Reloading config file failed. Check server logs for more info."
}
```

- config file was not changed and model versions reloading failed (model directory was removed):
```JSON
{
  "error": "Reloading models versions failed. Check server logs for more info."
}
```

- retrieving status of one of the models failed:
```JSON
{
  "error": "Retrieving all model statuses failed. Check server logs for more info."
}
```

- converting model status responses to json failed:
```JSON
{
  "error": "Serializing model statuses to json failed. Check server logs for more info."
}
```
Even if one of models reload failed other may be working properly. To check state of loaded models use [Config Status API](./model_server_rest_api.md#config-status). To detect exact cause of errors described above analyzing sever logs may be necessary.

## Config Status API <a name="config-status"></a>
* Description

Sends requests via RESTful API to get response that contains aggregation of getModelStatus responses for all models and [DAGs](./dag_scheduler.md).

* URL  
```
GET http://${REST_URL}:${REST_PORT}/v1/config
```
* Request  
To trigger this API HTTP GET request should be sent on given URL.Example `curl` command:

```Bash
curl --request GET http://${REST_URL}:${REST_PORT}/v1/config
```

* Response  
In case of success, response contains JSON with aggregation of getModelStatus responses for all models and DAGs, along with operation status: 
```JSON
{ 
"<model name>": 
{ 
  "model_version_status": [     
  { 
     "version": <model version>|<string>,
     "state": <model state>|<string>, 
     "status":
{ 
  "error_code": <error code>|<string>, 
  "error_message": <error message>|<string>       
} 
  }, 
  ...  
] 
}, 
... 
} 
```

In case of any failure during execution:
 
```JSON
{ 
  "error": <error message>|<string> 
} 
```
When operation succeeded HTTP response status code is 200, otherwise another code is returned.
Possible messages returned on error:

- retrieving status of one of the models failed:
```JSON
{
  "error": "Retrieving all model statuses failed. Check server logs for more info."
}
```

- converting model status responses to json failed:
```JSON
{
  "error": "Serializing model statuses to json failed. Check server logs for more info."
}
```
