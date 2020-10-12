# OpenVINO&trade; Model Server RESTful API Documentation

## Introduction
In addition with [gRPC APIs](./ModelServerGRPCAPI.md) OpenVINO&trade; model server also supports RESTful APIs. This page describes these API endpoints and an end-to-end example on usage.

## Model Status API
* Description

Get information about the status of served models

* URL 

```Bash
GET http://${REST_URL}:${REST_PORT}/v1/models/${MODEL_NAME}/versions/${MODEL_VERSION}
```
> **Note** : Including /versions/${MODEL_VERSION} is optional. If omitted status for all versions is returned in the response.

* Response format

If successful, returns a JSON of following format
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

## Model Metadata API
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

## Model Predict Function
* Description

Sends requests via TensorFlow Serving RESTful API using images in numpy format. It displays performance statistics and optionally the model accuracy.

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

A request in [row format](https://www.tensorflow.org/tfx/serving/api_rest#specifying_input_tensors_in_row_format) has response formatted as follows:
```
{
  "predictions": <value>|<(nested)list>|<list-of-objects>
}
```
A request in [column format](https://www.tensorflow.org/tfx/serving/api_rest#specifying_input_tensors_in_column_format) has response formatted as follows:
```
{
  "outputs": <value>|<(nested)list>|<object>
}
```

* Command
```Bash
python rest_serving_client.py --help
usage: rest_serving_client.py [-h] --images_numpy_path IMAGES_NUMPY_PATH
                              [--labels_numpy_path LABELS_NUMPY_PATH]
                              [--rest_url REST_URL] [--rest_port REST_PORT]
                              [--input_name INPUT_NAME]
                              [--output_name OUTPUT_NAME]
                              [--transpose_input {False,True}]
                              [--transpose_method {nchw2nhwc,nhwc2nchw}]
                              [--iterations ITERATIONS]
                              [--batchsize BATCHSIZE]
                              [--model_name MODEL_NAME]
                              [--request_format {row_noname,row_name,column_noname,column_name}]
                              [--model_version MODEL_VERSION]
```
* Optional Arguments

| Argument      | Description |
| :---        |    :----   |
| -h, --help       | Show help message and exit       |
| --images_numpy_path IMAGES_NUMPY_PATH |   Numpy in shape [n,w,h,c] or [n,c,h,w]      |
| --labels_numpy_path LABELS_NUMPY_PATH| Numpy in shape [n,1] - can be used to check model accuracy |
| --rest_url REST_URL| Specify url to REST API service. Default: http://localhost | 
| --rest_port REST_PORT| Specify port to REST API service. Default: 5555 |
| --input_name INPUT_NAME| Specify input tensor name. Default: input |
| --output_name OUTPUT_NAME| Specify output name. Default: resnet_v1_50/predictions/Reshape_1 |
| --transpose_input {False,True}| Set to False to skip NHWC>NCHW or NCHW>NHWC input transposing. Default: True|
| --transpose_method {nchw2nhwc,nhwc2nchw} | How the input transposition should be executed: nhwc2nchw or nhwc2nchw |
| --iterations ITERATIONS| Number of requests iterations, as default use number of images in numpy memmap. Default: 0 (consume all frames)|
| --batchsize BATCHSIZE| Number of images in a single request. Default: 1 |
| --model_name MODEL_NAME| Define model name, must be same as is in service. Default: resnet|
| --request_format {row_noname,row_name,column_noname,column_name}| Request format according to TF Serving API:row_noname,row_name,column_noname,column_name|
| --model_version MODEL_VERSION| Model version to be used. Default: LATEST |


* Sample Response
```Bash
output shape: (1, 1000)
Iteration 1; Processing time: 57.42 ms; speed 17.41 fps
imagenet top results in a single batch:
('\t', 0, 'airliner', 404, '; Correct match.')
```
* Usage example 
```Bash
python rest_serving_client.py --images_numpy_path imgs.npy --labels_numpy_path lbs.npy --input_name data --output_name prob --rest_port 8000 --transpose_input False
('Image data range:', 0, ':', 255)
Start processing:
	Model name: resnet
	Iterations: 10
	Images numpy path: imgs.npy
	Images in shape: (10, 3, 224, 224)

output shape: (1, 1000)
Iteration 1; Processing time: 57.42 ms; speed 17.41 fps
imagenet top results in a single batch:
('\t', 0, 'airliner', 404, '; Correct match.')
output shape: (1, 1000)
Iteration 2; Processing time: 57.65 ms; speed 17.35 fps
imagenet top results in a single batch:
('\t', 0, 'Arctic fox, white fox, Alopex lagopus', 279, '; Correct match.')
output shape: (1, 1000)
Iteration 3; Processing time: 59.21 ms; speed 16.89 fps
imagenet top results in a single batch:
('\t', 0, 'bee', 309, '; Correct match.')
output shape: (1, 1000)
Iteration 4; Processing time: 59.64 ms; speed 16.77 fps
imagenet top results in a single batch:
('\t', 0, 'golden retriever', 207, '; Correct match.')
output shape: (1, 1000)
Iteration 5; Processing time: 59.96 ms; speed 16.68 fps
imagenet top results in a single batch:
('\t', 0, 'gorilla, Gorilla gorilla', 366, '; Correct match.')
output shape: (1, 1000)
Iteration 6; Processing time: 59.41 ms; speed 16.83 fps
imagenet top results in a single batch:
('\t', 0, 'magnetic compass', 635, '; Correct match.')
output shape: (1, 1000)
Iteration 7; Processing time: 59.45 ms; speed 16.82 fps
imagenet top results in a single batch:
('\t', 0, 'peacock', 84, '; Correct match.')
output shape: (1, 1000)
Iteration 8; Processing time: 59.91 ms; speed 16.69 fps
imagenet top results in a single batch:
('\t', 0, 'pelican', 144, '; Correct match.')
output shape: (1, 1000)
Iteration 9; Processing time: 63.17 ms; speed 15.83 fps
imagenet top results in a single batch:
('\t', 0, 'snail', 113, '; Correct match.')
output shape: (1, 1000)
Iteration 10; Processing time: 52.59 ms; speed 19.01 fps
imagenet top results in a single batch:
('\t', 0, 'zebra', 340, '; Correct match.')

processing time for all iterations
average time: 58.30 ms; average speed: 17.15 fps
median time: 59.00 ms; median speed: 16.95 fps
max time: 63.00 ms; max speed: 15.00 fps
min time: 52.00 ms; min speed: 19.00 fps
time percentile 90: 59.40 ms; speed percentile 90: 16.84 fps
time percentile 50: 59.00 ms; speed percentile 50: 16.95 fps
time standard deviation: 2.61
time variance: 6.81
Classification accuracy: 100.00
```



