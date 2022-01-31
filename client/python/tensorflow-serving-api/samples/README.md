# Samples based on tensorflow-serving-api package

This document contains examples to run *GetModelStatus*, *GetModelMetadata*, *Predict* functions over gRPC API and REST API.
Samples are based on [tensorflow-serving-api](https://pypi.org/project/tensorflow-serving-api/) package. 

> **Note** : _tensorflow-serving-api_ package is heavy as it includes _tensorflow_ as its dependency. For a lightweight alternative, see [ovmsclient](https://pypi.org/project/ovmsclient/) package (along with the [samples](../ovmsclient))

It covers following topics:
* <a href="#grpc-api">gRPC API Client Examples </a>
* <a href="#rest-api">REST API Client Examples  </a>

## Requirement

**Note**: Provided examples and their dependencies are updated and validated for Python 3.6+ version. For older versions of Python, dependencies versions adjustment might be required.

Install client dependencies:
```
pip3 install -r requirements.txt
```

Access to Google Cloud Storage might require proper configuration of https_proxy in the docker engine or in the docker container.
In the examples listed below, OVMS can be started using a command:
```bash
docker run -d --rm -e "http_proxy=$http_proxy" -e "https_proxy=$https_proxy" -p 8000:8000 -p 9000:9000 openvino/model_server:latest --model_name resnet --model_path gs://ovms-public-eu/resnet50 --port 9000 --rest_port 8000
```

## gRPC API Client Examples <a name="grpc-api"></a>

### Model Status API 

- Command

```bash
python grpc_get_model_status.py --help
usage: grpc_get_model_status.py [-h] [--grpc_address GRPC_ADDRESS]
                           [--grpc_port GRPC_PORT] [--model_name MODEL_NAME]
                           [--model_version MODEL_VERSION]

```

- Arguments

| Argument      | Description |
| :---        |    :----   |
| -h, --help       | Show help message and exit       |
| --grpc_address GRPC_ADDRESS   |   Specify url to grpc service. Default:localhost      |
| --grpc_port GRPC_PORT | Specify port to grpc service. Default: 9000 |
| --model_name MODEL_NAME | Model name to query. Default: resnet | 
| --model_version MODEL_VERSION | Model version to query. Lists all versions if not specified |


- Usage Example

```bash
python grpc_get_model_status.py --grpc_port 9000 --model_name resnet

Getting model status for model: resnet

Model version: 1
State AVAILABLE
Error code:  0
Error message:
```


### Model Metadata API


- Command

```bash
python grpc_get_model_metadata.py --help
usage: grpc_get_model_metadata.py [-h] [--grpc_address GRPC_ADDRESS]
                           [--grpc_port GRPC_PORT] [--model_name MODEL_NAME]
                           [--model_version MODEL_VERSION]

```

- Arguments

| Argument      | Description |
| :---        |    :----   |
| -h, --help       | Show this help message and exit       |
| --grpc_address GRPC_ADDRESS   |   Specify url to grpc service. Default:localhost      |
| --grpc_port GRPC_PORT | Specify port to grpc service. Default: 9000 |
| --model_name MODEL_NAME | Define model name, must be same as is in service. Default: resnet | 
| --model_version MODEL_VERSION | Define model version - must be numerical |


- Usage Example

```bash
python grpc_get_model_metadata.py --grpc_port 9000 --model_name resnet --model_version 1

Getting model metadata for model: resnet
Inputs metadata:
	Input name: data; shape: [1, 3, 224, 224]; dtype: DT_FLOAT
Outputs metadata:
	Output name: prob; shape: [1, 1000]; dtype: DT_FLOAT
```

### Predict API 

#### **Submitting gRPC requests based on a dataset from numpy files:**

- Command

```bash
usage: grpc_predict_resnet.py [-h] --images_numpy_path IMAGES_NUMPY_PATH
                              [--labels_numpy_path LABELS_NUMPY_PATH]
                              [--grpc_address GRPC_ADDRESS]
                              [--grpc_port GRPC_PORT]
                              [--input_name INPUT_NAME]
                              [--output_name OUTPUT_NAME]
                              [--transpose_input {False,True}]
                              [--transpose_method {nchw2nhwc,nhwc2nchw}]
                              [--iterations ITERATIONS]
                              [--batchsize BATCHSIZE]
                              [--model_name MODEL_NAME]
                              [--pipeline_name PIPELINE_NAME] 
                              [--tls]
                              [--server_cert SERVER_CERT]
                              [--client_cert CLIENT_CERT]
                              [--client_key CLIENT_KEY]
```

- Arguments

| Argument      | Description |
| :---        |    :----   |
| -h,--help       | Show help message and exit       |
| --images_numpy_path   |   Numpy in shape [n,w,h,c] or [n,c,h,w]      |
| --labels_numpy_path | Numpy in shape [n,1] - can be used to check model accuracy |
| --grpc_address GRPC_ADDRESS | Specify url to grpc service. Default:localhost | 
| --grpc_port GRPC_PORT | Specify port to grpc service. Default: 9000 |
| --input_name | Specify input tensor name. Default: input |
| --output_name | Specify output name. Default: resnet_v1_50/predictions/Reshape_1 |
| --transpose_input {False,True}|  Set to False to skip NHWC>NCHW or NCHW>NHWC input transposing. Default: True|
| --transpose_method {nchw2nhwc,nhwc2nchw} | How the input transposition should be executed: nhwc2nchw or nhwc2nchw. Default nhwc2nchw|
| --iterations | Number of requests iterations, as default use number of images in numpy memmap. Default: 0 (consume all frames)|
| --batchsize | Number of images in a single request. Default: 1 |
| --model_name | Define model name, must be same as is in service. Default: resnet|
| --pipeline_name | Define pipeline name, must be same as is in service |
| --tls | enables TLS communication with gRPC endpoint |
| --server_cert SERVER_CERT | Path to the server certificate, used only with TLS communication |
| --client_cert CLIENT_CERT | Path to the client certificate, used only with TLS communication |
| --client_key CLIENT_KEY | Path to the client key, used only with TLS communication |


- Usage example

```bash
python grpc_predict_resnet.py --grpc_port 9000 --images_numpy_path ../../imgs.npy --input_name data --output_name prob --transpose_input False --labels_numpy_path ../../lbs.npy
Start processing:
	Model name: resnet
	Iterations: 10
	Images numpy path: imgs.npy
	Images in shape: (10, 3, 224, 224)

Iteration 1; Processing time: 26.54 ms; speed 37.67 fps
imagenet top results in a single batch:
	 0 airliner 404 ; Correct match.
Iteration 2; Processing time: 22.23 ms; speed 44.99 fps
imagenet top results in a single batch:
	 0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Iteration 3; Processing time: 21.72 ms; speed 46.03 fps
imagenet top results in a single batch:
	 0 bee 309 ; Correct match.
Iteration 4; Processing time: 20.71 ms; speed 48.28 fps
imagenet top results in a single batch:
	 0 golden retriever 207 ; Correct match.
Iteration 5; Processing time: 20.53 ms; speed 48.71 fps
imagenet top results in a single batch:
	 0 gorilla, Gorilla gorilla 366 ; Correct match.
Iteration 6; Processing time: 20.37 ms; speed 49.08 fps
imagenet top results in a single batch:
	 0 magnetic compass 635 ; Correct match.
Iteration 7; Processing time: 20.97 ms; speed 47.68 fps
imagenet top results in a single batch:
	 0 peacock 84 ; Correct match.
Iteration 8; Processing time: 22.82 ms; speed 43.83 fps
imagenet top results in a single batch:
	 0 pelican 144 ; Correct match.
Iteration 9; Processing time: 22.16 ms; speed 45.13 fps
imagenet top results in a single batch:
	 0 snail 113 ; Correct match.
Iteration 10; Processing time: 21.17 ms; speed 47.24 fps
imagenet top results in a single batch:
	 0 zebra 340 ; Correct match.

processing time for all iterations
average time: 21.40 ms; average speed: 46.73 fps
median time: 21.00 ms; median speed: 47.62 fps
max time: 26.00 ms; min speed: 38.46 fps
min time: 20.00 ms; max speed: 50.00 fps
time percentile 90: 22.40 ms; speed percentile 90: 44.64 fps
time percentile 50: 21.00 ms; speed percentile 50: 47.62 fps
time standard deviation: 1.74
time variance: 3.04
Classification accuracy: 100.00

```

#### **Submitting gRPC requests with data in binary format:**
```bash
usage: grpc_predict_binary_resnet.py [-h] [--images_list IMAGES_LIST]
                                     [--grpc_address GRPC_ADDRESS]
                                     [--grpc_port GRPC_PORT]
                                     [--input_name INPUT_NAME]
                                     [--output_name OUTPUT_NAME]
                                     [--model_name MODEL_NAME]
                                     [--batchsize BATCHSIZE]
```

- Arguments

| Argument      | Description |
| :---        |    :----   |
|  -h, --help | show this help message and exit
|  --images_list IMAGES_LIST | path to a file with a list of labeled images |
|  --grpc_address GRPC_ADDRESS | Specify url to grpc service. default:localhost |
|  --grpc_port GRPC_PORT | Specify port to grpc service. default: 9000 |
|  --input_name INPUT_NAME | Specify input tensor name. default: image_bytes |
|  --output_name OUTPUT_NAME | Specify output name. default: probabilities |
|  --model_name MODEL_NAME | Define model name, must be same as is in service default: resnet |
|  --batchsize BATCHSIZE | Number of images in a single request. default: 1 |


- Usage example
```bash
python grpc_predict_binary_resnet.py --grpc_address localhost --model_name resnet --input_name 0 --output_name 1463 --grpc_port 9000 --images input_images.txt  --batchsize 2
Start processing:
	Model name: resnet
	Images list file: input_images.txt
Batch: 0; Processing time: 22.04 ms; speed 45.38 fps
	 1 airliner 404 ; Incorrect match. Should be 279 Arctic fox, white fox, Alopex lagopus
	 2 white wolf, Arctic wolf, Canis lupus tundrarum 270 ; Incorrect match. Should be 279 Arctic fox, white fox, Alopex lagopus
Batch: 1; Processing time: 15.58 ms; speed 64.16 fps
	 3 bee 309 ; Correct match.
	 4 golden retriever 207 ; Correct match.
Batch: 2; Processing time: 17.93 ms; speed 55.79 fps
	 5 gorilla, Gorilla gorilla 366 ; Correct match.
	 6 magnetic compass 635 ; Correct match.
Batch: 3; Processing time: 17.14 ms; speed 58.36 fps
	 7 peacock 84 ; Correct match.
	 8 pelican 144 ; Correct match.
Batch: 4; Processing time: 15.56 ms; speed 64.25 fps
	 9 snail 113 ; Correct match.
	 10 zebra 340 ; Correct match.
Overall accuracy= 80.0 %
Average latency= 17.2 ms
```


## REST API Client Examples<a name="rest-api"></a>

### Model Status API
- Command
```Bash 
python rest_get_model_status.py --help
usage: rest_get_model_status.py [-h] [--rest_url REST_URL]
                                [--rest_port REST_PORT]
                                [--model_name MODEL_NAME]
                                [--model_version MODEL_VERSION]
                                [--client_cert CLIENT_CERT]
                                [--client_key CLIENT_KEY]
                                [--ignore_server_verification]
                                [--server_cert SERVER_CERT]
```
- Arguments 
```
| Argument      | Description |
| :---        |    :----   |
| -h, --help | Show help message and exit|
| --rest_url REST_URL | Specify url to REST API service. Default:http://localhost|
| --rest_port REST_PORT | Specify port to REST API service. Default: 5555|
| --model_name MODEL_NAME| Model name to query, must be same as is in service. Default : resnet|
| --model_version MODEL_VERSION | Model version to query - must be numerical. List all version if omitted|
```

- Usage Example 
```bash
python rest_get_model_status.py --rest_port 8000 --model_version 1
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
}
```

### Model Metadata API
- Command
```Bash
python rest_get_model_metadata.py --help
usage: rest_get_model_metadata.py [-h] [--rest_url REST_URL]
                                  [--rest_port REST_PORT]
                                  [--model_name MODEL_NAME]
                                  [--model_version MODEL_VERSION]
                                  [--client_cert CLIENT_CERT]
                                  [--client_key CLIENT_KEY]
                                  [--ignore_server_verification]
                                  [--server_cert SERVER_CERT]
```
- Arguments 

| Argument      | Description |
| :---        |    :----   |
|  -h, --help | show this help message and exit |
|  --rest_url REST_URL | Specify url to REST API service. default: http://localhost |
|  --rest_port REST_PORT | Specify port to REST API service. default: 5555 |
|  --model_name MODEL_NAME | Define model name, must be same as is in service. default: resnet |
|  --model_version MODEL_VERSION | Define model version - must be numerical |
|  --client_cert CLIENT_CERT | Specify mTLS client certificate file. Default: None. |
|  --client_key CLIENT_KEY | Specify mTLS client key file. Default: None. |
|  --ignore_server_verification | Skip TLS host verification. Do not use in production. Default: False. |
|  --server_cert SERVER_CERT | Path to a custom directory containing trusted CA certificates, server certificate, or a CA_BUNDLE file. |

- Usage Example
```Bash
python rest_get_model_metadata.py --rest_port 8000
{
 "modelSpec": {
  "name": "resnet",
  "signatureName": "",
  "version": "1"
 },
 "metadata": {
  "signature_def": {
   "@type": "type.googleapis.com/tensorflow.serving.SignatureDefMap",
   "signatureDef": {
    "serving_default": {
     "inputs": {
      "data": {
       "dtype": "DT_FLOAT",
       "tensorShape": {
        "dim": [
         {
          "size": "1",
          "name": ""
         },
         {
          "size": "3",
          "name": ""
         },
         {
          "size": "224",
          "name": ""
         },
         {
          "size": "224",
          "name": ""
         }
        ],
        "unknownRank": false
       },
       "name": "data"
      }
     },
     "outputs": {
      "prob": {
       "dtype": "DT_FLOAT",
       "tensorShape": {
        "dim": [
         {
          "size": "1",
          "name": ""
         },
         {
          "size": "1000",
          "name": ""
         }
        ],
        "unknownRank": false
       },
       "name": "prob"
      }
     },
     "methodName": ""
    }
   }
  }
 }
}
```

### Predict API

- Command :
```bash
python rest_predict_resnet.py --help
usage: rest_predict_resnet.py [-h] --images_numpy_path IMAGES_NUMPY_PATH
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
                              [--client_cert CLIENT_CERT]
                              [--client_key CLIENT_KEY]
                              [--ignore_server_verification]
                              [--server_cert SERVER_CERT]
```

- Arguments :

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
| --client_cert CLIENT_CERT | Specify mTLS client certificate file. Default: None |
| --client_key CLIENT_KEY | Specify mTLS client key file. Default: None |
| --ignore_server_verification | Skip TLS host verification. Do not use in production. Default: False |
| --server_cert SERVER_CERT | Path to a custom directory containing trusted CA certificates, server certificate, or a CA_BUNDLE file. Default: None, will use default system CA cert store |

- Usage Example
```bash
python rest_predict_resnet.py --images_numpy_path ../../imgs.npy --labels_numpy_path ../../lbs.npy --input_name data --output_name prob --rest_port 8000 --transpose_input False
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
#### **Submitting gRPC requests with data in binary format:**
```bash
usage: rest_predict_binary_resnet.py [-h] [--images_list IMAGES_LIST]
                                     [--rest_url REST_URL]
                                     [--input_name INPUT_NAME]
                                     [--output_name OUTPUT_NAME]
                                     [--model_name MODEL_NAME]
                                     [--request_format {row_noname,row_name,column_noname,column_name}]
                                     [--batchsize BATCHSIZE]
```

- Arguments


| Argument      | Description |
| :---        |    :----   |
|  -h, --help | show this help message and exit |
|  --images_list IMAGES_LIST | path to a file with a list of labeled images |
|  --rest_url REST_URL | Specify url to REST API service. default: http://localhost:8000 |
|  --input_name INPUT_NAME | Specify input tensor name. default: image_bytes |
|  --output_name OUTPUT_NAME | Specify output name. default: probabilities |
|  --model_name MODEL_NAME | Define model name, must be same as is in service default: resnet |
|  --request_format {row_noname,row_name,column_noname,column_name} | Request format according to TF Serving API: row_noname,row_name,column_noname,column_name |
|  --batchsize BATCHSIZE | Number of images in a single request. default: 1 |


- Usage example
```bash
python rest_predict_binary_resnet.py --rest_url http://localhost:8000 --model_name resnet --input_name 0 --output_name 1463  --images input_images.txt  --batchsize 2
Start processing:
	Model name: resnet
	Images list file: input_images.txt
Batch: 0; Processing time: 17.73 ms; speed 56.42 fps
output shape: (2, 1000)
	 1 airliner 404 ; Incorrect match. Should be 279 Arctic fox, white fox, Alopex lagopus
	 2 white wolf, Arctic wolf, Canis lupus tundrarum 270 ; Incorrect match. Should be 279 Arctic fox, white fox, Alopex lagopus
Batch: 1; Processing time: 14.06 ms; speed 71.11 fps
output shape: (2, 1000)
	 3 bee 309 ; Correct match.
	 4 golden retriever 207 ; Correct match.
Batch: 2; Processing time: 14.78 ms; speed 67.66 fps
output shape: (2, 1000)
	 5 gorilla, Gorilla gorilla 366 ; Correct match.
	 6 magnetic compass 635 ; Correct match.
Batch: 3; Processing time: 20.56 ms; speed 48.64 fps
output shape: (2, 1000)
	 7 peacock 84 ; Correct match.
	 8 pelican 144 ; Correct match.
Batch: 4; Processing time: 23.04 ms; speed 43.41 fps
output shape: (2, 1000)
	 9 snail 113 ; Correct match.
	 10 zebra 340 ; Correct match.
Overall accuracy= 80.0 %
Average latency= 17.6 ms
```