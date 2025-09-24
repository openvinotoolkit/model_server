# Samples based on tensorflow-serving-api package

This document contains examples to run *GetModelStatus*, *GetModelMetadata*, *Predict* functions over gRPC API and REST API.
Samples are based on [tensorflow-serving-api](https://pypi.org/project/tensorflow-serving-api/) package.

It covers following topics:
* [gRPC API Client Examples](#grpc-api-client-examples)
* [REST API Client Examples](#rest-api-client-examples)

## Requirement

**Note**: Provided examples and their dependencies are updated and validated for Python 3.7+ version. For older versions of Python, dependencies versions adjustment might be required.

Clone the repository and enter directory:

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/client/python/tensorflow-serving-api/samples
```

Install client dependencies:
```bash
pip3 install -r requirements.txt
```

In the examples listed below, OVMS can be started using a command:
```bash
wget -N https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.{xml,bin} -P models/resnet50/1
docker run -d -u $(id -u) -v $(pwd)/models:/models -p 8000:8000 -p 9000:9000 openvino/model_server:latest --model_name resnet --model_path /models/resnet50 --port 9000 --rest_port 8000
```

## gRPC API Client Examples

### Model Status API

- Command

```bash
python grpc_get_model_status.py --help
usage: grpc_get_model_status.py [-h] [--grpc_address GRPC_ADDRESS]
                           [--grpc_port GRPC_PORT]
                           [--model_name MODEL_NAME]
                           [--model_version MODEL_VERSION]

```

- Arguments

| Argument      | Description |
| :---        |    :----   |
| -h, --help       | Show help message and exit       |
| --grpc_address GRPC_ADDRESS   |   Specify url to grpc service. Default: localhost      |
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
Error message:  OK
```


### Model Metadata API


- Command

```bash
python grpc_get_model_metadata.py --help
usage: grpc_get_model_metadata.py [-h] [--grpc_address GRPC_ADDRESS]
                           [--grpc_port GRPC_PORT]
                           [--model_name MODEL_NAME]
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
	Input name: 0; shape: [1, 3, 224, 224]; dtype: DT_FLOAT
Outputs metadata:
	Output name: 1463; shape: [1, 1000]; dtype: DT_FLOAT
```

### Predict API

#### **Submitting gRPC requests based on a dataset from numpy files:**

- Command

```bash
python grpc_predict_resnet.py --help
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
                              [--dag-batch-size-auto] [--tls]
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
| --dag-batch-size-auto | Add demultiplexer dimension at front |
| --tls | enables TLS communication with gRPC endpoint |
| --server_cert SERVER_CERT | Path to the server certificate, used only with TLS communication |
| --client_cert CLIENT_CERT | Path to the client certificate, used only with TLS communication |
| --client_key CLIENT_KEY | Path to the client key, used only with TLS communication |


- Usage example

```bash
python grpc_predict_resnet.py --grpc_port 9000 --images_numpy_path ../../imgs.npy --input_name 0 --output_name 1463 --transpose_input False --labels_numpy_path ../../lbs.npy

Image data range: 0.0 : 255.0
Start processing:
        Model name: resnet
        Iterations: 10
        Images numpy path: ../../imgs.npy
        Numpy file shape: (10, 3, 224, 224)

Iteration 1; Processing time: 29.75 ms; speed 33.62 fps
imagenet top results in a single batch:
         0 airliner 404 ; Correct match.
Iteration 2; Processing time: 27.06 ms; speed 36.96 fps
imagenet top results in a single batch:
         0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Iteration 3; Processing time: 26.81 ms; speed 37.30 fps
imagenet top results in a single batch:
         0 bee 309 ; Correct match.
Iteration 4; Processing time: 27.25 ms; speed 36.70 fps
imagenet top results in a single batch:
         0 golden retriever 207 ; Correct match.
Iteration 5; Processing time: 25.64 ms; speed 39.00 fps
imagenet top results in a single batch:
         0 gorilla, Gorilla gorilla 366 ; Correct match.
Iteration 6; Processing time: 34.41 ms; speed 29.06 fps
imagenet top results in a single batch:
         0 magnetic compass 635 ; Correct match.
Iteration 7; Processing time: 23.39 ms; speed 42.75 fps
imagenet top results in a single batch:
         0 peacock 84 ; Correct match.
Iteration 8; Processing time: 23.82 ms; speed 41.98 fps
imagenet top results in a single batch:
         0 pelican 144 ; Correct match.
Iteration 9; Processing time: 25.66 ms; speed 38.97 fps
imagenet top results in a single batch:
         0 snail 113 ; Correct match.
Iteration 10; Processing time: 22.44 ms; speed 44.57 fps
imagenet top results in a single batch:
         0 zebra 340 ; Correct match.

processing time for all iterations
average time: 26.10 ms; average speed: 38.31 fps
median time: 25.50 ms; median speed: 39.22 fps
max time: 34.00 ms; min speed: 29.41 fps
min time: 22.00 ms; max speed: 45.45 fps
time percentile 90: 29.50 ms; speed percentile 90: 33.90 fps
time percentile 50: 25.50 ms; speed percentile 50: 39.22 fps
time standard deviation: 3.33
time variance: 11.09
Classification accuracy: 100.00
```

#### **Submitting gRPC requests with data in binary format:**

Using binary inputs feature requires loading model with layout set to `--layout NHWC:NCHW`:
```bash
wget -N https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.{xml,bin} -P models/resnet50/1
docker run -d -u $(id -u) -v $(pwd)/models:/models -p 8000:8000 -p 9000:9000 openvino/model_server:latest --model_name resnet --model_path /models/resnet50 --port 9000 --rest_port 8000 --layout NHWC:NCHW
```
```bash
python grpc_predict_binary_resnet.py --help
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
python grpc_predict_binary_resnet.py --grpc_address localhost --model_name resnet --input_name 0 --output_name 1463 --grpc_port 9000 --images ../../resnet_input_images.txt

Start processing:
        Model name: resnet
        Images list file: ../../resnet_input_images.txt
Batch: 0; Processing time: 31.59 ms; speed 31.65 fps
         1 airliner 404 ; Correct match.
Batch: 1; Processing time: 30.08 ms; speed 33.24 fps
         2 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Batch: 2; Processing time: 28.80 ms; speed 34.72 fps
         3 bee 309 ; Correct match.
Batch: 3; Processing time: 29.39 ms; speed 34.03 fps
         4 golden retriever 207 ; Correct match.
Batch: 4; Processing time: 29.91 ms; speed 33.43 fps
         5 gorilla, Gorilla gorilla 366 ; Correct match.
Batch: 5; Processing time: 28.52 ms; speed 35.07 fps
         6 magnetic compass 635 ; Correct match.
Batch: 6; Processing time: 28.69 ms; speed 34.85 fps
         7 peacock 84 ; Correct match.
Batch: 7; Processing time: 26.02 ms; speed 38.43 fps
         8 pelican 144 ; Correct match.
Batch: 8; Processing time: 24.28 ms; speed 41.19 fps
         9 snail 113 ; Correct match.
Batch: 9; Processing time: 29.93 ms; speed 33.41 fps
         10 zebra 340 ; Correct match.
Overall accuracy= 100.0 %
Average latency= 28.2 ms
```


## REST API Client Examples

Access to Google Cloud Storage might require proper configuration of https_proxy in the docker engine or in the docker container.
In the examples listed below, OVMS can be started using a command:
```bash
wget -N https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.{xml,bin} -P models/resnet50/1
docker run -d -u $(id -u) -v $(pwd)/models:/models -p 8000:8000 -p 9000:9000 openvino/model_server:latest --model_name resnet --model_path /models/resnet50 --port 9000 --rest_port 8000
```

### Model Status API
- Command
```bash
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

| Argument      | Description |
| :---        |    :----   |
| -h, --help | Show help message and exit|
| --rest_url REST_URL | Specify url to REST API service. Default:http://localhost|
| --rest_port REST_PORT | Specify port to REST API service. Default: 8000|
| --model_name MODEL_NAME| Model name to query, must be same as is in service. Default : resnet|
| --model_version MODEL_VERSION | Model version to query - must be numerical. List all version if omitted |
| --client_cert CLIENT_CERT | Specify mTLS client certificate file. Default: None.|
| --client_key CLIENT_KEY | Specify mTLS client key file. Default: None. |
| --ignore_server_verification | Skip TLS host verification. Do not use in production. Default: False. |
| --server_cert SERVER_CERT | Path to a custom directory containing trusted CA certificates, server certificate, or a CA_BUNDLE file. Default: None, will use default system CA cert store. |

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
```bash
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
|  --rest_port REST_PORT | Specify port to REST API service. default: 8000 |
|  --model_name MODEL_NAME | Define model name, must be same as is in service. default: resnet |
|  --model_version MODEL_VERSION | Define model version - must be numerical |
|  --client_cert CLIENT_CERT | Specify mTLS client certificate file. Default: None. |
|  --client_key CLIENT_KEY | Specify mTLS client key file. Default: None. |
|  --ignore_server_verification | Skip TLS host verification. Do not use in production. Default: False. |
|  --server_cert SERVER_CERT | Path to a custom directory containing trusted CA certificates, server certificate, or a CA_BUNDLE file. |

- Usage Example
```bash
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
      "0": {
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
       "name": "0"
      }
     },
     "outputs": {
      "1463": {
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
       "name": "1463"
      }
     },
     "methodName": "",
     "defaults": {}
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
| --rest_port REST_PORT| Specify port to REST API service. Default: 8000 |
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
python rest_predict_resnet.py --images_numpy_path ../../imgs.npy --labels_numpy_path ../../lbs.npy --input_name 0 --output_name 1463 --rest_port 8000 --transpose_input False

Image data range: 0 : 255
Start processing:
        Model name: resnet
        Iterations: 10
        Images numpy path: ../../imgs.npy
        Images in shape: (10, 3, 224, 224)

output shape: (1, 1000)
Iteration 1; Processing time: 54.98 ms; speed 18.19 fps
imagenet top results in a single batch:
         0 airliner 404 ; Correct match.
output shape: (1, 1000)
Iteration 2; Processing time: 46.54 ms; speed 21.49 fps
imagenet top results in a single batch:
         0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
output shape: (1, 1000)
Iteration 3; Processing time: 50.70 ms; speed 19.73 fps
imagenet top results in a single batch:
         0 bee 309 ; Correct match.
output shape: (1, 1000)
Iteration 4; Processing time: 46.89 ms; speed 21.33 fps
imagenet top results in a single batch:
         0 golden retriever 207 ; Correct match.
output shape: (1, 1000)
Iteration 5; Processing time: 45.78 ms; speed 21.84 fps
imagenet top results in a single batch:
         0 gorilla, Gorilla gorilla 366 ; Correct match.
output shape: (1, 1000)
Iteration 6; Processing time: 48.72 ms; speed 20.53 fps
imagenet top results in a single batch:
         0 magnetic compass 635 ; Correct match.
output shape: (1, 1000)
Iteration 7; Processing time: 45.20 ms; speed 22.12 fps
imagenet top results in a single batch:
         0 peacock 84 ; Correct match.
output shape: (1, 1000)
Iteration 8; Processing time: 45.50 ms; speed 21.98 fps
imagenet top results in a single batch:
         0 pelican 144 ; Correct match.
output shape: (1, 1000)
Iteration 9; Processing time: 45.30 ms; speed 22.08 fps
imagenet top results in a single batch:
         0 snail 113 ; Correct match.
output shape: (1, 1000)
Iteration 10; Processing time: 44.19 ms; speed 22.63 fps
imagenet top results in a single batch:
         0 zebra 340 ; Correct match.

processing time for all iterations
average time: 46.80 ms; average speed: 21.37 fps
median time: 45.50 ms; median speed: 21.98 fps
max time: 54.00 ms; min speed: 18.52 fps
min time: 44.00 ms; max speed: 22.73 fps
time percentile 90: 50.40 ms; speed percentile 90: 19.84 fps
time percentile 50: 45.50 ms; speed percentile 50: 21.98 fps
time standard deviation: 2.93
time variance: 8.56
Classification accuracy: 100.00
```
#### **Submitting REST requests with data in binary format:**

Using binary inputs feature requires loading model with layout set to `--layout NHWC:NCHW`:
```bash
wget -N https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.{xml,bin} -P models/resnet50/1
docker run -d -u $(id -u) -v $(pwd)/models:/models -p 8000:8000 -p 9000:9000 openvino/model_server:latest --model_name resnet --model_path /models/resnet50 --port 9000 --rest_port 8000 --layout NHWC:NCHW
```

```bash
python rest_predict_binary_resnet.py --help
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
python rest_predict_binary_resnet.py --rest_url http://localhost:8000 --model_name resnet --input_name 0 --output_name 1463  --images ../../resnet_input_images.txt

Start processing:
        Model name: resnet
        Images list file: ../../resnet_input_images.txt
Batch: 0; Processing time: 21.47 ms; speed 46.57 fps
output shape: (1, 1000)
         1 airliner 404 ; Correct match.
Batch: 1; Processing time: 22.43 ms; speed 44.58 fps
output shape: (1, 1000)
         2 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Batch: 2; Processing time: 24.57 ms; speed 40.71 fps
output shape: (1, 1000)
         3 bee 309 ; Correct match.
Batch: 3; Processing time: 30.92 ms; speed 32.34 fps
output shape: (1, 1000)
         4 golden retriever 207 ; Correct match.
Batch: 4; Processing time: 30.70 ms; speed 32.58 fps
output shape: (1, 1000)
         5 gorilla, Gorilla gorilla 366 ; Correct match.
Batch: 5; Processing time: 30.26 ms; speed 33.05 fps
output shape: (1, 1000)
         6 magnetic compass 635 ; Correct match.
Batch: 6; Processing time: 31.26 ms; speed 31.99 fps
output shape: (1, 1000)
         7 peacock 84 ; Correct match.
Batch: 7; Processing time: 29.17 ms; speed 34.28 fps
output shape: (1, 1000)
         8 pelican 144 ; Correct match.
Batch: 8; Processing time: 25.83 ms; speed 38.71 fps
output shape: (1, 1000)
         9 snail 113 ; Correct match.
Batch: 9; Processing time: 32.50 ms; speed 30.77 fps
output shape: (1, 1000)
         10 zebra 340 ; Correct match.
Overall accuracy= 100.0 %
Average latency= 27.4 ms
```
