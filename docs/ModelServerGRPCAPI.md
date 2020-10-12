# OpenVINO&trade; Model Server gRPC API Documentation

## Introduction 
This documents gives information about OpenVINO&trade; Model Server gRPC API. It is documented in the proto buffer files in [tensorflow_serving_api](https://github.com/tensorflow/serving/tree/r1.14/tensorflow_serving/apis). 
Using the gRPC interface is recommended for optimal performance due to its faster implementation of input data deserialization. gRPC achieves lower latency, especially with larger input messages like images. 

This document covers following API:
* <a href="#model-status">Model Status API</a>
* <a href="#model-metadata">Model MetaData API </a>
* <a href="#predict">Predict API </a>


> **Note:** The implementations for *Predict*, *GetModelMetadata* and *GetModelStatus* function calls are currently available. 
These are the most generic function calls and should address most of the usage scenarios.



## Model Status API <a name="model-status"></a>

- Description

Gets information about the status of served models including Model Version

> **Note:** [get model status function spec](https://github.com/tensorflow/serving/blob/r1.14/tensorflow_serving/apis/get_model_status.proto) can be used to report all exposed versions including their state in their lifecycle.

- Command

```bash
python get_model_status.py --help
usage: get_model_status.py [-h] [--grpc_address GRPC_ADDRESS]
                           [--grpc_port GRPC_PORT] [--model_name MODEL_NAME]
                           [--model_version MODEL_VERSION]

```

- Optional Arguments

| Argument      | Description |
| :---        |    :----   |
| -h, --help       | Show help message and exit       |
| --grpc_address GRPC_ADDRESS   |   Specify url to grpc service. Default:localhost      |
| --grpc_port GRPC_PORT | Specify port to grpc service. Default: 9000 |
| --model_name MODEL_NAME | Model name to query. Default: resnet | 
| --model_version MODEL_VERSION | Model version to query. Lists all versions if not specified |

- Sample Response

```bash
Getting model status for model: vehicle-detection

Model version: 1
State AVAILABLE
Error code:  0
Error message:
```

- Usage Example

```bash
python get_model_status.py --grpc_port 9000 --model_name resnet

Getting model status for model: resnet

Model version: 2
State AVAILABLE
Error code:  0
Error message:

Model version: 1
State AVAILABLE
Error code:  0
Error message:
```


## Model MetaData API <a name="model-metadata"></a>

- Description

Gets information about the served models. A function call GetModelMetadata accepts model spec information as input and returns Signature Definition content in the format similar to TensorFlow Serving.
 
> **Note:** [get_model_metadata function spec](https://github.com/tensorflow/serving/blob/r1.14/tensorflow_serving/apis/get_model_metadata.proto) has three message definitions: *SignatureDefMap*, *GetModelMetadataRequest*, *GetModelMetadataResponse*. 

- Command

```bash
python get_serving_meta.py --help
usage: get_serving_meta.py [-h] [--grpc_address GRPC_ADDRESS]
                           [--grpc_port GRPC_PORT] [--model_name MODEL_NAME]
                           [--model_version MODEL_VERSION]

```

- Optional Arguments

| Argument      | Description |
| :---        |    :----   |
| -h, --help       | Show this help message and exit       |
| --grpc_address GRPC_ADDRESS   |   Specify url to grpc service. Default:localhost      |
| --grpc_port GRPC_PORT | Specify port to grpc service. Default: 9000 |
| --model_name MODEL_NAME | Define model name, must be same as is in service. Default: resnet | 
| --model_version MODEL_VERSION | Define model version - must be numerical |


- Sample Response

```bash
Getting model metadata for model: vehicle-detection
Inputs metadata:
        Input name: data; shape: [1, 3, 384, 672]; dtype: DT_FLOAT
Outputs metadata:
        Output name: detection_out; shape: [1, 1, 200, 7]; dtype: DT_FLOAT
```

- Usage Example

```bash
python get_serving_meta.py --grpc_port 9001 --model_name resnet --model_version 1

Getting model metadata for model: resnet
Inputs metadata:
	Input name: data; shape: [1, 3, 224, 224]; dtype: DT_FLOAT
Outputs metadata:
	Output name: prob; shape: [1, 1000]; dtype: DT_FLOAT
```

## Predict API <a name="predict"></a>

- Description

Sends requests via TFS gRPC API using images in numpy format. It displays performance statistics and optionally the model accuracy.


> **Note:** [predict function spec](https://github.com/tensorflow/serving/blob/r1.14/tensorflow_serving/apis/predict.proto) has two message definitions: *PredictRequest* and  *PredictResponse*.  
> * *PredictRequest* specifies information about the model spec, a map of input data serialized via 
[TensorProto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto) to a string format.
> * *PredictResponse* includes a map of outputs serialized by 
[TensorProto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto) and information about the used model spec.

### **Submitting gRPC requests based on a dataset from numpy files:**

- Command

```bash
usage: grpc_serving_client.py [-h] --images_numpy_path IMAGES_NUMPY_PATH
                              [--labels_numpy_path LABELS_NUMPY_PATH]
                              [--grpc_address GRPC_ADDRESS]
                              [--grpc_port GRPC_PORT]
                              [--input_name INPUT_NAME]
                              [--output_name OUTPUT_NAME]
                              [--transpose_input {False,True}]
                              [--iterations ITERATIONS]
                              [--batchsize BATCHSIZE]
                              [--model_name MODEL_NAME]
```

- Optional Arguments

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
| --transpose_method {nchw2nhwc,nhwc2nchw} | How the input transposition should be executed: nhwc2nchw or nhwc2nchw |
| --iterations | Number of requests iterations, as default use number of images in numpy memmap. Default: 0 (consume all frames)|
| --batchsize | Number of images in a single request. Default: 1 |
| --model_name | Define model name, must be same as is in service. Default: resnet|

- Sample Response

```bash
Iteration 1; Processing time: 71.97 ms; speed 13.89 fps
imagenet top results in a single batch:
	 0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
```

- Usage example

```bash
python grpc_serving_client.py --grpc_port 9001 --images_numpy_path imgs.npy --input_name data --output_name prob --transpose_input False --labels_numpy lbs.npy
Start processing:
	Model name: resnet
	Iterations: 10
	Images numpy path: imgs.npy
	Images in shape: (10, 3, 224, 224)

Iteration 1; Processing time: 55.45 ms; speed 18.03 fps
imagenet top results in a single batch:
	 0 warplane, military plane 895 ; Incorrect match. Should be 404 airliner
Iteration 2; Processing time: 71.97 ms; speed 13.89 fps
imagenet top results in a single batch:
	 0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Iteration 3; Processing time: 69.82 ms; speed 14.32 fps
imagenet top results in a single batch:
	 0 bee 309 ; Correct match.
Iteration 4; Processing time: 68.95 ms; speed 14.50 fps
imagenet top results in a single batch:
	 0 golden retriever 207 ; Correct match.
Iteration 5; Processing time: 49.82 ms; speed 20.07 fps
imagenet top results in a single batch:
	 0 gorilla, Gorilla gorilla 366 ; Correct match.
Iteration 6; Processing time: 56.90 ms; speed 17.58 fps
imagenet top results in a single batch:
	 0 magnetic compass 635 ; Correct match.
Iteration 7; Processing time: 122.50 ms; speed 8.16 fps
imagenet top results in a single batch:
	 0 peacock 84 ; Correct match.
Iteration 8; Processing time: 50.65 ms; speed 19.74 fps
imagenet top results in a single batch:
	 0 pelican 144 ; Correct match.
Iteration 9; Processing time: 56.45 ms; speed 17.71 fps
imagenet top results in a single batch:
	 0 snail 113 ; Correct match.
Iteration 10; Processing time: 58.95 ms; speed 16.96 fps
imagenet top results in a single batch:
	 0 zebra 340 ; Correct match.

processing time for all iterations
average time: 65.40 ms; average speed: 15.29 fps
median time: 57.00 ms; median speed: 17.54 fps
max time: 122.00 ms; max speed: 8.20 fps
min time: 49.00 ms; min speed: 20.41 fps
time percentile 90: 76.10 ms; speed percentile 90: 13.14 fps
time percentile 50: 57.00 ms; speed percentile 50: 17.54 fps
time standard deviation: 20.25
time variance: 410.04
Classification accuracy: 90.00

```

### **Submitting gRPC requests based on a dataset from a list of jpeg files:**

- Command

```bash
usage: jpeg_classification.py [-h] [--images_list IMAGES_LIST]
                              [--grpc_address GRPC_ADDRESS]
                              [--grpc_port GRPC_PORT]
                              [--input_name INPUT_NAME]
                              [--output_name OUTPUT_NAME]
                              [--model_name MODEL_NAME] [--size SIZE]
```

- Optional Argument

| Argument      | Description |
| :---        |    :----   |
| -h, --help       | Show help message and exit       |
| --images_list   |   Path to a file with a list of labeled images      |
| --grpc_address GRPC_ADDRESS | Specify url to grpc service. Default:localhost | 
| --grpc_port GRPC_PORT | Specify port to grpc service. Default: 9000 |
| --input_name | Specify input tensor name. Default: input |
| --output_name | Specify output name. Default: resnet_v1_50/predictions/Reshape_1 |
| --model_name | Define model name, must be same as is in service. Default: resnet|
| --size SIZE  | The size of the image in the model|



- Sample Response

```bash
Processing time: 73.00 ms; speed 2.00 fps 13.79
Detected: 895  Should be: 404
images/arctic-fox.jpeg (1, 3, 224, 224) ; data range: 7.0 : 255.0
```

- Usage example

```bash
python jpeg_classification.py --grpc_port 9001 --input_name data --output_name prob
	Model name: resnet
	Images list file: input_images.txt

images/airliner.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 73.00 ms; speed 2.00 fps 13.79
Detected: 895  Should be: 404
images/arctic-fox.jpeg (1, 3, 224, 224) ; data range: 7.0 : 255.0
Processing time: 52.00 ms; speed 2.00 fps 19.06
Detected: 279  Should be: 279
images/bee.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 82.00 ms; speed 2.00 fps 12.2
Detected: 309  Should be: 309
images/golden_retriever.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 86.00 ms; speed 2.00 fps 11.69
Detected: 207  Should be: 207
images/gorilla.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 65.00 ms; speed 2.00 fps 15.39
Detected: 366  Should be: 366
images/magnetic_compass.jpeg (1, 3, 224, 224) ; data range: 0.0 : 247.0
Processing time: 51.00 ms; speed 2.00 fps 19.7
Detected: 635  Should be: 635
images/peacock.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 61.00 ms; speed 2.00 fps 16.28
Detected: 84  Should be: 84
images/pelican.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 61.00 ms; speed 2.00 fps 16.41
Detected: 144  Should be: 144
images/snail.jpeg (1, 3, 224, 224) ; data range: 0.0 : 248.0
Processing time: 56.00 ms; speed 2.00 fps 17.74
Detected: 113  Should be: 113
images/zebra.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 73.00 ms; speed 2.00 fps 13.68
Detected: 340  Should be: 340

Overall accuracy= 90.0
```
## Multiple input example for HDDL

- Description

The purpose of this example is to show how to send inputs from multiple sources(cameras, video files) to a model served from
inside the OpenVINO model server(inside docker)

- Pre-requisite

To run this example you will need to run the OpenVINO hddldaemon and OpenVINO model server separately. Below are the steps
to install and run them (provided for Linux OS):
 1. Setup [OpenVINO](https://docs.openvinotoolkit.org/2019_R2/_docs_install_guides_installing_openvino_linux.html) & [HDDL](https://docs.openvinotoolkit.org/2019_R2/_docs_install_guides_installing_openvino_linux_ivad_vpu.html):

 2. Setup [OVMS](https://github.com/IntelAI/OpenVINO-model-server/blob/master/docs/docker_container.md#starting-docker-container-with-hddl)  to use HDDL:


- Command
```bash
python multi_inputs.py --help
```

- Optional Arguments

| Argument      | Description |
| :---        |    :----   |
| -h,--help       | Show help message and exit       |
| -n NETWORK_NAME, --network_name NETWORK_NAME   |   Network name      |
| -l INPUT_LAYER, --input_layer INPUT_LAYER | Input layer name |
| -o OUTPUT_LAYER, --output_layer OUTPUT_LAYER | Output layer name | 
| -d INPUT_DIMENSION, --input_dimension INPUT_DIMENSION | Input image dimension |
| -c NUM_CAMERAS, --num_cameras NUM_CAMERAS | Number of cameras to be used |
| -f FILE, --file FILE | Path to the video file |
| -i IP, --ip IP| IP address of the ovms|
| -p PORT, --port PORT | Port of the ovms |

- Sample Response

```bash
==============
TERMINAL 1: <openvino_installation_root>/openvino/inference_engine/external/hddl/bin/hddldaemon
TERMINAL 2: docker run --rm -it --privileged --device /dev/ion:/dev/ion -v /var/tmp:/var/tmp -v /opt/ml:/opt/ml -e DEVICE=HDDL
            -e FILE_SYSTEM_POLL_WAIT_SECONDS=0 -p 8001:8001 -p 9001:9001 ie-serving-py:latest /ie-serving-py/start_server.sh
            ie_serving model --model_path /opt/ml/model5 --model_name SSDMobileNet --port 9001 --rest_port 8001
TERMINAL 3: python3.6 multi_inputs.py -n SSDMobileNet -l image_tensor -o DetectionOutput -d 300 -c 1
            -f /var/repos/github/sample-videos/face-demographics-walking.mp4 -i 127.0.0.1 -p 9001

Console logs:
============
[$(levelname)s ] Video1 fps: 7, Inf fps: 7, dropped fps: 0
[$(levelname)s ] Camera0 fps: 7, Inf fps: 7, dropped fps: 0
[$(levelname)s ] Video1 fps: 7, Inf fps: 7, dropped fps: 0
[$(levelname)s ] Camera0 fps: 7, Inf fps: 7, dropped fps: 0
[$(levelname)s ] Video1 fps: 7, Inf fps: 7, dropped fps: 0
[$(levelname)s ] Camera0 fps: 8, Inf fps: 8, dropped fps: 0
[$(levelname)s ] Exiting thread 0
[$(levelname)s ] Good Bye!

```
> **NOTE:** You should also be seeing the GUI showing the video frame and bounding boxes drawn with the detected class name



## See Also

- [Example client code](https://github.com/openvinotoolkit/model_server/tree/main/example_client) shows how to use these API and submit the requests using the gRPC interface.
- [TensorFlow Serving](https://github.com/tensorflow/serving)
- [gRPC](https://grpc.io/)




 




