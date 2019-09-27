# gRPC client examples

This client can be used to demonstrate connectivity with ie-serving service over gRPC API and
 TensorFlow server using Predict call.


## Requirements

Install client pip dependencies  

## Submitting gRPC requests based on a dataset from numpy files:

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

Sends requests via TFS gRPC API using images in numpy format. It displays
performance statistics and optionally the model accuracy

optional arguments:
  -h, --help            show this help message and exit
  --images_numpy_path IMAGES_NUMPY_PATH
                        numpy in shape [n,w,h,c] or [n,c,h,w]
  --labels_numpy_path LABELS_NUMPY_PATH
                        numpy in shape [n,1] - can be used to check model
                        accuracy
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT
                        Specify port to grpc service. default: 9000
  --input_name INPUT_NAME
                        Specify input tensor name. default: input
  --output_name OUTPUT_NAME
                        Specify output name. default:
                        resnet_v1_50/predictions/Reshape_1
  --transpose_input {False,True}
                        Set to False to skip NHWC>NCHW or NCHW>NHWC input
                        transposing. default: True
  --transpose_method {nchw2nhwc,nhwc2nchw}
                        How the input transposition should be executed:
                        nhwc2nchw or nchw2nhwc
  --iterations ITERATIONS
                        Number of requests iterations, as default use number
                        of images in numpy memmap. default: 0 (consume all
                        frames)
  --batchsize BATCHSIZE
                        Number of images in a single request. default: 1
  --model_name MODEL_NAME
                        Define model name, must be same as is in service.
                        default: resnet
```
Usage example:
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

## Submitting gRPC requests based on a dataset from a list of jpeg files:

```bash
usage: jpeg_classification.py [-h] [--images_list IMAGES_LIST]
                              [--grpc_address GRPC_ADDRESS]
                              [--grpc_port GRPC_PORT]
                              [--input_name INPUT_NAME]
                              [--output_name OUTPUT_NAME]
                              [--model_name MODEL_NAME] [--size SIZE]

Do requests to ie_serving and tf_serving using images in numpy format

optional arguments:
  -h, --help            show this help message and exit
  --images_list IMAGES_LIST
                        path to a file with a list of labeled images
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT
                        Specify port to grpc service. default: 9000
  --input_name INPUT_NAME
                        Specify input tensor name. default: input
  --output_name OUTPUT_NAME
                        Specify output name. default:
                        resnet_v1_50/predictions/Reshape_1
  --model_name MODEL_NAME
                        Define model name, must be same as is in service.
                        default: resnet
  --size SIZE           The size of the image in the model
```
Usage example:
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

## Getting info about served models
```bash
python get_serving_meta.py --help
usage: get_serving_meta.py [-h] [--grpc_address GRPC_ADDRESS]
                           [--grpc_port GRPC_PORT] [--model_name MODEL_NAME]
                           [--model_version MODEL_VERSION]

Get information about served models

optional arguments:
  -h, --help            show this help message and exit
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT
                        Specify port to grpc service. default: 9000
  --model_name MODEL_NAME
                        Define model name, must be same as is in service.
                        default: resnet
  --model_version MODEL_VERSION
                        Define model version - must be numerical
```
Usage example:
```bash
python get_serving_meta.py --grpc_port 9001 --model_name resnet --model_version 1
Getting model metadata for model: resnet
Inputs metadata:
	Input name: data; shape: [1, 3, 224, 224]; dtype: DT_FLOAT
Outputs metadata:
	Output name: prob; shape: [1, 1000]; dtype: DT_FLOAT
```
```bash
python rest_get_serving_meta.py --help
usage: rest_get_serving_meta.py [-h] [--rest_url REST_URL]
                                [--rest_port REST_PORT]
                                [--model_name MODEL_NAME]
                                [--model_version MODEL_VERSION]

Get information about served models

optional arguments:
  -h, --help            show this help message and exit
  --rest_url REST_URL   Specify url to REST API service. default:
                        http://localhost
  --rest_port REST_PORT
                        Specify port to REST API service. default: 5555
  --model_name MODEL_NAME
                        Define model name, must be same as is in service.
                        default: resnet
  --model_version MODEL_VERSION
                        Define model version - must be numerical
```

```json
python rest_get_serving_meta.py --rest_port 8000
{
  "modelSpec": {
    "name": "resnet",
    "version": "1"
  },
  "metadata": {
    "signature_def": {
      "@type": "type.googleapis.com/tensorflow.serving.SignatureDefMap",
      "signatureDef": {
#
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
                    "size": "224"
                  },
                  {
                    "size": "224"
                  }
                ]
              }
            }
          },
          "outputs": {
            "prob": {
              "name": "prob_2:0",
              "dtype": "DT_FLOAT",
              "tensorShape": {
                "dim": [
                  {
                    "size": "1"
                  },
                  {
                    "size": "1000"
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

## REST API client to predict function
```bash
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

Sends requests via TensorFlow Serving RESTfull API using images in numpy
format. It displays performance statistics and optionally the model accuracy

optional arguments:
  -h, --help            show this help message and exit
  --images_numpy_path IMAGES_NUMPY_PATH
                        numpy in shape [n,w,h,c] or [n,c,h,w]
  --labels_numpy_path LABELS_NUMPY_PATH
                        numpy in shape [n,1] - can be used to check model
                        accuracy
  --rest_url REST_URL   Specify url to REST API service. default:
                        http://localhost
  --rest_port REST_PORT
                        Specify port to REST API service. default: 5555
  --input_name INPUT_NAME
                        Specify input tensor name. default: input
  --output_name OUTPUT_NAME
                        Specify output name. default:
                        resnet_v1_50/predictions/Reshape_1
  --transpose_input {False,True}
                        Set to False to skip NHWC>NCHW or NCHW>NHWC input
                        transposing. default: True
  --transpose_method {nchw2nhwc,nhwc2nchw}
                        How the input transposition should be executed:
                        nhwc2nchw or nhwc2nchw
  --iterations ITERATIONS
                        Number of requests iterations, as default use number
                        of images in numpy memmap. default: 0 (consume all
                        frames)
  --batchsize BATCHSIZE
                        Number of images in a single request. default: 1
  --model_name MODEL_NAME
                        Define model name, must be same as is in service.
                        default: resnet
  --request_format {row_noname,row_name,column_noname,column_name}
                        Request format according to TF Serving API:
                        row_noname,row_name,column_noname,column_name
  --model_version MODEL_VERSION
                        Model version to be used. Default: LATEST
```
   
```bash
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

## Getting model serving status

```bash
python get_model_status.py --help
usage: get_model_status.py [-h] [--grpc_address GRPC_ADDRESS]
                           [--grpc_port GRPC_PORT] [--model_name MODEL_NAME]
                           [--model_version MODEL_VERSION]

Get information about the status of served models

optional arguments:
  -h, --help            show this help message and exit
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT
                        Specify port to grpc service. default: 9000
  --model_name MODEL_NAME
                        Model name to query. default: resnet
  --model_version MODEL_VERSION
                        Model version to query. Lists all versions if not
                        specified
```

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

```bash
python rest_get_model_status.py --help
usage: rest_get_model_status.py [-h] [--rest_url REST_URL]
                                [--rest_port REST_PORT]
                                [--model_name MODEL_NAME]
                                [--model_version MODEL_VERSION]

Get served models status

optional arguments:
  -h, --help            show this help message and exit
  --rest_url REST_URL   Specify url to REST API service. default:
                        http://localhost
  --rest_port REST_PORT
                        Specify port to REST API service. default: 5555
  --model_name MODEL_NAME
                        Model name to query, must be same as is in service.
                        default: resnet
  --model_version MODEL_VERSION
                        Model version to query - must be numerical. List all
                        version if omitted
```

```bash
python rest_get_model_status.py --rest_port 8000 --model_version 1
{'model_version_status': [{'version': '1', 'state': 'AVAILABLE', 'status': {'error_code': 'OK', 'error_message': ''}}]}
```

# Multiple input example for HDDL
```bash
The purpose of this example is to show how send inputs from multiple sources(cameras, video files) to a model served from
inside the OpenVINO model server(inside docker)

python multi_inputs.py --help
Options:
  -h, --help            Show this help message and exit.
  -n NETWORK_NAME, --network_name NETWORK_NAME
                        Network name
  -l INPUT_LAYER, --input_layer INPUT_LAYER
                        Input layer name
  -o OUTPUT_LAYER, --output_layer OUTPUT_LAYER
                        Output layer name
  -d INPUT_DIMENSION, --input_dimension INPUT_DIMENSION
                        Input image dimension
  -c NUM_CAMERAS, --num_cameras NUM_CAMERAS
                        Number of cameras to be used
  -f FILE, --file FILE  Path to the video file
  -i IP, --ip IP        ip address of the ovms
  -p PORT, --port PORT  port of the ovms

To run this example you will need to run the OpenVINO hddldaemon and OpenVINO model server separately. Below are the steps
to install and run them(provided for Linux OS):

* Setup OpenVINO & HDDL:
  - https://docs.openvinotoolkit.org/2019_R2/_docs_install_guides_installing_openvino_linux.html
  - https://docs.openvinotoolkit.org/2019_R2/_docs_install_guides_installing_openvino_linux_ivad_vpu.html

* Setup OVMS to use HDDL:
  - https://github.com/IntelAI/OpenVINO-model-server/blob/master/docs/docker_container.md#starting-docker-container-with-hddl

Sample output:
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

You should also be seeing the GUI showing the video frame and bounding boxes drawn with the detected class name

```

Refer also to the usage demo in the [jupyter notebook](../example_k8s/OVMS_demo.ipynb).
