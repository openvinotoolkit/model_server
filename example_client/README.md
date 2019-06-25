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
                        Set to False to skip NHWC->NCHW input transposing.
                        default: True
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

Refer also to the usage demo in the [jupyter notebook](../example_k8s/OVMS_demo.ipynb).

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
python rest_serving_client.py --images_numpy_path imgs.npy --output_name outputs --input_name in --transpose_input True --labels_numpy_path lbs.npy --rest_port 8500  --batchsize 1 --request_format row_noname
Image data range: 0.0 : 255.0
Start processing:
	Model name: resnet
	Iterations: 10
	Images numpy path: imgs.npy
	Images in shape: (10, 224, 224, 3)

Iteration 1; Processing time: 100.51 ms; speed 9.95 fps
imagenet top results in a single batch:
	 0 space shuttle 812 ; Incorrect match. Should be 404 airliner
Iteration 2; Processing time: 95.58 ms; speed 10.46 fps
imagenet top results in a single batch:
	 0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Iteration 3; Processing time: 103.17 ms; speed 9.69 fps
imagenet top results in a single batch:
	 0 hair slide 584 ; Incorrect match. Should be 309 bee
Iteration 4; Processing time: 90.85 ms; speed 11.01 fps
imagenet top results in a single batch:
	 0 clumber, clumber spaniel 216 ; Incorrect match. Should be 207 golden retriever
Iteration 5; Processing time: 89.56 ms; speed 11.17 fps
imagenet top results in a single batch:
	 0 gorilla, Gorilla gorilla 366 ; Correct match.
Iteration 6; Processing time: 96.82 ms; speed 10.33 fps
imagenet top results in a single batch:
	 0 analog clock 409 ; Incorrect match. Should be 635 magnetic compass
Iteration 7; Processing time: 90.46 ms; speed 11.06 fps
imagenet top results in a single batch:
	 0 peacock 84 ; Correct match.
Iteration 8; Processing time: 101.48 ms; speed 9.85 fps
imagenet top results in a single batch:
	 0 pelican 144 ; Correct match.
Iteration 9; Processing time: 89.02 ms; speed 11.23 fps
imagenet top results in a single batch:
	 0 snail 113 ; Correct match.
Iteration 10; Processing time: 100.40 ms; speed 9.96 fps
imagenet top results in a single batch:
	 0 zebra 340 ; Correct match.

processing time for all iterations
average time: 95.30 ms; average speed: 10.49 fps
median time: 95.50 ms; median speed: 10.47 fps
max time: 103.00 ms; max speed: 9.71 fps
min time: 89.00 ms; min speed: 11.24 fps
time percentile 90: 101.20 ms; speed percentile 90: 9.88 fps
time percentile 50: 95.50 ms; speed percentile 50: 10.47 fps
time standard deviation: 5.22
time variance: 27.21
Classification accuracy: 60.00
```