# Dynamic batch size with automatic model reloading{#ovms_docs_dynamic_bs_auto_reload}

## Introduction
This document guides how to configure model to accept input data in different batch sizes. In this case it's done by reloading the model with new batch size every time it receives the request with batch size different than the one currently set. 

The enabling of dynamic batch size via model reload is as simple as setting `batch_size` parameter to `auto`. To show how to configure dynamic batch size and make use of it let's take adventage of:

- Example client in python [grpc_serving_client.py](https://github.com/openvinotoolkit/model_server/blob/main/example_client/grpc_serving_client.py), that can be used to request inference on desired batch size.

- The example model [resnet](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/resnet50-binary-0001/README.md).

- While using resnet model with grpc_serving_client.py the script processes the output from the server and displays the inference results using previously prepared file with labels. Inside this file each image has assigned number, which indicates the correct classification result.  

## Steps
Clone OpenVINO&trade; Model Server github repository and enter `model_server` directory.
```
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
```
#### Download the pretrained model
Download model files and store it in `models` directory
```Bash
mkdir -p models/resnet/1
curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml -o models/resnet/1/resnet50-binary-0001.bin -o models/resnet/1/resnet50-binary-0001.xml
```

#### Pull the latest OVMS image from dockerhub
Pull the latest version of OpenVINO&trade; Model Server from Dockerhub :
```Bash
docker pull openvino/model_server:latest
```

#### Start ovms docker container with downloaded model and dynamic batch size
Start ovms container with image pulled in previous step and mount `models` directory :
```Bash
docker run --rm -d -v $(pwd)/models:/models -p 9000:9000 openvino/model_server:latest --model_name resnet --model_path /models/resnet --batch_size auto --port 9000
```

#### Run the client
```Bash
cd example_client
virtualenv .venv
. .venv/bin/activate
pip install -r client_requirements.txt

python grpc_serving_client.py --grpc_port 9000 --images_numpy_path imgs.npy --labels_numpy_path lbs.npy --input_name 0 --output_name 1463 --model_name resnet --transpose_input False --batchsize 1 > b1.txt && python grpc_serving_client.py --grpc_port 9000 --images_numpy_path imgs.npy --labels_numpy_path lbs.npy --input_name 0 --output_name 1463 --model_name resnet --transpose_input False --batchsize 8 > b8.txt;
```
*Note:* Results of running the client will be available in .txt files in current directory.

#### Output of the script
Output with `batchsize 1` stored in `b1.txt`:
```Bash
Image data range: 0.0 : 255.0
Start processing:
	Model name: resnet50DAG
	Iterations: 10
	Images numpy path: imgs.npy
	Numpy file shape: (10, 3, 224, 224)

Iteration 1; Processing time: 21.16 ms; speed 47.25 fps
imagenet top results in a single batch:
	 0 airliner 404 ; Correct match.
Iteration 2; Processing time: 8.08 ms; speed 123.79 fps
imagenet top results in a single batch:
	 0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Iteration 3; Processing time: 104.76 ms; speed 9.55 fps
imagenet top results in a single batch:
	 0 bee 309 ; Correct match.
Iteration 4; Processing time: 8.86 ms; speed 112.83 fps
imagenet top results in a single batch:
	 0 golden retriever 207 ; Correct match.
Iteration 5; Processing time: 19.05 ms; speed 52.48 fps
imagenet top results in a single batch:
	 0 gorilla, Gorilla gorilla 366 ; Correct match.
Iteration 6; Processing time: 9.31 ms; speed 107.47 fps
imagenet top results in a single batch:
	 0 magnetic compass 635 ; Correct match.
Iteration 7; Processing time: 7.10 ms; speed 140.81 fps
imagenet top results in a single batch:
	 0 peacock 84 ; Correct match.
Iteration 8; Processing time: 6.83 ms; speed 146.50 fps
imagenet top results in a single batch:
	 0 pelican 144 ; Correct match.
Iteration 9; Processing time: 6.74 ms; speed 148.26 fps
imagenet top results in a single batch:
	 0 snail 113 ; Correct match.
Iteration 10; Processing time: 7.08 ms; speed 141.26 fps
imagenet top results in a single batch:
	 0 zebra 340 ; Correct match.

processing time for all iterations
average time: 19.50 ms; average speed: 51.28 fps
median time: 8.00 ms; median speed: 125.00 fps
max time: 104.00 ms; min speed: 9.62 fps
min time: 6.00 ms; max speed: 166.67 fps
time percentile 90: 29.30 ms; speed percentile 90: 34.13 fps
time percentile 50: 8.00 ms; speed percentile 50: 125.00 fps
time standard deviation: 28.63
time variance: 819.45
Classification accuracy: 100.00

```
Output with `batchsize 8` stored in `b8.txt`:
```Bash
Image data range: 0.0 : 255.0
Start processing:
	Model name: resnet50DAG
	Iterations: 1
	Images numpy path: imgs.npy
	Numpy file shape: (10, 3, 224, 224)

Iteration 1; Processing time: 121.12 ms; speed 66.05 fps
imagenet top results in a single batch:
	 0 airliner 404 ; Correct match.
	 1 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
	 2 bee 309 ; Correct match.
	 3 golden retriever 207 ; Correct match.
	 4 gorilla, Gorilla gorilla 366 ; Correct match.
	 5 magnetic compass 635 ; Correct match.
	 6 peacock 84 ; Correct match.
	 7 pelican 144 ; Correct match.

processing time for all iterations
average time: 121.00 ms; average speed: 66.12 fps
median time: 121.00 ms; median speed: 66.12 fps
max time: 121.00 ms; min speed: 66.12 fps
min time: 121.00 ms; max speed: 66.12 fps
time percentile 90: 121.00 ms; speed percentile 90: 66.12 fps
time percentile 50: 121.00 ms; speed percentile 50: 66.12 fps
time standard deviation: 0.00
time variance: 0.00
Classification accuracy: 100.00

```
Each iteration presents the results of each infer request and details for each image in batch.

> Note that reloading the model takes time and during the reload new requests get queued up. Therefore, frequent model reloading may negatively affect overall performance. 