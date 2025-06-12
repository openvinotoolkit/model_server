# Dynamic batch size with OpenVINO&trade; Model Server Demultiplexer {#ovms_docs_dynamic_bs_demultiplexer}

## Introduction
This document shows how to configure a DAG Scheduler pipeline to send predict requests with an arbitrary batch size without model reloading.

With OpenVINO&trade; Model Server Demultiplexing infer request sent from client application can have various batch sizes and changing batch size does not require model reload.

More information about this feature can be found in [dynamic batch size in demultiplexing](./demultiplexing.md)

> **NOTE**: Only one dynamic demultiplexer (`demultiply_count` with value `-1`) can exist in the pipeline.

- Example client in python [grpc_predict_resnet.py](https://github.com/openvinotoolkit/model_server/blob/releases/2025/2/client/python/tensorflow-serving-api/samples/grpc_predict_resnet.py) can be used to request the pipeline. Use `--dag-batch-size-auto` flag to add an additional dimension to the input shape which is required for demultiplexing feature.

- The example uses model [resnet](https://github.com/openvinotoolkit/open_model_zoo/blob/2022.1.0/models/intel/resnet50-binary-0001/README.md).

- While using resnet model with grpc_predict_resnet.py the script processes the output from the server and displays the inference results using the previously prepared file with labels. Inside this file, each image has an assigned number, which indicates the correct recognition answer.  

## Steps
Clone OpenVINO&trade; Model Server GitHub repository and enter `model_server` directory.
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
```
#### Download the pretrained model
Download model files and store them in the `models` directory
```bash
mkdir -p models/resnet/1
curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml -o models/resnet/1/resnet50-binary-0001.bin -o models/resnet/1/resnet50-binary-0001.xml

chmod -R 755 ./models
```

#### Pull the latest OVMS image from dockerhub
Pull the latest version of OpenVINO&trade; Model Server from Dockerhub :
```bash
docker pull openvino/model_server:latest
```

#### OVMS configuration file
Go to the `models` directory:
```bash
cd models
```

Create a new file named `config.json` there:
```bash
echo '{
   "model_config_list": [
       {
           "config": {
               "name": "resnet",
               "base_path": "/models/resnet",
               "plugin_config": {
                   "NUM_STREAMS" : "1"
               }
           }
       }
   ],
   "pipeline_config_list": [
       {
           "name": "resnet50DAG",
           "inputs": [
               "0"
           ],
           "demultiply_count" : -1,
           "nodes": [
               {
                   "name": "resnetNode",
                   "model_name": "resnet",
                   "type": "DL model",
                   "inputs": [
                       {
                           "0": {
                               "node_name": "request",
                               "data_item": "0"
                           }
                       }
                   ],
                   "outputs": [
                       {
                           "data_item": "1463",
                           "alias": "1463"
                       }
                   ]
               }
           ],
           "outputs": [
               {"1463": {
                       "node_name": "resnetNode",
                       "data_item": "1463"}}
           ]
       }
   ]
}' >> config.json
```

#### Start ovms docker container with downloaded model
Start ovms container with image pulled in previous step and mount `models` directory :
```bash
docker run --rm -d -v $(pwd):/models -p 9000:9000 openvino/model_server:latest --config_path /models/config.json --port 9000
```

#### Checking metadata
```bash
cd ../client/python/tensorflow-serving-api/samples
virtualenv .venv
. .venv/bin/activate
pip install -r requirements.txt

python grpc_get_model_metadata.py --grpc_port 9000 --model_name resnet50DAG
```

```bash
...
Getting model metadata for model: resnet50DAG
Inputs metadata:
        Input name: 0; shape: [-1, 1, 3, 224, 224]; dtype: DT_FLOAT
Outputs metadata:
        Output name: 1463; shape: [-1, 1, 1000]; dtype: DT_FLOAT
```

> **NOTE**: While using the dynamic batching feature both input and output shape has an additional dimension, which represents split batch size. Setting batch size parameter to `--batchsize 8` would set input shape to `[8,1,3,244,244]` and output shape to `[8,1,1000]`.

#### Run the client
```bash
python grpc_predict_resnet.py --grpc_port 9000 --images_numpy_path ../../imgs.npy --labels_numpy_path ../../lbs.npy --input_name 0 --output_name 1463 --model_name resnet50DAG --dag-batch-size-auto --transpose_input False --batchsize 1 > b1.txt && python grpc_predict_resnet.py --grpc_port 9000 --images_numpy_path ../../imgs.npy --labels_numpy_path ../../lbs.npy --input_name 0 --output_name 1463 --model_name resnet50DAG --dag-batch-size-auto --transpose_input False --batchsize 8 > b8.txt;
```
*Note:* Results of running the client will be available in .txt files in the current directory.

#### Output of the script
Output with `batchsize 1` stored in `b1.txt`:
```bash
cat b1.txt
Image data range: 0.0 : 255.0
Start processing:
	Model name: resnet50DAG
	Iterations: 10
	Images numpy path: ../../imgs.npy
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
```bash
cat b8.txt
Image data range: 0.0 : 255.0
Start processing:
	Model name: resnet50DAG
	Iterations: 1
	Images numpy path: ../../imgs.npy
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

With this feature, we were able to successfully run client simultaneously with different batch size parameters without performance impact from the model reloading.