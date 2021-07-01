# Dynamic batch size with OpenVINO&trade; Model Server Demultiplexer

## Intoduction
This document guides how to configure DAG Scheduler pipeline to be able to send predict request with arbitrary batch size without model reloading.

With OpenVINO&trade; Model Server Demultiplexing infer request sent from client application can have various batch sizes and changing batch size does not require model reload.

*Note:* When using `demultiply_count` parameters, only one demultiplexer can exist in pipeline.

- Example client in python grpc_serving_client.py can be used to request the pipeline. Use `--dag-batch-size-auto` flag to add additional dimension to the input shape which is required for demultiplexing feature.

- The example uses model [resnet](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/resnet50-binary-0001/README.md).

- While using resnet model with grpc_serving_client.py the script processes the output from the server and displays the inference results using previously prepared file with labels. Inside this file each image has assigned number, which indicates the correct recognition answer.  

## Steps
Clone OpenVINO&trade; Model Server github repository and enter `model_server` directory.
#### Download the pretrained model
Download model files and store it in `models` directory
```Bash
mkdir -p models/resnet/1
curl https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml -o models/resnet/1/resnet50-binary-0001.bin -o models/resnet/1/resnet50-binary-0001.xml
```

#### Pull the latest OVMS image from dockerhub
Pull the latest version of OpenVINO&trade; Model Server from Dockerhub :
```Bash
docker pull openvino/model_server:latest
```

#### OVMS configuration file
Create new file named `config.json` :
```json
{
   "model_config_list": [
       {
           "config": {
               "name": "resnet",
               "base_path": "/models/resnet"
           }
       }
   ],
   "pipeline_config_list": [
       {
           "name": "resnet50DAG",
           "inputs": [
               "0"
           ],
           "demultiply_count" : 0,
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
}
```

#### Start ovms docker container with downloaded model
Start ovms container with image pulled in previous step and mount `models` directory :
```Bash
docker run --rm -d -v $(pwd)/models:/models -v $(pwd)/config.json:/config.json -p 9000:9000 openvino/model_server:latest --config_path config.json --port 9000
```

#### Run the client :
```Bash
cd example_client
virtualenv .venv
. .venv/bin/activate
pip install -r client_requirements.txt

python grpc_serving_client.py --images_numpy_path imgs.npy --labels_numpy_path lbs.npy --input_name 0 --output_name 1463 --model_name resnet50DAG --dag-batch-size-auto --transpose_input False --batchsize 1
```
#### Output of the script
```Bash
...
Start processing:
        Model name: resnet50DAG
        Iterations: 10
        Images numpy path: imgs.npy
        Images in shape: (10, 3, 224, 224)

Iteration 1; Processing time: 50.44 ms; speed 19.83 fps
imagenet top results in a single batch:
         0 airliner 404 ; Correct match.
Iteration 2; Processing time: 47.74 ms; speed 20.95 fps
imagenet top results in a single batch:
         0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Iteration 3; Processing time: 43.93 ms; speed 22.76 fps
imagenet top results in a single batch:
         0 bee 309 ; Correct match.
Iteration 4; Processing time: 48.04 ms; speed 20.82 fps
imagenet top results in a single batch:
         0 golden retriever 207 ; Correct match.
Iteration 5; Processing time: 45.71 ms; speed 21.88 fps
imagenet top results in a single batch:
         0 gorilla, Gorilla gorilla 366 ; Correct match.
Iteration 6; Processing time: 45.31 ms; speed 22.07 fps
imagenet top results in a single batch:
         0 magnetic compass 635 ; Correct match.
Iteration 7; Processing time: 45.71 ms; speed 21.88 fps
imagenet top results in a single batch:
         0 peacock 84 ; Correct match.
Iteration 8; Processing time: 45.31 ms; speed 22.07 fps
imagenet top results in a single batch:
         0 pelican 144 ; Correct match.
Iteration 9; Processing time: 44.89 ms; speed 22.28 fps
imagenet top results in a single batch:
         0 snail 113 ; Correct match.
Iteration 10; Processing time: 46.63 ms; speed 21.45 fps
imagenet top results in a single batch:
         0 zebra 340 ; Correct match.

processing time for all iterations
average time: 45.80 ms; average speed: 21.83 fps
median time: 45.00 ms; median speed: 22.22 fps
max time: 50.00 ms; min speed: 20.00 fps
min time: 43.00 ms; max speed: 23.26 fps
time percentile 90: 48.20 ms; speed percentile 90: 20.75 fps
time percentile 50: 45.00 ms; speed percentile 50: 22.22 fps
time standard deviation: 1.94
time variance: 3.76
Classification accuracy: 100.00
```
Each iteration presents the results of each infer request and details for each image in batch.

#### Run the client with different batch size :
```Bash
python grpc_serving_client.py --images_numpy_path imgs.npy --labels_numpy_path lbs.npy --input_name 0 --output_name 1463 --model_name resnet50DAG --dag-batch-size-auto --transpose_input False --batchsize 2
```

#### Output of the script
```Bash
Start processing:
        Model name: resnet50DAG
        Iterations: 5
        Images numpy path: imgs.npy
        Images in shape: (10, 3, 224, 224)

Iteration 1; Processing time: 51.59 ms; speed 38.77 fps
imagenet top results in a single batch:
         0 airliner 404 ; Correct match.
         1 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Iteration 2; Processing time: 48.00 ms; speed 41.67 fps
imagenet top results in a single batch:
         0 bee 309 ; Correct match.
         1 golden retriever 207 ; Correct match.
Iteration 3; Processing time: 47.77 ms; speed 41.87 fps
imagenet top results in a single batch:
         0 gorilla, Gorilla gorilla 366 ; Correct match.
         1 magnetic compass 635 ; Correct match.
Iteration 4; Processing time: 47.16 ms; speed 42.41 fps
imagenet top results in a single batch:
         0 peacock 84 ; Correct match.
         1 pelican 144 ; Correct match.
Iteration 5; Processing time: 45.93 ms; speed 43.54 fps
imagenet top results in a single batch:
         0 snail 113 ; Correct match.
         1 zebra 340 ; Correct match.

processing time for all iterations
average time: 47.40 ms; average speed: 42.19 fps
median time: 47.00 ms; median speed: 42.55 fps
max time: 51.00 ms; min speed: 39.22 fps
min time: 45.00 ms; max speed: 44.44 fps
time percentile 90: 49.40 ms; speed percentile 90: 40.49 fps
time percentile 50: 47.00 ms; speed percentile 50: 42.55 fps
time standard deviation: 1.96
time variance: 3.84
Classification accuracy: 100.00
```