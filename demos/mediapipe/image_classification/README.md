# MediaPipe Image Classification Demo {#ovms_docs_demo_mediapipe_image_classification}

This guide shows how to implement [MediaPipe](../../../docs/mediapipe.md) graph using OVMS.

Example usage of graph that contains only one model - resnet:

## Prepare the repository

Clone the repository and enter mediapipe image_classification directory
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/mediapipe/image_classification
```

## Download ResNet50 model

```bash
mkdir -p model/1
wget -P model/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin
wget -P model/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml
```

## Run OpenVINO Model Server
```bash
docker run -d -v $PWD:/mediapipe -p 9000:9000 openvino/model_server:latest --config_path /mediapipe/config.json --port 9000
```

## Run the client:
```bash
cd model_server/client/python/kserve-api/samples

python grpc_infer_resnet.py --model_name resnetMediapipe --grpc_port 9008 --images_numpy_path . --transpose_input False
./../imgs.npy --input_name in --output_name out --labels_numpy_path ../../lbs.npy
Image data range: 0.0 : 255.0
Start processing:
        Model name: resnetMediapipe
        Iterations: 10
        Images numpy path: ../../imgs.npy
        Numpy file shape: (10, 3, 224, 224)

Iteration 1; Processing time: 14.40 ms; speed 69.46 fps
imagenet top results in a single batch:
         0 airliner 404 ; Correct match.
Iteration 2; Processing time: 10.72 ms; speed 93.32 fps
imagenet top results in a single batch:
         0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Iteration 3; Processing time: 9.27 ms; speed 107.83 fps
imagenet top results in a single batch:
         0 bee 309 ; Correct match.
Iteration 4; Processing time: 8.47 ms; speed 118.02 fps
imagenet top results in a single batch:
         0 golden retriever 207 ; Correct match.
Iteration 5; Processing time: 9.17 ms; speed 109.03 fps
imagenet top results in a single batch:
         0 gorilla, Gorilla gorilla 366 ; Correct match.
Iteration 6; Processing time: 8.56 ms; speed 116.78 fps
imagenet top results in a single batch:
         0 magnetic compass 635 ; Correct match.
Iteration 7; Processing time: 8.39 ms; speed 119.16 fps
imagenet top results in a single batch:
         0 peacock 84 ; Correct match.
Iteration 8; Processing time: 8.44 ms; speed 118.44 fps
imagenet top results in a single batch:
         0 pelican 144 ; Correct match.
Iteration 9; Processing time: 8.36 ms; speed 119.55 fps
imagenet top results in a single batch:
         0 snail 113 ; Correct match.
Iteration 10; Processing time: 9.16 ms; speed 109.19 fps
imagenet top results in a single batch:
         0 zebra 340 ; Correct match.

processing time for all iterations
average time: 9.10 ms; average speed: 109.89 fps
median time: 8.50 ms; median speed: 117.65 fps
max time: 14.00 ms; min speed: 71.43 fps
min time: 8.00 ms; max speed: 125.00 fps
time percentile 90: 10.40 ms; speed percentile 90: 96.15 fps
time percentile 50: 8.50 ms; speed percentile 50: 117.65 fps
time standard deviation: 1.76
time variance: 3.09
Classification accuracy: 100.00
```
