# Preview support of KServe gRPC API{#ovms_docs_kserve}

OpenVINO Model Server 2022.2 release introduce preview of support for [KServe gRPC API](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2).

Inference supports putting tensor buffers either in `ModelInferRequest`'s [`InferTensorContents`](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/grpc_predict_v2.proto#L155) and [`raw_input_contents`](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/grpc_predict_v2.proto#L202). There is no support for BF16 data type and there is no support for using FP16 in `InferTensorContents`.
In case of sending raw images jpeg files BYTES data type should be used and data should be put in `InferTensorContents`'s `bytes_contents`.

In current release there is support for:
* Model Metadata
* Model Ready
* Inference

In current release there is no support for:
* Server Live
* Server Ready
* Server Metadata

## Introduction
This guide shows how to get model metadata and perform basic inference with [ResNet50](https://github.com/openvinotoolkit/open_model_zoo/blob/2022.1.0/models/intel/resnet50-binary-0001/README.md) model using KServe API and example client in Python.

The script `kfs_grpc_predict_resnet.py` performs image classification task using ResNet50 model.

## Steps
Clone OpenVINO&trade; Model Server GitHub repository and enter `model_server` directory.
```
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
```
#### Download the Pretrained Model
Download the model files and store them in the `models` directory
```Bash
curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml -o models/resnet/1/resnet50-binary-0001.bin -o models/resnet/1/resnet50-binary-0001.xml
```

#### Pull the Latest Model Server Image
Pull the latest version of OpenVINO&trade; Model Server from Docker Hub :
```Bash
docker pull openvino/model_server:latest
```

#### Start the Model Server Container with Downloaded Model and Dynamic Batch Size
Start the server container with the image pulled in the previous step and mount the `models` directory :
```Bash
docker run --rm -d -v $(pwd)/models:/models -p 9000:9000 openvino/model_server:latest --model_name resnet --model_path /models/resnet --batch_size auto --port 9000
```

#### Prepare virtualenv
```Bash
cd client/python/kserve-api/samples
virtualenv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

#### Run the Client to get model readiness
```Bash
python3 ./kfs_grpc_model_ready.py --grpc_port 9000 --grpc_address localhost --model_name resnet
```

#### Script Output
```Bash
Model Ready:
ready: true
```

#### Run the Client to get metadata
```Bash
python3 ./kfs_grpc_model_metadata.py --grpc_port 9000 --grpc_address localhost --model_name resnet
```

#### Script Output
```Bash
server metadata:
name: "resnet"
versions: "1"
platform: "OpenVINO"
inputs {
  name: "0"
  datatype: "FP32"
  shape: 1
  shape: 3
  shape: 224
  shape: 224
}
outputs {
  name: "1463"
  datatype: "FP32"
  shape: 1
  shape: 1000
}
```

#### Run the Client to perform inference
```Bash
python kfs_grpc_predict_resnet.py --grpc_port 9000 --images_numpy_path ../../imgs.npy --labels_numpy_path ../../lbs.npy --input_name 0 --output_name 1463 --model_name resnet --transpose_input False;
```

#### Script Output
```Bash
Image data range: 0.0 : 255.0
Start processing:
        Model name: resnet
        Iterations: 10
        Images numpy path: ../../imgs.npy
        Numpy file shape: (10, 3, 224, 224)

Iteration 1; Processing time: 29.98 ms; speed 33.36 fps
imagenet top results in a single batch:
         0 airliner 404 ; Correct match.
Iteration 2; Processing time: 23.50 ms; speed 42.56 fps
imagenet top results in a single batch:
         0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Iteration 3; Processing time: 23.42 ms; speed 42.71 fps
imagenet top results in a single batch:
         0 bee 309 ; Correct match.
Iteration 4; Processing time: 23.25 ms; speed 43.01 fps
imagenet top results in a single batch:
         0 golden retriever 207 ; Correct match.
Iteration 5; Processing time: 24.08 ms; speed 41.53 fps
imagenet top results in a single batch:
         0 gorilla, Gorilla gorilla 366 ; Correct match.
Iteration 6; Processing time: 26.38 ms; speed 37.91 fps
imagenet top results in a single batch:
         0 magnetic compass 635 ; Correct match.
Iteration 7; Processing time: 25.46 ms; speed 39.27 fps
imagenet top results in a single batch:
         0 peacock 84 ; Correct match.
Iteration 8; Processing time: 24.60 ms; speed 40.65 fps
imagenet top results in a single batch:
         0 pelican 144 ; Correct match.
Iteration 9; Processing time: 23.52 ms; speed 42.52 fps
imagenet top results in a single batch:
         0 snail 113 ; Correct match.
Iteration 10; Processing time: 22.40 ms; speed 44.64 fps
imagenet top results in a single batch:
         0 zebra 340 ; Correct match.

processing time for all iterations
average time: 24.20 ms; average speed: 41.32 fps
median time: 23.50 ms; median speed: 42.55 fps
max time: 29.00 ms; min speed: 34.48 fps
min time: 22.00 ms; max speed: 45.45 fps
time percentile 90: 26.30 ms; speed percentile 90: 38.02 fps
time percentile 50: 23.50 ms; speed percentile 50: 42.55 fps
time standard deviation: 1.94
time variance: 3.76
Classification accuracy: 100.00
```
