# Model Ensemble Pipeline Demo {#ovms_docs_demo_ensemble}

This guide shows how to implement a model ensemble using the [DAG Scheduler](../../../docs/dag_scheduler.md).

- Let's consider you develop an application to perform image classification. There are many different models that can be used for this task. The goal is to combine results from inferences executed on two different models and calculate argmax to pick the most probable classification label. 
- For this task, select two models: [googlenet-v2](https://docs.openvino.ai/2023.0/omz_models_model_googlenet_v2_tf.html) and [resnet-50](https://docs.openvino.ai/2022.2/omz_models_model_resnet_50_tf.html#doxid-omz-models-model-resnet-50-tf). Additionally, create own model **argmax** to combine and select top result. The aim is to perform this task on the server side with no intermediate results passed over the network. The server should take care of feeding inputs/outputs in subsequent models. Both - googlenet and resnet predictions should run in parallel. 
- Diagram for this pipeline would look like this: 

![diagram](model_ensemble_diagram.svg)

## Step 1: Prepare the repository

Clone the repository and enter model_ensemble directory
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/model_ensemble/python
```

Repository preparation is simplified with `make` script, just run `make` in this repository.
```bash
make
```

The steps in `Makefile` are:

1. Download and use the models from [open model zoo](https://github.com/openvinotoolkit/open_model_zoo).
2. Use [python script](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/tests/models/argmax_sum.py) located in this repository. Since it uses tensorflow to create models in _saved model_ format, hence tensorflow pip package is required.
3. Prepare argmax model with `(1, 1001)` input shapes to match output of the googlenet and resnet output shapes. The generated model will sum inputs and calculate the index with the highest value. The model output will indicate the most likely predicted class from the ImageNet* dataset.
4. Convert models to IR format and [prepare models repository](../../../docs/models_repository.md).

```bash
...
models
├── argmax
│   └── 1
│       ├── saved_model.bin
│       ├── saved_model.mapping
│       └── saved_model.xml
├── config.json
├── googlenet-v2-tf
│   └── 1
│       ├── googlenet-v2-tf.bin
│       ├── googlenet-v2-tf.mapping
│       └── googlenet-v2-tf.xml
└── resnet-50-tf
    └── 1
        ├── resnet-50-tf.bin
        ├── resnet-50-tf.mapping
        └── resnet-50-tf.xml

6 directories, 10 files
```

## Step 2: Define required models and pipeline <a name="define-models"></a>
Pipelines need to be defined in the configuration file to use them. The same configuration file is used to define served models and served pipelines.

Use the [config.json located here](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/demos/model_ensemble/python/config.json), the content is as follows:
```bash
cat config.json
{
    "model_config_list": [
        {
            "config": {
                "name": "googlenet",
                "base_path": "/models/googlenet-v2-tf"
            }
        },
        {
            "config": {
                "name": "resnet",
                "base_path": "/models/resnet-50-tf"
            }
        },
        {
            "config": {
                "name": "argmax",
                "base_path": "/models/argmax"
            }
        }
    ],
    "pipeline_config_list": [
        {
            "name": "image_classification_pipeline",
            "inputs": ["image"],
            "nodes": [
                {
                    "name": "googlenet_node",
                    "model_name": "googlenet",
                    "type": "DL model",
                    "inputs": [
                        {"input": {"node_name": "request",
                                   "data_item": "image"}}
                    ], 
                    "outputs": [
                        {"data_item": "InceptionV2/Predictions/Softmax",
                         "alias": "probability"}
                    ] 
                },
                {
                    "name": "resnet_node",
                    "model_name": "resnet",
                    "type": "DL model",
                    "inputs": [
                        {"map/TensorArrayStack/TensorArrayGatherV3": {"node_name": "request",
                                                                      "data_item": "image"}}
                    ], 
                    "outputs": [
                        {"data_item": "softmax_tensor",
                         "alias": "probability"}
                    ] 
                },
                {
                    "name": "argmax_node",
                    "model_name": "argmax",
                    "type": "DL model",
                    "inputs": [
                        {"input1": {"node_name": "googlenet_node",
                                    "data_item": "probability"}},
                        {"input2": {"node_name": "resnet_node",
                                    "data_item": "probability"}}
                    ], 
                    "outputs": [
                        {"data_item": "argmax:0",
                         "alias": "most_probable_label"}
                    ] 
                }
            ],
            "outputs": [
                {"label": {"node_name": "argmax_node",
                           "data_item": "most_probable_label"}}
            ]
        }
    ]
}
```
In the `model_config_list` section, three models are defined as usual. We can refer to them by name in the pipeline definition but we can also request single inference on them separately. The same inference gRPC and REST API is used to request models and pipelines. OpenVINO&trade; Model Server will first try to search for a model with the requested name. If not found, it will try to find pipeline.


## Step 3: Start the Model Server

Run command to start the Model Server
```bash
docker run --rm -v $(pwd)/models/:/models:ro -p 9100:9100 -p 8100:8100 openvino/model_server:latest --config_path /models/config.json --port 9100 --rest_port 8100 --log_level DEBUG
```

## Step 4: Requesting the service

Input images can be sent to the service requesting resource name `image_classification_pipeline`. There is an example client doing that:

Check accuracy of the pipeline by running the client in another terminal:
```bash
cd ../../../client/python/tensorflow-serving-api/samples
virtualenv .venv
. .venv/bin/activate && pip3 install -r requirements.txt
python3 grpc_predict_resnet.py --pipeline_name image_classification_pipeline --images_numpy_path ../../imgs.npy \
    --labels_numpy_path ../../lbs.npy --grpc_port 9100 --input_name image --output_name label --transpose_input True --transpose_method nchw2nhwc --iterations 10
Image data range: 0.0 : 255.0
Start processing:
        Model name: image_classification_pipeline
        Iterations: 10
        Images numpy path: ../../imgs.npy
        Numpy file shape: (10, 224, 224, 3)

Iteration 1; Processing time: 33.51 ms; speed 29.85 fps
imagenet top results in a single batch:
response shape (1,)
         0 airliner 404 ; Correct match.
Iteration 2; Processing time: 42.52 ms; speed 23.52 fps
imagenet top results in a single batch:
response shape (1,)
         0 Arctic fox, white fox, Alopex lagopus 279 ; Correct match.
Iteration 3; Processing time: 34.42 ms; speed 29.05 fps
imagenet top results in a single batch:
response shape (1,)
         0 bee 309 ; Correct match.
Iteration 4; Processing time: 32.34 ms; speed 30.92 fps
imagenet top results in a single batch:
response shape (1,)
         0 golden retriever 207 ; Correct match.
Iteration 5; Processing time: 35.92 ms; speed 27.84 fps
imagenet top results in a single batch:
response shape (1,)
         0 gorilla, Gorilla gorilla 366 ; Correct match.
Iteration 6; Processing time: 33.63 ms; speed 29.74 fps
imagenet top results in a single batch:
response shape (1,)
         0 magnetic compass 635 ; Correct match.
Iteration 7; Processing time: 37.22 ms; speed 26.86 fps
imagenet top results in a single batch:
response shape (1,)
         0 peacock 84 ; Correct match.
Iteration 8; Processing time: 35.84 ms; speed 27.90 fps
imagenet top results in a single batch:
response shape (1,)
         0 pelican 144 ; Correct match.
Iteration 9; Processing time: 33.69 ms; speed 29.68 fps
imagenet top results in a single batch:
response shape (1,)
         0 snail 113 ; Correct match.
Iteration 10; Processing time: 46.54 ms; speed 21.49 fps
imagenet top results in a single batch:
response shape (1,)
         0 zebra 340 ; Correct match.

processing time for all iterations
average time: 36.00 ms; average speed: 27.78 fps
median time: 34.50 ms; median speed: 28.99 fps
max time: 46.00 ms; min speed: 21.74 fps
min time: 32.00 ms; max speed: 31.25 fps
time percentile 90: 42.40 ms; speed percentile 90: 23.58 fps
time percentile 50: 34.50 ms; speed percentile 50: 28.99 fps
time standard deviation: 4.31
time variance: 18.60
Classification accuracy: 100.00
```

## Step 5: Analyze pipeline execution in server logs

By analyzing debug logs and timestamps it is seen that GoogleNet and ResNet model inferences were started in parallel. Just after all inputs became ready - argmax node has started its job.
```bash
docker logs <container_id>
[2022-02-28 11:30:20.159][485][serving][debug][prediction_service.cpp:69] Processing gRPC request for model: image_classification_pipeline; version: 0
[2022-02-28 11:30:20.159][485][serving][debug][prediction_service.cpp:80] Requested model: image_classification_pipeline does not exist. Searching for pipeline with that name...
[2022-02-28 11:30:20.160][485][dag_executor][debug][pipeline.cpp:83] Started execution of pipeline: image_classification_pipeline
[2022-02-28 11:30:20.160][485][serving][debug][modelmanager.cpp:1280] Requesting model: resnet; version: 0.
[2022-02-28 11:30:20.160][485][serving][debug][modelmanager.cpp:1280] Requesting model: googlenet; version: 0.
[2022-02-28 11:30:20.194][485][serving][debug][modelmanager.cpp:1280] Requesting model: argmax; version: 0.
```

## Step 6: Requesting pipeline metadata

We can use the same gRPC/REST example client as we use for requesting model metadata. The only difference is we specify pipeline name instead of the model name.
```bash
python3 grpc_get_model_metadata.py --grpc_port 9100 --model_name image_classification_pipeline
Getting model metadata for model: image_classification_pipeline
Inputs metadata:
        Input name: image; shape: [1, 224, 224, 3]; dtype: DT_FLOAT
Outputs metadata:
        Output name: label; shape: [1]; dtype: DT_INT64
```
