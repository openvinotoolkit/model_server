# Multi Object Classification Demo {#ovms_demo_multi_object_classification_pipeline}
This document demonstrates how to create complex pipelines using object detection and object recognition models from OpenVINO Model Zoo. As an example, we will use [ssdlite_mobilenet_v2](https://github.com/openvinotoolkit/open_model_zoo/blob/2022.1.0/models/public/ssdlite_mobilenet_v2/README.md) to detect objects on the image. Then, for each detected object we will crop it using [model_zoo_intel_object_detection](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/custom_nodes/model_zoo_intel_object_detection) example custom node. Finally, each object image will be forwarded to [efficientnet-b0](https://github.com/openvinotoolkit/open_model_zoo/blob/2022.1.0/models/public/efficientnet-b0/README.md) model.

TODO: Add visualisation

Using such pipeline, a single request to OVMS can perform a complex set of operations to determine all objects and their classes.

## Pipeline Configuration Graph

Below is depicted graph implementing multi object classification pipeline execution. 

TODO: Add visualisation

It includes the following Nodes:
- Model `ssdlite_mobilenet_v2` - deep learning model which takes user image as input. Its outputs contain information about object coordinates and confidence levels.
- Custom node `model_zoo_intel_object_detection` - it includes C++ implementation of common object detection models results processing. By analysing the output it produces cropped object images based on the configurable score level threshold. Custom node also resizes them to the target resolution and combines into a single output of a dynamic batch size. The output batch size is determined by the number of detected
boxes according to the configured criteria. All operations on the images employ OpenCV libraries which are preinstalled in the OVMS. Learn more about the [model_zoo_intel_object_detection custom node](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/src/custom_nodes/model_zoo_intel_object_detection).
- demultiplexer - outputs from the custom node model_zoo_intel_object_detection have variable batch size. In order to match it with the sequential `efficientnet-b0` models, data is split into individuial images with each batch size equal to 1.
Such smaller requests can be submitted for inference in parallel to the next Model Nodes. Learn more about the [demultiplexing](../../../docs/demultiplexing.md).
- Model `efficientnet-b0` - this model classifies given object image
- Response - the output of the whole pipeline contains probability distribution across all classes for each detected object on the image. 

## Prepare workspace to run the demo

To successfully deploy multi object classification pipeline you need to have a workspace that contains:
- Deep learning models for inference
- Custom node for image processing
- Configuration file

Clone the repository and enter multi_object_classification_pipeline directory
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/multi_object_classification_pipeline/python
```

You can prepare the workspace that contains all the above by just running

```
make
```

### Final directory structure

Once the `make` procedure is finished, you should have `workspace` directory ready with the following content.
```
workspace
├── config.json
├── lib
│   └── lib
│       └── libcustom_node_model_zoo_intel_object_detection.so
└── models
    ├── efficientnet-b0
    │   └── 1
    │       ├── efficientnet-b0.bin
    │       ├── efficientnet-b0.mapping
    │       └── efficientnet-b0.xml
    └── ssdlite_mobilenet_v2
        └── 1
            ├── ssdlite_mobilenet_v2.bin
            ├── ssdlite_mobilenet_v2.mapping
            └── ssdlite_mobilenet_v2.xml
```

## Deploying OVMS

Deploy OVMS with prepared pipeline using the following command:

```bash
docker run -p 9000:9000 -d -v ${PWD}/workspace:/workspace openvino/model_server --config_path /workspace/config.json --port 9000
```

## Requesting the Service

Install python dependencies:
```bash
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
``` 

Now you can run the client:
```bash
python3 client.py --pipeline_name multi_object_classification --grpc_port 9000 --image_input_path ../../common/static/images/dogs/multiple_dogs.jpg
no: 0; label id: 220; name: Sussex spaniel
no: 1; label id: 188; name: wire-haired fox terrier
no: 2; label id: 207; name: golden retriever
no: 3; label id: 262; name: Brabancon griffon
