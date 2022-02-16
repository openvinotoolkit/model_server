# Vehicle Analysis Pipeline {#ovms_docs_demo_vehicle_analysis}

## Analyze Multiple Vehicles in a Single Image Frame
This document demonstrates how to create complex pipelines using object detection and object recognition models from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo). As an example, we use the [vehicle-detection-0202](https://github.com/openvinotoolkit/open_model_zoo/blob/2021.4/models/intel/vehicle-detection-0202/README.md) to detect multiple vehicles on the image. Next, for each detected vehicle we crop using the [model_zoo_intel_object_detection](https://github.com/openvinotoolkit/model_server/blob/v2021.4.2/src/custom_nodes/model_zoo_intel_object_detection) custom node. Finally, each vehicle image is sent to the [vehicle-attributes-recognition-barrier-0042](https://github.com/openvinotoolkit/open_model_zoo/blob/2021.4/models/intel/vehicle-attributes-recognition-barrier-0042/README.md) model.

![Vehicles analysis visualization](vehicles_analysis.png)

Using such pipeline, a single request to OVMS can perform a complex set of operations to determine all vehicles and their properties.

## Pipeline Configuration Graph

Below is depicted graph implementing vehicles analysis pipeline execution. 

![Vehicles Analysis Pipeline Graph](vehicles_analysis_graph.svg)

It includes the following Nodes:
- Model `vehicle_detection` - deep learning model which takes user image as input. Its outputs contain information about vehicle coordinates and confidence levels.
- Custom node `model_zoo_intel_object_detection` - it includes C++ implementation of common object detection models results processing. By analysing the output it produces cropped vehicle images based on the configurable score level threshold. Custom node also resizes them to the target resolution and combines into a single output of a dynamic batch size. The output batch size is determined by the number of detected
boxes according to the configured criteria. All operations on the images employ OpenCV libraries which are preinstalled in the OVMS. Learn more about the [model_zoo_intel_object_detection custom node](https://github.com/openvinotoolkit/model_server/blob/v2021.4.2/src/custom_nodes/model_zoo_intel_object_detection).
- demultiplexer - outputs from the custom node model_zoo_intel_object_detection have variable batch size. In order to match it with the sequential recognition models, data is split into individuial images with each batch size equal to 1.
Such smaller requests can be submitted for inference in parallel to the next Model Nodes. Learn more about the [demultiplexing](./demultiplexing.md).
- Model `vehicle_attributes_recognition` - this model recognizes type and color for given vehicle image
- Response - the output of the whole pipeline combines the recognized vehicle images with their metadata: coordinates, type, color, and detection confidence level. 

## Prepare the models from OpenVINO Model Zoo
### Vehicle detection model
```
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/vehicle-detection-0202/FP32/vehicle-detection-0202.xml
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/vehicle-detection-0202/FP32/vehicle-detection-0202.bin
```
### Vehicle attributes recognition model
```
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/vehicle-attributes-recognition-barrier-0042/FP32/vehicle-attributes-recognition-barrier-0042.xml
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/vehicle-attributes-recognition-barrier-0042/FP32/vehicle-attributes-recognition-barrier-0042.bin
```

## Building the Custom Node "model_zoo_intel_object_detection" Library 

Custom nodes are loaded into OVMS as dynamic libraries implementing OVMS API from [custom_node_interface.h](https://github.com/openvinotoolkit/model_server/blob/v2021.4.2/src/custom_node_interface.h).
It can use OpenCV libraries included in OVMS or it could use other thirdparty components.

The custom node `model_zoo_intel_object_detection` can be built inside a docker container via the following procedure:
- go to the custom node source code folder [src/custom_nodes/model_zoo_intel_object_detection](https://github.com/openvinotoolkit/model_server/blob/v2021.4.2/src/custom_nodes/model_zoo_intel_object_detection)
- run `make` command

This command will export the compiled library in `./lib` folder.
Copy this `lib` folder to the same location with previously downloaded models.

## OVMS Configuration File

The configuration file for running the vehicles analysis demo is stored in [config.json](https://github.com/openvinotoolkit/model_server/blob/v2021.4.2/src/custom_nodes/model_zoo_intel_object_detection/config_vehicles_example.json).
Copy this file along with the model files.

## Final directory structure
```
workspace
├── config.json
├── lib
│   └── libcustom_node_model_zoo_intel_object_detection.so
├── vehicle-attributes-recognition-barrier-0042
│   └── 1
│       ├── vehicle-attributes-recognition-barrier-0042.bin
│       └── vehicle-attributes-recognition-barrier-0042.xml
└── vehicle-detection-0202
    └── 1
        ├── vehicle-detection-0202.bin
        └── vehicle-detection-0202.xml
```

## Deploying OVMS

Deploy OVMS with vehicles analysis pipeline using the following command:

```bash
docker run -p 9000:9000 -d -v ${PWD}/workspace:/workspace openvino/model_server --config_path /workspace/config.json --port 9000
```

## Requesting the Service

Exemplary client [vehicles_analysis_pipeline_client.py](https://github.com/openvinotoolkit/model_server/blob/v2021.4.2/example_client/vehicles_analysis_pipeline_client.py) can be used to request pipeline deployed in previous step.

From the context of [example_client](https://github.com/openvinotoolkit/model_server/blob/v2021.4.2/example_client) folder install python3 requirements:
```bash
pip install -r client_requirements.txt
``` 

Now you can create a directory for text images and run the client:
```bash
mkdir results
```
```bash
python3 vehicles_analysis_pipeline_client.py --pipeline_name multiple_vehicle_recognition --grpc_port 9000 --image_input_path ./images/cars/road1.jpg --vehicle_images_output_name vehicle_images --vehicle_images_save_path ./results --image_width 512 --image_height 512 --input_image_layout NHWC
Output: name[types]
    numpy => shape[(37, 1, 4)] data[float32]
Output: name[vehicle_coordinates]
    numpy => shape[(37, 1, 4)] data[float32]
Output: name[colors]
    numpy => shape[(37, 1, 7)] data[float32]
Output: name[confidence_levels]
    numpy => shape[(37, 1, 1)] data[float32]
Output: name[vehicle_images]
    numpy => shape[(37, 1, 72, 72, 3)] data[float32]

Found 37 vehicles:
0 Type: van Color: gray
1 Type: car Color: gray
2 Type: car Color: black
3 Type: car Color: white
4 Type: car Color: black
5 Type: truck Color: red
6 Type: truck Color: gray
7 Type: car Color: white
8 Type: car Color: blue
9 Type: car Color: red
10 Type: car Color: yellow
11 Type: car Color: gray
12 Type: van Color: gray
13 Type: car Color: gray
14 Type: car Color: red
15 Type: truck Color: gray
16 Type: truck Color: red
17 Type: car Color: red
18 Type: car Color: gray
19 Type: car Color: black
20 Type: truck Color: gray
21 Type: car Color: red
22 Type: truck Color: blue
23 Type: truck Color: red
24 Type: car Color: gray
25 Type: truck Color: red
26 Type: car Color: black
27 Type: truck Color: gray
28 Type: truck Color: blue
29 Type: truck Color: gray
30 Type: car Color: gray
31 Type: car Color: white
32 Type: car Color: yellow
33 Type: car Color: red
34 Type: truck Color: gray
35 Type: truck Color: gray
36 Type: car Color: red
```

With additional parameter `--vehicle_images_save_path`, the client script saves all detected vehicle images to jpeg files into directory path to confirm
if the image was analyzed correctly.
