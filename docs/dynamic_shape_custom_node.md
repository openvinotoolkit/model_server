# Dynamic Shape with a Custom Node{#ovms_docs_dynamic_shape_custom_node}

## Introduction
This guide shows how to configure a simple Directed Acyclic Graph (DAG) with a custom node that performs input resizing before passing input data to the model.

The node below is provided as a demonstration. See instructions for how to build and use the custom node: [Image Transformation](https://github.com/openvinotoolkit/model_server/tree/main/src/custom_nodes/image_transformation).


To run inference with this setup, we will use the following:

- Example client in Python [face_detection.py](https://github.com/openvinotoolkit/model_server/blob/main/demos/face_detection/python/face_detection.py) that can be used to request inference on with the desired input shape.

- An example [face_detection_retail_0004](https://github.com/openvinotoolkit/open_model_zoo/blob/releases/2021/4/models/intel/face-detection-retail-0004/README.md) model.

When using the `face_detection_retail_0004` model with the `face_detection.py` script, images are loaded and resized to the desired width and height. Then the output from the server is processed and inference results are displayed with bounding boxes drawn around the detected faces.

## Steps
Clone OpenVINO&trade; Model Server GitHub repository and enter `model_server` directory.
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
```

#### Download the Pretrained Model
Download the model files and store them in the `models` directory:
```bash
mkdir -p models/face_detection/1
curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.xml -o models/face_detection/1/face-detection-retail-0004.bin -o models/face_detection/1/face-detection-retail-0004.xml
```

#### Pull the Latest Model Server Image
Pull the latest version of OpenVINO&trade; Model Server from Docker Hub :
```bash
docker pull openvino/model_server:latest
```

### Build a Custom Node

1. Go to the custom node examples directory:
    ```bash
    cd src/custom_nodes
    ```

3. Build the custom node:
    ```bash
    # replace to 'redhat` if using UBI base image
    export BASE_OS=ubuntu

    make BASE_OS=${BASE_OS} NODES=image_transformation
    ```

4. Copy the custom node to the `models` repository:
    ```bash
    cp lib/${BASE_OS}/libcustom_node_image_transformation.so ../../models/libcustom_node_image_transformation.so
    ```

#### Create Model Server Configuration File
Go to the `models` directory:
```bash
cd ../../models
```

Create a new file named `config.json` in the `models` directory:
```bash
echo '{
    "model_config_list": [
        {
            "config": {
                "name": "face_detection_retail",
                "base_path": "/models/face_detection"
            }
        }
    ],
    "custom_node_library_config_list": [
        {"name": "image_transformation",
            "base_path": "/models/libcustom_node_image_transformation.so"}
    ],
    "pipeline_config_list": [
        {
            "name": "face_detection",
            "inputs": ["data"],
            "nodes": [
                {
                    "name": "image_transformation_node",
                    "library_name": "image_transformation",
                    "type": "custom",
                    "params": {
                        "target_image_width": "300",
                        "target_image_height": "300",

                        "mean_values": "[123.675,116.28,103.53]",
                        "scale_values": "[58.395,57.12,57.375]",

                        "original_image_color_order": "BGR",
                        "target_image_color_order": "BGR",

                        "original_image_layout": "NCHW",
                        "target_image_layout": "NCHW"
                    },
                    "inputs": [
                        {"image": {
                                "node_name": "request",
                                "data_item": "data"}}],
                    "outputs": [
                        {"data_item": "image",
                            "alias": "transformed_image"}]
                },
                {
                    "name": "face_detection_node",
                    "model_name": "face_detection_retail",
                    "type": "DL model",
                    "inputs": [
                        {"data":
                            {
                             "node_name": "image_transformation_node",
                             "data_item": "transformed_image"
                            }
                        }
                    ],
                    "outputs": [
                        {"data_item": "detection_out",
                         "alias": "face_detection_output"}
                    ]
                }
            ],
            "outputs": [
                {"detection_out": {
                        "node_name": "face_detection_node",
                        "data_item": "face_detection_output"}}
            ]
        }
    ]
}' >> config.json
```

#### Start Model Server Container with Downloaded Model
Start the container with the image pulled in the previous step and mount `<models_dir>` :
```bash
docker run --rm -d -v ${PWD}:/models -p 9000:9000 openvino/model_server:latest --config_path /models/config.json --port 9000
```

#### Run the Client
```bash
cd ../demos/face_detection/python
virtualenv .venv
. .venv/bin/activate
pip install -r ../../common/python/requirements.txt
mkdir results_500x500

python face_detection.py --grpc_port 9000 --width 500 --height 500 --input_images_dir ../../common/static/images/people --output_dir results_500x500 --model_name face_detection

mkdir results_600x400

python face_detection.py --grpc_port 9000 --width 600 --height 400 --input_images_dir ../../common/static/images/people --output_dir results_600x400 --model_name face_detection
```
Results of running the client will be available in directories specified in `--output_dir`