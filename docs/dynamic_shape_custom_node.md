# Dynamic shape with custom node{#ovms_docs_dynamic_shape_custom_node}

## Introduction
This document guides how to configure a simple DAG with custom node that performs input resizing before passing data to the actual model. 

Such custom node has been created by the model server team. The description of how to build and use it is avaiable on: https://github.com/openvinotoolkit/model_server/tree/main/src/custom_nodes/image_transformation.


To show inference running on such setup let's take adventage of:

- Example client in python [face_detection.py](https://github.com/openvinotoolkit/model_server/blob/main/example_client/face_detection.py), that can be used to request inference on desired input shape.

- The example model [face_detection_retail_0004](https://docs.openvinotoolkit.org/2021.4/omz_models_model_face_detection_retail_0004.html).

- While using face_detection_retail_0004 model with face_detection.py the script loads images and resizes them to desired width and height. Then it processes the output from the server and displays the inference results by drawing bounding boxes around predicted faces. 

## Steps
Clone OpenVINO&trade; Model Server github repository and enter `model_server` directory.

#### Download the pretrained model
Download model files and store it in <models_dir> directory
```Bash
mkdir -p <models_dir>/face_detection/1
curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.xml -o models/face_detection/1/face-detection-retail-0004.bin -o models/face_detection/1/face-detection-retail-0004.xml
```

#### Pull the latest OVMS image from dockerhub
Pull the latest version of OpenVINO&trade; Model Server from Dockerhub :
```Bash
docker pull openvino/model_server:latest
```

### Build the custom node
1. Clone model server repository  
    ```
    git clone https://github.com/openvinotoolkit/model_server.git
    ```

2. Go to custom node directory
    ```
    cd model_server/src/custom_nodes/image_transformation/
    ``` 

3. Build the custom node
    ```
    make build
    ```

4. Copy the custom node to models repository
    ```
    cp lib/libcustom_node_image_transformation.so <models_dir>
    ```

#### OVMS configuration file
Create new file named `config.json` in <models_dir> :
```json
{
    "model_config_list": [
        {
            "config": {
                "name": "face_detection_retail",
                "base_path": "/models/face_detection",
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
            "inputs": ["image"],
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
                        "target_image_layout": "NCHW",
                    },
                    "inputs": [
                        {"image": {
                                "node_name": "request",
                                "data_item": "image"}}],
                    "outputs": [
                        {"data_item": "image",
                            "alias": "transformed_image"}]
                },
                {
                    "name": "face_detection_node",
                    "model_name": "face_detection_retail",
                    "type": "DL model",
                    "inputs": [
                        {"input": 
                            {
                             "node_name": "image_transformation_node",
                             "data_item": "transformed_image"
                            }
                        }
                    ],
                    "outputs": [
                        {"data_item": "detection",
                         "alias": "face_detection_output"}
                    ]
                }
            ],
            "outputs": [
                {"detection": {
                        "node_name": "face_detection_node",
                        "data_item": "face_detection_output"}}
            ]
        }
    ]
}
```

#### Start ovms docker container with downloaded model
Start ovms container with image pulled in previous step and mount <models_dir> :
```Bash
docker run --rm -d -v <models_dir>:/models -p 9000:9000 openvino/model_server:latest --config_path /models/config.json --port 9000
```

#### Run the client
```Bash
cd example_client
virtualenv .venv
. .venv/bin/activate
pip install -r client_requirements.txt
mkdir results_500x500 results_600x400

python face_detection.py --width 500 --height 500 --input_images_dir images/people --output_dir results_500x500

python face_detection.py --width 600 --height 400 --input_images_dir images/people --output_dir results_600x400
```
Results of running the client will be available in directories specified in `--output_dir`