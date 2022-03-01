# Face Blur With Person Detection Pipeline Demo with OVMS {#ovms_demo_face_blur_with_person_detection_pipeline}

This document demonstrates how to create pipelines using object detection models from OpenVINO Model Zoo in order to blur the image. As an example, we will use [face-detection-retail-0004](https://github.com/openvinotoolkit/open_model_zoo/blob/2021.4/models/intel/face-detection-retail-0004/README.md) to detect multiple faces on the image. Then, for each detected face we will blur it using [face_blur](https://github.com/openvinotoolkit/model_server/blob/develop/src/custom_nodes/face_blur) example custom node.

Additionally the same image will be used to make an inference with [person-detection-retail-0013](https://github.com/openvinotoolkit/open_model_zoo/blob/2021.4/models/intel/person-detection-retail-0013/README.md) model to detect people on it.

## Pipeline Configuration Graph

Below is depicted graph implementing face blur with person detection pipeline execution.

![Face Blur With Person Detection Pipeline Graph](https://github.com/openvinotoolkit/model_server/blob/develop/demos/face_blur_with_person_detection/python/face_blur_with_person_detection.svg)

It include the following Nodes:
- Model `face-detection-retail-0004` - deep learning model which takes user image as input. Its output contain information about faces coordinates and confidence levels.
- Custom node `face_blur` - it includes C++ implementation of image blurring. By analysing the output it produces image blurred in spots detected by object detection model based on the configurable score level threshold. Custom node also resizes it to the target resolution. All operations on the images employ OpenCV libraries which are preinstalled in the OVMS. Learn more about the [face_blur custom node](https://github.com/openvinotoolkit/model_server/blob/develop/src/custom_nodes/face_blur).
- Model `person-detection-retail-0013` - deep learning model which takes user image as input. Its output contain information about people coordinates and confidence levels.
- Response - image blurred in spots detected by object detection model and person detection model output with detected people.

### Inputs
- `'image' with shape (1, 400, 600, 3)` - original image with `BGR` color model
### Outputs
- `'image' with shape (1, target_image_height, target_image_width, 3)` - image with blurred faces with `BGR` color model
- `'detection' with shape (1, 1, 200, 7)` - person-detection model output

**NOTE** `target_image_width` and `target_image_height` are [face_blur custom node](https://github.com/openvinotoolkit/model_server/blob/develop/src/custom_nodes/face_blur) parameters.


## Prepare workspace to run the demo

To successfully delpoy face blur pipeline you need to have a workspace that contains:
- [face-detection-retail-0004](https://github.com/openvinotoolkit/open_model_zoo/blob/2021.4/models/intel/face-detection-retail-0004/README.md)
- Custom node for image blurring
- [person-detection-retail-0013](https://github.com/openvinotoolkit/open_model_zoo/blob/2021.4/models/intel/person-detection-retail-0013/README.md)
- Configuration file

You can prepare the workspace that contains all the above by just running

```
make
```

### Final directory structure

Once the `make` procedure is finished, you should have `workspace` directory ready with the following content.
```
workspace
|── config.json
|── lib
│   └── libcustom_node_face_blur.so
|── person-detection-retail-0013
│   └── 1
│       ├── person-detection-retail-0013.bin
│       └── person-detection-retail-0013.xml
└── face-detection-retail-0004
    └── 1
        ├── face-detection-retail-0004.bin
        └── face-detection-retail-0004.xml
```

## Deploying OVMS

Deploy OVMS with face blur pipeline using the following command:

```bash
docker run -p 9000:9000 -d -v ${PWD}/workspace:/workspace openvino/model_server --config_path /workspace/config.json --port 9000
```

## Requesting the Service

Install python dependencies:
```bash
pip3 install -r requirements.txt
``` 

Now you can create a directory for output images and run the client:
```bash
mkdir results
```

```bash
python face_blur_with_person_detection.py --grpc_port 9000 --image_input_path ../../common/static/images/people/people1.jpeg --blurred_image_save_path ./results --image_width 600 --image_height 400 --image_layout NHWC --detection_image_save_path ./results
```

Examplary result of running the demo:

![Blurred Image](https://github.com/openvinotoolkit/model_server/blob/develop/demos/face_blur_with_person_detection/python/face_blur_image.jpg) ![Image With Detections](https://github.com/openvinotoolkit/model_server/blob/develop/demos/face_blur_with_person_detection/python/image_with_detections.jpg)

**NOTE** `--save_detection_with_blur True` parameter can be used to save detections with blurred faces combined.
