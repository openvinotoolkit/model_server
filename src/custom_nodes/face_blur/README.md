# Custom node for image blurring with object detection models from OpenVINO Model Zoo

This custom node analyses the response of models from OpenVINO Model Zoo. Based on the inference results and the original image, it blurs the image where boxes were detected.

# Supported models

All [OpenVINO Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/2022.1.0/models/intel) object detection models with specific output tensor `detection` with shape: `[1, 1, 200, 7]`:
- face-detection
- face-detection-adas
- face-detection-retail
- person-detection
- person-detection-adas
- person-detection-retail
- vehicle-detection
- vehicle-detection-adas
- product-detection
- person-vehicle-bike-detection
- vehicle-license-plate-detection
- pedestrian-and-vehicle-detector

**NOTE** Exemplary [configuration file](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/demos/face_blur/python/config.json) is available in [face_blur demo](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/demos/face_blur/python/).

# Building custom node library

You can build the shared library of the custom node simply by running command in the context of custom node examples directory:
```bash
git clone https://github.com/openvinotoolkit/model_server && cd model_server/src/custom_nodes
make NODES=face_blur
```
It will compile the library inside a docker container and save the results in `lib/<OS>/` folder.

You can also select base OS between RH 8.5 (redhat) and Ubuntu 20.04 (ubuntu) by setting `BASE_OS` environment variable.
```bash
make BASE_OS=redhat NODES=face_blur
```

# Custom node inputs

| Input name       | Description           | Shape | Precision |
| ------------- |:-------------:| -----:| -----:|
| image      | Input image in an array format. Only batch size 1 is supported and images must have 3 channels. Resolution is configurable via parameters `original_image_width` and `original_image_height`. Color data required only in BGR format. | `1,3,H,W` | FP32 |
| detection      | object detection model output | `1,1,200,7` | FP32 |

# Custom node outputs

| Output name        | Description           | Shape  | Precision |
| ------------- |:-------------:| -----:| -----:|
| image | Returns blurred image in place of detected boxes. Boxes are filtered based on confidence_threshold param. Resolution is defined by the node parameters.   | `N,C,H,W` | FP32  |

# Custom node parameters
Parameters can be defined in pipeline definition in OVMS configuration file. [Read more](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/docs/custom_node_development.md) about node parameters.
| Parameter        | Description           | Default  | Required |
| ------------- | ------------- | ------------- | ------------ |
| original_image_width  | Required input image width |  | &check; |
| original_image_height  | Required input image height |  | &check; |
| target_image_width | Target width of the blurred image.   |  | &check; |
| target_image_height  | Target width of the blurred image. |  | &check; |
| original_image_layout | Defines input image layout | NCHW | |
| target_image_layout | Defines the data layout of blurred image | NCHW | |
| confidence_threshold | Number in a range of 0-1 |  | &check; |
| debug  | Defines if debug messages should be displayed | false | |
| gaussian_blur_kernel_size  | Kernel size used in gaussian blur that should be positive and odd | | &check; |
