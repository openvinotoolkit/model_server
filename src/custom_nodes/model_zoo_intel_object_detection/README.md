# Custom node for object detection models from OpenVINO Model Zoo

This custom node analyses the response of models from OpenVINO Model Zoo. Based on the inference results and the original image,
it generates a list of detected boxes for following object recognition models. 
Each image in the output will be resized to the predefined target size to fit the input of the next model in the 
DAG pipeline.
Additionally to the detected text boxes, two additional outputs are returned - information about coordinates and confidence levels of each box detection.

# Supported models

All [OpenVINO Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel) object detection models with specific output tensor `detection` with shape: `[1, 1, 200, 7]`:
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

# Building custom node library

You can build the shared library of the custom node simply by running command in this custom node folder context:
```
make
```
It will compile the library inside a docker container and save the results in `lib` folder.

You can also select base OS between RH 8.4 (redhat), CentOS 7 (centos) and Ubuntu 20.04 (ubuntu) by setting `BASE_OS` environment variable.
```
make BASE_OS=redhat
```

# Custom node inputs

| Input name       | Description           | Shape | Precision |
| ------------- |:-------------:| -----:| -----:|
| image      | Input image in an array format. Only batch size 1 is supported and images must have 3 channels. Resolution is configurable via parameters `original_image_width` and `original_image_height`. Color data required only in BGR format. | `1,3,H,W` | FP32 |
| detection      | object detection model output | `1,1,200,7` | FP32 |


# Custom node outputs

| Output name        | Description           | Shape  | Precision |
| ------------- |:-------------:| -----:| -----:|
| images      | Returns images representing detected text boxes. Boxes are filtered based on confidence_threshold param. Resolution is defined by the node parameters. All images are in a single batch. Batch size depend on the number of detected objects.  | `N,1,C,H,W` | FP32 |
| coordinates      | For every detected box `N` the following info is added: x coordinate for the box center, y coordinate for the box center, box original width, box original height | `N,1,4` | FP32 |
| confidences |   For every detected box `N` information about confidence level (N - number of detected boxes; more about demultiplexing [here](./../../../docs/demultiplexing.md)) | `N,1,1` | FP32 |
| label_ids   |   For every detected box `N` information about label id (N - number of detected boxes; more about demultiplexing [here](./../../../docs/demultiplexing.md)) | `N,1,1` | I32 |

# Custom node parameters
Parameters can be defined in pipeline definition in OVMS configuration file. [Read more](./../../../docs/custom_node_development.md) about node parameters.

| Parameter        | Description           | Default  | Required |
| ------------- | ------------- | ------------- | ------------ |
| original_image_width  | Required input image width |  | &check; |
| original_image_height  | Required input image height |  | &check; |
| target_image_width | Target width of the boxes in output. Boxes in the original image will be resized to that value.  |  | &check; |
| target_image_height  | Target width of the boxes in output. Boxes in the original image will be resized to that value. |  | &check; |
| original_image_layout | Defines input image layout | NCHW | |
| target_image_layout | Defines the data layout of detected object images in the node output | NCHW | |
| convert_to_gray_scale  | Defines if output images should be in grayscale or in color  | false | |
| confidence_threshold | Number in a range of 0-1 |  | &check; |
| debug  | Defines if debug messages should be displayed | false | |
| max_output_batch  | Prevents too big batches with incorrect confidence level. It can avoid exceeding RAM resources | 100 | |
| filter_label_id  | For object detection models with multiple label IDs results, use this parameter to filter the ones with desired ID | | |
