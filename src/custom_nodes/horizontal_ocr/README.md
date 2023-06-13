# Custom node for OCR implementation with horizontal-text-detection and text-recognition 

This custom node analyses the response of horizontal-text-detection model. Based on the inference results and the original image,
it generates a list of detected boxes for text recognition. 
Each image in the output will be resized to the predefined target size to fit the next inference model in the 
DAG pipeline.
Additionally to the detected text boxes, in the two additional outputs are returned their coordinates and confidence levels.  

This custom node can be used to process video frames via [camera example](../../../demos/horizontal_text_detection/python/README.md).

**NOTE** Exemplary [configuration file](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/demos/horizontal_text_detection/python/config.json) is available in [demo with camera](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/demos/horizontal_text_detection/python/).

# Building custom node library

You can build the shared library of the custom node simply by running command in the context of custom node examples directory:
```bash
git clone https://github.com/openvinotoolkit/model_server && cd model_server/src/custom_nodes
make NODES=horizontal_ocr
```
It will compile the library inside a docker container and save the results in `lib/<OS>/` folder.

You can also select base OS between RH 8.5 (redhat) and Ubuntu 20.04 (ubuntu) by setting `BASE_OS` environment variable.
```bash
make BASE_OS=redhat NODES=horizontal_ocr
```

# Custom node inputs

| Input name       | Description           | Shape  | Precision |
| ------------- |:-------------:| -----:| ------:|
| image      | Input image in an array format. Only batch size 1 is supported and images must have 3 channels. Resolution is configurable via parameters original_image_width and original_image_height. | `1,3,H,W` | FP32 |
| boxes      | horizontal-text-detection model output `boxes`, where `X` is number of boxes and second dimension contains box info and confidence level | `X,5` | FP32 |


# Custom node outputs

| Output name        | Description           | Shape  | Precision |
| ------------- |:-------------:| -----:| -------:|
| text_images      | Returns images representing detected text boxes. Boxes are filtered based on confidence_threshold param. Resolution is defined by the node parameters. All images are in a single batch. Batch size depend on the number of detected objects.  | `N,1,C,H,W` | FP32 |
| text_coordinates      | For every detected box `N` the following info is added: x coordinate for the box center, y coordinate for the box center, box original width, box original height | `N,1,4` | I32 |
| confidence_levels |   For every detected box `N` information about score result | `N,1,1` | FP32 |

# Custom node parameters

| Parameter        | Description           | Default  | Required |
| ------------- | ------------- | ------------- | ----------- |
| original_image_width  | Required input image width |  | &check; |
| original_image_height  | Required input image height |  | &check; |
| original_image_layout  | Input image layout. Possible layouts: NCHW/NHWC. When using NHWC, it is possible to accept binary inputs. | NCHW | |
| target_image_width | Target width of the text boxes in output. Boxes in the original image will be resized to that value.  |  | &check; |
| target_image_height  | Target width of the text boxes in output. Boxes in the original image will be resized to that value. |  | &check; |
| target_image_layout  | Output images layout. Possible layouts: NCHW/NHWC. When using NHWC, it is possible to accept binary inputs. | NCHW | |
| convert_to_gray_scale  | Defines if output images should be in grayscale or in color  | false | |
| confidence_threshold | Number in a range of 0-1 |  | &check; |
| debug  | Defines if debug messages should be displayed | false | |
| max_output_batch  | Prevents too big batches with incorrect confidence level. It can avoid exceeding RAM resources | 100 | |
