# Custom node for OCR implementation with east-resnet50 and crnn models 

This custom node analyses the response of east-resnet50 model. Based on the inference results and the original image,
it generates a list of detected boxes for text recognition. 
Each image in the output will be resized to the predefined target size to fit the next inference model in the 
DAG pipeline.
Additionally to the detected text boxes, in the two additional outputs are returned their coordinates with information about geometry
and confidence levels for the filtered list of detections.  

**NOTE** Exemplary [configuration file](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/demos/optical_character_recognition/python/config.json) is available in [optical character recognition demo](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/demos/optical_character_recognition/python/).

# Building custom node library

You can build the shared library of the custom node simply by running command in the context of custom node examples directory:
```bash
git clone https://github.com/openvinotoolkit/model_server && cd model_server/src/custom_nodes
make NODES=east_ocr
```
It will compile the library inside a docker container and save the results in `lib/<OS>/` folder.

You can also select base OS between RH 8.5 (redhat) and Ubuntu 20.04 (ubuntu) by setting `BASE_OS` environment variable.
```bash
make BASE_OS=redhat NODES=east_ocr
```

# Custom node inputs

| Input name       | Description           | Shape  | Precision |
| ------------- |:-------------:| -----:| ------:|
| image      | Input image in an array format. Only batch size 1 is supported and images must have 3 channels. Resolution is configurable via parameters original_image_width and original_image_height. | `1,3,H,W` | FP32 |
| scores      | east-resnet50 model output `feature_fusion/Conv_7/Sigmoid` | `1,256,480,1` | FP32 |
| geometry | east-resnet50 model output `feature_fusion/concat_3` | `1,256,480,5` | FP32 |


# Custom node outputs

| Output name        | Description           | Shape  | Precision |
| ------------- |:-------------:| -----:| -------:|
| text_images      | Returns images representing detected text boxes. Boxes are filtered based on confidence_threshold and overlap_threshold params. Resolution is defined by the node parameters. All images are in a single batch. Batch size depend on the number of detected objects.  | `N,1,C,H,W` | FP32 |
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
| overlap_threshold | a ratio in a range of 0-1 for non-max suppression algorithm. Defines the overlapping ratio to reject detection as duplicated  | 0.3 | |
| debug  | Defines if debug messages should be displayed | false | |
| max_output_batch  | Prevents too big batches with incorrect confidence level. It can avoid exceeding RAM resources | 100 | |
| box_width_adjustment | Horizontal size expansion level for text images to compensate cut letter. Letters might be cut on the edges in case of the EAST model accuracy problems. That parameter defines how much horizontal size should be expanded comparing to the original width | 0 | |
| box_height_adjustment | Vertical size expansion level for text images to compensate cut letter. Letters might be cut on the edges in case of the EAST model accuracy problems. That parameter defines how much vertical size should be expanded comparing to the original height | 0 | |
| rotation_angle_threshold | For detections with angled text boxes node applies rotation to display text vertically. Parameters allows disabling rotation for angles below this value.  | 0 | |
