# Custom node for OCR implementation with east-resnet50 and crnn models 

This custom node analyses the response of east-resnet50 model. Based on the inference results and the original image,
it generates a list of detected boxes for text recognition. 
Each image in the output will be resized to the predefined target size to fit the next inference model in the 
DAG pipeline.
Beside the set of detected text boxes, in the second output are returned their labels with information about geometry
and scores but only for the filtered list of detections.  

# Building custom node library

You can build the shared library of the custom node simply by command in this custom node folder context:
```
make
```
It will compile the library inside a docker container and save the results in `lib` folder.

# Custom node inputs

| Input name       | Description           | Format  |
| ------------- |:-------------:| -----:|
| image      | Input image in an array format. Only batch size 1 is supported and images must have 3 channels. Resolution and layout is configurable via parameters original_image_width and original_image_height | 13HW or 1HW3, precision FP32 |
| scores      | east-resnet50 model output `feature_fusion/Conv_7/Sigmoid` | shape: [1 1 256 480], precision: FP32 |
| geometry | east-resnet50 model output `feature_fusion/concat_3` | shape: [1 5 256 480], precision: FP32 |


# Custom node outputs

| Output name        | Description           | Format  |
| ------------- |:-------------:| -----:|
| text_images      | Returns images representing detected text boxes. Boxes are filtered based on confidence_threshold and overlap_threshold params. Resolution and layout is defined by the node parameters. All images are in a single batch. Batch size depend on the number of detected objects.  | shape: [1,N,C,H,W] or [1,N,H,W,C], precision FP32 |
| text_coordinates      | For every detected box `N` the following info is added: x coordinate for the box center, y coordinate for the box center, box original width, box original height, box angle in radians | [1,N,5] |
| confidence_levels |   For every detected box `N` information about score result | [1,N,1] |

# Custom node parameters

| Parameter        | Description           | Default  |
| ------------- | ------------- | ------------- |
| original_image_width  | Required input image width | 1920 |
| original_image_height  | Required input image height | 1024 |
| target_image_width | Target width of the text boxes in output. Boxes in the original image will be resized to that value.  | 32 |
| target_image_height  | Target width of the text boxes in output. Boxes in the original image will be resized to that value. | 100 |
| convert_to_gray_scale  | Defines if output images should be in grayscale or in color  | false |
| confidence_threshold | Number in a range of 0-1 | 0.9 |
| overlap_threshold | a ratio in a range of 0-1 for non-max suppression algorithm. Defines the overlapping ratio to reject detection as duplicated  | 0.3 |
| debug  | Defines if debug messages should be displayed | false |
| max_output_batch  | Prevents too big batches with incorrect confidence level. It can avoid exceeding RAM resources | 100 |
| image_layout | Defines data layout in input image and output images. NCHW or NHWC only are supported | NCHW |