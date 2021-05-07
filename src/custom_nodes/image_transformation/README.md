# Custom node for general purpose image transformations 

This custom node takes image with dynamic shape (color, width, height) as an input. It performs multiple operations and produces an output:
- resize to desired width and height
- layout change between NCHW and NHWC
- color ordering between BGR, RGB (3 color channels) and GRAY (1 color channel)

Important to note that this node uses OpenCV for processing so for good performance results prefers NHWC layout.
In other cases conversion applies which reduces performance of this node.
# Building custom node library

You can build the shared library of the custom node simply by running command in this custom node folder context:
```
make
```
It will compile the library inside a docker container and save the results in `lib` folder.

# Custom node inputs

| Input name       | Description           | Shape  | Precision |
| ------------- |:-------------:| -----:| ------:|
| image      | Input image in an array format. Only batch size 1 is supported. Resolution is dynamic (node takes any width and height) but should be greater than 0. 1 and 3 color channels are supported. Data might be either in NCHW or NHWC format. | `1,C,H,W` or `1,H,W,C` (configurable via parameter) | FP32 |


# Custom node outputs

| Output name        | Description           | Shape  | Precision |
| ------------- |:-------------:| -----:| -------:|
| image      | Returns image after transformation. Transformations are configurable via parameters.  | `1,C,H,W` or `1,H,W,C` (configurable via parameter) | FP32 |

# Custom node parameters

| Parameter        | Description           | Default  | Required |
| ------------- | ------------- | ------------- | ----------- |
| target_image_width  | Desired image width after transformation. If not specified, width will not be changed. |  |  |
| target_image_height  | Desired image height after transformation. If not specified, height will not be changed. |  |  |
| original_image_color_order  | Input image color order | `BGR` |  |
| target_image_color_order  | Output image color order. If specified and differs from original_image_color_order, color order conversion will be performed | `BGR` |  |
| original_image_layout  | Input image layout. This is required to determine image shape from input shape | | &check; |
| target_image_layout  | Output image layout. If specified and differs from original_image_layout, layout conversion will be performed | | |
| debug  | Defines if debug messages should be displayed | false | |
