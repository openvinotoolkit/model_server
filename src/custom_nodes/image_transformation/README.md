# Custom node for general purpose image transformations 

This custom node takes image with dynamic shape (color, width, height) as an input. It performs multiple operations and produces an output:
- resize to desired width and height
- layout change between NCHW and NHWC
- color ordering between BGR, RGB (3 color channels) and GRAY (1 color channel)
- change data value range per channel: `[0;255]`, `[0;1]`, `[-1;1]`

Important to note that this node uses OpenCV for processing so for good performance results prefers NHWC layout.
In other cases conversion applies which reduces performance of this node.

**NOTE** Exemplary configuration files are available in [onnx model with server preprocessing demo](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/demos/using_onnx_model/python) and [config with single node](example_config.json).

# Building custom node library

You can build the shared library of the custom node simply by running command in the context of custom node examples directory:
```bash
git clone https://github.com/openvinotoolkit/model_server && cd model_server/src/custom_nodes
make NODES=image_transformation
```
It will compile the library inside a docker container and save the results in `lib/<OS>/` folder.

You can also select base OS between RH 8.5 (redhat) and Ubuntu 20.04 (ubuntu) by setting `BASE_OS` environment variable.
```bash
make BASE_OS=redhat NODES=image_transformation
```

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
| scale  | All values will be divided by this value. When `scale_values` is specified, this value is ignored. [read more](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Additional_Optimization_Use_Cases.html#specifying-mean-and-scale-values) | | |
| scale_values  | Scale values to be used for the input image per channel. Input data will be divided by those values. Values should be provided in the same order as output image color order. [read more](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Additional_Optimization_Use_Cases.html#specifying-mean-and-scale-values) | | |
| mean_values  | Mean values to be used for the input image per channel. Values will be subtracted from each input image data value. Values should be provided in the same order as output image color order. [read more](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Additional_Optimization_Use_Cases.html#specifying-mean-and-scale-values) | | |
| debug  | Defines if debug messages should be displayed | false | |

> **_NOTE:_**  Subtracting mean values is performed before division by scale values.
