# Batch, Shape and Layout {#ovms_docs_shape_batch_layout}

## Batch Processing in OpenVINO&trade; Model Server

- `batch_size` parameter is optional. By default, the batch size is derived from the model. It is set by the model optimizer tool.
- When that parameter is set to a numerical value, it is changing the model batch size at the service start. 
It accepts also a value `auto` - this command makes the served model set the batch size automatically based on the incoming data at run time.
- Each time the input data change the batch size, the model is reloaded. It might have an extra response delay for the first request.
This feature is useful for sequential inference requests of the same batch size.

*Note:* In case of frequent batch size changes in predict requests, consider using [demultiplexing feature](./demultiplexing.md) from [Directed Acyclic Graph Scheduler](./dag_scheduler.md) which is more
performant in such situations because it is not adding an extra overhead with model reloading between requests like --batch_size auto setting. Exemplary usage of this feature can be found in [dynamic_batch_size](./dynamic_bs_demultiplexer.md) document.

- OpenVINO&trade; Model Server determines the batch size based on the size of the first dimension in the first input.
For example with the input shape (1, 3, 225, 225), the batch size is set to 1. With input shape (8, 3, 225, 225) the batch size is set to 8.

*Note:* Some models like object detection do not work correctly with batch size changed with the `batch_size` parameter. Typically those are the models,
whose output's first dimension is not representing the batch size like on the input side.
Changing batch size in this models can be done with the network reshaping by setting `shape` parameter appropriately or specify batch position using `--layout` parameter (read below).

## Model reshaping in OpenVINO&trade; Model Server
- `shape` parameter is optional and it takes precedence over the batch_size parameter. When the shape is defined as an argument,
it ignores the batch_size value.

- The shape argument can change the model enabled in the model server to fit the required parameters. It accepts 3 forms of the values:
    - `"auto"` phrase - model server will be reloading the model with the shape matching the input data matrix. 
    - a tuple e.g. `"(1,3,224,224)"` - it defines the shape to be used for all incoming requests for models with a single input
    - JSON object e.g. `{"input1":"(1,3,224,224)","input2":"(1,3,50,50)"}` - it defines a shape of every included input in the model

*Note:* Some models do not support the reshape operation. Learn more about supported model graph layers including all limitations
on [Shape Inference Document](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_ShapeInference.html).
In case the model can't be reshaped, it will remain in the original parameters and all requests with incompatible input format
will get an error. The model server will also report such problems in the logs.

## Changing model input/output layout
Starting from 2022.1, Model Optimizer by default preserves the layout of original model. If the model uses NHWC layout before conversion to IR format, it will be preserved after conversion. Model Optimizer can also be used to add transposition step to change the layout to desired one. Models which process image data are usually exported with NCHW or NHWC layout. Image transformation libraries like OpenCV or Pillow use NHWC layout. To minimize amount of overhead caused by transposition operation, it is suggested to export your model with NHWC layout.

In case you already have model with NCHW layout and are not willing to re-export it, OpenVINO™ Model Server allows changing input/output layout at runtime with `--layout` parameter via CLI or `config.json`. Please note that it modifies the model by adding transposition operations as pre-processing step.

Layout parameter is optional. By default layout is inherited from OpenVINO™ model. You can use this parameter to adjust models both in ONNX and Intermediate Representation format. In case no layout was specified during model export phase, the default layout OpenVINO™ Model Server sets is `N...`. It means that the only known dimensions is the first one - batch (`N`), and there is undefined amount of dimensions following the batch.

Layout change is supported for variety of combinations accepting following characters: `N`, `C`, `H`, `W`, `D`, `?`, `...`. Each dimension can appear only once. The exception is `?` which can appear multiple times meaning that there is unknown dimension on given position. `...` means that there is unknown number of dimensions in between given dimensions.

Examples:
- `NHW` - means there are 3 dimensions: _batch_, _height_ and _width_.
- `N??C` - means there are 4 dimensions: first being _batch_, 4th being the _channels_ and 2 unknown dimensions on second and third position
- `NC...W` - means there are undefined amount of dimensions: first being _batch_, second being the _channels_ and the last dimension is _width_.

You can specify 2 forms of values:
  * string - e.g. `NCHW` (which expands to `NCHW:NCHW`) or `NHWC:NCHW`; both applicable only for models with single input tensor
  * JSON object - e.g. `{"input1":"NHWC:NCHW", "input2":"NHWC:NCHW", "output1":"CN:NC"}`; allows to specify layout for multiple inputs and outputs by name.

The `<target_layout>:<source_layout>` notation means:
- `<target_layout>` request input data layout
- `<source_layout>` layout expected by model

After the model layout is changed, the requests must match the new, (updated by transposition) shape matching `<target_layout>` parameter.

Example, given the parameter `--layout NHWC:NCHW`:

| | Inherited from model  | New expected metadata |
|---|---|---|
| shape | `(1, 3, 224, 224)` | `(1, 224, 224, 3)`  |
| layout | `N...` | `NHWC` |

This is also possible to omit the colon (`:`) and pass single layout parameter: e.g. `--layout CN`. This does not add transposition step, but allows guiding Model Server to treat inputs as if batch size was on second position. This is significant when exported model has batch on arbitrary position and `--batch_size auto` is used, which makes the model reload with new batch size to match request batch dimension.

### Important
Changing layout is not supported for models with input names the same as output names. <br>
For model included in DAG, layouts of subsequent nodes must match, similarly to network shape and precision.

> **WARNING**: Beginning with 2022.1 release, the `--layout` parameter has changed meaning and the setting is not back compatible. Previously to change from `NCHW` to `NHWC` it was required to pass `--layout NHWC`, now it is required to pass `--layout NHWC:NCHW`. The prior version does not add preprocessing step and just informs about incorrect layout of exported model.
