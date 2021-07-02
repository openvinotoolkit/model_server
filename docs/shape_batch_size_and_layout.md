# Batch Processing in OpenVINO&trade; Model Server

- `batch_size` parameter is optional. By default, is accepted the batch size derived from the model. It is set by the model optimizer tool.
- When that parameter is set to numerical value, it is changing the model batch size at service start up. 
It accepts also a value `auto` - this special phrase make the served model to set the batch size automatically based on the incoming data at run time.
- Each time the input data change the batch size, the model is reloaded. It might have extra response delay for the first request.
This feature is useful for sequential inference requests of the same batch size.

*Note:* In case of frequent batch size changes in predict requests, consider using [demultiplexing feature](./demultiplexing.md#dynamic-batch-handling-with-demultiplexing) from [Directed Acyclic Graph Scheduler](./dag_scheduler.md) which is more
performant in such situations because it is not adding extra overhead with model reloading between requests like --batch_size auto setting. Examplary usage of this feature can be found in [dynamic_batch_size](./dynamic_batch_size.md) document.

- OpenVINO&trade; Model Server determines the batch size based on the size of the first dimension in the first input.
For example with the input shape (1, 3, 225, 225), the batch size is set to 1. With input shape (8, 3, 225, 225) the batch size is set to 8.

*Note:* Some models like object detection do not work correctly with batch size changed with `batch_size` parameter. Typically those are the models,
whose output's first dimension is not representing the batch size like on the input side.
Changing batch size in this kind of models can be done with network reshaping by setting `shape` parameter appropriately.

# Model reshaping in OpenVINO&trade; Model Server
- `shape` parameter is optional and it takes precedence over batch_size parameter. When the shape is defined as an argument,
it ignores the batch_size value.

- The shape argument can change the model enabled in the model server to fit the required parameters. It accepts 3 forms of the values:
    - "auto" phrase - model server will be reloading the model with the shape matching the input data matrix. 
    - a tuple e.g. (1,3,224,224) - it defines the shape to be used for all incoming requests for models with a single input
    - a dictionary of tuples e.g. {input1:(1,3,224,224),input2:(1,3,50,50)} - it defines a shape of every included input in the model

*Note:* Some models do not support reshape operation. Learn more about supported model graph layers including all limitations
on [Shape Inference Document](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_ShapeInference.html).
In case the model can't be reshaped, it will remain in the original parameters and all requests with incompatible input format
will get an error. The model server will also report such problem in the logs.

# Changing model input/output layout
OpenVINO models which process image data are generated via the model optimizer with NCHW layout. Image transformation libraries like OpenCV or Pillow use NHWC layout. This makes it required to transpose the data in the client application before it can be sent to OVMS. Custom node example implementations internally also use NHWC format to perform image transformations. Transposition operations increase the overall processing latency. Layout parameter reduces the latency by changing the model in runtime to accept NHWC layout instead of NCHW. That way the whole processing cycle is more effective by avoiding unnecessary data transpositions. That is especially beneficial for models with high resolution images, where data transposition could be more expensive in processing.<br><br>

Layout parameter is optional. By default layout is inherited from OpenVINO™ model. You can specify layout during conversion to IR format via Model Optimizer. You can also use this parameter for ONNX models.<br>

Layout change is only supported to `NCHW` or `NHWC`. You can specify 2 forms of values:
  * string - either `NCHW` or `NHWC`; applicable only for models with single input tensor
  * dictionary of strings - e.g. `{"input1":"NHWC", "input2":"NCHW", "output1":"NHWC"}`; allows to specify layout for multiple inputs and outputs by name.

After the model layout is changed, the requests must match the new updated shape in order NHWC instead of NCHW. For NCHW inputs it should be: `(batch, channels, height, width)` but for NHWC this is: `(batch, height, width, channels)`.

Changing layout is not supported for models with input names the same as output names.<br>
For model included in DAG, layouts of subsequent nodes must match similary to network shape and precision.
