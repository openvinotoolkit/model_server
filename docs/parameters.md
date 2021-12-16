# Model Server Parameters {#ovms_docs_parameters}


## Model configuration options

| Option  | Value format | Description | Required |
|---|---|---|---|
| `"model_name"/"name"` | `string` | model name exposed over gRPC and REST API.(use `model_name` in command line, `name` in json config)   | Yes |
| `"model_path"/"base_path"` | `"/opt/ml/models/model"`"gs://bucket/models/model""s3://bucket/models/model""azure://bucket/models/model" | If using a Google Cloud Storage, Azure Storage or S3 path, see the requirements below.(use `model_path` in command line, `base_path` in json config)  | Yes |
| `"shape"` | `tuple, json or "auto"` | `shape` is optional and takes precedence over `batch_size`. The `shape` argument changes the model that is enabled in the model server to fit the parameters. `shape` accepts three forms of the values:* `auto` - The model server reloads the model with the shape that matches the input data matrix.* a tuple, such as `(1,3,224,224)` - The tuple defines the shape to use for all incoming requests for models with a single input.* A dictionary of shapes, such as `{"input1":"(1,3,224,224)","input2":"(1,3,50,50)", "input3":"auto"}` - This option defines the shape of every included input in the model.Some models don't support the reshape operation.If the model can't be reshaped, it remains in the original parameters and all requests with incompatible input format result in an error. See the logs for more information about specific errors.Learn more about supported model graph layers including all limitations at [Shape Inference Document](https://docs.openvinotoolkit.org/2021.4/_docs_IE_DG_ShapeInference.html). | No |
| `"batch_size"` | `integer / "auto"` | Optional. By default, the batch size is derived from the model, defined through the OpenVINO Model Optimizer. `batch_size` is useful for sequential inference requests of the same batch size.Some models, such as object detection, don't work correctly with the `batch_size` parameter. With these models, the output's first dimension doesn't represent the batch size. You can set the batch size for these models by using network reshaping and setting the `shape` parameter appropriately.The default option of using the Model Optimizer to determine the batch size uses the size of the first dimension in the first input for the size. For example, if the input shape is `(1, 3, 225, 225)`, the batch size is set to `1`. If you set `batch_size` to a numerical value, the model batch size is changed when the service starts.`batch_size` also accepts a value of `auto`. If you use `auto`, then the served model batch size is set according to the incoming data at run time. The model is reloaded each time the input data changes the batch size. You might see a delayed response upon the first request.  | No |
| `"layout" `| `json / string` | `layout` is optional argument which allows to change the layout of model input and output tensors. Only `NCHW` and `NHWC` layouts are supported.When specified with single string value - layout change is only applied to single model input. To change multiple model inputs or outputs, you can specify json object with mapping, such as: `{"input1":"NHWC","input2":"NHWC","output1":"NHWC"}`.If not specified, layout is inherited from model. | No |
| `"model_version_policy"` | `{ "all": {} }``{ "latest": { "num_versions":2 } }``{ "specific": { "versions":[1, 3] } }` | Optional.The model version policy lets you decide which versions of a model that the OpenVINO Model Server is to serve. By default, the server serves the latest version. One reason to use this argument is to control the server memory consumption.The accepted format is in json.Examples:{"latest": { "num_versions":2 } # server will serve only two latest versions of model{"specific": { "versions":[1, 3] } } # server will serve only versions 1 and 3 of given model{"all": {} } # server will serve all available versions of given model | No |
| `"plugin_config"` | json with plugin config mappings like`{"CPU_THROUGHPUT_STREAMS": "CPU_THROUGHPUT_AUTO"}` |  List of device plugin parameters. For full list refer to [OpenVINO documentation](https://docs.openvinotoolkit.org/2021.4/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html) and [performance tuning guide](./performance_tuning.md)  | No |
| `"nireq"` | `integer` | The size of internal request queue. When set to 0 or no value is set value is calculated automatically based on available resources.| No |
| `"target_device"` | `"CPU"/"HDDL"/"GPU"/"NCS"/"MULTI"/"HETERO"` | Device name to be used to execute inference operations. Refer to AI accelerators support below. | No |
| `stateful` | `bool` | If set to true, model is loaded as stateful. | No |
| `idle_sequence_cleanup` | `bool` | If set to true, model will be subject to periodic sequence cleaner scans.  See [idle sequence cleanup](stateful_models.md#stateful_cleanup). | No |
| `max_sequence_number` | `uint32` | Determines how many sequences can be handled concurrently by a model instance. | No |
| `low_latency_transformation` | `bool` | If set to true, model server will apply [low latency transformation](https://docs.openvinotoolkit.org/2021.4/openvino_docs_IE_DG_network_state_intro.html#lowlatency_transformation) on model load. | No |

### Batch Processing in OpenVINO&trade; Model Server

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

### Model reshaping in OpenVINO&trade; Model Server
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

### Changing model input/output layout
OpenVINO models which process image data are generated via the model optimizer with NCHW layout. Image transformation libraries like OpenCV or Pillow use NHWC layout. This makes it required to transpose the data in the client application before it can be sent to OVMS. Custom node example implementations internally also use NHWC format to perform image transformations. Transposition operations increase the overall processing latency. Layout parameter reduces the latency by changing the model in runtime to accept NHWC layout instead of NCHW. That way the whole processing cycle is more effective by avoiding unnecessary data transpositions. That is especially beneficial for models with high resolution images, where data transposition could be more expensive in processing.

Layout parameter is optional. By default layout is inherited from OpenVINO™ model. You can specify layout during conversion to IR format via Model Optimizer. You can also use this parameter for ONNX models.

Layout change is only supported to `NCHW` or `NHWC`. You can specify 2 forms of values:
  * string - either `NCHW` or `NHWC`; applicable only for models with single input tensor
  * dictionary of strings - e.g. `{"input1":"NHWC", "input2":"NCHW", "output1":"NHWC"}`; allows to specify layout for multiple inputs and outputs by name.

After the model layout is changed, the requests must match the new updated shape in order NHWC instead of NCHW. For NCHW inputs it should be: `(batch, channels, height, width)` but for NHWC this is: `(batch, height, width, channels)`.

Changing layout is not supported for models with input names the same as output names.
For model included in DAG, layouts of subsequent nodes must match similary to network shape and precision.


## Server configuration options

Configuration options for server are defined only via command line options and determine configuration common for all served models. 

| Option  | Value format  | Description  | Required  |
|---|---|---|---|
| `port` | `integer` | Number of the port used by gRPC sever. | Yes |
| `rest_port` | `integer` | Number of the port used by HTTP server (if not provided or set to 0, HTTP server will not be launched). | No |
| `grpc_bind_address` | `string` | Network interface address or a hostname, to which gRPC server will bind to. Default: all interfaces: 0.0.0.0 | No |
| `rest_bind_address` | `string` | Network interface address or a hostname, to which REST server will bind to. Default: all interfaces: 0.0.0.0 | No |
| `grpc_workers` | `integer` | Number of the gRPC server instances (must be from 1 to CPU core count). Default value is 1 and it's optimal for most use cases. Consider setting higher value while expecting heavy load. | No |
| `rest_workers` | `integer` | Number of HTTP server threads. Effective when `rest_port` > 0. Default value is set based on the number of CPUs. | No |
| `file_system_poll_wait_seconds` | `integer` | Time interval between config and model versions changes detection in seconds. Default value is 1. Zero value disables changes monitoring. | No |
| `sequence_cleaner_poll_wait_minutes` | `integer` | Time interval (in minutes) between next sequence cleaner scans. Sequences of the models that are subjects to idle sequence cleanup that have been inactive since the last scan are removed. Zero value disables sequence cleaner. See [idle sequence cleanup](stateful_models.md#stateful_cleanup). | No |
| `cpu_extension` | `string` | Optional path to a library with [custom layers implementation](https://docs.openvinotoolkit.org/2021.4/openvino_docs_IE_DG_Extensibility_DG_Intro.html) (preview feature in OVMS).
| `log_level` | `"DEBUG"/"INFO"/"ERROR"` | Serving logging level | No |
| `log_path` | `string` | Optional path to the log file. | No |



