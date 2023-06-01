# Model Server Parameters {#ovms_docs_parameters}


## Model Configuration Options

| Option  | Value format | Description |
|---|---|---|
| `"model_name"/"name"` | `string` | Model name exposed over gRPC and REST API.(use `model_name` in command line, `name` in json config)   |
| `"model_path"/"base_path"` | `string` | If using a Google Cloud Storage, Azure Storage or S3 path, see [cloud storage guide](./using_cloud_storage.md). The path may look as follows:<br>`"/opt/ml/models/model"`<br>`"gs://bucket/models/model"`<br>`"s3://bucket/models/model"`<br>`"azure://bucket/models/model"`<br>The path can be also relative to the config.json location<br>(use `model_path` in command line, `base_path` in json config)  |
| `"shape"` | `tuple/json/"auto"` | `shape` is optional and takes precedence over `batch_size`. The `shape` argument changes the model that is enabled in the model server to fit the parameters. `shape` accepts three forms of the values: * `auto` - The model server reloads the model with the shape that matches the input data matrix. * a tuple, such as `(1,3,224,224)` - The tuple defines the shape to use for all incoming requests for models with a single input. * A dictionary of shapes, such as `{"input1":"(1,3,224,224)","input2":"(1,3,50,50)", "input3":"auto"}` - This option defines the shape of every included input in the model.Some models don't support the reshape operation.If the model can't be reshaped, it remains in the original parameters and all requests with incompatible input format result in an error. See the logs for more information about specific errors.Learn more about supported model graph layers including all limitations at [Shape Inference Document](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_ShapeInference.html). |
| `"batch_size"` | `integer/"auto"` | Optional. By default, the batch size is derived from the model, defined through the OpenVINO Model Optimizer. `batch_size` is useful for sequential inference requests of the same batch size.Some models, such as object detection, don't work correctly with the `batch_size` parameter. With these models, the output's first dimension doesn't represent the batch size. You can set the batch size for these models by using network reshaping and setting the `shape` parameter appropriately.The default option of using the Model Optimizer to determine the batch size uses the size of the first dimension in the first input for the size. For example, if the input shape is `(1, 3, 225, 225)`, the batch size is set to `1`. If you set `batch_size` to a numerical value, the model batch size is changed when the service starts.`batch_size` also accepts a value of `auto`. If you use `auto`, then the served model batch size is set according to the incoming data at run time. The model is reloaded each time the input data changes the batch size. You might see a delayed response upon the first request.  |
| `"layout" `| `json/string` | `layout` is optional argument which allows to define or change the layout of model input and output tensors. To change the layout (add the transposition step), specify `<target layout>:<source layout>`. Example: `NHWC:NCHW` means that user will send input data in `NHWC` layout while the model is in `NCHW` layout.<br><br>When specified without colon separator, it doesn't add a transposition but can determine the batch dimension. E.g. `--layout CN` makes prediction service treat second dimension as batch size.<br><br>When the model has multiple inputs or the output layout has to be changed, use a json format. Set the mapping, such as: `{"input1":"NHWC:NCHW","input2":"HWN:NHW","output1":"CN:NC"}`.<br><br>If not specified, layout is inherited from model.<br><br>[Read more](shape_batch_size_and_layout.md#changing-model-inputoutput-layout) |
| `"model_version_policy"` | `json/string` | Optional. The model version policy lets you decide which versions of a model that the OpenVINO Model Server is to serve. By default, the server serves the latest version. One reason to use this argument is to control the server memory consumption.The accepted format is in json or string. Examples: <br> `{"latest": { "num_versions":2 }` <br> `{"specific": { "versions":[1, 3] } }` <br> `{"all": {} }` |
| `"plugin_config"` | `json/string`  |  List of device plugin parameters. For full list refer to [OpenVINO documentation](https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html) and [performance tuning guide](./performance_tuning.md). Example: <br> `{"PERFORMANCE_HINT": "LATENCY"}`  |
| `"nireq"` | `integer` | The size of internal request queue. When set to 0 or no value is set value is calculated automatically based on available resources.|
| `"target_device"` | `string` | Device name to be used to execute inference operations. Accepted values are: `"CPU"/"GPU"/"MULTI"/"HETERO"` |
| `"stateful"` | `bool` | If set to true, model is loaded as stateful. |
| `"idle_sequence_cleanup"` | `bool` | If set to true, model will be subject to periodic sequence cleaner scans.  See [idle sequence cleanup](stateful_models.md). |
| `"max_sequence_number"` | `uint32` | Determines how many sequences can be handled concurrently by a model instance. |
| `"low_latency_transformation"` | `bool` | If set to true, model server will apply [low latency transformation](https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_lowlatency2.html) on model load. |
| `"metrics_enable"` | `bool` | Flag enabling [metrics](https://docs.openvino.ai/2022.3/ovms_docs_metrics.html) endpoint on rest_port. |    
| `"metrics_list"` | `string` | Comma separated list of [metrics](https://docs.openvino.ai/2022.3/ovms_docs_metrics.html). If unset, only default metrics will be enabled.|



> **Note** : Specifying config_path is mutually exclusive with putting model parameters in the CLI ([serving multiple models](./starting_server.md)).

| Option  | Value format  | Description  |
|---|---|---|
| `config_path` | `string` |  Absolute path to json configuration file |

## Server configuration options

Configuration options for the server are defined only via command-line options and determine configuration common for all served models. 

| Option  | Value format  | Description  |
|---|---|---|
| `port` | `integer` | Number of the port used by gRPC sever. |
| `rest_port` | `integer` | Number of the port used by HTTP server (if not provided or set to 0, HTTP server will not be launched). |
| `grpc_bind_address` | `string` | Network interface address or a hostname, to which gRPC server will bind to. Default: all interfaces: 0.0.0.0 |
| `rest_bind_address` | `string` | Network interface address or a hostname, to which REST server will bind to. Default: all interfaces: 0.0.0.0 |
| `grpc_workers` | `integer` | Number of the gRPC server instances (must be from 1 to CPU core count). Default value is 1 and it's optimal for most use cases. Consider setting higher value while expecting heavy load. |
| `rest_workers` | `integer` | Number of HTTP server threads. Effective when `rest_port` > 0. Default value is set based on the number of CPUs. |
| `file_system_poll_wait_seconds` | `integer` | Time interval between config and model versions changes detection in seconds. Default value is 1. Zero value disables changes monitoring. |
| `sequence_cleaner_poll_wait_minutes` | `integer` | Time interval (in minutes) between next sequence cleaner scans. Sequences of the models that are subjects to idle sequence cleanup that have been inactive since the last scan are removed. Zero value disables sequence cleaner. See [idle sequence cleanup](stateful_models.md). |
| `custom_node_resources_cleaner_interval_seconds` | `integer` | Time interval (in seconds) between two consecutive resources cleanup scans. Default is 1. Must be greater than 0. See [custom node development](custom_node_development.md). |
| `cpu_extension` | `string` | Optional path to a library with [custom layers implementation](https://docs.openvino.ai/2023.0/openvino_docs_Extensibility_UG_Intro.html). |
| `log_level` | `"DEBUG"/"INFO"/"ERROR"` | Serving logging level |
| `log_path` | `string` | Optional path to the log file. |
| `cache_dir` | `string` | Path to the model cache storage. Caching will be enabled if this parameter is defined or the default path /opt/cache exists |
| `grpc_channel_arguments` | `string` |   A comma separated list of arguments to be passed to the grpc server. (e.g. grpc.max_connection_age_ms=2000) |
| `help` | `NA` |  Shows help message and exit |
| `version` | `NA` |  Shows binary version |


