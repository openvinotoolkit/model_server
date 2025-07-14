# Model Server Parameters {#ovms_docs_parameters}


## Model Configuration Options

| Option  | Value format | Description |
|---|---|---|
| `"model_name"/"name"` | `string` | Model name exposed over gRPC and REST API.(use `model_name` in command line, `name` in json config)   |
| `"model_path"/"base_path"` | `string` | If using a Google Cloud Storage, Azure Storage or S3 path, see [cloud storage guide](./using_cloud_storage.md). The path may look as follows:<br>`"/opt/ml/models/model"`<br>`"gs://bucket/models/model"`<br>`"s3://bucket/models/model"`<br>`"azure://bucket/models/model"`<br>The path can be also relative to the config.json location<br>(use `model_path` in command line, `base_path` in json config)  |
| `"shape"` | `tuple/json/"auto"` | `shape` is optional and takes precedence over `batch_size`. The `shape` argument changes the model that is enabled in the model server to fit the parameters. `shape` accepts three forms of the values: * `auto` - The model server reloads the model with the shape that matches the input data matrix. * a tuple, such as `(1,3,224,224)` - The tuple defines the shape to use for all incoming requests for models with a single input. * A dictionary of shapes, such as `{"input1":"(1,3,224,224)","input2":"(1,3,50,50)", "input3":"auto"}` - This option defines the shape of every included input in the model.Some models don't support the reshape operation.If the model can't be reshaped, it remains in the original parameters and all requests with incompatible input format result in an error. See the logs for more information about specific errors.Learn more about supported model graph layers including all limitations at [Shape Inference Document](https://docs.openvino.ai/2025/openvino-workflow/running-inference/model-input-output/changing-input-shape.html). |
| `"batch_size"` | `integer/"auto"` | Optional. By default, the batch size is derived from the model, defined through the OpenVINO Model Optimizer. `batch_size` is useful for sequential inference requests of the same batch size.Some models, such as object detection, don't work correctly with the `batch_size` parameter. With these models, the output's first dimension doesn't represent the batch size. You can set the batch size for these models by using network reshaping and setting the `shape` parameter appropriately.The default option of using the Model Optimizer to determine the batch size uses the size of the first dimension in the first input for the size. For example, if the input shape is `(1, 3, 225, 225)`, the batch size is set to `1`. If you set `batch_size` to a numerical value, the model batch size is changed when the service starts.`batch_size` also accepts a value of `auto`. If you use `auto`, then the served model batch size is set according to the incoming data at run time. The model is reloaded each time the input data changes the batch size. You might see a delayed response upon the first request.  |
| `"layout" `| `json/string` | `layout` is optional argument which allows to define or change the layout of model input and output tensors. To change the layout (add the transposition step), specify `<target layout>:<source layout>`. Example: `NHWC:NCHW` means that user will send input data in `NHWC` layout while the model is in `NCHW` layout.<br><br>When specified without colon separator, it doesn't add a transposition but can determine the batch dimension. E.g. `--layout CN` makes prediction service treat second dimension as batch size.<br><br>When the model has multiple inputs or the output layout has to be changed, use a json format. Set the mapping, such as: `{"input1":"NHWC:NCHW","input2":"HWN:NHW","output1":"CN:NC"}`.<br><br>If not specified, layout is inherited from model.<br><br> [Read more](shape_batch_size_and_layout.md#changing-model-input-output-layout) |
| `"model_version_policy"` | `json/string` | Optional. The model version policy lets you decide which versions of a model that the OpenVINO Model Server is to serve. By default, the server serves the latest version. One reason to use this argument is to control the server memory consumption.The accepted format is in json or string. Examples: <br> `{"latest": { "num_versions":2 }` <br> `{"specific": { "versions":[1, 3] } }` <br> `{"all": {} }` |
| `"plugin_config"` | `json/string`  |  List of device plugin parameters. For full list refer to [OpenVINO documentation](https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-devices.html) and [performance tuning guide](./performance_tuning.md). Example: <br> `{"PERFORMANCE_HINT": "LATENCY"}`  |
| `"nireq"` | `integer` | The size of internal request queue. When set to 0 or no value is set value is calculated automatically based on available resources.|
| `"target_device"` | `string` | Device name to be used to execute inference operations. Accepted values are: `"CPU"/"GPU"/"MULTI"/"HETERO"` |
| `"stateful"` | `bool` | If set to true, model is loaded as stateful. |
| `"idle_sequence_cleanup"` | `bool` | If set to true, model will be subject to periodic sequence cleaner scans.  See [idle sequence cleanup](stateful_models.md). |
| `"max_sequence_number"` | `uint32` | Determines how many sequences can be handled concurrently by a model instance. |
| `"low_latency_transformation"` | `bool` | If set to true, model server will apply [low latency transformation](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-request/stateful-models/obtaining-stateful-openvino-model.html#lowlatency2-transformation) on model load. |
| `"metrics_enable"` | `bool` | Flag enabling [metrics](metrics.md) endpoint on rest_port. |
| `"metrics_list"` | `string` | Comma separated list of [metrics](metrics.md). If unset, only default metrics will be enabled.|
| `"allowed_local_media_path"` | `string` | Path to the directory containing images to include in requests. If unset, local filesystem images in requests are not supported.|

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
| `sequence_cleaner_poll_wait_minutes` | `integer` | Time interval (in minutes) between next sequence cleaner scans. Sequences of the models that are subjects to idle sequence cleanup that have been inactive since the last scan are removed. Zero value disables sequence cleaner. See [idle sequence cleanup](stateful_models.md). It also sets the schedule for releasing free memory from the heap. |
| `custom_node_resources_cleaner_interval_seconds` | `integer` | Time interval (in seconds) between two consecutive resources cleanup scans. Default is 1. Must be greater than 0. See [custom node development](custom_node_development.md). |
| `cpu_extension` | `string` | Optional path to a library with [custom layers implementation](https://docs.openvino.ai/2025/documentation/openvino-extensibility.html). |
| `log_level` | `"DEBUG"/"INFO"/"ERROR"` | Serving logging level |
| `log_path` | `string` | Optional path to the log file. |
| `cache_dir` | `string` | Path to the model cache storage. Caching will be enabled if this parameter is defined or the default path /opt/cache exists |
| `grpc_channel_arguments` | `string` |   A comma separated list of arguments to be passed to the grpc server. (e.g. grpc.max_connection_age_ms=2000) |
| `grpc_max_threads` | `string` |   Maximum number of threads which can be used by the grpc server. Default value depends on number of CPUs. |
| `grpc_memory_quota` | `string` |   GRPC server buffer memory quota. Default value set to 2147483648 (2GB). |
| `help` | `NA` |  Shows help message and exit |
| `version` | `NA` |  Shows binary version |
| `allow_credentials` | `bool` (default: false) | Whether to allow credentials in CORS requests. |
| `allow_headers` | `string` (default: *) | Comma-separated list of allowed headers in CORS requests. |
| `allow_methods` | `string` (default: *) | Comma-separated list of allowed methods in CORS requests. |
| `allow_origins` | `string` (default: *) | Comma-separated list of allowed origins in CORS requests. |

## Config management mode options

Configuration options for the config management mode, which is used to manage config file in the model repository.
| Option  | Value format  | Description  |
|---|---|---|
| `model_repository_path` | `string` | Path to the model repository. This path is prefixed to the relative model path. Use|
| `list_models`| `NA` | List all models paths in the model repository. |
| `model_name` | `string` | Name of the model as visible in serving. If ```--model_path``` is not provided, path is deduced from name. |
| `model_path` | `string` | Optional. Path to the model repository. If path is relative then it is prefixed with ```--model_repository_path```. |
| `add_to_config` | `string` |  Either path to directory containing config.json file for OVMS, or path to ovms configuration file, to add specific model to. |
| `remove_from_config` | `string` |  Either path to directory containing config.json file for OVMS, or path to ovms configuration file, to remove specific model from. |

## Pull mode configuration options

Shared configuration options for the pull, and pull & start mode. In the presence of ```--pull``` parameter OVMS will only pull model without serving.

### Pull Mode Options

| Option                      | Value format | Description                                                                                                   |
|-----------------------------|--------------|---------------------------------------------------------------------------------------------------------------|
| `--pull`                    | `NA`         | Runs the server in pull mode to download the model from the Hugging Face repository.  |
| `--source_model`            | `string`     | Name of the model in the Hugging Face repository. If not set, `model_name` is used.                |
| `--model_repository_path`   | `string`     | Directory where all required model files will be saved.                                                       |
| `--model_name`              | `string`     | Name of the model as exposed externally by the server.                                                        |
| `"target_device"` | `string` | Device name to be used to execute inference operations. Accepted values are: `"CPU"/"GPU"/"MULTI"/"HETERO"` |
| `--task`                    | `string`     | Task type the model will support (`text_generation`, `embedding`, `rerank`, `image_generation`).  Default: `text_generation` |

There are also additional environment variables that may change the behavior of pulling:

### Basic Environment Variables for Pull Mode

| Variable        | Value format | Description                                                                                                              |
|-----------------|--------------|--------------------------------------------------------------------------------------------------------------------------|
| `HF_ENDPOINT`   | `string`     | Default: `huggingface.co`. For users in China, set to `https://hf-mirror.com` if needed.                                 |
| `HF_TOKEN`      | `string`     | Authentication token required for accessing some models from Hugging Face.                                               |
| `https_proxy`   | `string`     | If set, model downloads will use this proxy.                                                                             |

### Advanced Environment Variables for Pull Mode
| Variable                            | Format  | Description                                                                                                |
| `GIT_OPT_SET_SERVER_CONNECT_TIMEOUT`| `int`   | Timeout to attempt connections to a remote server. Default value 4000 ms.                                  |
| `GIT_OPT_SET_SERVER_TIMEOUT`        | `int`   | Timeout for reading from and writing to a remote server. Default value 4000 ms.                            |
| `GIT_OPT_SET_SSL_CERT_LOCATIONS`    | `string`| Path to check for ssl certificates.                                                                        |

### Advanced Environment Variables for Pull Mode in 2025.2 release
| Variable                       | Format  | Description                                                                                                |
| `GIT_SERVER_CONNECT_TIMEOUT_MS`| `int`   | Timeout to attempt connections to a remote server. Default value 4000 ms.                                  |
| `GIT_SERVER_TIMEOUT_MS`        | `int`   | Timeout for reading from and writing to a remote server. Default value 0 - using system sesttings          |

Task specific parameters for different tasks (text generation/image generation/embeddings/rerank) are listed below:

### Text generation
| option                        | Value format | Description                                                                                                    |
|-------------------------------|--------------|----------------------------------------------------------------------------------------------------------------|
| `--max_num_seqs`              | `integer`    | The maximum number of sequences that can be processed together. Default: 256.                                  |
| `--pipeline_type`             | `string`     | Type of the pipeline to be used. Choices: `LM`, `LM_CB`, `VLM`, `VLM_CB`, `AUTO`. Default: `AUTO`.             |
| `--enable_prefix_caching`     | `bool`       | Enables algorithm to cache the prompt tokens. Default: true.                                                   |
| `--max_num_batched_tokens`    | `integer`    | The maximum number of tokens that can be batched together.                                                     |
| `--cache_size`                | `integer`    | Cache size in GB. Default: 10.                                                                                 |
| `--draft_source_model`        | `string`     | HF model name or path to the local folder with PyTorch or OpenVINO draft model.                                |
| `--dynamic_split_fuse`        | `bool`       | Enables dynamic split fuse algorithm. Default: true.                                                           |
| `--max_prompt_len`            | `integer`    | Sets NPU specific property for maximum number of tokens in the prompt.                                         |
| `--kv_cache_precision`        | `string`     | Reduced kv cache precision to `u8` lowers the cache size consumption. Accepted values: `u8` or empty (default).|
| `--response_parser`           | `string`     | Type of parser to use for tool calls and reasoning in model output. Currently supported: [qwen3, llama3, hermes3, phi4] |

### Image generation
| option                            | Value format | Description                                                                                                         |
|-----------------------------------|--------------|---------------------------------------------------------------------------------------------------------------------|
| `--max_resolution`                | `string`     | Maximum allowed resolution in the format `WxH` (W = width, H = height). If not specified, inherited from the model. |
| `--default_resolution`            | `string`     | Default resolution in the format `WxH` when not specified by the client. If not specified, inherited from the model.|
| `--max_num_images_per_prompt`     | `integer`    | Maximum number of images a client can request per prompt in a single request. In 2025.2 release only 1 image generation per request is supported. |
| `--default_num_inference_steps`   | `integer`    | Default number of inference steps when not specified by the client.                                                 |
| `--max_num_inference_steps`       | `integer`    | Maximum number of inference steps a client can request for a given model.                                           |
| `--num_streams`                   | `integer`    | Number of parallel execution streams for image generation models. Use at least 2 on 2-socket CPU systems.           |

### Embeddings
| option                    | Value format | Description                                                                    |
|---------------------------|--------------|--------------------------------------------------------------------------------|
| `--num_streams`           | `integer`    | The number of parallel execution streams to use for the model. Use at least 2 on 2 socket CPU systems. Default: 1. |
| `--normalize`             | `bool`       | Normalize the embeddings. Default: true.                                       |
| `--mean_pooling`          | `bool`       | Mean pooling option. Default: false.                                           |

### Rerank
| option                    | Value format | Description                                                                    |
|---------------------------|--------------|--------------------------------------------------------------------------------|
| `--num_streams`           | `integer`    | The number of parallel execution streams to use for the model. Use at least 2 on 2 socket CPU systems. Default: 1. |
| `--max_allowed_chunks`    | `integer`    | Maximum allowed chunks. Default: 10000.                                        |
