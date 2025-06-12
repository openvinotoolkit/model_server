# Starting the Server  {#ovms_docs_serving_model}

There are two method for passing to the model server information about the models and their configuration:
- via CLI parameters - for a single model or pipeline
- via config file in json format - for any number of models and pipelines

Note that changing configuration in runtime while serving is possible only with the config file.
When deploying model(s) with a configuration file, you can add or delete models, as well as update their configurations in runtime, without needing to restart the server.

## Serving a Classic Model

Before starting the container, make sure you have [prepared the model for serving](models_repository.md).

Start the model server by running the following command with your parameters: 

```
docker run -d --rm -v <models_repository>:/models -p 9000:9000 -p 8000:8000 openvino/model_server:latest \
--model_path <path_to_model> --model_name <model_name> --port 9000 --rest_port 8000 --log_level DEBUG
```
or for binary package:
```
ovms --model_path <path_to_model> --model_name <model_name> --port 9000 --rest_port 8000 --log_level DEBUG
```

Example using a ResNet model:

```bash
mkdir -p models/resnet/1
wget -P models/resnet/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin
wget -P models/resnet/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml

docker run -d --rm -v ${PWD}/models:/models -p 9000:9000 -p 8000:8000 openvino/model_server:latest \
--model_path /models/resnet/ --model_name resnet --port 9000 --rest_port 8000 --log_level DEBUG
```

The required Model Server parameters are listed below. For additional configuration options, see the [Model Server Parameters](parameters.md) section.

| option                         | description                                                            |
|--------------------------------|------------------------------------------------------------------------|
| `--rm`                         | remove the container when exiting the Docker container                 |
| `-d`                           | runs the container in the background                                   |
| `-v`                           | defines how to mount the model folder in the Docker container          |
| `-p`                           | exposes the model serving port outside the Docker container            |
| `openvino/model_server:latest` | represents the image name; the ovms binary is the Docker entry point   |
| `--model_path`                 | model location                                                         |
| `--model_name`                 | the name of the model in the model_path                                |
| `--port`                       | the gRPC server port                                                   |
| `--rest_port`                  | the REST server port                                                   |

Possible model locations (`--model_path`):
* Docker container path that is mounted during start-up
* Google Cloud Storage path `gs://<bucket>/<model_path>`
* AWS S3 path `s3://<bucket>/<model_path>`   
* Azure blob path `az://<container>/<model_path>`

`openvino/model_server:latest` varies by tag and build process - see tags: https://hub.docker.com/r/openvino/model_server/tags/ for a full tag list.

- Expose the container ports to **open ports** on your host or virtual machine. 
- In the command above, port 9000 is exposed for gRPC and port 8000 is exposed for REST API calls.
- Add model_name for the client gRPC/REST API calls.

## Serving GenAI models and mediapipes

### Starting the mediapipe graph or LLM models
You can start server with single mediapipe graph, or LLM model that is already configured in local filesystem with:

::::{tab-set}
:::{tab-item} With Docker
**Required:** Docker Engine installed

```bash
docker run -d --rm -v <model_repository_path>:/models -p 9000:9000 -p 8000:8000 openvino/model_server:latest \
--model_path <path_to_model> --model_name <model_name> --port 9000 --rest_port 8000
```
:::

:::{tab-item} On Baremetal Host
**Required:** OpenVINO Model Server package - see [deployment instructions](../deploying_server_baremetal.md) for details.

```bat
ovms.exe --model_path <path_to_model> --model_name <model_name> --port 9000 --rest_port 8000
```
:::
::::

Server will detect the type of requested servable (model or mediapipe graph) and load it accordingly. This detection is based on the presence of a `.pbtxt` file, which defines the Mediapipe graph structure.

*Note*: There is no online model modification nor versioning capability as of now for graphs, LLM like models.

### Starting the LLM model from HF directly

In case you do not want to prepare model repository before starting the server, you can just run OVMS with:

::::{tab-set}
:::{tab-item} With Docker
**Required:** Docker Engine installed

```bash
docker run -d --rm -v <model_repository_path>:/models openvino/model_server:latest --source_model <model_name_in_HF> --model_repository_path /models --model_name <ovms_servable_name> --task <task> [TASK_SPECIFIC_OPTIONS]
```
:::

:::{tab-item} On Baremetal Host
**Required:** OpenVINO Model Server package - see [deployment instructions](../deploying_server_baremetal.md) for details.

```bat
ovms.exe --source_model <model_name_in_HF> --model_repository_path /models --model_name <ovms_servable_name> --task <task> [TASK_SPECIFIC_OPTIONS]
```
:::
::::

It will download required model files, prepare configuration for OVMS and start serving the model.

In case of GenAI models, startup may require additional parameters specific to task. For details refer [here](./parameters.md).

*Note:*
When using ```--task``` you need both read and write access rights to models repository.

## Serving Multiple Models 

To serve multiple models and pipelines from the same container you will need an additional JSON configuration file that defines each model. To use a container with several models, you need an additional JSON configuration file defining each model. `model_config_list` array that includes a collection of config objects for each served model. The `name` and the `base_path` values of the model are required for each config object.

```json
{
   "model_config_list":[
      {
         "config":{
            "name":"model_name1",
            "base_path":"/opt/ml/models/model1",
            "batch_size": "16",
            "model_version_policy": {"all": {}}
         }
      },
      {
         "config":{
            "name":"model_name3",
            "base_path":"gs://bucket/models/model3",
            "model_version_policy": {"specific": { "versions":[1, 3] }}
         }
      },
      {
         "config":{
             "name":"model_name4",
             "base_path":"s3://bucket/models/model4",
             "shape": {
                "input1": "(1,3,200,200)",
                "input2": "(1,3,50,50)"
             },
             "plugin_config": {"PERFORMANCE_HINT": "THROUGHPUT"}
         }
      },
      {
         "config":{
             "name":"model_name5",
             "base_path":"s3://bucket/models/model5",
             "nireq": 32,
             "target_device": "GPU"
         }
      }
   ]
}
```
In case of deploying a complete pipelines defined by a MediaPipe graph, each of them should be added to the configuration file as addition section:

```json
    "mediapipe_config_list": [
    {
        "name":"mediapipe_graph_name"
    },
    {
        "name":"mediapipe2",
        "base_path":"non_default_path"
    }
    ]
```
Check more info about [MediaPipe graphs](./mediapipe.md)


`base_path` in the config.json can be absolute or relative to the configuration file. This is helpful when models are distributed together with the config file, the paths do not need to be adjusted.

Examples:
```
docker run -d --rm -v <models_repository>:/models -p 9000:9000 -p 8000:8000 openvino/model_server:latest \
--config_path /models/config.json --port 9000 --rest_port 8000
```
or for binary package:
```
ovms --config_path <path_to_config_file> --port 9000 --rest_port 8000
```

### List models

Assuming you have models repository already prepared, to check what models/graphs are servable from specified repository:
```
docker run -d --rm -v <model_repository_path>:/models openvino/model_server:latest \
--model_repository_path /models --list_models
```

For following directory structure:
```text
/models
├── meta
│   ├── llama4
│   │   └── graph.pbtxt
│   ├── llama3.1
│   │   └── graph.pbtxt
├── LLama3.2
│   └── graph.pbtxt
└── resnet
    └── 1
        └── saved_model.pb
```

The output would be:
```text
meta/llama4
meta/llama3.1
LLama3.2
resnet
```

### Enable model

To add model to ovms configuration file you can either do it manually or use:

::::{tab-set}
:::{tab-item} With Docker
**Required:** Docker Engine installed

```bash
docker run -d --rm -v <model_repository_path>:/models openvino/model_server:latest \
--model_repository_path /models/<model_path> --add_to_config <config_file_directory_path> --model_name <name>
```
:::

:::{tab-item} On Baremetal Host
**Required:** OpenVINO Model Server package - see [deployment instructions](../deploying_server_baremetal.md) for details.

```bat
ovms.exe --model_repository_path /models/<model_path> --add_to_config <config_file_directory_path> --model_name <name>
```
:::
::::

When model is directly inside models repository.

*Note*:
If you want to add model with specific path you can use:
```text
docker run -d --rm -v <model_repository_path>:/models openvino/model_server:latest \
--add_to_config <config_file_directory_path> --model_name <name> --model_path <model_path>
```

*Note:* Use relative paths to make the config.json transferable in model_repository across ovms instances.
For example:
```text
cd model_repository_path
ovms --add_to_config . --model_name OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov --model_repository_path .
```

### Disable model

If you want to remove model from configuration file you can do it either manually or use command:

```text
docker run -d --rm -v <model_repository_path>:/models openvino/model_server:latest \
--remove_from_config <config_file_directory_path> --model_name <name>
```

::::{tab-set}
:::{tab-item} With Docker
**Required:** Docker Engine installed

```bash
docker run -d --rm -v <model_repository_path>:/models openvino/model_server:latest \
--remove_from_config <config_file_directory_path> --model_name <name>
```
:::

:::{tab-item} On Baremetal Host
**Required:** OpenVINO Model Server package - see [deployment instructions](../deploying_server_baremetal.md) for details.

```bat
ovms.exe --remove_from_config <config_file_directory_path> --model_name <name>
```
:::
::::