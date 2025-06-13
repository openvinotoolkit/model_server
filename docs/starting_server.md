# Starting the Server  {#ovms_docs_serving_model}

There are two method for passing to the model server information about the models and their configuration:
- via CLI parameters - for a single model or pipeline
- via config file in json format - for any number of models and pipelines

Note that changing configuration in runtime while serving is possible only with the config file.
When deploying model(s) with a configuration file, you can add or delete models, as well as update their configurations in runtime, without needing to restart the server.

## Serving a single Classic Model, Mediapipe, GenAI Model

### Starting with prepared models
Before starting the container, make sure you have [reviewed preparing model repository](models_repository.md).

Start the model server by running the following command with your parameters: 

::::{tab-set}
:::{tab-item} With Docker
**Required:** Docker Engine installed

```bash
docker run -d --rm -v <models_repository>:/models -p 9000:9000 -p 8000:8000 openvino/model_server:latest \
--model_path <path_to_model> --model_name <model_name> --port 9000 --rest_port 8000 --log_level DEBUG
```
:::

:::{tab-item} On Baremetal Host
**Required:** OpenVINO Model Server package - see [deployment instructions](../deploying_server_baremetal.md) for details.

```bash
ovms --model_path <path_to_model> --model_name <model_name> --port 9000 --rest_port 8000 --log_level DEBUG
```
:::
::::

Server will detect the type of requested servable (model or mediapipe graph) and load it accordingly. This detection is based on the presence of a `graph.pbtxt` file, which defines the Mediapipe graph structure, presence of versions directory for classic models.

Example using a ResNet model:

docker run w tabach
```bash
mkdir -p models/resnet/1
wget -P models/resnet/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin
wget -P models/resnet/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml
```

::::{tab-set}
:::{tab-item} With Docker
**Required:** Docker Engine installed

```bash
docker run -d --rm -v ${PWD}/models:/models -p 9000:9000 -p 8000:8000 openvino/model_server:latest \
--model_path /models/resnet/ --model_name resnet --port 9000 --rest_port 8000 --log_level DEBUG
```
:::

:::{tab-item} On Baremetal Host

```bat
ovms --model_path /models/resnet/ --model_name resnet --port 9000 --rest_port 8000 --log_level DEBUG
```
:::
::::

The required Model Server parameters are listed below. For additional configuration options, see the [Model Server Parameters](parameters.md) section.

`openvino/model_server:latest` varies by tag and build process - see tags: https://hub.docker.com/r/openvino/model_server/tags/ for a full tag list.

- Expose the container ports to **open ports** on your host or virtual machine. 
- In the command above, port 9000 is exposed for gRPC and port 8000 is exposed for REST API calls.
- Add model_name for the client gRPC/REST API calls.

### Starting the GenAI model from HF directly

Notka ze z OV organization

In case you do not want to prepare model repository before starting the server, you can just run OVMS with:

::::{tab-set}
:::{tab-item} With Docker
**Required:** Docker Engine installed

```bash
docker run --user $(id -u):$(id -g) -p 9000:9000 -p 8000:8000 --rm -v <model_repository_path>:/models openvino/model_server:latest \
--port 8000 --rest_port 9000 --source_model <model_name_in_HF> --model_repository_path /models --model_name <ovms_servable_name> --task <task> [TASK_SPECIFIC_OPTIONS]
```
:::

:::{tab-item} On Baremetal Host
**Required:** OpenVINO Model Server package - see [deployment instructions](./deploying_server_baremetal.md) for details.

```bat
ovms --source_model <model_name_in_HF> --model_repository_path /models --model_name <ovms_servable_name> --task <task> [TASK_SPECIFIC_OPTIONS]
```
:::
::::

It will download required model files, prepare configuration for OVMS and start serving the model.

In case of GenAI models, startup can use additional parameters specific to task. For details refer [here](./parameters.md).

*Note:*
When using pull during startup you need both read and write access rights to models repository.

Example using `Phi-3-mini-FastDraft-50M-int8-ov` model:


::::{tab-set}
:::{tab-item} With Docker
**Required:** Docker Engine installed
```bash
docker run --user $(id -u):$(id -g) -p 9000:9000 -p 8000:8000 --rm -v <model_repository_path>:/models openvino/model_server:latest \
--port 8000 --rest_port 9000 --source_model "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov" --model_repository_path /models/ --model_name Phi-3-mini-FastDraft-50M-int8-ov --task text_generation
```
:::

:::{tab-item} On Baremetal Host
**Required:** OpenVINO Model Server package - see [deployment instructions](./deploying_server_baremetal.md) for details.

```bash
ovms --source_model "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov" --model_repository_path /models/ --model_name Phi-3-mini-FastDraft-50M-int8-ov --task text_generation --port 8000 --rest_port 9000
```
:::
::::

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
            "name":"model_name3",
            "base_path":"gs://bucket/models/model3",
         }
      }
   ]
}
```

How to run with ```config.json```:
```text
docker run --user $(id -u):$(id -g) --rm -v <models_repository>:/models:ro -p 9000:9000 -p 8000:8000 openvino/model_server:latest \
--config_path /models/config.json --port 9000 --rest_port 8000
```
or for binary package:
```text
ovms --config_path <path_to_config_file> --port 9000 --rest_port 8000
```

## Config management

### List models

Assuming you have models repository already prepared, to check what models/graphs are servable from specified repository:
```text
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
ovms --model_repository_path /models/<model_path> --add_to_config <config_file_directory_path> --model_name <name>
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
ovms --remove_from_config <config_file_directory_path> --model_name <name>
```
:::
::::