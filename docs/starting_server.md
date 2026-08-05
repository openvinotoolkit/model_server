# Starting the Server  {#ovms_docs_serving_model}

There are three methods for passing model information and configuration to the model server:
- via CLI parameters, for a single model or pipeline
- via a config file in JSON format, for any number of classic models
- via `graph.pbtxt` in the model folder, for all MediaPipe pipelines including generative models

Note that changing configuration at runtime while serving is possible only with the config file.
When deploying models with a configuration file, you can add or delete models, as well as update their configurations at runtime, without restarting the server.
Updating `graph.pbtxt` runtime parameters takes effect after the model is unloaded and reloaded (removed from `config.json` and added again, or after the service is restarted).

## Serving a Single Classic Model, MediaPipe, or GenAI Model

### Starting with prepared models
Before starting the container, make sure you have reviewed [preparing the model repository](models_repository.md).

Start the model server by running the following command with your parameters:

::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed

```text
docker run -d --rm -v <models_repository>:/models -p 9000:9000 -p 8000:8000 openvino/model_server:latest \
--model_path <path_to_model> --model_name <model_name> --port 9000 --rest_port 8000 --log_level DEBUG
```
:::

:::{tab-item} On Bare-metal Host
:sync: baremetal
**Required:** OpenVINO Model Server package - see [deployment instructions](./deploying_server_baremetal.md) for details.

```text
ovms --model_path <path_to_model> --model_name <model_name> --port 9000 --rest_port 8000 --log_level DEBUG
```
:::
::::

OVMS detects the type of requested servable (classic model, generative model, or MediaPipe graph) and loads it accordingly. This detection is based on the presence of a `graph.pbtxt` file, which defines the MediaPipe graph structure, and on the presence of a versions directory for classic models.

When `graph.pbtxt` is missing in a generative model, OVMS attempts to determine the model use case and required parameters based on included configuration files that follow Hugging Face standards. In most cases, no extra parameters are required to deploy the model.

If automatic detection cannot be done, parameters can be provided explicitly with `--task <TASK>` followed by task-specific options.

::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed

```text
docker run -d --rm -v ${PWD}/<model>:/model -p 8000:8000 openvino/model_server:latest \
--model_path /model --model_name <model_name> --rest_port 8000 --log_level DEBUG \
--task <TASK> --target_device <DEVICE> ........
```
:::

:::{tab-item} On Bare-metal Host
:sync: baremetal
**Required:** OpenVINO Model Server package - see [deployment instructions](./deploying_server_baremetal.md) for details.

```text
ovms --model_path <path_to_model> --model_name <model_name> --rest_port 8000 --log_level DEBUG --task <TASK> --target_device <DEVICE> .....
```
:::
::::


**Example using a ResNet model:**

```bash
mkdir -p models/resnet/1
wget -P models/resnet/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin
wget -P models/resnet/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml
```

::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed

```bash
docker run -d --rm -v ${PWD}/models:/models -p 9000:9000 -p 8000:8000 openvino/model_server:latest \
--model_path /models/resnet/ --model_name resnet --port 9000 --rest_port 8000 --log_level DEBUG
```
:::

:::{tab-item} On Bare-metal Host
:sync: baremetal
```text
ovms --model_path models/resnet/ --model_name resnet --port 9000 --rest_port 8000 --log_level DEBUG
```
:::
::::


**Example using OpenVINO/Qwen3-0.6B-int4-ov model from a local folder:**

```bash
pip install huggingface_hub
hf download OpenVINO/Qwen3-0.6B-int4-ov --local-dir Qwen3-0.6B-int4-ov
```

::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed

```bash
docker run -d --rm -v ${PWD}/Qwen3-0.6B-int4-ov:/model -p 8000:8000 openvino/model_server:latest \
--model_path /model/ --model_name qwen3-0.6 --rest_port 8000 --task text_generation --target_device CPU
```
:::

:::{tab-item} On Bare-metal Host
:sync: baremetal
```text
ovms --model_path Qwen3-0.6B-int4-ov --model_name qwen3-0.6 --rest_port 8000 --task text_generation --target_device CPU
```
:::
::::

The required Model Server parameters are listed below. For additional configuration options, see the [Model Server Parameters](parameters.md) section.

`openvino/model_server:latest` varies by tag and build process. See https://hub.docker.com/r/openvino/model_server/tags/ for a full tag list.

- In the command above, port 9000 is exposed for gRPC and port 8000 is exposed for REST API calls.
- Add `model_name` for client gRPC/REST API calls.

### Starting the GenAI model from Hugging Face directly

For models outside the OpenVINO organization, follow the additional prerequisites described in [OVMS pull mode](./pull_optimum_cli.md).

If you do not want to prepare a model repository before starting the server and you want to serve a model directly from [Hugging Face](https://huggingface.co/), run OVMS with:

::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed

```text
docker run --user $(id -u):$(id -g) -p 9000:9000 -p 8000:8000 --rm -v <model_repository_path>:/models openvino/model_server:latest \
--port 9000 --rest_port 8000 --source_model <model_name_in_HF> --model_repository_path /models --model_name <ovms_servable_name> --target_device <DEVICE> --task <task> [TASK_SPECIFIC_OPTIONS]
```
:::

:::{tab-item} On Bare-metal Host
:sync: baremetal
**Required:** OpenVINO Model Server package - see [deployment instructions](./deploying_server_baremetal.md) for details.

```text
ovms --source_model <model_name_in_HF> --model_repository_path /models --model_name <ovms_servable_name> --target_device <DEVICE> --task <task> [TASK_SPECIFIC_OPTIONS]
```
:::
::::

It will download required model files, prepare configuration for OVMS and start serving the model.

For GenAI models, startup can use additional task-specific parameters. For details, see [parameters](./parameters.md).

> **Note:** When using pull during startup, you need both read and write permissions for the model repository.

Example using `Phi-3-mini-FastDraft-50M-int8-ov` model:

::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed
```text
docker run --user $(id -u):$(id -g) -p 9000:9000 -p 8000:8000 --rm -v <model_repository_path>:/models openvino/model_server:latest \
--port 9000 --rest_port 8000 --source_model "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov" --model_repository_path /models/ --model_name Phi-3-mini-FastDraft-50M-int8-ov --target_device CPU --task text_generation
```
:::

:::{tab-item} On Bare-metal Host
:sync: baremetal
**Required:** OpenVINO Model Server package - see [deployment instructions](./deploying_server_baremetal.md) for details.

```bat
ovms --source_model "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov" --model_repository_path models/ --model_name Phi-3-mini-FastDraft-50M-int8-ov --target_device GPU --task text_generation --port 9000 --rest_port 8000
```
:::
::::

## Serving Multiple Models

To serve multiple models and pipelines from the same container, you need an additional JSON configuration file that defines each model. The `model_config_list` array includes a collection of config objects for each served model. The `name` and `base_path` values are required for each config object.

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
            "name":"Phi-3-mini-FastDraft-50M-int8-ov",
            "base_path":"/models/OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov/"
         }
      }
   ]
}
```

**How to run with `config.json`:**

::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed
```text
docker run --user $(id -u):$(id -g) --rm -v <models_repository>:/models:ro -p 9000:9000 -p 8000:8000 openvino/model_server:latest \
--config_path /models/config.json --port 9000 --rest_port 8000
```
:::

:::{tab-item} On Bare-metal Host
:sync: baremetal
**Required:** OpenVINO Model Server package - see [deployment instructions](./deploying_server_baremetal.md) for details.

```text
ovms --config_path /models/config.json --port 9000 --rest_port 8000
```
:::
::::

### Config management

#### List models

Assuming you have a prepared model repository, use the following command to check which models/graphs are servable from the specified repository:

::::{tab-set}
:::{tab-item} With Docker
**Required:** Docker Engine installed
:sync: docker

```text
docker run -d --rm -v <model_repository_path>:/models openvino/model_server:latest \
--model_repository_path /models --list_models
```
:::

:::{tab-item} On Bare-metal Host
:sync: baremetal
**Required:** OpenVINO Model Server package - see [deployment instructions](./deploying_server_baremetal.md) for details.

```text
ovms --model_repository_path <model_repository_path> --list_models
```
:::
::::

For the following directory structure:
```text
/models
├── meta
│   ├── llama4
│   │   └── graph.pbtxt
│   ├── llama3.1
│   │   └── graph.pbtxt
├── llama3.2
│   └── graph.pbtxt
└── resnet
    └── 1
        └── saved_model.pb
```

The output would be:
```text
meta/llama4
meta/llama3.1
llama3.2
resnet
```

#### Enable model

To add a model to an OVMS configuration file, you can either do it manually or use:

::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed

```text
docker run -d --rm -v <model_repository_path>:/models openvino/model_server:latest \
--model_repository_path /models/ --add_to_config --config_path <config_file_path> --model_name <name>
```
:::

:::{tab-item} On Bare-metal Host
:sync: baremetal
**Required:** OpenVINO Model Server package - see [deployment instructions](./deploying_server_baremetal.md) for details.
Note: the `OVMS_MODEL_REPOSITORY_PATH` environment variable can determine default values for `--model_repository_path` and `--config_path`, where `--config_path` defaults to `config.json` inside the model repository path.
```text
export OVMS_MODEL_REPOSITORY_PATH=/models
ovms --add_to_config --model_name <name>
```
:::
::::


Use this when the model is directly inside the model repository.

> **Note:** If you want to add a model with a specific path, you can use the `--model_path` parameter:

```text
docker run -d --rm -v <model_repository_path>:/models openvino/model_server:latest \
--add_to_config --config_path <config_file_path> --model_name <name> --model_path <model_path>
```

> **Note:** Use relative or absolute paths. `config_path` is relative to the current folder. `model_path` is relative to the config file. With `model_repository_path`, `model_name` represents a folder relative to the model repository path.
For example:
```text
ovms --add_to_config --config_path models/config.json --model_name OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov --model_repository_path models
or
ovms --add_to_config --config_path models/config.json --model_name OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov --model_path OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov
```


Adding classic models to `config.json` is also possible with extra runtime parameters:
```text
ovms --add_to_config --config_path config.json --model_name resnet --model_path models/resnet/resnet.xml --batch_size 1 --target_device CPU

```

#### Disable model

If you want to remove a model from the configuration file, you can do it manually or use the command below:

::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed
```text
docker run -d --rm -v <model_repository_path>:/models openvino/model_server:latest \
--remove_from_config --config_path <config_file_path> --model_name <name>
```
:::

:::{tab-item} On Bare-metal Host
:sync: baremetal
**Required:** OpenVINO Model Server package - see [deployment instructions](./deploying_server_baremetal.md) for details.

```text
ovms --remove_from_config --config_path <config_file_path> --model_name <name>
```
:::
::::
