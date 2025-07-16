# OVMS Pull mode {#ovms_docs_pull}

This documents describes how to leverage OpenVINO Model Server (OVMS) pull feature to automate deployment configuration with Generative AI models. When pulling from [OpenVINO organization](https://huggingface.co/OpenVINO) from HF no additional steps are required. However, when pulling models [outside of the OpenVINO](https://github.com/openvinotoolkit/model_server/blob/main/docs/pull_optimum_cli.md) organization you have to install additional python dependencies when using baremetal execution so that optimum-cli is available for ovms executable or build the OVMS python container for docker deployments. In summary you have 2 options:

- pulling preconfigured models in IR format from OpenVINO organization
- pulling models with automatic conversion and quantization (requires optimum-cli). Include additional consideration like longer time for deployment and pulling model data (original model) from HF, model memory for conversion, diskspace - described [here](https://github.com/openvinotoolkit/model_server/blob/main/docs/pull_optimum_cli.md)

### Pulling the models

There is a special mode to make OVMS pull the model from Hugging Face before starting the service:

::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed

```text
docker run $(id -u):$(id -g) --rm -v <model_repository_path>:/models:rw openvino/model_server:latest --pull --source_model <model_name_in_HF> --model_repository_path /models --model_name <external_model_name> --target_device <DEVICE> --task <task> [TASK_SPECIFIC_PARAMETERS]
```
:::

:::{tab-item} On Baremetal Host
:sync: baremetal
**Required:** OpenVINO Model Server package - see [deployment instructions](./deploying_server_baremetal.md) for details.

```text
ovms --pull --source_model <model_name_in_HF> --model_repository_path <model_repository_path> --model_name <external_model_name> --target_device <DEVICE> --task <task> [TASK_SPECIFIC_PARAMETERS]
```
:::
::::

Example for pulling `OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov`:

```text
ovms --pull --source_model "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov" --model_repository_path /models --model_name Phi-3-mini-FastDraft-50M-int8-ov --target_device CPU --task text_generation 
```
::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed

```text
docker run $(id -u):$(id -g) --rm -v <model_repository_path>:/models:rw openvino/model_server:latest --pull --source_model "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov" --model_repository_path /models --model_name Phi-3-mini-FastDraft-50M-int8-ov --task text_generation
```
:::

:::{tab-item} On Baremetal Host
:sync: baremetal
**Required:** OpenVINO Model Server package - see [deployment instructions](./deploying_server_baremetal.md) for details.

```text
ovms --pull --source_model "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov" --model_repository_path /models --model_name Phi-3-mini-FastDraft-50M-int8-ov --task text_generation 
```
:::
::::


It will prepare all needed configuration files to support LLMS with OVMS in the model repository. Check [parameters page](./parameters.md) for detailed descriptions of configuration options and parameter usage.

In case you want to setup model and start server in one step follow instructions on [this page](./starting_server.md).

*Note:*
When using pull mode you need both read and write access rights to models repository.
