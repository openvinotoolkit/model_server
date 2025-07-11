# OVMS Pull mode {#ovms_docs_pull}

This documents describes how leverage OpenVINO Model Server (OVMS) pull feature to automate deployment configuration with Generative AI models from or outside of OpenVINO organization in HuggingFace (HF). When pulling from [OpenVINO organization](https://huggingface.co/OpenVINO) from HF no additional steps are required. However, when pulling models outside of the OpenVINO organization you have to install additional python dependencies when using baremetal execution so that optimum-cli is available for ovms executable or build the OVMS python container for docker deployments.

Specific OVMS pull mode example for models outside of OpenVINO organization is described in section `2. Download the preconfigured models using ovms --pull option for models outside OpenVINO organization` in the [RAG demo](https://github.com/openvinotoolkit/model_server/blob/main/demos/continuous_batching/rag/README.md)

## Pulling models outside of OpenVINO organization
### Build python docker image
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make python_image
```
Then use the docker commands described in `Pulling the models` section with `openvino/model_server:py` container instead of `openvino/model_server:latest`.

### Install optimum-cli
Install python on your baremetal system from `https://www.python.org/downloads/`
```console
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/2/demos/common/export_models/requirements.txt
pip3 install -q -r requirements.txt
```
Then use the docker commands described in `Pulling the models` section

### Ovms pull mode alternative
You can use the `export_models.py` script described in [this document](../demos/common/export_models/README.md).

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
**Required:** OpenVINO Model Server package - see [deployment instructions](../deploying_server_baremetal.md) for details.

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
**Required:** OpenVINO Model Server package - see [deployment instructions](../deploying_server_baremetal.md) for details.

```text
ovms --pull --source_model "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov" --model_repository_path /models --model_name Phi-3-mini-FastDraft-50M-int8-ov --task text_generation 
```
:::
::::


It will prepare all needed configuration files to support LLMS with OVMS in the model repository. Check [parameters page](./parameters.md) for detailed descriptions of configuration options and parameter usage.

In case you want to setup model and start server in one step follow instructions on [this page](./starting_server.md).

*Note:*
When using pull mode you need both read and write access rights to models repository.
