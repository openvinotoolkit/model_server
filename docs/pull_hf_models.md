# Generative AI Models {#ovms_docs_pull}

```{toctree}
---
maxdepth: 1
hidden:
---
ovms_demos_common_export
```

Generative AI models require additional configuration steps before deployment with OpenVINO Model Server (OVMS). There are two primary methods for preparing generative AI models:

## 1. Model preparation with OVMS:

OVMS can automatically download models from the Hugging Face (HF) repository and configure them for serving. This approach leverages built-in OVMS functionality to streamline model preparation.

*Note:* This approach will work assuming you are pulling from [OpenVINO organization](https://huggingface.co/OpenVINO) from HF. If the model is not from this organization, additional configuration steps may be required to ensure compatibility with OVMS.

### Pulling the models

There is a special mode to make OVMS pull the model from Hugging Face before starting the service:

::::{tab-set}
:::{tab-item} With Docker
**Required:** Docker Engine installed

```bash
docker run -d --rm -v <model_repository_path>:/models openvino/model_server:latest --pull --source_model <model_name_in_HF> --model_repository_path /models --model_name <external_model_name> --task <task> [TASK_SPECIFIC_PARAMETERS]
```
:::

:::{tab-item} On Baremetal Host
**Required:** OpenVINO Model Server package - see [deployment instructions](../deploying_server_baremetal.md) for details.

```bat
ovms.exe --pull --source_model <model_name_in_HF> --model_repository_path <model_repository_path> --model_name <external_model_name> --task <task> [TASK_SPECIFIC_PARAMETERS]
```
:::
::::


Example for pulling `OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov`:

```bash
docker run -d --rm -v /models:/models openvino/model_server:latest --source_model "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov" --model_repository_path /models --model_name Phi-3-mini-FastDraft-50M-int8-ov --task text_generation 
```

It will prepare all needed configuration files to support LLMS with OVMS in the model repository. Check [parameters page](./parameters.md) for detailed descriptions of configuration options and parameter usage.

*Note:*
When using ```--task``` you need both read and write access rights to models repository.

## 2. Preprocessing with Python Script:
Alternatively, users can utilize the provided Python script to export, optimize and configure models prior to server deployment. This approach is more flexible as it allows for using models that were not optimized beforehand, but requires having Python set up to work. You can find the script [here](./../demos/common/export_models/export_models.py) and its README [here](./../demos/common/export_models/README.md).

*Note:*
Use the integrated OVMS download for streamlined setups or the Python script for more flexibility in handling non-optimized models.

