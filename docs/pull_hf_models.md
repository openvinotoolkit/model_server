# Pull mode {#ovms_docs_pull}

This documents describes how leverage OpenVINO Model Server (OVMS) pull feature to automate deployment configuration with Generative AI models from HuggingFace (HF). This approach assumes that you are pulling from [OpenVINO organization](https://huggingface.co/OpenVINO) from HF. If the model is not from that organization, follow steps described in [this document](./export_model_script.md).

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
