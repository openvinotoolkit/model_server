# OVMS Pull mode {#ovms_docs_pull}

This document describes how to leverage OpenVINO Model Server (OVMS) pull feature to automate deployment configuration with Generative AI models. When pulling from [Hugging Face Hub](https://huggingface.co/) or when pulling GGUF model, no additional steps are required. However, when pulling models in Pytorch format, you have to install additional python dependencies when using baremetal execution so that optimum-cli is available for ovms executable or rely on the docker image `openvino/model_server:latest-py`. In summary you have 3 options:

- pulling pre-configured models in IR format (described below)
- pulling GGUF models from Hugging Face
- pulling models with automatic conversion and quantization via optimum-cli. Described in the [pulling with conversion](https://github.com/openvinotoolkit/model_server/blob/releases/2025/4/docs/pull_optimum_cli.md)

> **Note:** Models in IR format must be exported using `optimum-cli` including tokenizer and detokenizer files also in IR format, if applicable. If missing, tokenizer and detokenizer should be added using `convert_tokenizer --with-detokenizer` tool.

## Pulling pre-configured models

There is a special OVMS mode to pull the model from Hugging Face without starting the service. It is triggered by `--pull` parameter. The application quits after the model is downloaded. Without `--pull` option, the model will be deployed and server started.

::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed
```text
docker run $(id -u):$(id -g) --rm -v <model_repository_path>:/models:rw openvino/model_server:weekly --pull --source_model <model_name_in_HF> --model_repository_path /models --model_name <external_model_name> --target_device <DEVICE> [--gguf_filename SPECIFIC_QUANTIZATION_FILENAME.gguf] --task <task> [TASK_SPECIFIC_PARAMETERS]
```
:::

:::{tab-item} On Baremetal Host
:sync: baremetal
**Required:** OpenVINO Model Server package - see [deployment instructions](./deploying_server_baremetal.md) for details.

```text
ovms --pull --source_model <model_name_in_HF> --model_repository_path <model_repository_path> --model_name <external_model_name> --target_device <DEVICE> [--gguf_filename SPECIFIC_QUANTIZATION_FILENAME.gguf] --task <task> [TASK_SPECIFIC_PARAMETERS]
```
:::
::::

*Note:* GGUF format model is only supported with `--task text_generation`. For list of supported models check [blog](https://blog.openvino.ai/blog-posts/openvino-genai-supports-gguf-models).

Example for pulling `OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov`:

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

Example for pulling GGUF model `unsloth/Llama-3.2-1B-Instruct-GGUF` with Q4_K_M quantization on baremetal host:

::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed

```text
docker run $(id -u):$(id -g) --rm -v <model_repository_path>:/models:rw openvino/model_server:weekly --pull --source_model "unsloth/Llama-3.2-1B-Instruct-GGUF" --model_repository_path /models --model_name unsloth/Llama-3.2-1B-Instruct-GGUF --task text_generation --gguf_filename Llama-3.2-1B-Instruct-Q4_K_M.gguf
```
:::

:::{tab-item} On Baremetal Host
:sync: baremetal
**Required:** OpenVINO Model Server package - see [deployment instructions](./deploying_server_baremetal.md) for details.
```text
ovms --pull --source_model "unsloth/Llama-3.2-1B-Instruct-GGUF" --model_repository_path /models --model_name unsloth/Llama-3.2-1B-Instruct-GGUF --task text_generation --gguf_filename Llama-3.2-1B-Instruct-Q4_K_M.gguf
```
:::
::::

## Pulling models outside OpenVINO organization

It is possible to pull models outside of OpenVINO organization. 

Example for pulling `Echo9Zulu/phi-4-int4_asym-awq-ov`:

```text
ovms --pull --source_model Echo9Zulu/phi-4-int4_asym-awq-ov --model_repository_path /models --model_name phi-4-int4_asym-awq-ov --target_device CPU --task text_generation 
```

> **Note:** These models aren't tested properly and their accuracy or performance may be low.

Check [parameters page](./parameters.md) for detailed descriptions of configuration options and parameter usage.

In case you want to setup model and start server in one step, follow [instructions](./starting_server.md).

> **Note:**  When using pull mode you need both read and write access rights to models repository.
