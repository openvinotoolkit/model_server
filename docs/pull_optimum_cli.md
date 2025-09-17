# OVMS Pull mode with optimum cli {#ovms_docs_pull_optimum}

This documents describes how to leverage OpenVINO Model Server (OVMS) pull feature to automate deployment of generative models from HF with conversion and quantization in runtime.

You have to use docker image with optimum-cli `openvino/model_server:latest-py` or install additional python dependencies to the baremetal package. Follow the steps described below.

> **Note:** This procedure might increase memory usage during the model conversion and requires downloading the original model. Expect memory usage at least to the level of original model size during the conversion.

## Add optimum-cli to OVMS installation on windows

```bat
curl -L https://github.com/openvinotoolkit/model_server/releases/download/v2025.3/ovms_windows_python_on.zip -o ovms.zip
tar -xf ovms.zip
ovms\setupvars.bat
ovms\python\python -m pip install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/requirements.txt
```
Then use the ovms cli commands described in `Pulling the models` section

## Pulling the models

Using `--pull` parameter, we can use OVMS to download the model, quantize and compress to required format. The model will be prepared to the deployment in the configured `--model_repository_path`. Without `--pull` parameter, OVMS will start serving immediately after the model is prepared.

::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed

```text
docker run $(id -u):$(id -g) --rm -v <model_repository_path>:/models:rw openvino/model_server:latest-py --pull --source_model <model_name_in_HF> --model_repository_path /models --model_name <external_model_name> --target_device <DEVICE> --task <task> [TASK_SPECIFIC_PARAMETERS]
```
:::

:::{tab-item} On Baremetal Host
:sync: baremetal
```text
ovms --pull --source_model <model_name_in_HF> --model_repository_path <model_repository_path> --model_name <external_model_name> --target_device <DEVICE> --task <task> [TASK_SPECIFIC_PARAMETERS]
```
:::
::::

Example for pulling `Qwen/Qwen3-8B`:

```bat
ovms --pull --source_model "Qwen/Qwen3-8B" --model_repository_path /models --model_name Qwen3-8B --target_device CPU --task text_generation --weight-format int8 
```
::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed

```text
docker run $(id -u):$(id -g) --rm -v <model_repository_path>:/models:rw openvino/model_server:latest-py --pull --source_model "Qwen/Qwen3-8B" --model_repository_path /models --model_name Qwen3-8B --task text_generation --weight-format int8
```
:::

:::{tab-item} On Baremetal Host
:sync: baremetal
**Required:** OpenVINO Model Server package - see [deployment instructions](./deploying_server_baremetal.md) for details.

```bat
ovms --pull --source_model "Qwen/Qwen3-8B" --model_repository_path /models --model_name Qwen3-8B --task text_generation --weight-format int8
```
:::
::::


Check [parameters page](./parameters.md) for detailed descriptions of configuration options and parameter usage.


## Additional considerations

When using pull mode you need both read and write access rights to local models repository.

You need read permissions to the source model in Hugging Face Hub. Pass the access token via environment variable just like with hf_cli application.

You can mount the HuggingFace cache to avoid downloading the original model in case it was pulled earlier.

Below is an example pull command with optimum model cache directory sharing and setting HF_TOKEN environment variable for model download authentication:

```bash
docker run -e HF_TOKEN=hf_YOURTOKEN -e HF_HOME=/hf_home/cache --user $(id -u):$(id -g) --group-add=$(id -g) -v /opt/home/user/.cache/huggingface/:/hf_home/cache -v $(pwd)/models:/models:rw openvino/model_server:latest-py --pull --model_repository_path /models --source_model meta-llama/Meta-Llama-3-8B-Instruct
```

