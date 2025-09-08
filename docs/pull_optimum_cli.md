# OVMS Pull mode with optimum cli {#ovms_docs_pull_optimum}

This documents describes how to leverage OpenVINO Model Server (OVMS) pull feature to automate deployment configuration with Generative AI models when pulling outside of [OpenVINO organization](https://huggingface.co/OpenVINO) from HF.

You have to use docker image with optimum-cli or install additional python dependencies to the baremetal package. Follow the steps described below.

Pulling models with automatic conversion and quantization (requires optimum-cli). Include additional consideration like longer time for deployment and pulling model data (original model) from HF, model memory for conversion, diskspace.

Note: Pulling the models from HuggingFace Hub can automate conversion and compression. It might however increase memory usage during the model conversion and requires downloading the original model.

## OVMS building and installation for optimum-cli integration
### Build python docker image
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make python_image
```

Example pull command with optimum model cache directory sharing and setting HF_TOKEN environment variable for model download authentication.

```bash
docker run -e HF_TOKEN=hf_YOURTOKEN -e HF_HOME=/hf_home/cache --user $(id -u):$(id -g) --group-add=$(id -g) -v /opt/home/user/.cache/huggingface/:/hf_home/cache -v $(pwd)/models:/models:rw openvino/model_server:py --pull --model_repository_path /models --source_model meta-llama/Meta-Llama-3-8B-Instruct
```

### Install optimum-cli
Install python on your baremetal system from `https://www.python.org/downloads/` and run the commands:
```console
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/3/demos/common/export_models/requirements.txt
```

or use the python binary from the ovms_windows_python_on.zip or ovms.tar.gz package - see [deployment instructions](deploying_server_baremetal.md) for details.

```bat
curl -L https://github.com/openvinotoolkit/model_server/releases/download/v2025.3/ovms_windows_python_on.zip -o ovms.zip
tar -xf ovms.zip
```
```bat
ovms\setupvars.bat
ovms\python\python -m pip install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/3/demos/common/export_models/requirements.txt
```
Then use the ovms cli commands described in `Pulling the models` section

## Pulling the models

Using `--pull` parameter, we can use OVMS to download the model, quantize and compress to required format. The model will be prepared to the deployment in the configured `--model_repository_path`. Without `--pull` parameter, OVMS will start serving immediately after the model is prepared.

::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed

```text
docker run $(id -u):$(id -g) --rm -v <model_repository_path>:/models:rw openvino/model_server:py --pull --source_model <model_name_in_HF> --model_repository_path /models --model_name <external_model_name> --target_device <DEVICE> --task <task> [TASK_SPECIFIC_PARAMETERS]
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

Example for pulling `Qwen/Qwen3-8B`:

```bat
ovms --pull --source_model "Qwen/Qwen3-8B" --model_repository_path /models --model_name Qwen3-8B --target_device CPU --task text_generation 
```
::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed

```text
docker run $(id -u):$(id -g) --rm -v <model_repository_path>:/models:rw openvino/model_server:py --pull --source_model "Qwen/Qwen3-8B" --model_repository_path /models --model_name Qwen3-8B --task text_generation
```
:::

:::{tab-item} On Baremetal Host
:sync: baremetal
**Required:** OpenVINO Model Server package - see [deployment instructions](./deploying_server_baremetal.md) for details.

```bat
ovms --pull --source_model "Qwen/Qwen3-8B" --model_repository_path /models --model_name Qwen3-8B --task text_generation 
```
:::
::::


It will prepare all needed configuration files to support LLMS with OVMS in the model repository. Check [parameters page](./parameters.md) for detailed descriptions of configuration options and parameter usage.


*Note:*
When using pull mode you need both read and write access rights to models repository.
