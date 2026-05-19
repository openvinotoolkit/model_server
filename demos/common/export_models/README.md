# Exporting models using script {#ovms_demos_common_export}

This documents describes how to export, optimize and configure models prior to server deployment with provided python script. This approach is more flexible than using [pull feature](../../../docs/pull_hf_models.md) from OVMS as it allows for using models that were not optimized beforehand and provided in [OpenVINO organization](https://huggingface.co/OpenVINO) in HuggingFace, but requires having Python set up to work.
> **Warning:** This script uses option `--trust-remote-code`, which allows model-provided Python code to run on your machine during export. Use only trusted models/sources, review repository code before running, and avoid executing this script in sensitive environments.

## What it does

This script automates exporting models from Hugging Faces hub or fine-tuned in PyTorch format to the `models` repository for deployment with OpenVINO Model Server. In one step it prepares a complete set of resources in the `models` repository for a supported GenAI use case.

## Quick Start
```console
git clone https://github.com/openvinotoolkit/model_server
cd model_server/demos/common/export_models
pip install -q -r requirements.txt
mkdir models
python export_model.py --help
```
Expected Output:
```console
usage: export_model.py [-h] {text_generation,embeddings_ov,rerank,rerank_ov,image_generation,text2speech,speech2text} ...

Export Hugging face models to OVMS models repository including all configuration for deployments

positional arguments:
  {text_generation,embeddings_ov,rerank,rerank_ov,image_generation,text2speech,speech2text}
                        subcommand help
    text_generation     export model for chat and completion endpoints
    embeddings_ov       export model for embeddings endpoint with directory structure aligned with OpenVINO tools
    rerank              [deprecated] export model for rerank endpoint with models split into separate, versioned directories
    rerank_ov           export model for rerank endpoint with directory structure aligned with OpenVINO tools
    image_generation    export model for image generation endpoint
    text2speech         export model for text2speech endpoint
    speech2text         export model for speech2text endpoint

options:
  -h, --help            show this help message and exit
```
For every use case subcommand there is adjusted list of parameters:

```console
python export_model.py text_generation --help
```

> Note: Exporting some models might require different transformers version than specified in requirements.txt. Check [supported models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/). If custom transformers version is required, install it afterwards via `pip install transformers==<version>`



## Model Export Examples

### Text Generation Models

#### Text Generation CPU Deployment
```console
python export_model.py text_generation --source_model meta-llama/Meta-Llama-3-8B-Instruct --weight-format fp16 --kv_cache_precision u8 --config_file_path models/config_all.json --model_repository_path models
```

#### GPU Deployment (Low Concurrency, Limited Memory)
Text generation for GPU target device with limited memory without dynamic split fuse algorithm (recommended for usage in low concurrency):
```console
python export_model.py text_generation --source_model meta-llama/Meta-Llama-3-8B-Instruct --weight-format int4 --config_file_path models/config_all.json --model_repository_path models --target_device GPU --disable_dynamic_split_fuse --max_num_batched_tokens 8192
```
#### GPU Deployment (High Concurrency, Dynamic Split Fuse Enabled)
Text generation for GPU target device with limited memory with enabled dynamic split fuse algorithm (recommended for usage in high concurrency):
```console
python export_model.py text_generation --source_model meta-llama/Meta-Llama-3-8B-Instruct --weight-format int4 --config_file_path models/config_all.json --model_repository_path models --target_device GPU
```
#### NPU Deployment
Text generation for NPU target device. Command below sets max allowed prompt size and configures model compilation directory to speedup initialization time:
```console
python export_model.py text_generation --source_model meta-llama/Llama-3.2-3B-Instruct --config_file_path models/config_all.json --model_repository_path models --target_device NPU --max_prompt_len 2048 --ov_cache_dir ./models/.ov_cache
```
> **Note:** Some models like `mistralai/Mistral-7B-Instruct-v0.3` might fail to export because the task can't be determined automatically. In such situation it can be set in `--extra_quantization_parameters`. For example:
```console
python export_model.py text_generation --source_model mistralai/Mistral-7B-Instruct-v0.3 --model_repository_path models --extra_quantization_params "--task text-generation-with-past"
```
> **Note:** Model `microsoft/Phi-3.5-vision-instruct` requires one manual adjustments after export in the file `generation_config.json` like in the [PR](https://huggingface.co/microsoft/Phi-3.5-vision-instruct/discussions/40/files).
It will ensure, the generation stops after eos token.

> **Note:** In order to export GPTQ models, you need to install also package `auto_gptq` via command `BUILD_CUDA_EXT=0 pip install auto_gptq` on Linux and `set BUILD_CUDA_EXT=0 && pip install auto_gptq` on Windows. 


### Embedding Models

#### Embeddings with deployment on a single CPU host:
```console
python export_model.py embeddings_ov --source_model BAAI/bge-large-en-v1.5 --weight-format int8 --config_file_path models/config_all.json
```

#### Embeddings with deployment on a dual CPU host:
```console
python export_model.py embeddings_ov --source_model BAAI/bge-large-en-v1.5 --weight-format int8 --config_file_path models/config_all.json --num_streams 2
```

#### Embeddings with pooling parameter
Supported poolings: `LAST`, `MEAN`, `CLS` (default).
```console
python export_model.py embeddings_ov --source_model Qwen/Qwen3-Embedding-0.6B --pooling LAST --weight-format fp16 --config_file_path models/config_all.json
```

#### Embeddings with `sentence_transformers` library
Some embedding models require special handling during export. For example:
```console
python export_model.py embeddings_ov --source_model Alibaba-NLP/gte-large-en-v1.5 --extra_quantization_params "--library sentence_transformers" --weight-format fp16 --config_file_path models/config_all.json
```
Known models that require it:
- Alibaba-NLP/gte-large-en-v1.5
- nomic-ai/nomic-embed-text-v1.5



#### With Input Truncation
By default, embeddings endpoint returns an error when the input exceed the maximum model context length.
It is possible to change the behavior to truncate prompts automatically to fit the model. Add `--truncate` option in the export command.
```console
python export_model.py embeddings_ov \
    --source_model BAAI/bge-large-en-v1.5 \
    --weight-format int8 \
    --config_file_path models/config_all.json \
    --truncate
```
> **Note:** When using `--truncate`, inputs exceeding the model's context length will be automatically shortened rather than producing an error. While this prevents failures, it may impact accuracy as only a portion of the input is analyzed.

### Reranking Models
```console
python export_model.py rerank_ov \
    --source_model BAAI/bge-reranker-large \
    --weight-format int8 \
    --config_file_path models/config_all.json \
    --num_streams 2
```

### Image Generation Models
```console
python export_model.py image_generation \
    --source_model dreamlike-art/dreamlike-anime-1.0 \
    --weight-format int8 \
    --config_file_path models/config_all.json \
    --max_resolution 2048x2048
```

## Deployment example

After exporting models using the commands above (which use `--model_repository_path models` and `--config_file_path models/config_all.json`), you can deploy them with either with Docker or on Baremetal.

### CPU Deployment with Docker
```bash
docker run -d --rm -p 8000:8000 \
    -v $(pwd)/models:/workspace:ro \
    openvino/model_server:latest \
    --port 9000 --rest_port 8000 \
    --config_path /workspace/config_all.json
```

### GPU Deployment with Docker
```bash
docker run -d --rm -p 8000:8000 \
    --device /dev/dri \
    --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) \
    -v $(pwd)/models:/workspace:ro \
    openvino/model_server:latest-gpu \
    --port 9000 --rest_port 8000 \
    --config_path /workspace/config_all.json
```

### Baremetal Deployment
```bat
ovms --port 9000 --rest_port 8000 --config_path models/config_all.json
```

## Memory Requirements

Exporting large models requires substantial host memory, especially during quantization. For systems with limited RAM, consider configuring additional swap space or virtual memory.

:::{dropdown} Virtual Memory Configuration in Windows

  1.  Open System Properties. Right-click on Start, click on System, then Click "Advanced system settings"
  3.  Under "Performance", click "Settings"
  4.  Navigate to the "Advanced" tab → click "Change" under Virtual Memory
  5.  Uncheck "Automatically manage paging file size for all drives"
  6.  Select the desired drive and choose "Custom size:"
  7.  Set:
  - Initial size (MB): 2000 
  - Maximum size (MB): 20000 MB (adjust depending on model size)
  8.  Click "Set", then "OK"
  9.  You may need to Restart your computer for changes to take effect

:::

