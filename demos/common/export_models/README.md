# Using GEN AI Models {#ovms_demos_common_export}

Generative AI models require additional configuration steps before deployment with OpenVINO Model Server (OVMS). There are two primary methods for preparing generative AI models:

1. **Integrated OVMS Download**: OVMS can automatically download models from the Hugging Face (HF) repository and configure them for serving. This approach leverages built-in OVMS functionality to streamline model preparation.

2. **Preprocessing with Python Script**: Alternatively, users can utilize the provided Python script to export and configure models prior to server deployment.

Choose the method that best fits your deployment workflow and requirements.

## 1. Using GEN AI models with OVMS binary

### Starting the mediapipe graph or LLM models
Since 2025.2 you can start server with single mediapipe graph, or LLM model that is already present in local filesystem with:

```{code}
docker run -d --rm -v <model_repository_path>:/models -p 9000:9000 -p 8000:8000 openvino/model_server:latest \
--model_path <path_to_model> --model_name <model_name> --port 9000 --rest_port 8000
```

Server will detect the type of requested servable (model or mediapipe graph) and load it accordingly. This detection is based on the presence of a `.pbtxt` file, which defines the Mediapipe graph structure.

*Note*: There is no online model modification nor versioning capability as of now for graphs, LLM like models.

### Starting the LLM model from HF directly

There is a special mode to make OVMS pull the model from Hugging Face before starting the service:

```{code}
docker run -d --rm -v <model_repository_path>:/models openvino/model_server:latest --pull --source_model <model_name_in_HF> --model_repository_path /models --model_name <external_model_name> --task <task> [TASK_SPECIFIC_OPTIONS]
```

| option                    | description                                                                                   |
|---------------------------|-----------------------------------------------------------------------------------------------|
| `--pull`                  | Instructs the server to run in pulling mode to get the model from the Hugging Face repository |
| `--source_model`          | Specifies the model name in the Hugging Face model repository (optional - if empty model_name is used) |
| `--model_repository_path` | Directory where all required model files will be saved                                        |
| `--model_name`            | Name of the model as exposed externally by the server                                         |
| `--task`                  | Defines the task the model will support (e.g., text_generation/embedding, rerank, image_generation)                       |

It will prepare all needed configuration files to support LLMS with OVMS in model repository

In case you do not want to prepare model repository before starting the server in one command you can run OVMS with:

```{code}
docker run -d --rm -v <model_repository_path>:/models openvino/model_server:latest --source_model <model_name_in_HF> --model_repository_path /models --model_name <ovms_servable_name> --task <task> --task_params <task_params>
```

It will download required model files, prepare configuration for OVMS and start serving the model. Available tasks are:

- `text_generation`: Generates text based on input prompts, useful for applications like chatbots or creative writing. Example: Provide a prompt and receive a coherent paragraph as output.
- `rerank`: Reorders search results or predictions based on relevance or specific criteria. Example: Improve search engine results by reranking based on user preferences.
- `embeddings`: Converts input data (e.g., text or images) into numerical representations for machine learning tasks. Example: Generate embeddings for text to use in similarity searches.
- `image_generation`: Creates images based on textual descriptions or other inputs. Example: Generate an image of a "sunset over mountains" from a text prompt.

Each task has its own specific additional parameters that can be used. Refer to the `--help` command for details.

### Starting the LLM model from local storage

In case you have predownloaded the model files from HF but you lack OVMS configuration files you can start OVMS with
```{code}
docker run -d --rm -v <model_repository_path>:/models openvino/model_server:latest --source_model <model_name_in_HF> --model_repository_path <path_where_to_store_ovms_config_files> --model_name <external_model_name> --task <task> [TASK_SPECIFIC_OPTIONS]
```
This command will create graph.pbtxt in the ```model_repository_path/source_model``` path.

## 2. Preparing models using python script

This script automates exporting models from Hugging Faces hub or fine-tuned in PyTorch format to the `models` repository for deployment with OpenVINO Model Server. In one step it prepares a complete set of resources in the `models` repository for a supported GenAI use case.

### Quick Start
```console
git clone https://github.com/openvinotoolkit/model_server
cd model_server/demos/common/export_models
pip install -q -r requirements.txt
mkdir models
python export_model.py --help
```
Expected Output:
```console
usage: export_model.py [-h] {text_generation,embeddings,rerank,image_generation} ...

Export Hugging face models to OVMS models repository including all configuration for deployments

positional arguments:
  {text_generation,embeddings,rerank,image_generation}
                        subcommand help
    text_generation     export model for chat and completion endpoints
    embeddings          export model for embeddings endpoint
    rerank              export model for rerank endpoint
    image_generation    export model for image generation endpoint
```
For every use case subcommand there is adjusted list of parameters:

```console
python export_model.py text_generation --help
```
Expected Output:
```console
usage: export_model.py text_generation [-h] [--model_repository_path MODEL_REPOSITORY_PATH] --source_model SOURCE_MODEL [--model_name MODEL_NAME] [--weight-format PRECISION] [--config_file_path CONFIG_FILE_PATH] [--overwrite_models] [--target_device TARGET_DEVICE]
                                       [--ov_cache_dir OV_CACHE_DIR] [--pipeline_type {LM,LM_CB,VLM,VLM_CB,AUTO}] [--kv_cache_precision {u8}] [--extra_quantization_params EXTRA_QUANTIZATION_PARAMS] [--enable_prefix_caching] [--disable_dynamic_split_fuse]
                                       [--max_num_batched_tokens MAX_NUM_BATCHED_TOKENS] [--max_num_seqs MAX_NUM_SEQS] [--cache_size CACHE_SIZE] [--draft_source_model DRAFT_SOURCE_MODEL] [--draft_model_name DRAFT_MODEL_NAME] [--max_prompt_len MAX_PROMPT_LEN]

options:
  -h, --help            show this help message and exit
  --model_repository_path MODEL_REPOSITORY_PATH
                        Where the model should be exported to
  --source_model SOURCE_MODEL
                        HF model name or path to the local folder with PyTorch or OpenVINO model
  --model_name MODEL_NAME
                        Model name that should be used in the deployment. Equal to source_model if HF model name is used
  --weight-format PRECISION
                        precision of the exported model
  --config_file_path CONFIG_FILE_PATH
                        path to the config file
  --overwrite_models    Overwrite the model if it already exists in the models repository
  --target_device TARGET_DEVICE
                        CPU, GPU, NPU or HETERO, default is CPU
  --ov_cache_dir OV_CACHE_DIR
                        Folder path for compilation cache to speedup initialization time
  --pipeline_type {LM,LM_CB,VLM,VLM_CB,AUTO}
                        Type of the pipeline to be used. AUTO is used by default
  --kv_cache_precision {u8}
                        u8 or empty (model default). Reduced kv cache precision to u8 lowers the cache size consumption.
  --extra_quantization_params EXTRA_QUANTIZATION_PARAMS
                        Add advanced quantization parameters. Check optimum-intel documentation. Example: "--sym --group-size -1 --ratio 1.0 --awq --scale-estimation --dataset wikitext2"
  --enable_prefix_caching
                        This algorithm is used to cache the prompt tokens.
  --disable_dynamic_split_fuse
                        The maximum number of tokens that can be batched together.
  --max_num_batched_tokens MAX_NUM_BATCHED_TOKENS
                        empty or integer. The maximum number of tokens that can be batched together.
  --max_num_seqs MAX_NUM_SEQS
                        256 by default. The maximum number of sequences that can be processed together.
  --cache_size CACHE_SIZE
                        KV cache size in GB
  --draft_source_model DRAFT_SOURCE_MODEL
                        HF model name or path to the local folder with PyTorch or OpenVINO draft model. Using this option will create configuration for speculative decoding
  --draft_model_name DRAFT_MODEL_NAME
                        Draft model name that should be used in the deployment. Equal to draft_source_model if HF model name is used. Available only in draft_source_model has been specified.
  --max_prompt_len MAX_PROMPT_LEN
                        Sets NPU specific property for maximum number of tokens in the prompt. Not effective if target device is not NPU
```

### Model Export Examples

#### Text Generation Models

##### Text Generation CPU Deployment
```console
python export_model.py text_generation --source_model meta-llama/Meta-Llama-3-8B-Instruct --weight-format fp16 --kv_cache_precision u8 --config_file_path models/config_all.json --model_repository_path models
```

##### GPU Deployment (Low Concurrency, Limited Memory)
Text generation for GPU target device with limited memory without dynamic split fuse algorithm (recommended for usage in low concurrency):
```console
python export_model.py text_generation --source_model meta-llama/Meta-Llama-3-8B-Instruct --weight-format int4 --config_file_path models/config_all.json --model_repository_path models --target_device GPU --disable_dynamic_split_fuse --max_num_batched_tokens 8192 --cache_size 1
```
##### GPU Deployment (High Concurrency, Dynamic Split Fuse Enabled)
Text generation for GPU target device with limited memory with enabled dynamic split fuse algorithm (recommended for usage in high concurrency):
```console
python export_model.py text_generation --source_model meta-llama/Meta-Llama-3-8B-Instruct --weight-format int4 --config_file_path models/config_all.json --model_repository_path models --target_device GPU --cache_size 3
```
##### NPU Deployment
Text generation for NPU target device. Command below sets max allowed prompt size and configures model compilation directory to speedup initialization time:
```console
python export_model.py text_generation --source_model meta-llama/Llama-3.2-3B-Instruct --config_file_path models/config_all.json --model_repository_path models --target_device NPU --max_prompt_len 2048 --ov_cache_dir ./models/.ov_cache
```

#### Embedding Models

##### Embeddings with deployment on a single CPU host:
```console
python export_model.py embeddings --source_model Alibaba-NLP/gte-large-en-v1.5 --weight-format int8 --config_file_path models/config_all.json
```

##### Embeddings with deployment on a dual CPU host:
```console
python export_model.py embeddings --source_model Alibaba-NLP/gte-large-en-v1.5 --weight-format int8 --config_file_path models/config_all.json --num_streams 2
```

##### With Input Truncation
By default, embeddings endpoint returns an error when the input exceed the maximum model context length.
It is possible to change the behavior to truncate prompts automatically to fit the model. Add `--truncate` option in the export command.
```console
python export_model.py embeddings \
    --source_model BAAI/bge-large-en-v1.5 \
    --weight-format int8 \
    --config_file_path models/config_all.json \
    --truncate
```
> **Note:** When using `--truncate`, inputs exceeding the model's context length will be automatically shortened rather than producing an error. While this prevents failures, it may impact accuracy as only a portion of the input is analyzed.

#### Reranking Models
```console
python export_model.py rerank \
    --source_model BAAI/bge-reranker-large \
    --weight-format int8 \
    --config_file_path models/config_all.json \
    --num_streams 2
```

#### Image Generation Models
```console
python export_model.py image_generation \
    --source_model dreamlike-art/dreamlike-anime-1.0 \
    --weight-format int8 \
    --config_file_path models/config_all.json \
    --max_resolution 2048x2048
```

### Deployment example

After exporting models using the commands above (which use `--model_repository_path models` and `--config_file_path models/config_all.json`), you can deploy them with either with Docker or on Baremetal.

#### CPU Deployment with Docker
```bash
docker run -d --rm -p 8000:8000 \
    -v $(pwd)/models:/workspace:ro \
    openvino/model_server:latest \
    --port 9000 --rest_port 8000 \
    --config_path /workspace/config_all.json
```

#### GPU Deployment with Docker
```bash
docker run -d --rm -p 8000:8000 \
    --device /dev/dri \
    --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) \
    -v $(pwd)/models:/workspace:ro \
    openvino/model_server:latest-gpu \
    --port 9000 --rest_port 8000 \
    --config_path /workspace/config_all.json
```

#### Baremetal Deployment
```bat
ovms --port 9000 --rest_port 8000 --config_path models/config_all.json
```

### Memory Requirements

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

