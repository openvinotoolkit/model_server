# Exporting GEN AI models for deployments in OpenVINO Model Server

This script automates exporting models from Hugging Faces hub or fine-tuned in PyTorch format to the models repository for deployment for the model serving.
In one step it prepares a complete set of resources in the models repository for a supported GenAI use case.

```console
git clone https://github.com/openvinotoolkit/model_server
cd model_server/demos/common/export_models
pip install -q -r requirements.txt
python export_model.py --help
usage: export_model.py [-h] {text_generation,embeddings,rerank} ...

Export Hugging face models to OVMS models repository including all configuration for deployments

positional arguments:
  {text_generation,embeddings,rerank}
                        subcommand help
    text_generation     export model for chat and completion endpoints
    embeddings          export model for embeddings endpoint
    rerank              export model for rerank endpoint
```
For every use case subcommand there is adjusted list of parameters:

```console
python export_model.py text_generation --help 
usage: export_model.py text_generation [-h] [--model_repository_path MODEL_REPOSITORY_PATH] --source_model SOURCE_MODEL [--model_name MODEL_NAME] [--weight-format PRECISION] [--config_file_path CONFIG_FILE_PATH]
                                       [--overwrite_models] [--target_device TARGET_DEVICE] [--kv_cache_precision {u8}] [--enable_prefix_caching] [--disable_dynamic_split_fuse]
                                       [--max_num_batched_tokens MAX_NUM_BATCHED_TOKENS] [--max_num_seqs MAX_NUM_SEQS] [--cache_size CACHE_SIZE]

options:
  -h, --help            show this help message and exit
  --model_repository_path MODEL_REPOSITORY_PATH
                        Where the model should be exported to
  --source_model SOURCE_MODEL
                        HF model name or path to the local folder with PyTorch or OpenVINO model
  --model_name MODEL_NAME
                        Model name that should be used in the deployment. Equal to source_name if HF model name is used
  --weight-format PRECISION
                        precision of the exported model
  --config_file_path CONFIG_FILE_PATH
                        path to the config file
  --overwrite_models    Overwrite the model if it already exists in the models repository
  --target_device TARGET_DEVICE
                        CPU or GPU, default is CPU
  --kv_cache_precision {u8}
                        u8 or empty (model default). Reduced kv cache precision to u8 lowers the cache size consumption.
  --enable_prefix_caching
                        This algorithm is used to cache the prompt tokens.
  --disable_dynamic_split_fuse
                        The maximum number of tokens that can be batched together.
  --max_num_batched_tokens MAX_NUM_BATCHED_TOKENS
                        empty or integer. The maximum number of tokens that can be batched together.
  --max_num_seqs MAX_NUM_SEQS
                        256 by default. The maximum number of sequences that can be processed together.
  --cache_size CACHE_SIZE
                        cache size in GB
```

## Examples how models can be exported

Text generation for CPU target device:
```console
mkdir -p models
python export_model.py text_generation --source_model meta-llama/Meta-Llama-3-8B-Instruct --weight-format fp16 --kv_cache_precision u8 --config_file_path models/config_all.json --model_repository_path models 
```

Text generation for GPU target device with limited memory without dynamic split fuse algorithm (recommended for usage in low concurrency):
```console
mkdir -p models
python export_model.py text_generation --source_model meta-llama/Meta-Llama-3-8B-Instruct --weight-format int4 --config_file_path models/config_all.json --model_repository_path models --target_device GPU --disable_dynamic_split_fuse --max_num_batched_tokens 8192 --cache_size 2
```

Text generation for GPU target device with limited memory with enabled dynamic split fuse algorithm (recommended for usage in high concurrency):
```console
mkdir -p models
python export_model.py text_generation --source_model meta-llama/Meta-Llama-3-8B-Instruct --weight-format int4 --config_file_path models/config_all.json --model_repository_path models --target_device GPU --cache_size 2
```

Embeddings with deployment on a single CPU host:
```console
mkdir -p models
python export_model.py embeddings --source_model Alibaba-NLP/gte-large-en-v1.5 --weight-format int8  --config_file_path models/config_all.json
```

Embeddings with deployment on a dual CPU host:
```console
mkdir -p models
python export_model.py embeddings --source_model Alibaba-NLP/gte-large-en-v1.5 --weight-format int8  --config_file_path models/config_all.json --num_streams 2
```

By default, embeddings endpoint returns an error when the input exceed the maximum model context length.
It is possible to change the behavior to truncate prompts automatically to fit the model. Add `--truncate` option in the export command.
```console
mkdir -p models
python export_model.py embeddings --source_model BAAI/bge-large-en-v1.5 --weight-format int8 --config_file_path models/config_all.json --truncate
```
Note, that truncating input will prevent errors but the accuracy might be impacted as only part of the input will be analyzed.

Reranking:
```console
mkdir -p models
python export_model.py rerank --source_model BAAI/bge-reranker-large --weight-format int8  --config_file_path models/config_all.json --num_streams 2
```

## Deployment example

The export commands above deploy the models in `models/` folder and the configuration file is created in `models/config_all.json`.

```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/workspace:ro openvino/model_server:latest --port 9000 --rest_port 8000 --config_path /workspace/config_all.json
```

In case GPU is the target device in any model, the following command can be applied:
```bash
docker run -d --rm -p 8000:8000 --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -v $(pwd)/models:/workspace:ro openvino/model_server:latest-gpu --port 9000 --rest_port 8000 --config_path /workspace/config_all.json
```

For baremetal deployment, the equivalent command would be:
```console
ovms --port 9000 --rest_port 8000 --config_path models/config_all.json
```