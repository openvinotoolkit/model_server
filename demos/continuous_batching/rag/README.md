# RAG demo with all execution steps delegated to the OpenVINO Model Server {#ovms_demos_continuous_batching_rag}

## Creating models repository for all the endpoints

### 1. Download the preconfigured models using ovms --pull option with form [HugginFaces Hub OpenVINO organization](https://huggingface.co/OpenVINO)
::::{tab-set}

:::{tab-item} With Docker
**Required:** Docker Engine installed

```bash
mkdir models
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/Qwen3-8B-int4-ov
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/bge-base-en-v1.5-fp16-ov --task embeddings
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/bge-reranker-base-fp16-ov --task rerank

docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --add_to_config /models --model_name OpenVINO/Qwen3-8B-int4-ov --model_path OpenVINO/Qwen3-8B-int4-ov
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --add_to_config /models --model_name OpenVINO/bge-base-en-v1.5-fp16-ov --model_path OpenVINO/bge-base-en-v1.5-fp16-ov
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --add_to_config /models --model_name OpenVINO/bge-reranker-base-fp16-ov --model_path OpenVINO/bge-reranker-base-fp16-ov

docker run -d --rm -p 8023:8023 -v $(pwd)/models:/workspace:ro openvino/model_server:latest --rest_port 8023 --config_path /workspace/config.json
```
:::

:::{tab-item} On Baremetal Host
**Required:** OpenVINO Model Server package - see [deployment instructions](../deploying_server_baremetal.md) for details.

```bat
mkdir models
ovms.exe --pull --model_repository_path models --source_model OpenVINO/Qwen3-8B-int4-ov
ovms.exe --pull --model_repository_path models --source_model OpenVINO/bge-base-en-v1.5-fp16-ov --task embeddings
ovms.exe --pull --model_repository_path models --source_model OpenVINO/bge-reranker-base-fp16-ov --task rerank

ovms.exe --add_to_config models --model_name OpenVINO/Qwen3-8B-int4-ov --model_path OpenVINO/Qwen3-8B-int4-ov
ovms.exe --add_to_config models --model_name OpenVINO/bge-base-en-v1.5-fp16-ov --model_path OpenVINO/bge-base-en-v1.5-fp16-ov
ovms.exe --add_to_config models --model_name OpenVINO/bge-reranker-base-fp16-ov --model_path OpenVINO/bge-reranker-base-fp16-ov

ovms.exe --rest_port 8000 --config_path %cd%\models\config.json
```
:::
::::

## Using RAG

When the model server is deployed and serving all 3 endpoints, run the [jupyter notebook](https://github.com/openvinotoolkit/model_server/blob/main/demos/continuous_batching/rag/rag_demo.ipynb) to use RAG chain with a fully remote execution.

## Alternate way of creating models repository for all the endpoints using export_model.py with models conversion

```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/1/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/1/demos/common/export_models/requirements.txt

mkdir models
python export_model.py text_generation --source_model meta-llama/Meta-Llama-3-8B-Instruct --weight-format int8 --kv_cache_precision u8 --config_file_path models/config_all.json --model_repository_path models
python export_model.py embeddings --source_model Alibaba-NLP/gte-large-en-v1.5 --weight-format int8 --config_file_path models/config_all.json
python export_model.py rerank --source_model BAAI/bge-reranker-large --weight-format int8  --config_file_path models/config_all.json
```

## Deploying the model server

### With Docker
```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/workspace:ro openvino/model_server:latest --rest_port 8000 --config_path /workspace/config_all.json
```

### On Baremetal
```bat
ovms --rest_port 8000 --config_path ./models/config_all.json
```