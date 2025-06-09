# RAG demo with all execution steps delegated to the OpenVINO Model Server {#ovms_demos_continuous_batching_rag}


## Creating models repository for all the endpoints

### 1. Deploy the Model
::::{tab-set}

:::{tab-item} With Docker
**Required:** Docker Engine installed

```bash
mkdir models
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --source_model OpenVINO/Phi-3.5-mini-instruct-int4-ov --model_repository_path models

```
:::

:::{tab-item} On Baremetal Host
**Required:** OpenVINO Model Server package - see [deployment instructions](../deploying_server_baremetal.md) for details.

```bat
mkdir models
ovms.exe --pull --model_repository_path models --source_model OpenVINO/Qwen3-8B-int4-ov
ovms.exe --pull --model_repository_path models --source_model OpenVINO/bge-base-en-v1.5-fp16-ov --task embeddings
ovms.exe --pull --model_repository_path models --source_model OpenVINO/bge-reranker-base-fp16-ov --task rerank

ovms.exe --add_to_config models --model_name llm_text --model_path OpenVINO/Qwen3-8B-int4-ov
ovms.exe --add_to_config models --model_name embeddings --model_path OpenVINO/bge-base-en-v1.5-fp16-ov
ovms.exe --add_to_config models --model_name rerank --model_path OpenVINO/bge-reranker-base-fp16-ov

ovms.exe --rest_port 8000 --config_path models\config.json
```
:::
::::

## Deploying the model server


### With Docker
```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/workspace:ro openvino/model_server:latest --rest_port 8000 --config_path /workspace/config_all.json
```

### On Baremetal
```bat
ovms --rest_port 8000 --config_path ./models/config_all.json
```

## Using RAG

When the model server is deployed and serving all 3 endpoints, run the [jupyter notebook](https://github.com/openvinotoolkit/model_server/blob/main/demos/continuous_batching/rag/rag_demo.ipynb) to use RAG chain with a fully remote execution.