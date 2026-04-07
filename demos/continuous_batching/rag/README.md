# RAG demo with OpenVINO Model Server {#ovms_demos_continuous_batching_rag}

## Creating models repository for all the endpoints with ovms --pull or python export_model.py script

### Download the preconfigured models using ovms --pull option from [Hugging Face Hub OpenVINO organization](https://huggingface.co/OpenVINO) (Simple usage)
::::{tab-set}

:::{tab-item} With Docker
**Required:** Docker Engine installed

```bash
mkdir models
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/Qwen3-8B-int4-ov --task text_generation
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/bge-base-en-v1.5-fp16-ov --task embeddings
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/bge-reranker-base-fp16-ov --task rerank

docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name OpenVINO/Qwen3-8B-int4-ov --model_path OpenVINO/Qwen3-8B-int4-ov
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name OpenVINO/bge-base-en-v1.5-fp16-ov --model_path OpenVINO/bge-base-en-v1.5-fp16-ov
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name OpenVINO/bge-reranker-base-fp16-ov --model_path OpenVINO/bge-reranker-base-fp16-ov
```
:::

:::{tab-item} On Baremetal Host
**Required:** OpenVINO Model Server package - see [deployment instructions](../../../docs/deploying_server_baremetal.md) for details.

```bat
mkdir models

ovms --pull --model_repository_path models --source_model OpenVINO/Qwen3-8B-int4-ov --task text_generation
ovms --pull --model_repository_path models --source_model OpenVINO/bge-base-en-v1.5-fp16-ov --task embeddings
ovms --pull --model_repository_path models --source_model OpenVINO/bge-reranker-base-fp16-ov --task rerank

ovms --add_to_config --config_path models/config.json --model_name OpenVINO/Qwen3-8B-int4-ov --model_path OpenVINO/Qwen3-8B-int4-ov
ovms --add_to_config --config_path models/config.json --model_name OpenVINO/bge-base-en-v1.5-fp16-ov --model_path OpenVINO/bge-base-en-v1.5-fp16-ov
ovms --add_to_config --config_path models/config.json --model_name OpenVINO/bge-reranker-base-fp16-ov --model_path OpenVINO/bge-reranker-base-fp16-ov
```
:::

:::{tab-item} Windows service
**Required:** OpenVINO Model Server package - see [deployment instructions](../../../docs/deploying_server_baremetal.md) for details.
**Assumption:** install_ovms_service.bat was called without additional parameters - using default c:\models config path.
```bat
mkdir c:\models

ovms --pull --model_repository_path c:\models --source_model OpenVINO/Qwen3-8B-int4-ov --task text_generation
ovms --pull --model_repository_path c:\models --source_model OpenVINO/bge-base-en-v1.5-fp16-ov --task embeddings
ovms --pull --model_repository_path c:\models --source_model OpenVINO/bge-reranker-base-fp16-ov --task rerank

ovms --add_to_config --config_path c:\models\config.json --model_name OpenVINO/Qwen3-8B-int4-ov --model_path OpenVINO/Qwen3-8B-int4-ov
ovms --add_to_config --config_path c:\models\config.json --model_name OpenVINO/bge-base-en-v1.5-fp16-ov --model_path OpenVINO/bge-base-en-v1.5-fp16-ov
ovms --add_to_config --config_path c:\models\config.json --model_name OpenVINO/bge-reranker-base-fp16-ov --model_path OpenVINO/bge-reranker-base-fp16-ov
```
:::
::::


### Optionally, if you want to deploy different models use the build-in ovms functionality in openvino/model_server:latest-py described here [pull mode with optimum cli](../../../docs/pull_optimum_cli.md)

## Deploying the model server

### With Docker
```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/workspace:ro openvino/model_server:latest --rest_port 8000 --config_path /workspace/config.json
```
### On Baremetal Unix
```bash
ovms --rest_port 8000 --config_path models/config.json
```
### Windows
```bat
ovms --rest_port 8000 --config_path models\config.json
```

### Server as Windows Service
```bat
sc start ovms
```

## Readiness Check

Wait for the models to load. You can check the status with a simple command:
```console
curl http://localhost:8000/v3/models
```
```
{
  "data": [
    {
      "id": "OpenVINO/Qwen3-8B-int4-ov",
      "object": "model",
      "created": 1775552853,
      "owned_by": "OVMS"
    },
    {
      "id": "OpenVINO/bge-base-en-v1.5-fp16-ov",
      "object": "model",
      "created": 1775552853,
      "owned_by": "OVMS"
    },
    {
      "id": "OpenVINO/bge-reranker-base-fp16-ov",
      "object": "model",
      "created": 1775552853,
      "owned_by": "OVMS"
    }
  ],
  "object": "list"
}
```

## Using RAG

When the model server is deployed and serving all 3 endpoints, run the [jupyter notebook](https://github.com/openvinotoolkit/model_server/blob/main/demos/continuous_batching/rag/rag_demo.ipynb) to use RAG chain with a fully remote execution.
