# RAG demo with OpenVINO Model Server {#ovms_demos_continuous_batching_rag}

## Creating models repository for all the endpoints

::::{tab-set}

:::{tab-item} With Docker
**Required:** Docker Engine installed

```bash
mkdir models
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:2026.1 --pull --model_repository_path /models --source_model OpenVINO/Qwen3-8B-int4-ov --task text_generation
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:2026.1 --pull --model_repository_path /models --source_model OpenVINO/bge-base-en-v1.5-fp16-ov --task embeddings
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:2026.1 --pull --model_repository_path /models --source_model OpenVINO/bge-reranker-base-fp16-ov --task rerank

docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:2026.1 --add_to_config --config_path /models/config.json --model_name OpenVINO/Qwen3-8B-int4-ov --model_path OpenVINO/Qwen3-8B-int4-ov
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:2026.1 --add_to_config --config_path /models/config.json --model_name OpenVINO/bge-base-en-v1.5-fp16-ov --model_path OpenVINO/bge-base-en-v1.5-fp16-ov
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:2026.1 --add_to_config --config_path /models/config.json --model_name OpenVINO/bge-reranker-base-fp16-ov --model_path OpenVINO/bge-reranker-base-fp16-ov
```
:::

:::{tab-item} On Baremetal Windows
**Required:** OpenVINO Model Server package - see [deployment instructions](../../../docs/deploying_server_baremetal.md) for details.

```bat
mkdir models

ovms --pull --model_repository_path models --source_model OpenVINO/Qwen3-8B-int4-ov --task text_generation --target_device GPU
ovms --pull --model_repository_path models --source_model OpenVINO/bge-base-en-v1.5-fp16-ov --task embeddings --target_device GPU
ovms --pull --model_repository_path models --source_model OpenVINO/bge-reranker-base-fp16-ov --task rerank --target_device GPU

ovms --add_to_config --config_path models/config.json --model_name OpenVINO/Qwen3-8B-int4-ov --model_path OpenVINO/Qwen3-8B-int4-ov
ovms --add_to_config --config_path models/config.json --model_name OpenVINO/bge-base-en-v1.5-fp16-ov --model_path OpenVINO/bge-base-en-v1.5-fp16-ov
ovms --add_to_config --config_path models/config.json --model_name OpenVINO/bge-reranker-base-fp16-ov --model_path OpenVINO/bge-reranker-base-fp16-ov
```
:::
::::


> NOTE: If you want to deploy models in pytorch format you can use the built-in OVMS optimum-cli functionality of `openvino/model_server:2026.1-py` described in [pull mode with optimum cli](../../../docs/pull_optimum_cli.md)

> NOTE: You can also use [the windows service](../../../docs/windows_service.md) setup for the ease of use and shorter commands - with default model_repository_path and config_path

## Deploying the model server

::::{tab-set}

:::{tab-item} With Docker
```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/workspace:ro openvino/model_server:2026.1 --rest_port 8000 --config_path /workspace/config.json
```
:::

:::{tab-item} On Baremetal Windows
```bat
ovms --rest_port 8000 --config_path models\config.json
```
:::
::::

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

When the model server is deployed and serving all 3 endpoints, run the [jupyter notebook](https://github.com/openvinotoolkit/model_server/blob/releases/2026/1/demos/continuous_batching/rag/rag_demo.ipynb) to use RAG chain with a fully remote execution.
