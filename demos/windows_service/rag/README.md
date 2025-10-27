# RAG demo with OpenVINO Model Server as service on Windows {#ovms_demos_continuous_batching_rag_windows_service}

## Creating models repository for all the endpoints with ovms --pull or python export_model.py script

### 1. Download the preconfigured models using ovms --pull option from [HugginFaces Hub OpenVINO organization](https://huggingface.co/OpenVINO) (Simple usage)
::::{tab-set}

:::{tab-item} On Baremetal Host
**Required:** OpenVINO Model Server package - see [deployment instructions](../../../docs/deploying_server_baremetal.md) for details.

```bat
mkdir models

ovms --pull --model_repository_path models --source_model OpenVINO/Qwen3-8B-int4-ov --task text_generation
ovms --pull --model_repository_path models --source_model OpenVINO/bge-base-en-v1.5-fp16-ov --task embeddings
ovms --pull --model_repository_path models --source_model OpenVINO/bge-reranker-base-fp16-ov --task rerank

ovms --add_to_config models --model_name OpenVINO/Qwen3-8B-int4-ov --model_path OpenVINO/Qwen3-8B-int4-ov
ovms --add_to_config models --model_name OpenVINO/bge-base-en-v1.5-fp16-ov --model_path OpenVINO/bge-base-en-v1.5-fp16-ov
ovms --add_to_config models --model_name OpenVINO/bge-reranker-base-fp16-ov --model_path OpenVINO/bge-reranker-base-fp16-ov
```
:::
::::

## Deploying the model server service

### Install service
```bat
sc create ovms binPath= "%cd%\ovms\ovms.exe --rest_port 8000 --config_path %cd%\models\config.json --log_level INFO --log_path %cd%\ovms_server.log" DisplayName= "OpenVino Model Server"
```

## Optionally set your own service description
```bat
sc description ovms "Hosts models and makes them accessible to software components over standard network protocols."
```

### Start the service
```bat
sc start ovms
```

### Stop the service
```bat
sc stop ovms
```

### Stop the service
```bat
sc delete ovms
```

## Using RAG

When the model server is deployed and serving all 3 endpoints, run the [jupyter notebook](https://github.com/openvinotoolkit/model_server/blob/main/demos/continuous_batching/rag/rag_demo.ipynb) to use RAG chain with a fully remote execution.
