# RAG demo with all execution steps delegated to the OpenVINO Model Server {#ovms_demos_continuous_batching_rag}


## Creating models repository for all the endpoints

```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/requirements.txt

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

## Using RAG

When the model server is deployed and serving all 3 endpoints, run the [jupyter notebook](https://github.com/openvinotoolkit/model_server/blob/main/demos/continuous_batching/rag/rag_demo.ipynb) to use RAG chain with a fully remote execution.