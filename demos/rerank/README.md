# How to serve Rerank models via Cohere API {#ovms_demos_rerank}

## Get the docker image

Build the image from source to try this new feature. It will be included in the public image in the coming version 2024.5.
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make release_image GPU=1
```
It will create an image called `openvino/model_server:latest`.
> **Note:** This operation might take 40min or more depending on your build host.
> **Note:** `GPU` parameter in image build command is needed to include dependencies for GPU device.

## Model preparation
> **Note** Python 3.9 or higher is needed for that step
Here, the original Pytorch LLM model and the tokenizer will be converted to IR format and optionally quantized.
That ensures faster initialization time, better performance and lower memory consumption.

Install python dependencies for the conversion script:
```bash
pip3 install -r demos/common/export_models/requirements.txt
```

Run optimum-cli to download and quantize the model:
```bash
mkdir models
python demos/common/export_models/export_model.py rerank --source_model BAAI/bge-reranker-large --weight-format int8 --config_file_path models/config.json --model_repository_path models 
```

You should have a model folder like below:
```bash
tree models
models
├── BAAI
│   └── bge-reranker-large
│       ├── graph.pbtxt
│       ├── rerank
│       │   └── 1
│       │       ├── model.bin
│       │       └── model.xml
│       ├── subconfig.json
│       └── tokenizer
│           └── 1
│               ├── model.bin
│               └── model.xml
└── config.json

```
> **Note** The actual models support version management and can be automatically swapped to newer version when new model is uploaded in newer version folder.


## Deployment 

```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/workspace:ro openvino/model_server:latest --port 9000 --rest_port 8000 --config_path /workspace/config.json
```

## Client code


```bash
curl http://localhost:8000/v3/rerank  -H "Content-Type: application/json" \
-d '{ "model": "BAAI/bge-reranker-large", "query": "welcome", "documents":["good morning","farewell"]}' | jq .
```
```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.3886180520057678
    },
    {
      "index": 1,
      "relevance_score": 0.0055549247190356255
    }
  ]
}
```


## Comparison with Hugging Faces

```bash
python demos/embeddings/compare_results.py --query "hello" --document "welcome" --document "farewell"
query hello
documents ['welcome', 'farewell']
HF Duration: 145.731 ms
OVMS Duration: 23.227 ms
HF reranking: [0.99640983 0.08154089]
OVMS reranking: [0.9968273 0.0913821]
```

## Tested models

```
BAAI/bge-reranker-large
BAAI/bge-reranker-v2-m3
BAAI/bge-reranker-base
```

## Integration with Langchain

Check [RAG demo](../continuous_batching/rag/README.md) which employs `rerank` endpoint together with `chat/completions` and `embeddings`. 


