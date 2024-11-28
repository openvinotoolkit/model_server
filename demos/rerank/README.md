# How to serve Rerank models via Cohere API {#ovms_demos_rerank}

## Model preparation
> **Note** Python 3.9 or higher is needed for that step
Here, the original Pytorch LLM model and the tokenizer will be converted to IR format and optionally quantized.
That ensures faster initialization time, better performance and lower memory consumption.

Install python dependencies for the conversion script:
```bash
pushd .
cd demos/common/export_models
pip3 install -r requirements.txt
```

Run optimum-cli to download and quantize the model:
```bash
mkdir models
python export_model.py rerank --source_model BAAI/bge-reranker-large --weight-format int8 --config_file_path models/config.json --model_repository_path models 
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

Readiness of the model can be reported with a simple curl command. 
```bash
curl -i http://localhost:8000/v2/models/BAAI%2Fbge-reranker-large/ready
HTTP/1.1 200 OK
Content-Type: application/json
Date: Sat, 09 Nov 2024 23:19:27 GMT
Content-Length: 0
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

Alternatively there could be used cohere python client like in the example below:
```bash
pip3 install cohere
```
```bash
echo '
import cohere
client = cohere.Client(base_url="http://localhost:8000/v3", api_key="not_used")
responses = client.rerank(query="hello",documents=["welcome","farewell"], model="BAAI/bge-reranker-large")
for response in responses.results:
    print(f"index {response.index}, relevance_score {response.relevance_score}")' > rerank_client.py

python3 rerank_client.py 
```
It will return response similar to:
```
index 0, relevance_score 0.9968273043632507
index 1, relevance_score 0.09138210117816925
```

## Comparison with Hugging Faces

```bash
popd
pushd .
cd demos/rerank/
python demos/rerank/compare_results.py --query "hello" --document "welcome" --document "farewell" --base_url http://localhost:8000/v3/
query hello
documents ['welcome', 'farewell']
HF Duration: 145.731 ms
OVMS Duration: 23.227 ms
HF reranking: [0.99640983 0.08154089]
OVMS reranking: [0.9968273 0.0913821]
```

## Performance benchmarking

An asynchronous benchmarking client can be used to access the model server performance with various load conditions. Below are execution examples captured on dual Intel(R) Xeon(R) CPU Max 9480.
```bash
popd
pushd .
cd demos/benchmark/embeddings/
pip install -r requirements.txt
python benchmark_embeddings.py --backend ovms_rerank --dataset synthetic --synthetic_length 500 --request_rate inf --batch_size 20 --model BAAI/bge-reranker-large 
Number of documents: 1000
100%|██████████████████████████████████████| 50/50 [00:19<00:00,  2.53it/s]
Tokens: 501000
Success rate: 100.0%. (50/50)
Throughput - Tokens per second: 25325.17484336458
Mean latency: 10268 ms
Median latency: 10249 ms
Average document length: 501.0 tokens

python benchmark_embeddings.py --backend ovms_rerank --dataset synthetic --synthetic_length 500 --request_rate inf --batch_size 20 --model BAAI/bge-reranker-large 
Number of documents: 1000
100%|██████████████████████████████████████| 50/50 [00:19<00:00,  2.53it/s]
Tokens: 501000
Success rate: 100.0%. (50/50)
Throughput - Tokens per second: 25325.17484336458
Mean latency: 10268 ms
Median latency: 10249 ms
Average document length: 501.0 tokens

python benchmark_embeddings.py --backend ovms_rerank --dataset Cohere/wikipedia-22-12-simple-embeddings --request_rate inf 
--batch_size 20 --model BAAI/bge-reranker-large 
Number of documents: 1000
100%|██████████████████████████████████████| 50/50 [00:09<00:00,  5.55it/s]
Tokens: 92248
Success rate: 100.0%. (50/50)
Throughput - Tokens per second: 10236.429922338193
Mean latency: 4511 ms
Median latency: 4309 ms
Average document length: 92.248 tokens


```
## Tested models

```
BAAI/bge-reranker-large
BAAI/bge-reranker-v2-m3
BAAI/bge-reranker-base
```

## Integration with Langchain

Check [RAG demo](../continuous_batching/rag/README.md) which employs `rerank` endpoint together with `chat/completions` and `embeddings`. 


