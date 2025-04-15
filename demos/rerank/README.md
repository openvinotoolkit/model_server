# How to serve Rerank models via Cohere API {#ovms_demos_rerank}

## Prerequisites

**Model preparation**: Python 3.9 or higher with pip 

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../docs/deploying_server_baremetal.md)

**(Optional) Client**: Python with pip

## Model preparation

Here, the original Pytorch LLM model and the tokenizer will be converted to IR format and optionally quantized.
That ensures faster initialization time, better performance and lower memory consumption.

Download export script, install it's dependencies and create directory for the models:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/requirements.txt
mkdir models 
```

Run `export_model.py` script to download and quantize the model:

**CPU**

```console
python export_model.py rerank --source_model BAAI/bge-reranker-large --weight-format int8 --config_file_path models/config.json --model_repository_path models 
```

**GPU**:
```console
python export_model.py rerank --source_model BAAI/bge-reranker-large --weight-format int8 --target_device GPU --config_file_path models/config.json --model_repository_path models 
```
> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.

You should have a model folder like below:
```
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


## Server Deployment

:::{dropdown} **Deploying with Docker**

**CPU**
```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/workspace:ro openvino/model_server:latest --port 9000 --rest_port 8000 --config_path /workspace/config.json
```
**GPU**

In case you want to use GPU device to run the embeddings model, add extra docker parameters `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)` 
to `docker run` command, use the image with GPU support and make sure set the target_device in subconfig.json to GPU. Also make sure the export model quantization level and cache size fit to the GPU memory. All of that can be applied with the commands:

```bash
docker run -d --rm -p 8000:8000 --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -v $(pwd)/models:/workspace:ro openvino/model_server:latest-gpu --rest_port 8000 --config_path /workspace/config.json
```
:::

:::{dropdown} **Deploying On Bare Metal**

Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.

Depending on how you prepared models in the first step of this demo, they are deployed to either CPU or GPU (it's defined in `config.json`). If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

```bat
ovms --rest_port 8000 --config_path ./models/config.json
```
:::

## Readiness Check

Readiness of the model can be reported with a simple curl command. 
```bash
curl -i http://localhost:8000/v2/models/BAAI%2Fbge-reranker-large/ready
HTTP/1.1 200 OK
content-length: 0
content-type: application/json; charset=utf-8
content-type: application/json
```

## Client code

:::{dropdown} **Requesting rerank score with cURL**

```bash
curl http://localhost:8000/v3/rerank  -H "Content-Type: application/json" -d "{ \"model\": \"BAAI/bge-reranker-large\", \"query\": \"welcome\", \"documents\":[\"good morning\",\"farewell\"]}"
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
:::

:::{dropdown} **Requesting rerank score with Cohere Python package**
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

python rerank_client.py
```
It will return response similar to:
```
index 0, relevance_score 0.9968273043632507
index 1, relevance_score 0.09138210117816925
```
:::

## Comparison with Hugging Faces

```bash
git clone https://github.com/openvinotoolkit/model_server
python model_server/demos/rerank/compare_results.py --query "hello" --document "welcome" --document "farewell" --base_url http://localhost:8000/v3/
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
cd model_server/demos/benchmark/embeddings/
pip install -r requirements.txt
python benchmark_embeddings.py --api_url http://localhost:8000/v3/rerank --backend ovms_rerank --dataset synthetic --synthetic_length 500 --request_rate inf --batch_size 20 --model BAAI/bge-reranker-large 
Number of documents: 1000
100%|██████████████████████████████████████| 50/50 [00:19<00:00,  2.53it/s]
Tokens: 501000
Success rate: 100.0%. (50/50)
Throughput - Tokens per second: 25325.17484336458
Mean latency: 10268 ms
Median latency: 10249 ms
Average document length: 501.0 tokens

python benchmark_embeddings.py --api_url http://localhost:8000/v3/rerank --backend ovms_rerank --dataset synthetic --synthetic_length 500 --request_rate inf --batch_size 20 --model BAAI/bge-reranker-large 
Number of documents: 1000
100%|██████████████████████████████████████| 50/50 [00:19<00:00,  2.53it/s]
Tokens: 501000
Success rate: 100.0%. (50/50)
Throughput - Tokens per second: 25325.17484336458
Mean latency: 10268 ms
Median latency: 10249 ms
Average document length: 501.0 tokens

python benchmark_embeddings.py --api_url http://localhost:8000/v3/rerank --backend ovms_rerank --dataset Cohere/wikipedia-22-12-simple-embeddings --request_rate inf --batch_size 20 --model BAAI/bge-reranker-large
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


