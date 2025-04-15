# How to serve Embeddings models via OpenAI API {#ovms_demos_embeddings}
This demo shows how to deploy embeddings models in the OpenVINO Model Server for text feature extractions.
Text generation use case is exposed via OpenAI API `embeddings` endpoint.

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
python export_model.py embeddings --source_model Alibaba-NLP/gte-large-en-v1.5 --weight-format int8 --config_file_path models/config.json --model_repository_path models
```

**GPU**
```console
python export_model.py embeddings --source_model Alibaba-NLP/gte-large-en-v1.5 --weight-format int8 --target_device GPU --config_file_path models/config.json --model_repository_path models
```

> **Note** Change the `--weight-format` to quantize the model to `fp16`, `int8` or `int4` precision to reduce memory consumption and improve performance.
> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.

You should have a model folder like below:

```
tree models
models
├── Alibaba-NLP
│   └── gte-large-en-v1.5
│       ├── embeddings
│       │   └── 1
│       │       ├── model.bin
│       │       └── model.xml
│       ├── graph.pbtxt
│       ├── subconfig.json
│       └── tokenizer
│           └── 1
│               ├── model.bin
│               └── model.xml
└── config.json

```
> **Note** The actual models support version management and can be automatically swapped to newer version when new model is uploaded in newer version folder.
> In case you trained the pytorch model it can be exported like below:
> `python export_model.py embeddings --source_model <pytorch model> --model_name Alibaba-NLP/gte-large-en-v1.5 --precision int8 --config_file_path models/config.json --version 2`

The default configuration of the `EmbeddingsCalculator` should work in most cases but the parameters can be tuned inside the `node_options` section in the `graph.pbtxt` file. Runtime configuration for both models can be tuned in `subconfig.json` file. They can be set automatically via export parameters in the `export_model.py` script.

For example:
`python export_model.py embeddings --source_model Alibaba-NLP/gte-large-en-v1.5 --precision int8 --num_streams 2 --skip_normalize --config_file_path models/config.json`


## Tested models
All models supported by [optimum-intel](https://github.com/huggingface/optimum-intel) should be compatible. In serving validation are included Hugging Face models:
```
    nomic-ai/nomic-embed-text-v1.5
    Alibaba-NLP/gte-large-en-v1.5
    BAAI/bge-large-en-v1.5
    BAAI/bge-large-zh-v1.5
    thenlper/gte-small
```

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

:::{dropdown} **Deploying on Bare Metal**

Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.

Depending on how you prepared models in the first step of this demo, they are deployed to either CPU or GPU (it's defined in `config.json`). If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

```bat
ovms --rest_port 8000 --config_path ./models/config.json
```
:::

### Readiness Check

Wait for the model to load. You can check the status with a simple command below. Note that the slash `/` in the model name needs to be escaped with `%2F`:
```bash
curl -i http://localhost:8000/v2/models/Alibaba-NLP%2Fgte-large-en-v1.5/ready
HTTP/1.1 200 OK
content-length: 0
content-type: application/json; charset=utf-8
content-type: application/json
```

## Client code

:::{dropdown} **Request embeddings with cURL**
```bash
curl http://localhost:8000/v3/embeddings -H "Content-Type: application/json" -d "{ \"model\": \"Alibaba-NLP/gte-large-en-v1.5\", \"input\": \"hello world\"}"
```
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [
        -0.03440694510936737,
        -0.02553200162947178,
        -0.010130723007023335,
        -0.013917984440922737,
...
        0.02722850814461708,
        -0.017527244985103607,
        -0.0053995149210095406
      ],
      "index": 0
    }
  ]
}

```
:::

:::{dropdown} **Request embeddings with OpenAI Python package**

```bash
pip3 install openai
```
```bash
echo '
from openai import OpenAI
import numpy as np

client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)
model = "Alibaba-NLP/gte-large-en-v1.5"
embedding_responses = client.embeddings.create(
    input=[
        "That is a happy person",
        "That is a happy very person"
    ],
    model=model,
)
embedding_from_string1 = np.array(embedding_responses.data[0].embedding)
embedding_from_string2 = np.array(embedding_responses.data[1].embedding)
cos_sim = np.dot(embedding_from_string1, embedding_from_string2)/(np.linalg.norm(embedding_from_string1)*np.linalg.norm(embedding_from_string2))
print("Similarity score as cos_sim", cos_sim)' >> openai_client.py

python openai_client.py
```
It will report results like `Similarity score as cos_sim 0.97654650115054`.

:::

## Benchmarking feature extraction

An asynchronous benchmarking client can be used to access the model server performance with various load conditions. Below are execution examples captured on dual Intel(R) Xeon(R) CPU Max 9480.
```console
git clone https://github.com/openvinotoolkit/model_server
pushd .
cd model_server/demos/benchmark/embeddings/
pip install -r requirements.txt
python benchmark_embeddings.py --api_url http://localhost:8000/v3/embeddings --dataset synthetic --synthetic_length 5 --request_rate 10 --batch_size 1 --model Alibaba-NLP/gte-large-en-v1.5
Number of documents: 1000
100%|████████████████████████████████████████████████████████████████| 1000/1000 [01:45<00:00,  9.50it/s]
Tokens: 5000
Success rate: 100.0%. (1000/1000)
Throughput - Tokens per second: 48.588129701166125
Mean latency: 17 ms
Median latency: 16 ms
Average document length: 5.0 tokens


python benchmark_embeddings.py --api_url http://localhost:8000/v3/embeddings --request_rate inf --batch_size 32 --dataset synthetic --synthetic_length 510 --model Alibaba-NLP/gte-large-en-v1.5
Number of documents: 1000
100%|████████████████████████████████████████████████████████████████| 50/50 [00:21<00:00,  2.32it/s]
Tokens: 510000
Success rate: 100.0%. (32/32)
Throughput - Tokens per second: 27995.652060806977
Mean latency: 10113 ms
Median latency: 10166 ms
Average document length: 510.0 tokens


python benchmark_embeddings.py --api_url http://localhost:8000/v3/embeddings --request_rate inf --batch_size 1 --dataset Cohere/wikipedia-22-12-simple-embeddings
Number of documents: 1000
100%|████████████████████████████████████████████████████████████████| 1000/1000 [00:15<00:00, 64.02it/s]
Tokens: 83208
Success rate: 100.0%. (1000/1000)
Throughput - Tokens per second: 5433.913083411673
Mean latency: 1424 ms
Median latency: 1451 ms
Average document length: 83.208 tokens
```

## RAG with Model Server

Embeddings endpoint can be applied in RAG chains to delegated text feature extraction both for documented vectorization and in context retrieval.
Check this demo to see the langchain code example which is using OpenVINO Model Server both for text generation and embedding endpoint in [RAG application demo](https://github.com/openvinotoolkit/model_server/tree/main/demos/continuous_batching/rag)


## Testing the model accuracy over serving API

A simple method of testing the response accuracy is via comparing the response for a sample prompt from the model server and with local python execution based on HuggingFace python code.

The script [compare_results.py](./compare_results.py) can assist with such experiment.
```bash
popd
cd model_server/demos/embeddings
python compare_results.py --model Alibaba-NLP/gte-large-en-v1.5 --service_url http://localhost:8000/v3/embeddings --input "hello world" --input "goodbye world"

input ['hello world', 'goodbye world']
HF Duration: 50.626 ms NewModel
OVMS Duration: 20.219 ms
Batch number: 0
OVMS embeddings: shape: (1024,) emb[:20]:
 [-0.0349 -0.0256 -0.0102 -0.0139 -0.0175 -0.0015 -0.0297 -0.0002 -0.0424
 -0.0145 -0.0141  0.0101  0.0057  0.0001  0.0316 -0.03   -0.04   -0.0474
  0.0084 -0.0097]
HF AutoModel: shape: (1024,) emb[:20]:
 [-0.0345 -0.0252 -0.0106 -0.0124 -0.0167 -0.0018 -0.0301  0.0002 -0.0408
 -0.0139 -0.015   0.0104  0.0054 -0.0006  0.0326 -0.0296 -0.04   -0.0457
  0.0087 -0.0102]
Difference score with HF AutoModel: 0.02175156185021083
Batch number: 1
OVMS embeddings: shape: (1024,) emb[:20]:
 [-0.0141 -0.0332 -0.0041 -0.0205 -0.0008  0.0189 -0.0278 -0.0083 -0.0511
  0.0043  0.0262 -0.0079  0.016   0.0084  0.0123 -0.0414 -0.0314 -0.0332
  0.0101 -0.0052]
HF AutoModel: shape: (1024,) emb[:20]:
 [-0.0146 -0.0333 -0.005  -0.0194  0.0004  0.0197 -0.0281 -0.0069 -0.0511
  0.005   0.0253 -0.0067  0.0167  0.0079  0.0128 -0.0407 -0.0317 -0.0329
  0.0095 -0.0051]
Difference score with HF AutoModel: 0.024787274668209857

```

It is easy also to run model evaluation using [MTEB](https://github.com/embeddings-benchmark/mteb) framework using a custom class based on openai model:
```bash
pip install mteb --extra-index-url "https://download.pytorch.org/whl/cpu"
python ovms_mteb.py --model Alibaba-NLP/gte-large-en-v1.5 --service_url http://localhost:8000/v3/embeddings
```
Results will be stored in `results` folder:
```json
{
  "dataset_revision": "0fd18e25b25c072e09e0d92ab615fda904d66300",
  "task_name": "Banking77Classification",
  "mteb_version": "1.31.6",
  "scores": {
    "test": [
      {
        "accuracy": 0.849416,
        "f1": 0.845058,
        "f1_weighted": 0.845058,
        "scores_per_experiment": [
          {
            "accuracy": 0.854545,
            "f1": 0.850033,
            "f1_weighted": 0.850033
          },
          {
            "accuracy": 0.86461,
            "f1": 0.860671,
            "f1_weighted": 0.860671
          },
          {
            "accuracy": 0.847403,
            "f1": 0.843897,
            "f1_weighted": 0.843897
          },
          {
            "accuracy": 0.856169,
            "f1": 0.853613,
            "f1_weighted": 0.853613
          },
          {
            "accuracy": 0.843831,
            "f1": 0.839043,
            "f1_weighted": 0.839043
          },
          {
            "accuracy": 0.847078,
            "f1": 0.844124,
            "f1_weighted": 0.844124
          },
          {
            "accuracy": 0.842208,
            "f1": 0.837938,
            "f1_weighted": 0.837938
          },
          {
            "accuracy": 0.843506,
            "f1": 0.837239,
            "f1_weighted": 0.837239
          },
          {
            "accuracy": 0.85,
            "f1": 0.844696,
            "f1_weighted": 0.844696
          },
          {
            "accuracy": 0.844805,
            "f1": 0.839321,
            "f1_weighted": 0.839321
          }
        ],
        "main_score": 0.849416,
        "hf_subset": "default",
        "languages": [
          "eng-Latn"
        ]
      }
    ]
  },
  "evaluation_time": 109.37459182739258,
  "kg_co2_emissions": null
}
```

