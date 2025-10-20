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
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/3/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/3/demos/common/export_models/requirements.txt
mkdir models 
```

Run `export_model.py` script to download and quantize the model:

**CPU**
```console
python export_model.py embeddings_ov --source_model BAAI/bge-large-en-v1.5 --weight-format int8 --config_file_path models/config.json --model_repository_path models
```

**GPU**
```console
python export_model.py embeddings_ov --source_model BAAI/bge-large-en-v1.5 --weight-format int8 --target_device GPU --config_file_path models/config.json --model_repository_path models
```

> **Note** Change the `--weight-format` to quantize the model to `fp16`, `int8` or `int4` precision to reduce memory consumption and improve performance.
> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.

You should have a model folder like below:

```
tree models
models
├── BAAI
│   └── bge-large-en-v1.5
│       ├── config.json
│       ├── graph.pbtxt
│       ├── openvino_model.bin
│       |── openvino_model.xml
│       ├── openvino_tokenizer.bin
│       ├── openvino_tokenizer.xml
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── vocab.txt
└── config.json

```

The default configuration of the `EmbeddingsCalculatorOV` should work in most cases but the parameters can be tuned inside the `node_options` section in the `graph.pbtxt` file. They can be set automatically via export parameters in the `export_model.py` script.

For example:
`python export_model.py embeddings_ov --source_model BAAI/bge-large-en-v1.5 --weight-format int8 --skip_normalize --config_file_path models/config.json`

> **Note:** By default OVMS returns first token embeddings as sequence embeddings (called CLS pooling). It can be changed using `--pooling` option if needed by the model. Supported values are CLS and LAST. For example:
```console
python export_model.py embeddings_ov --source_model Qwen/Qwen3-Embedding-0.6B --weight-format fp16 --pooling LAST --config_file_path models/config.json
```

## Tested models
All models supported by [optimum-intel](https://github.com/huggingface/optimum-intel) should be compatible. In serving validation are included Hugging Face models:
```
    BAAI/bge-large-en-v1.5
    BAAI/bge-large-zh-v1.5
    thenlper/gte-small
    Qwen/Qwen3-Embedding-0.6B
```

## Server Deployment

:::{dropdown} **Deploying with Docker**

**CPU**
```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/workspace:ro openvino/model_server:latest --rest_port 8000 --config_path /workspace/config.json
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
curl -s http://localhost:8000/v3/models | jq
{
  "object": "list",
  "data": [
    {
      "id": "BAAI/bge-large-en-v1.5",
      "object": "model",
      "created": 1760740840,
      "owned_by": "OVMS"
    }
  ]
}
```

## Client code

:::{dropdown} **Request embeddings with cURL**
```bash
curl http://localhost:8000/v3/embeddings -H "Content-Type: application/json" -d "{ \"model\": \"BAAI/bge-large-en-v1.5\", \"input\": \"hello world\"}"
```
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [
        0.0348474495112896,
        0.03174889087677002,
        0.020687419921159744,
        -0.03732980415225029,
...
        -0.006655215751379728,
        -0.003451703116297722,
        0.015204334631562233
      ],
      "index": 0
    }
  ],
  "usage":{"prompt_tokens":4,"total_tokens":4}
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
model = "BAAI/bge-large-en-v1.5"
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
It will report results like `Similarity score as cos_sim 0.9612974628414152`.

:::

## Benchmarking feature extraction

An asynchronous benchmarking client can be used to access the model server performance with various load conditions. Below are execution examples captured on dual Intel(R) Xeon(R) CPU Max 9480.
```console
git clone https://github.com/openvinotoolkit/model_server
pushd .
cd model_server/demos/benchmark/embeddings/
pip install -r requirements.txt
python benchmark_embeddings.py --api_url http://localhost:8000/v3/embeddings --dataset synthetic --synthetic_length 5 --request_rate 10 --batch_size 1 --model BAAI/bge-large-en-v1.5
Number of documents: 1000
100%|████████████████████████████████████████████████████████████████| 1000/1000 [01:45<00:00,  9.50it/s]
Tokens: 5000
Success rate: 100.0%. (1000/1000)
Throughput - Tokens per second: 48.588129701166125
Mean latency: 17 ms
Median latency: 16 ms
Average document length: 5.0 tokens


python benchmark_embeddings.py --api_url http://localhost:8000/v3/embeddings --request_rate inf --batch_size 32 --dataset synthetic --synthetic_length 510 --model BAAI/bge-large-en-v1.5
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
Check this demo to see the langchain code example which is using OpenVINO Model Server both for text generation and embedding endpoint in [RAG application demo](https://github.com/openvinotoolkit/model_server/tree/releases/2025/3/demos/continuous_batching/rag)


## Testing the model accuracy over serving API

A simple method of testing the response accuracy is via comparing the response for a sample prompt from the model server and with local python execution based on HuggingFace python code.

The script [compare_results.py](./compare_results.py) can assist with such experiment.
```bash
popd
cd model_server/demos/embeddings
python compare_results.py --model BAAI/bge-large-en-v1.5 --service_url http://localhost:8000/v3/embeddings --pooling CLS --input "Model Server hosts models and makes them accessible to software components over standard network protocols."
input ['Model Server hosts models and makes them accessible to software components over standard network protocols.']
HF Duration: 133.467 ms BertModel
OVMS Duration: 48.964 ms
Batch number: 0
OVMS embeddings: shape: (1024,) emb[:20]:
 [-0.0016  0.0049 -0.0257 -0.0273  0.0264  0.0313 -0.0177 -0.0102  0.0194
  0.0469 -0.0181  0.0092  0.0448 -0.0288 -0.01    0.0629 -0.0341 -0.0489
 -0.0557 -0.0283]
HF AutoModel: shape: (1024,) emb[:20]:
 [-0.0013  0.0053 -0.0264 -0.0281  0.0251  0.0311 -0.0176 -0.0108  0.0191
  0.0479 -0.0181  0.0092  0.0453 -0.0286 -0.0101  0.0631 -0.0338 -0.0493
 -0.0565 -0.0286]
Difference score with HF AutoModel: 0.025911861732258994

```

It is easy also to run model evaluation using [MTEB](https://github.com/embeddings-benchmark/mteb) framework using a custom class based on openai model:
```bash
pip install mteb --extra-index-url "https://download.pytorch.org/whl/cpu"
python ovms_mteb.py --model BAAI/bge-large-en-v1.5 --service_url http://localhost:8000/v3/embeddings
```
Results will be stored in `results` folder:
```json
{
  "dataset_revision": "0fd18e25b25c072e09e0d92ab615fda904d66300",
  "task_name": "Banking77Classification",
  "mteb_version": "1.34.11",
  "scores": {
    "test": [
      {
        "accuracy": 0.848571,
        "f1": 0.842365,
        "f1_weighted": 0.842365,
        "scores_per_experiment": [
          {
            "accuracy": 0.843831,
            "f1": 0.836592,
            "f1_weighted": 0.836592
          },
          {
            "accuracy": 0.850649,
            "f1": 0.84395,
            "f1_weighted": 0.84395
          },
          {
            "accuracy": 0.849675,
            "f1": 0.843094,
            "f1_weighted": 0.843094
          },
          {
            "accuracy": 0.853896,
            "f1": 0.850204,
            "f1_weighted": 0.850204
          },
          {
            "accuracy": 0.846753,
            "f1": 0.83981,
            "f1_weighted": 0.83981
          },
          {
            "accuracy": 0.85,
            "f1": 0.844339,
            "f1_weighted": 0.844339
          },
          {
            "accuracy": 0.844805,
            "f1": 0.838669,
            "f1_weighted": 0.838669
          },
          {
            "accuracy": 0.846104,
            "f1": 0.839095,
            "f1_weighted": 0.839095
          },
          {
            "accuracy": 0.852922,
            "f1": 0.847884,
            "f1_weighted": 0.847884
          },
          {
            "accuracy": 0.847078,
            "f1": 0.840013,
            "f1_weighted": 0.840013
          }
        ],
        "main_score": 0.848571,
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

