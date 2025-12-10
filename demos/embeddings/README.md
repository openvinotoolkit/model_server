# How to serve Embeddings models via OpenAI API {#ovms_demos_embeddings}
This demo shows how to deploy embeddings models in the OpenVINO Model Server for text feature extractions.
Text generation use case is exposed via OpenAI API `embeddings` endpoint.

## Prerequisites

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../docs/deploying_server_baremetal.md)

**(Optional) Model preparation**: Can be omitted when pulling models in IR format directly from HuggingFaces. Otherwise Python 3.9 or higher with pip for manual model export step.

**(Optional) Client**: Python with pip

## Model preparation

### Direct pulling of pre-configured HuggingFace models

This procedure can be used to pull preconfigured models from OpenVINO organization in HuggingFace Hub

**CPU**
::::{tab-set}
:::{tab-item} OpenVINO/Qwen3-Embedding-0.6B-int8-ov
:sync: Qwen3-Embedding-0.6B-int8-ov
**Using docker image**
```bash
mkdir -p models
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/Qwen3-Embedding-0.6B-int8-ov --pooling LAST --task embeddings
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name OpenVINO/Qwen3-Embedding-0.6B-int8-ov --model_path OpenVINO/Qwen3-Embedding-0.6B-int8-ov
```

**On Bare Metal (Windows/Linux)**
```console
mkdir models
ovms --pull --model_repository_path /models --source_model OpenVINO/Qwen3-Embedding-0.6B-int8-ov --pooling LAST --task embeddings
ovms --add_to_config --config_path /models/config.json --model_name OpenVINO/Qwen3-Embedding-0.6B-int8-ov --model_path OpenVINO/Qwen3-Embedding-0.6B-int8-ov
```
:::
:::{tab-item} OpenVINO/bge-base-en-v1.5-int8-ov
:sync: bge-base-en-v1.5-int8-ov
**Using docker image**
```bash
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/bge-base-en-v1.5-int8-ov --pooling CLS --task embeddings

docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name OpenVINO/bge-base-en-v1.5-int8-ov --model_path OpenVINO/bge-base-en-v1.5-int8-ov
```

**On Bare Metal (Windows/Linux)**
```console
ovms --pull --model_repository_path /models --source_model OpenVINO/bge-base-en-v1.5-int8-ov --pooling CLS --task embeddings

ovms --add_to_config --config_path /models/config.json --model_name OpenVINO/bge-base-en-v1.5-int8-ov --model_path OpenVINO/bge-base-en-v1.5-int8-ov
```
:::
::::

**GPU**
::::{tab-set}
:::{tab-item} OpenVINO/Qwen3-Embedding-0.6B-int8-ov
:sync: Qwen3-Embedding-0.6B-int8-ov
**Using docker image**
```bash
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/Qwen3-Embedding-0.6B-int8-ov --pooling LAST --target_device GPU --task embeddings

docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name OpenVINO/Qwen3-Embedding-0.6B-int8-ov --model_path OpenVINO/Qwen3-Embedding-0.6B-int8-ov
```

**On Bare Metal (Windows/Linux)**
```console
ovms --pull --model_repository_path /models --source_model OpenVINO/Qwen3-Embedding-0.6B-int8-ov --pooling LAST --target_device GPU --task embeddings

ovms --add_to_config --config_path /models/config.json --model_name OpenVINO/Qwen3-Embedding-0.6B-int8-ov --model_path OpenVINO/Qwen3-Embedding-0.6B-int8-ov
```
:::
:::{tab-item} OpenVINO/bge-base-en-v1.5-int8-ov
:sync: OpenVINO/bge-base-en-v1.5-int8-ov
**Using docker image**
```bash
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/bge-base-en-v1.5-int8-ov --pooling CLS --target_device GPU --task embeddings

docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name OpenVINO/bge-base-en-v1.5-int8-ov --model_path OpenVINO/bge-base-en-v1.5-int8-ov
```

**On Bare Metal (Windows/Linux)**
```console
ovms --pull --model_repository_path /models --source_model OpenVINO/bge-base-en-v1.5-int8-ov --pooling CLS --target_device GPU --task embeddings

ovms --add_to_config --config_path /models/config.json --model_name OpenVINO/bge-base-en-v1.5-int8-ov --model_path OpenVINO/bge-base-en-v1.5-int8-ov
```
:::
::::

### Export model

Here, the original Pytorch LLM model and the tokenizer will be converted to IR format and optionally quantized.
That ensures faster initialization time, better performance and lower memory consumption.

Download export script, install it's dependencies and create directory for the models:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/4/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/4/demos/common/export_models/requirements.txt
mkdir models 
```

Run `export_model.py` script to download and quantize the model:

**CPU**
::::{tab-set}
:::{tab-item} BAAI/bge-large-en-v1.5
:sync: bge-large-en-v1.5
```console
python export_model.py embeddings_ov --source_model BAAI/bge-large-en-v1.5 --pooling CLS --weight-format int8 --config_file_path models/config.json --model_repository_path models
```
:::
:::{tab-item} BAAI/bge-large-zh-v1.5
:sync: bge-large-zh-v1.5
```console
python export_model.py embeddings_ov --source_model BAAI/bge-large-zh-v1.5 --pooling CLS --weight-format int8 --config_file_path models/config.json --model_repository_path models
```
:::
:::{tab-item} thenlper/gte-small
:sync: gte-small
```console
python export_model.py embeddings_ov --source_model thenlper/gte-small --pooling CLS --weight-format int8 --config_file_path models/config.json --model_repository_path models
```
:::
:::{tab-item} sentence-transformers/all-MiniLM-L12-v2
:sync: all-MiniLM-L12-v2
```console
python export_model.py embeddings_ov --source_model sentence-transformers/all-MiniLM-L12-v2 --pooling MEAN --weight-format int8 --config_file_path models/config.json --model_repository_path models
```
:::
:::{tab-item} sentence-transformers/all-distilroberta-v1
:sync: all-distilroberta-v1
```console
python export_model.py embeddings_ov --source_model sentence-transformers/all-distilroberta-v1 --pooling MEAN --weight-format int8 --config_file_path models/config.json --model_repository_path models
```
:::
:::{tab-item} mixedbread-ai/deepset-mxbai-embed-de-large-v1
:sync: deepset-mxbai-embed-de-large-v1
```console
python export_model.py embeddings_ov --source_model mixedbread-ai/deepset-mxbai-embed-de-large-v1 --pooling MEAN --weight-format int8 --config_file_path models/config.json --model_repository_path models
```
:::
:::{tab-item} intfloat/multilingual-e5-large-instruct
:sync: multilingual-e5-large-instruc
```console
python export_model.py embeddings_ov --source_model intfloat/multilingual-e5-large-instruct --pooling MEAN --weight-format int8 --config_file_path models/config.json --model_repository_path models
```
:::
:::{tab-item} intfloat/multilingual-e5-large
:sync: multilingual-e5-large
```console
python export_model.py embeddings_ov --source_model intfloat/multilingual-e5-large --pooling MEAN --weight-format int8 --config_file_path models/config.json --model_repository_path models
```
:::
::::


**GPU**
::::{tab-set}
:::{tab-item} BAAI/bge-large-en-v1.5
:sync: bge-large-en-v1.5
```console
python export_model.py embeddings_ov --source_model BAAI/bge-large-en-v1.5 --pooling CLS --weight-format int8 --target_device GPU --config_file_path models/config.json --model_repository_path models
```
:::
:::{tab-item} BAAI/bge-large-zh-v1.5
:sync: bge-large-zh-v1.5
```console
python export_model.py embeddings_ov --source_model BAAI/bge-large-zh-v1.5 --pooling CLS --weight-format int8 --target_device GPU --config_file_path models/config.json --model_repository_path models
```
:::
:::{tab-item} thenlper/gte-small
:sync: gte-small
```console
python export_model.py embeddings_ov --source_model thenlper/gte-small --pooling CLS --weight-format int8 --target_device GPU --config_file_path models/config.json --model_repository_path models
```
:::
:::{tab-item} sentence-transformers/all-MiniLM-L12-v2
:sync: all-MiniLM-L12-v2
```console
python export_model.py embeddings_ov --source_model sentence-transformers/all-MiniLM-L12-v2 --pooling MEAN --weight-format int8 --target_device GPU --config_file_path models/config.json --model_repository_path models
```
:::
:::{tab-item} sentence-transformers/all-distilroberta-v1
:sync: all-distilroberta-v1
```console
python export_model.py embeddings_ov --source_model sentence-transformers/all-distilroberta-v1 --pooling MEAN --weight-format int8 --target_device GPU --config_file_path models/config.json --model_repository_path models
```
:::
:::{tab-item} mixedbread-ai/deepset-mxbai-embed-de-large-v1
:sync: deepset-mxbai-embed-de-large-v1
```console
python export_model.py embeddings_ov --source_model mixedbread-ai/deepset-mxbai-embed-de-large-v1 --pooling MEAN --weight-format int8 --target_device GPU --config_file_path models/config.json --model_repository_path models
```
:::
:::{tab-item} intfloat/multilingual-e5-large-instruct
:sync: multilingual-e5-large-instruc
```console
python export_model.py embeddings_ov --source_model intfloat/multilingual-e5-large-instruct --pooling MEAN --weight-format int8 --target_device GPU --config_file_path models/config.json --model_repository_path models
```
:::
:::{tab-item} intfloat/multilingual-e5-large
:sync: multilingual-e5-large
```console
python export_model.py embeddings_ov --source_model intfloat/multilingual-e5-large --pooling MEAN --weight-format int8 --target_device GPU --config_file_path models/config.json --model_repository_path models
```
:::
::::


> **Note** Change the `--weight-format` to quantize the model to `fp16`, `int8` or `int4` precision to reduce memory consumption and improve performance.
> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.

You should have a model folder like below:

```
tree models
models
├── BAAI
│   └── bge-large-en-v1.5
│       ├── config.json
│       ├── graph.pbtxt
│       ├── openvino_model.bin
│       ├── openvino_model.xml
│       ├── openvino_tokenizer.bin
│       ├── openvino_tokenizer.xml
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── vocab.txt
└── config.json
```

The default configuration of the `EmbeddingsCalculatorOV` should work in most cases but the parameters can be tuned inside the `node_options` section in the `graph.pbtxt` file. They can be set automatically via export parameters in the `export_model.py` script.

For example:
`python export_model.py embeddings_ov --source_model BAAI/bge-large-en-v1.5 --weight-format int8 --skip_normalize --config_file_path models/config.json`

> **Note:** By default OVMS returns first token embeddings as sequence embeddings (called CLS pooling). It can be changed using `--pooling` option if needed by the model. Supported values are CLS, MEAN and LAST. For example:
```console
python export_model.py embeddings_ov --source_model Qwen/Qwen3-Embedding-0.6B --weight-format fp16 --pooling LAST --config_file_path models/config.json
```

## Tested models
All models supported by [optimum-intel](https://github.com/huggingface/optimum-intel) should be compatible. The demo is validated against following Hugging Face models:

|Model name|Pooling|
|---|---|
|OpenVINO/Qwen3-Embedding-0.6B-int8-ov|LAST|
|OpenVINO/bge-base-en-v1.5-int8-ov|CLS|
|BAAI/bge-large-en-v1.5|CLS|
|BAAI/bge-large-zh-v1.5|CLS|
|thenlper/gte-small|CLS|
|sentence-transformers/all-MiniLM-L12-v2|MEAN|
|sentence-transformers/all-distilroberta-v1|MEAN|
|mixedbread-ai/deepset-mxbai-embed-de-large-v1|MEAN|
|intfloat/multilingual-e5-large-instruct|MEAN|
|intfloat/multilingual-e5-large|MEAN|


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
curl http://localhost:9999/v3/models/BAAI%2Fbge-large-en-v1.5

{"id":"BAAI/bge-large-en-v1.5","object":"model","created":1763997378,"owned_by":"OVMS"}
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
  ],
  "usage":{"prompt_tokens":4,"total_tokens":4}
}

```
:::

:::{dropdown} **Request embeddings with OpenAI Python package**

```bash
pip3 install openai "numpy<2"
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
It will report results like `Similarity score as cos_sim 0.9605122725993963`.

:::

## Benchmarking feature extraction

An asynchronous benchmarking client can be used to access the model server performance with various load conditions. Below are execution examples captured on dual Intel(R) Xeon(R) CPU Max 9480.
```console
git clone https://github.com/openvinotoolkit/model_server
pushd .
cd model_server/demos/benchmark/v3/
pip install -r requirements.txt
python benchmark.py --api_url http://localhost:8000/v3/embeddings --dataset synthetic --synthetic_length 5 --request_rate 10 --batch_size 1 --model BAAI/bge-large-en-v1.5
Number of documents: 1000
100%|████████████████████████████████████████████████████████████████| 1000/1000 [01:44<00:00,  9.56it/s]
Tokens: 5000
Success rate: 100.0%. (1000/1000)
Throughput - Tokens per second: 47.8
Mean latency: 14.40 ms
Median latency: 13.97 ms
Average document length: 5.0 tokens


python benchmark.py --api_url http://localhost:8000/v3/embeddings --request_rate inf --batch_size 32 --dataset synthetic --synthetic_length 510 --model BAAI/bge-large-en-v1.5
Number of documents: 1000
100%|████████████████████████████████████████████████████████████████| 32/32 [00:17<00:00,  1.82it/s]
Tokens: 510000
Success rate: 100.0%. (32/32)
Throughput - Tokens per second: 29,066.2
Mean latency: 9768.28 ms
Median latency: 9905.79 ms
Average document length: 510.0 tokens


python benchmark.py --api_url http://localhost:8000/v3/embeddings --request_rate inf --batch_size 1 --dataset Cohere/wikipedia-22-12-simple-embeddings --model BAAI/bge-large-en-v1.5
Number of documents: 1000
100%|████████████████████████████████████████████████████████████████| 1000/1000 [00:15<00:00, 64.02it/s]
Tokens: 83208
Success rate: 100.0%. (1000/1000)
Throughput - Tokens per second: 4,120.6
Mean latency: 1882.98 ms
Median latency: 1608.47 ms
Average document length: 83.208 tokens
```

## RAG with Model Server

Embeddings endpoint can be applied in RAG chains to delegated text feature extraction both for documented vectorization and in context retrieval.
Check this demo to see the langchain code example which is using OpenVINO Model Server both for text generation and embedding endpoint in [RAG application demo](https://github.com/openvinotoolkit/model_server/tree/releases/2025/4/demos/continuous_batching/rag)


## Testing the model accuracy over serving API

A simple method of testing the response accuracy is via comparing the response for a sample prompt from the model server and with local python execution based on HuggingFace python code.

The script [compare_results.py](https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/4/demos/embeddings/compare_results.py) can assist with such experiment.
```bash
popd
cd model_server/demos/embeddings
python compare_results.py --model BAAI/bge-large-en-v1.5 --service_url http://localhost:8000/v3/embeddings --pooling CLS --input "hello world" --input "goodbye world"

input ['hello world', 'goodbye world']
HF Duration: 93.921 ms BertModel
OVMS Duration: 160.806 ms
Batch number: 0
OVMS embeddings: shape: (1024,) emb[:20]:
 [ 0.0336  0.0321  0.0213 -0.0373 -0.0156 -0.0122  0.0246  0.0412  0.0492
  0.0207  0.0056  0.0169 -0.0133  0.0009 -0.0421  0.0206 -0.0222 -0.0291
 -0.0532  0.0382]
HF AutoModel: shape: (1024,) emb[:20]:
 [ 0.0343  0.0332  0.0219 -0.0371 -0.0158 -0.0131  0.0247  0.0408  0.0489
  0.0208  0.0053  0.0176 -0.0132  0.001  -0.0422  0.0208 -0.0213 -0.0278
 -0.0538  0.0388]
Difference score with HF AutoModel: 0.020708760995591734
Batch number: 1
OVMS embeddings: shape: (1024,) emb[:20]:
 [ 0.0161  0.0156  0.0235  0.0199  0.0005 -0.0559  0.0124  0.0122  0.0205
 -0.027   0.0152  0.0153 -0.0429 -0.0537 -0.0514 -0.0059 -0.0294 -0.0451
 -0.0371  0.0361]
HF AutoModel: shape: (1024,) emb[:20]:
 [ 0.0175  0.0161  0.0234  0.0196  0.0012 -0.0565  0.0109  0.0111  0.0194
 -0.0275  0.0148  0.0144 -0.0425 -0.0538 -0.0515 -0.0062 -0.0298 -0.0447
 -0.0376  0.0359]
Difference score with HF AutoModel: 0.020293646680283224

```

It is easy also to run model evaluation using [MTEB](https://github.com/embeddings-benchmark/mteb) framework using a custom class based on openai model:
```bash
pip install "mteb<2" einops openai --extra-index-url "https://download.pytorch.org/whl/cpu"
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/4/demos/embeddings/ovms_mteb.py -o ovms_mteb.py
python ovms_mteb.py --model BAAI/bge-large-en-v1.5 --service_url http://localhost:8000/v3/embeddings
```
Results will be stored in `results` folder:
```json
{
  "dataset_revision": "0fd18e25b25c072e09e0d92ab615fda904d66300",
  "task_name": "Banking77Classification",
  "mteb_version": "1.39.7",
  "scores": {
    "test": [
      {
        "accuracy": 0.848636,
        "f1": 0.842405,
        "f1_weighted": 0.842405,
        "scores_per_experiment": [
          {
            "accuracy": 0.842532,
            "f1": 0.835091,
            "f1_weighted": 0.835091
          },
          {
            "accuracy": 0.851299,
            "f1": 0.844622,
            "f1_weighted": 0.844622
          },
          {
            "accuracy": 0.849026,
            "f1": 0.842238,
            "f1_weighted": 0.842238
          },
          {
            "accuracy": 0.853571,
            "f1": 0.849815,
            "f1_weighted": 0.849815
          },
          {
            "accuracy": 0.846104,
            "f1": 0.839,
            "f1_weighted": 0.839
          },
          {
            "accuracy": 0.849675,
            "f1": 0.844259,
            "f1_weighted": 0.844259
          },
          {
            "accuracy": 0.846104,
            "f1": 0.840343,
            "f1_weighted": 0.840343
          },
          {
            "accuracy": 0.846753,
            "f1": 0.8397,
            "f1_weighted": 0.8397
          },
          {
            "accuracy": 0.853571,
            "f1": 0.848239,
            "f1_weighted": 0.848239
          },
          {
            "accuracy": 0.847727,
            "f1": 0.84074,
            "f1_weighted": 0.84074
          }
        ],
        "main_score": 0.848636,
        "hf_subset": "default",
        "languages": [
          "eng-Latn"
        ]
      }
    ]
  },
  "evaluation_time": 3841.1886789798737,
  "kg_co2_emissions": null
}
```
Compare against local HuggingFace execution for reference:
```console
mteb run -m thenlper/gte-small -t Banking77Classification --output_folder results
``` 

# Usage of tokenize endpoint (release 2025.4 or weekly)

The `tokenize` endpoint provides a simple API for tokenizing input text using the same tokenizer as the deployed embeddings model. This allows you to see how your text will be split into tokens before feature extraction or inference. The endpoint accepts a string or list of strings and returns the corresponding token IDs.

Example usage:
```console
curl http://localhost:8000/v3/tokenize -H "Content-Type: application/json" -d "{ \"model\": \"BAAI/bge-large-en-v1.5\", \"text\": \"hello world\" }"
```
Response:
```json
{
  "tokens": [101,7592,2088,102]
}
```

It's possible to use additional parameters:
 - `pad_to_max_length` - whether to pad the sequence to the maximum length. Default is False. 
 - `max_length` - maximum length of the sequence. If specified, it truncates the tokens to the provided number.
 - `padding_side` - side to pad the sequence, can be `left` or `right`. Default is `right`.
 - `add_special_tokens` - whether to add special tokens like BOS, EOS, PAD. Default is True. 

 Example usage:
```console
curl http://localhost:8000/v3/tokenize -H "Content-Type: application/json" -d "{ \"model\": \"BAAI/bge-large-en-v1.5\", \"text\": \"hello world\", \"max_length\": 10, \"pad_to_max_length\": true, \"padding_side\": \"left\", \"add_special_tokens\": true }"
```

Response:
```json
{
  "tokens":[0,0,0,0,0,0,101,7592,2088,102]
}
```
