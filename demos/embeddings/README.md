# Text Embeddings models via OpenAI API {#ovms_demos_embeddings}
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
mkdir -p ${HOME}/models
docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/Qwen3-Embedding-0.6B-int8-ov --pooling LAST --task embeddings
docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name Qwen3-Embedding-0.6B-int8-ov --model_path OpenVINO/Qwen3-Embedding-0.6B-int8-ov
```

**On Bare Metal (Windows)**
```bat
mkdir C:\models
ovms --pull --model_repository_path C:/models --source_model OpenVINO/Qwen3-Embedding-0.6B-int8-ov --pooling LAST --task embeddings
ovms --add_to_config --config_path C:/models/config.json --model_name OpenVINO/Qwen3-Embedding-0.6B-int8-ov --model_path OpenVINO/Qwen3-Embedding-0.6B-int8-ov
```
:::
:::{tab-item} OpenVINO/bge-base-en-v1.5-int8-ov
:sync: bge-base-en-v1.5-int8-ov
**Using docker image**
```bash
docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/bge-base-en-v1.5-int8-ov --pooling CLS --task embeddings

docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name bge-base-en-v1.5-int8-ov --model_path OpenVINO/bge-base-en-v1.5-int8-ov
```

**On Bare Metal (Windows)**
```bat
ovms --pull --model_repository_path C:/models --source_model OpenVINO/bge-base-en-v1.5-int8-ov --pooling CLS --task embeddings

ovms --add_to_config --config_path C:/models/config.json --model_name bge-base-en-v1.5-int8-ov --model_path OpenVINO/bge-base-en-v1.5-int8-ov
```
:::
:::{tab-item} OpenVINO/bge-m3-int4-ov
:sync: bge-m3-int4-ov
**Using docker image**
```bash
docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/bge-m3-int4-ov --pooling CLS --task embeddings

docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name bge-m3-int4-ov --model_path OpenVINO/bge-m3-int4-ov
```

**On Bare Metal (Windows)**
```bat
ovms --pull --model_repository_path C:/models --source_model OpenVINO/bge-m3-int4-ov --pooling CLS --task embeddings

ovms --add_to_config --config_path C:/models/config.json --model_name bge-m3-int4-ov --model_path OpenVINO/bge-m3-int4-ov
```
:::
::::

**GPU**
::::{tab-set}
:::{tab-item} OpenVINO/Qwen3-Embedding-0.6B-int8-ov
:sync: Qwen3-Embedding-0.6B-int8-ov
**Using docker image**
```bash
docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/Qwen3-Embedding-0.6B-int8-ov --pooling LAST --target_device GPU --task embeddings

docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name Qwen3-Embedding-0.6B-int8-ov --model_path OpenVINO/Qwen3-Embedding-0.6B-int8-ov
```

**On Bare Metal (Windows)**
```bat
ovms --pull --model_repository_path C:/models --source_model OpenVINO/Qwen3-Embedding-0.6B-int8-ov --pooling LAST --target_device GPU --task embeddings

ovms --add_to_config --config_path C:/models/config.json --model_name OpenVINO/Qwen3-Embedding-0.6B-int8-ov --model_path OpenVINO/Qwen3-Embedding-0.6B-int8-ov
```
:::
:::{tab-item} OpenVINO/bge-base-en-v1.5-int8-ov
:sync: OpenVINO/bge-base-en-v1.5-int8-ov
**Using docker image**
```bash
docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/bge-base-en-v1.5-int8-ov --pooling CLS --target_device GPU --task embeddings

docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name bge-base-en-v1.5-int8-ov --model_path OpenVINO/bge-base-en-v1.5-int8-ov
```

**On Bare Metal (Windows)**
```bat
ovms --pull --model_repository_path C:/models --source_model OpenVINO/bge-base-en-v1.5-int8-ov --pooling CLS --target_device GPU --task embeddings

ovms --add_to_config --config_path C:/models/config.json --model_name bge-base-en-v1.5-int8-ov --model_path OpenVINO/bge-base-en-v1.5-int8-ov
```
:::
:::{tab-item} OpenVINO/bge-m3-int4-ov
:sync: bge-m3-int4-ov
**Using docker image**
```bash
docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/bge-m3-int4-ov --pooling CLS --target_device GPU --task embeddings

docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name bge-m3-int4-ov --model_path OpenVINO/bge-m3-int4-ov
```

**On Bare Metal (Windows)**
```bat
ovms --pull --model_repository_path C:/models --source_model OpenVINO/bge-m3-int4-ov --pooling CLS --target_device GPU --task embeddings

ovms --add_to_config --config_path C:/models/config.json --model_name bge-m3-int4-ov --model_path OpenVINO/bge-m3-int4-ov
```
:::
::::

### Pulling OpenVINO IR with tokenizer conversion

Some HuggingFace repositories already include OpenVINO IR of the model in the `openvino` directory, but without the tokenizer and detokenizer models.
The script below downloads only that directory and generates the tokenizer and detokenizer with `convert_tokenizer`. It stops if the repository has no `openvino` directory.

Download the script and install its dependencies:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/embeddings/download_ir_with_tokenizer.py -o download_ir_with_tokenizer.py
pip install huggingface_hub "openvino-tokenizers[transformers]"
```

Download the model with the script. The command is the same on Linux and Windows:

::::{tab-set}
:::{tab-item} thenlper/gte-small
:sync: thenlper/gte-small
```console
python download_ir_with_tokenizer.py --model_id thenlper/gte-small
```
:::
:::{tab-item} sentence-transformers/all-MiniLM-L12-v2
:sync: sentence-transformers/all-MiniLM-L12-v2
```console
python download_ir_with_tokenizer.py --model_id sentence-transformers/all-MiniLM-L12-v2
```
:::
:::{tab-item} sentence-transformers/all-distilroberta-v1
:sync: sentence-transformers/all-distilroberta-v1
```console
python download_ir_with_tokenizer.py --model_id sentence-transformers/all-distilroberta-v1
```
:::
:::{tab-item} intfloat/multilingual-e5-large
:sync: intfloat/multilingual-e5-large
```console
python download_ir_with_tokenizer.py --model_id intfloat/multilingual-e5-large
```
:::
:::{tab-item} sentence-transformers/all-mpnet-base-v2
:sync: sentence-transformers/all-mpnet-base-v2
```console
python download_ir_with_tokenizer.py --model_id sentence-transformers/all-mpnet-base-v2
```
:::
::::

Then create `graph.pbtxt` and add the model to `config.json`:

::::{tab-set}
:::{tab-item} thenlper/gte-small
:sync: thenlper/gte-small
**Using docker image**
```bash
docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --configure --model_path /models/thenlper/gte-small --task embeddings --pooling CLS

docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name thenlper/gte-small --model_path thenlper/gte-small
```

**On Bare Metal (Windows)**
```bat
ovms --configure --model_path C:/models/thenlper/gte-small --task embeddings --pooling CLS

ovms --add_to_config --config_path C:/models/config.json --model_name thenlper/gte-small --model_path thenlper/gte-small
```
:::
:::{tab-item} sentence-transformers/all-MiniLM-L12-v2
:sync: sentence-transformers/all-MiniLM-L12-v2
**Using docker image**
```bash
docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --configure --model_path /models/sentence-transformers/all-MiniLM-L12-v2 --task embeddings --pooling MEAN

docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name sentence-transformers/all-MiniLM-L12-v2 --model_path sentence-transformers/all-MiniLM-L12-v2
```

**On Bare Metal (Windows)**
```bat
ovms --configure --model_path C:/models/sentence-transformers/all-MiniLM-L12-v2 --task embeddings --pooling MEAN

ovms --add_to_config --config_path C:/models/config.json --model_name sentence-transformers/all-MiniLM-L12-v2 --model_path sentence-transformers/all-MiniLM-L12-v2
```
:::
:::{tab-item} sentence-transformers/all-distilroberta-v1
:sync: sentence-transformers/all-distilroberta-v1
**Using docker image**
```bash
docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --configure --model_path /models/sentence-transformers/all-distilroberta-v1 --task embeddings --pooling MEAN

docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name sentence-transformers/all-distilroberta-v1 --model_path sentence-transformers/all-distilroberta-v1
```

**On Bare Metal (Windows)**
```bat
ovms --configure --model_path C:/models/sentence-transformers/all-distilroberta-v1 --task embeddings --pooling MEAN

ovms --add_to_config --config_path C:/models/config.json --model_name sentence-transformers/all-distilroberta-v1 --model_path sentence-transformers/all-distilroberta-v1
```
:::
:::{tab-item} intfloat/multilingual-e5-large
:sync: intfloat/multilingual-e5-large
**Using docker image**
```bash
docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --configure --model_path /models/intfloat/multilingual-e5-large --task embeddings --pooling MEAN

docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name intfloat/multilingual-e5-large --model_path intfloat/multilingual-e5-large
```

**On Bare Metal (Windows)**
```bat
ovms --configure --model_path C:/models/intfloat/multilingual-e5-large --task embeddings --pooling MEAN

ovms --add_to_config --config_path C:/models/config.json --model_name intfloat/multilingual-e5-large --model_path intfloat/multilingual-e5-large
```
:::
:::{tab-item} sentence-transformers/all-mpnet-base-v2
:sync: sentence-transformers/all-mpnet-base-v2
**Using docker image**
```bash
docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --configure --model_path /models/sentence-transformers/all-mpnet-base-v2 --task embeddings --pooling MEAN

docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name sentence-transformers/all-mpnet-base-v2 --model_path sentence-transformers/all-mpnet-base-v2
```

**On Bare Metal (Windows)**
```bat
ovms --configure --model_path C:/models/sentence-transformers/all-mpnet-base-v2 --task embeddings --pooling MEAN

ovms --add_to_config --config_path C:/models/config.json --model_name sentence-transformers/all-mpnet-base-v2 --model_path sentence-transformers/all-mpnet-base-v2
```
:::
::::

> **Note** To run the model on GPU, add `--target_device GPU` to the `--configure` command.

**NPU**
::::{tab-set}
:::{tab-item} Qwen/Qwen3-Embedding-0.6B
:sync: Qwen3-Embedding-0.6B-int8
```bash
docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/Qwen3-Embedding-0.6B-int8-ov --pooling LAST --task embeddings --target_device NPU --max_length 1000

docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name Qwen3-Embedding-0.6B-int8-ov --model_path OpenVINO/Qwen3-Embedding-0.6B-int8-ov
```
:::
:::{tab-item} OpenVINO/bge-base-en-v1.5-int8-ov
:sync: OpenVINO/bge-base-en-v1.5-int8-ov
```bash
docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/bge-base-en-v1.5-int8-ov --pooling CLS --task embeddings --target_device NPU --max_length 400

docker run --user $(id -u):$(id -g) --rm -v ${HOME}/models:/models:rw openvino/model_server:latest --add_to_config --config_path /models/config.json --model_name bge-base-en-v1.5-int8-ov --model_path OpenVINO/bge-base-en-v1.5-int8-ov
```
:::
::::
> **Note** For NPU Set `--max_length` as low as possible to speed up model loading and inference.

> **Note** For NPU the pooling mode --pooling LAST has the best accuracy.

> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.

## Server Deployment

:::{dropdown} **Deploying with Docker**

**CPU**
```bash
docker run -d --rm -p 8000:8000 -v ${HOME}/models:/workspace:ro openvino/model_server:latest --rest_port 8000 --config_path /workspace/config.json
```
**GPU**

In case you want to use GPU device to run the embeddings model, add extra docker parameters `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)` 
to `docker run` command, use the image with GPU support and make sure set the target_device in subconfig.json to GPU. Also make sure the export model quantization level and cache size fit to the GPU memory. All of that can be applied with the commands:

```bash
docker run -d --rm -p 8000:8000 --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -v ${HOME}/models:/workspace:ro openvino/model_server:latest-gpu --rest_port 8000 --config_path /workspace/config.json
```
**NPU**
NOTE: NPU execution for embeddings model is a preview feature.
In case you want to use NPU device to run the embeddings model, add extra docker parameters `--device /dev/accel --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)` 
to `docker run` command, use the image with NPU support and make sure set the target_device in subconfig.json to NPU. Also make sure the export model quantization level and cache size fit to the NPU memory. All of that can be applied with the commands:

```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000 --device /dev/accel --group-add=$(stat -c "%g" /dev/dri/render*  | head -1) -v ${HOME}/models:/workspace:ro openvino/model_server:latest-gpu --rest_port 8000 --config_path /workspace/config.json
```
:::

:::{dropdown} **Deploying on Bare Metal**

Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.

Depending on how you prepared models in the first step of this demo, they are deployed to either CPU or GPU (it's defined in `config.json`). If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

```bat
ovms --rest_port 8000 --config_path C:/models/config.json
```
:::

### Readiness Check

Wait for the model to load. You can check the status with a simple command below:
```bash
curl http://localhost:8000/v1/models/bge-base-en-v1.5-int8-ov

{"id":"bge-base-en-v1.5-int8-ov","object":"model","created":1763997378,"owned_by":"OVMS"}
```

## Client code

:::{dropdown} **Request embeddings with cURL**
```bash
curl http://localhost:8000/v1/embeddings -H "Content-Type: application/json" -d "{ \"model\": \"bge-base-en-v1.5-int8-ov\", \"input\": \"hello world\"}"
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
pip3 install openai numpy
```
```bash
echo '
from openai import OpenAI
import numpy as np

client = OpenAI(
  base_url="http://localhost:8000/v1",
  api_key="unused"
)
model = "bge-base-en-v1.5-int8-ov"
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
It will report results like `Similarity score as cos_sim 0.9640598645292134`.

:::

## Benchmarking feature extraction
For synthetic and Hugging Face dataset examples, setup, and token-counting details,
see [Text Embeddings](../benchmark/openai/README.md#text-embeddings) in the benchmark guide.

## RAG with Model Server

Embeddings endpoint can be applied in RAG chains to delegated text feature extraction both for documented vectorization and in context retrieval.
Check this demo to see the langchain code example which is using OpenVINO Model Server both for text generation and embedding endpoint in [RAG application demo](https://github.com/openvinotoolkit/model_server/tree/main/demos/continuous_batching/rag)


## Testing the model accuracy over serving API

A simple method of testing the response accuracy is via comparing the response for a sample prompt from the model server and with local python execution based on HuggingFace python code.

The script [compare_results.py](https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/embeddings/compare_results.py) can assist with such experiment.
```bash
curl -fL https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/embeddings/compare_results.py -o compare_results.py
python compare_results.py --model_name bge-base-en-v1.5-int8-ov --hf_model_name BAAI/bge-base-en-v1.5 --service_url http://localhost:8000/v1 --pooling CLS --input "hello world" --input "goodbye world"

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
```console
pip install mteb tiktoken einops openai --extra-index-url "https://download.pytorch.org/whl/cpu"
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/embeddings/ovms_mteb.py -o ovms_mteb.py
python ovms_mteb.py --model_name bge-base-en-v1.5-int8-ov --service_url http://localhost:8000/v1
```
Results will be stored in `results` folder:
```json
{
  "dataset_revision": "0fd18e25b25c072e09e0d92ab615fda904d66300",
  "task_name": "Banking77Classification",
  "mteb_version": "2.27.22",
  "scores": {
    "test": [
      {
        "scores_per_experiment": [
          {
            "accuracy": 0.835714,
            "f1": 0.82741,
            "f1_weighted": 0.82741,
            "precision": 0.842506,
            "precision_weighted": 0.842506,
            "recall": 0.835714,
            "recall_weighted": 0.835714,
            "ap": null,
            "ap_weighted": null
          },
          {
            "accuracy": 0.841234,
            "f1": 0.833948,
            "f1_weighted": 0.833948,
            "precision": 0.84872,
            "precision_weighted": 0.84872,
            "recall": 0.841234,
            "recall_weighted": 0.841234,
            "ap": null,
            "ap_weighted": null
          },
          {
            "accuracy": 0.841883,
            "f1": 0.834378,
            "f1_weighted": 0.834378,
            "precision": 0.854342,
            "precision_weighted": 0.854342,
            "recall": 0.841883,
            "recall_weighted": 0.841883,
            "ap": null,
            "ap_weighted": null
          },
          {
            "accuracy": 0.842532,
            "f1": 0.837997,
            "f1_weighted": 0.837997,
            "precision": 0.85386,
            "precision_weighted": 0.85386,
            "recall": 0.842532,
            "recall_weighted": 0.842532,
            "ap": null,
            "ap_weighted": null
          },
          {
            "accuracy": 0.83961,
            "f1": 0.833709,
            "f1_weighted": 0.833709,
            "precision": 0.854358,
            "precision_weighted": 0.854358,
            "recall": 0.83961,
            "recall_weighted": 0.83961,
            "ap": null,
            "ap_weighted": null
          },
          {
            "accuracy": 0.844156,
            "f1": 0.839925,
            "f1_weighted": 0.839925,
            "precision": 0.857884,
            "precision_weighted": 0.857884,
            "recall": 0.844156,
            "recall_weighted": 0.844156,
            "ap": null,
            "ap_weighted": null
          },
          {
            "accuracy": 0.837662,
            "f1": 0.832103,
            "f1_weighted": 0.832103,
            "precision": 0.850791,
            "precision_weighted": 0.850791,
            "recall": 0.837662,
            "recall_weighted": 0.837662,
            "ap": null,
            "ap_weighted": null
          },
          {
            "accuracy": 0.836039,
            "f1": 0.828496,
            "f1_weighted": 0.828496,
            "precision": 0.854481,
            "precision_weighted": 0.854481,
            "recall": 0.836039,
            "recall_weighted": 0.836039,
            "ap": null,
            "ap_weighted": null
          },
          {
            "accuracy": 0.843506,
            "f1": 0.835978,
            "f1_weighted": 0.835978,
            "precision": 0.85638,
            "precision_weighted": 0.85638,
            "recall": 0.843506,
            "recall_weighted": 0.843506,
            "ap": null,
            "ap_weighted": null
          },
          {
            "accuracy": 0.834416,
            "f1": 0.828135,
            "f1_weighted": 0.828135,
            "precision": 0.854075,
            "precision_weighted": 0.854075,
            "recall": 0.834416,
            "recall_weighted": 0.834416,
            "ap": null,
            "ap_weighted": null
          }
        ],
        "accuracy": 0.839675,
        "f1": 0.833208,
        "f1_weighted": 0.833208,
        "precision": 0.85274,
        "precision_weighted": 0.85274,
        "recall": 0.839675,
        "recall_weighted": 0.839675,
        "ap": NaN,
        "ap_weighted": NaN,
        "main_score": 0.839675,
        "hf_subset": "default",
        "languages": [
          "eng-Latn"
        ],
        "mteb_version": "2.27.22"
      }
    ]
  },
  "evaluation_time": 127.6319568157196,
  "kg_co2_emissions": null,
  "date": null,
  "evaluation_phases": null
}
```
Compare against local HuggingFace execution for reference:
```console
mteb run -m BAAI/bge-base-en-v1.5 -t Banking77Classification --output_folder results
``` 

# Usage of tokenize endpoint

The `tokenize` endpoint provides a simple API for tokenizing input text using the same tokenizer as the deployed embeddings model. This allows you to see how your text will be split into tokens before feature extraction or inference. The endpoint accepts a string or list of strings and returns the corresponding token IDs.

Example usage:
```console
curl http://localhost:8000/v1/tokenize -H "Content-Type: application/json" -d "{ \"model\": \"bge-base-en-v1.5-int8-ov\", \"text\": \"hello world\" }"
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
curl http://localhost:8000/v1/tokenize -H "Content-Type: application/json" -d "{ \"model\": \"bge-base-en-v1.5-int8-ov\", \"text\": \"hello world\", \"max_length\": 10, \"pad_to_max_length\": true, \"padding_side\": \"left\", \"add_special_tokens\": true }"
```

Response:
```json
{
  "tokens":[0,0,0,0,0,0,101,7592,2088,102]
}
```
