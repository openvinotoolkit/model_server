# How to serve Embeddings models via OpenAI API {#ovms_demos_embeddings}
This demo shows how to deploy embeddings models in the OpenVINO Model Server for text feature extractions.
Text generation use case is exposed via OpenAI API `embeddings` endpoint.

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
LLM engine parameters will be defined inside the `graph.pbtxt` file.

Install python dependencies for the conversion script:
```bash
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
pip3 install optimum-intel@git+https://github.com/huggingface/optimum-intel.git  openvino-tokenizers[transformers]==2024.4.* openvino==2024.4.* nncf>=2.11.0 sentence_transformers==3.1.1 openai
```

Run optimum-cli to download and quantize the model:
```bash
cd demos/embeddings
convert_tokenizer -o models/gte-large-en-v1.5-tokenizer/1 Alibaba-NLP/gte-large-en-v1.5
optimum-cli export openvino --disable-convert-tokenizer --model Alibaba-NLP/gte-large-en-v1.5 --task feature-extraction --weight-format int8 --trust-remote-code --library sentence_transformers  models/gte-large-en-v1.5-embeddings/1
rm models/gte-large-en-v1.5-embeddings/1/*.json models/gte-large-en-v1.5-embeddings/1/vocab.txt 
```
> **Note** Change the `--weight-format` to quantize the model to `fp16`, `int8` or `int4` precision to reduce memory consumption and improve performance.

You should have a model folder like below:
```bash
tree models/
models/
├── graph.pbtxt
├── gte-large-en-v1.5-embeddings
│   └── 1
│       ├── openvino_model.bin
│       └── openvino_model.xml
├── gte-large-en-v1.5-tokenizer
│   └── 1
│       ├── openvino_tokenizer.bin
│       └── openvino_tokenizer.xml
└── subconfig.json

```
> **Note** The actual models support version management and can be automatically swapped to newer version when new model is uploaded in newer version folder. The models can be also stored on the cloud storage like s3, gcs or azure storage.

> **Note** Check for other tested models in script `export_all_models.sh`.

The default configuration of the `LLMExecutor` should work in most cases but the parameters can be tuned inside the `node_options` section in the `graph.pbtxt` file. 
Runtime configuration for both models can be tuned in `subconfig.json` file. 


## Server configuration
Prepare config.json:
```bash
cat config.json
{
    "model_config_list": [],
    "mediapipe_config_list": [
        {
            "name": "Alibaba-NLP/gte-large-en-v1.5",
            "base_path": "models"
        }
    ]
}
```


## Start-up
```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/:/workspace:ro openvino/model_server:latest --port 9000 --rest_port 8000 --config_path /workspace/config.json
```
In case you want to use GPU device to run the embeddings model, add extra docker parameters `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)` 
to `docker run` command, use the image with GPU support and make sure set the target_device in subconfig.json to GPU. 
Also make sure the export model quantization level and cache size fit to the GPU memory.
```


Wait for the model to load. You can check the status with a simple command:
```bash
curl -s http://localhost:8000/v1/config | jq -c .
{"Alibaba-NLP/gte-large-en-v1.5":{"model_version_status":[{"version":"1","state":"AVAILABLE","status":{"error_code":"OK","error_message":"OK"}}]},"embeddings_model":{"model_version_status":[{"version":"1","state":"AVAILABLE","status":{"error_code":"OK","error_message":"OK"}}]},"tokenizer":{"model_version_status":[{"version":"1","state":"AVAILABLE","status":{"error_code":"OK","error_message":"OK"}}]}}
```

## Client code


```bash
curl http://localhost:8000/v3/embeddings \
  -H "Content-Type: application/json" -d '{ "model": "Alibaba-NLP/gte-large-en-v1.5", "input": "hello world"}' | jq .
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

Alternatively there could be used openai python client like in the example below:

```bash
pip3 install openai
```
```python
from openai import OpenAI

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
print("Similarity score as cos_sim", cos_sim)
```
It will report results like `Similarity score as cos_sim 0.97654650115054`.

## Benchmarking feature extraction

TBD

## RAG with Model Server

Embeddings endpoint can be applied in RAG chains to deletated text feature extraction both for documented vectorization and in context retrieval.
Check this demo to see the langchain code example which is using OpenVINO Model Server both for text generation and embedding endpoint in [RAG application demo](https://github.com/openvinotoolkit/model_server/tree/main/demos/continuous_batching/rag)

## Deploying multiple embedding models

It is possible to deploy multiple graphs and models on a single model server instance. For each model the same export steps should be repeated and each pipeline should be added to the configuration file.
The following script prepares the repository with all tested models:
```bash
./export_all_models.sh
```
It creates `config_all.json` with models structure including IR files, `graph.pbtxt` definitions and `subconfig.json` subconfigs.

All those models can be deployed together via:
```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/:/workspace:ro openvino/model_server:latest --port 9000 --rest_port 8000 --config_path /workspace/config_all.json
```


## Testing the model accuracy over serving API

A simple method of testing the response accuracy is via comparing the response for a sample prompt from the model server and with local python execution based on HuggingFace python code.

The script [compare_results.py](./compare_results.py) can assist with such experiment.
```bash
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
```
pip install mteb
python mteb_ovms.py --model Alibaba-NLP/gte-large-en-v1.5 --service_url http://localhost:8000/v3/embeddings
```


