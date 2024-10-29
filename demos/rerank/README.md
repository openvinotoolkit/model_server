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
LLM engine parameters will be defined inside the `graph.pbtxt` file.

Install python dependencies for the conversion script:
```bash
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
pip3 install optimum-intel@git+https://github.com/huggingface/optimum-intel.git  openvino-tokenizers[transformers]==2024.4.* openvino==2024.4.* nncf>=2.11.0 sentence_transformers==3.1.1 openai "transformers<4.45"
```

Run optimum-cli to download and quantize the model:
```bash
cd demos/rerank
convert_tokenizer -o models/BAAI/bge-reranker-large-tokenizer/1 BAAI/bge-reranker-large
optimum-cli export openvino --disable-convert-tokenizer --model BAAI/bge-reranker-large --task text-classification --trust-remote-code models/BAAI/bge-reranker-large-rerank/1
```

You should have a model folder like below:
```bash
tree models/BAAI
models/BAAI
├── bge-reranker-large-tokenizer
│   └── 1
│       ├── openvino_tokenizer.bin
│       └── openvino_tokenizer.xml
└── bge-reranker-large-rerank
    └── 1
        ├── config.json
        ├── openvino_model.bin
        ├── openvino_model.xml
        ├── sentencepiece.bpe.model
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        └── tokenizer.json
```
> **Note** The actual models support version management and can be automatically swapped to newer version when new model is uploaded in newer version folder. The models can be also stored on the cloud storage like s3, gcs or azure storage.


## Server configuration
Prepare config.json:
```bash
cat config.json
{
    "model_config_list": [
        {
            "config": {
                "name": "tokenizer",
                "base_path": "/workspace/models/BAAI/bge-reranker-large-tokenizer"
            }
        },
        {
            "config": {
                "name": "rerank_model",
                "base_path": "/workspace/models/BAAI/bge-reranker-large-rerank"
            }
        }
    ],
    "mediapipe_config_list": [
        {
            "name": "rerank",
            "graph_path": "/workspace/models/graph.pbtxt"
        }
    ]
}

```


## Start-up
```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/:/workspace:ro openvino/model_server:latest --port 9000 --rest_port 8000 --config_path /workspace/config.json
```

## Client code


```python
import requests

# Define the API endpoint and your API key
api_endpoint = "http://localhost:8000/v3/rerank"
api_key = "YOUR_API_KEY"

# Create the payload
payload = {
    "model": "rerank",
    "query": "what is it?",
    "documents": [
        "one text",
        "two text longer than one"
    ]
}

# Set up the headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Make the POST request
response = requests.post(api_endpoint, json=payload, headers=headers)

# Handle the response
if response.status_code == 200:
    reranked_results = response.json()
    print("Reranked Results:", reranked_results)
else:
    print("Error:", response.status_code, response.text)

```

Check OVMS logs for results (serialization is not implemented yet)

```
-0.64591
-6.86708
0.343912 0.00104043 
```

## Comparision with LangChain OpenVINOReranker

```python
from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker, RerankRequest

reranker = OpenVINOReranker(
    model_name_or_path='BAAI/bge-reranker-large',
    top_n=3,
)

print(reranker.rerank(RerankRequest(
    query="what is it?",
    passages=[dict(text="one text"), dict(text="two text longer than one")]
)))

```

Result is the same:

```
[{'text': 'one text', 'score': tensor(0.3439)}, {'text': 'two text longer than one', 'score': tensor(0.0010)}]
```
