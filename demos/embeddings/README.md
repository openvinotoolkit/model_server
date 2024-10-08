# How to serve Embeddings models via OpenAI API {#ovms_demos_embeddings}
This demo shows how to deploy embeddings models in the OpenVINO Model Server for text feature extractions.
Text generation use case is exposed via OpenAI API `embeddings` endpoint.

## Get the docker image

Until the feature is not included in public image, build the image from source to try the latest enhancements in this feature. In 2024.5 public release this command will be optional.
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
```
> **Note** Change the `--weight-format` to quantize the model to `fp16`, `int8` or `int4` precision to reduce memory consumption and improve performance.

You should have a model folder like below:
```bash
tree models/
models/
├── graph.pbtxt
├── gte-large-en-v1.5-embeddings
│   └── 1
│       ├── config.json
│       ├── openvino_model.bin
│       ├── openvino_model.xml
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── vocab.txt
├── gte-large-en-v1.5-tokenizer
│   └── 1
│       ├── openvino_tokenizer.bin
│       └── openvino_tokenizer.xml
└── subconfig.json

```

The default configuration of the `LLMExecutor` should work in most cases but the parameters can be tuned inside the `node_options` section in the `graph.pbtxt` file. 
Runtime configuration for both models can be tunned in `subconfig.json` file. 

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

A single servable exposes both `chat/completions` and `completions` endpoints with and without stream capabilities.
Chat endpoint is expected to be used for scenarios where conversation context should be pasted by the client and the model prompt is created by the server based on the jinja model template.
Completion endpoint should be used to pass the prompt directly by the client and for models without the jinja template.

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

Altenratively there could be used openai python client like in the example below:

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
responses = client.embeddings.create(
    input=[
        "Hello my name is",
        "Model server can support embeddings endpoint"
    ],
    model=model,
)
for data in responses.data:
    print(data.embedding)
```



## Benchmarking feature extraction

TBD

## RAG with Model Server

TBD

## Scaling the Model Server

TBD

## Testing the model accuracy over serving API

TBD