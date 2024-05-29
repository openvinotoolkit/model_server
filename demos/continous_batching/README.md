# How to serve LLM models with Continous Batching via OpenAI API {#ovms_demos_continous_batching}

## Model preparation
Download latest optimum-intel:
```bash
PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" python3 -m pip install "optimum-intel[nncf,openvino]"@git+https://github.com/huggingface/optimum-intel.git openvino-tokenizers
```

Run optimum-cli to download and quantize the model:
```bash
optimum-cli export openvino --model meta-llama/Llama-2-7b-chat-hf --weight-format int8 meta-llama/Llama-2-7b-chat-hf
```
Copy the graph to the model folder. The same graph can be used for a range of LLM models.
```bash
cat graph.pbtxt
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "LLMExecutor"
  calculator: "HttpLLMCalculator"
  input_stream: "LOOPBACK:loopback"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "LLM_NODE_RESOURCES:llm"
  output_stream: "LOOPBACK:loopback"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  input_stream_info: {
    tag_index: 'LOOPBACK:0',
    back_edge: true
  }
  node_options: {
      [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
          models_path: "./"
      }
  }
  input_stream_handler {
    input_stream_handler: "SyncSetInputStreamHandler",
    options {
      [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
        sync_set {
          tag_index: "LOOPBACK:0"
        }
      }
    }
  }
}

cp graph.pbtxt meta-llama/Llama-2-7b-chat-hf/

tree meta-llama/
meta-llama/
└── Llama-2-7b-chat-hf
    ├── config.json
    ├── generation_config.json
    ├── graph.pbtxt
    ├── openvino_detokenizer.bin
    ├── openvino_detokenizer.xml
    ├── openvino_model.bin
    ├── openvino_model.xml
    ├── openvino_tokenizer.bin
    ├── openvino_tokenizer.xml
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── tokenizer.model

```

## Server configuration
Prepare config.json:
```bash
cat config.json
{
    "model_config_list": [],
    "mediapipe_config_list": [
        {
            "name": "meta-llama/Llama-2-7b-chat-hf",
            "graph_path": "graph.pbtxt"
        }
    ]
}
```


## Start-up
```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/:/workspace:ro openvino/model_server --port 9000 --rest_port 8000 --config_path /workspace/config.json
```
Wait for the model to load. You can check the status with a simple command:
```bash
curl http://localhost:8000/v1/config
{
"meta-llama/Llama-2-7b-chat-hf" : 
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
}

## Client code

Both unary and streaming calls should be available via the same servable:

### Unary:
```bash
curl http://localhost:8000/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "max_tokens":30,
    "stream":false,
    "messages": [
      {
        "role": "user",
        "content": "What is OpenVINO?"
      }
    ]
  }'| jq .
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "\n\nOpenVINO is an open-source software library for deep learning inference that is designed to optimize and run deep learning models on a variety",
        "role": "assistant"
      }
    }
  ],
  "created": 1716825108,
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "object": "chat.completion"
}
```

## Streaming:

The endpoint `chat/completions` is compatible with OpenAI client so it can be easily used to generated code also in streaming mode:

Install the client library:
```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

stream = client.chat.completions.create(
    model="llama",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

Output:
```
This is a test.
```

## Benchmarking text generation with high concurrency

OpenVINO Model Server employs efficient parallelization for text generation. It can be used to generate text also in high concurrency in the environment shared by multiple clients.
It can be demostrated using benchmarking app from vLLM repository:
```bash
git clone https://github.com/vllm-project/vllm
cd vllm/benchmarks
pip install -r ../requirements-cpu.txt
sed -i -e 's|v1/chat/completions|v3/chat/completions|g' backend_request_func.py  # allow calls to endpoint with v3 instead of v1 like in vLLM
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json  # sample dataset
python benchmark_serving.py --host localhost --port 8000 --endpoint /v3/chat/completions --backend openai-chat --model meta-llama/Llama-2-7b-chat-hf --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json  --num-prompts 1000 --request-rate 1

```