# How to serve LLM models with Continuous Batching via OpenAI API {#ovms_demos_continuous_batching}
This demo shows how to deploy LLM models in the OpenVINO Model Server using continuous batching and paged attention algorithms.
Text generation use case is exposed via OpenAI API `chat/completions` and `completions` endpoints.
That makes it easy to use and efficient especially on on Intel® Xeon® processors.

> **Note:** This demo was tested on Intel® Xeon® processors Gen4 and Gen5.



## Get the docker image

This demo is supported starting from version 2024.2.

```bash
docker pull openvino/model_server:latest
```
It is optional but recommended to build the latest code version from the main branch.
That way you can take advantage of the latest enhancements in this feature.
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make release_image RUN_TESTS=0
```
It will create an image called `openvino/model_server:latest`.
> **Note:** This operation might take 40min or more depending on your build host.

## Model preparation
In this step the original Pytorch LLM model and the tokenizer will be converted to IR format and optionally quantized.
That ensures faster initialization time, better performance and lower memory consumption.
Here, we will also define the LLM engine parameters inside the `graph.pbtxt`.

Install python dependencies for the conversion script:
```bash
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2024/3/demos/continuous_batching/requirements.txt
```

Run optimum-cli to download and quantize the model:
```bash
cd demos/continuous_batching
optimum-cli export openvino --disable-convert-tokenizer --model meta-llama/Meta-Llama-3-8B-Instruct --weight-format fp16 Meta-Llama-3-8B-Instruct
convert_tokenizer -o Meta-Llama-3-8B-Instruct --with-detokenizer --skip-special-tokens --streaming-detokenizer --not-add-special-tokens meta-llama/Meta-Llama-3-8B-Instruct
```

> **Note:** Before downloading the model, access must be requested. Follow the instructions on the [HuggingFace model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B) to request access. When access is granted, create an authentication token in the HuggingFace account -> Settings -> Access Tokens page. Issue the following command and enter the authentication token. Authenticate via `huggingface-cli login`.

Copy the graph to the model folder. 
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
          models_path: "./",
          cache_size: 50
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

cp graph.pbtxt Meta-Llama-3-8B-Instruct/

tree Meta-Llama-3-8B-Instruct/
Meta-Llama-3-8B-Instruct
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


The default configuration of the `LLMExecutor` should work in most cases but the parameters can be tunned inside the `node_options` section in the `graph.pbtxt` file. 
Note that the `models_path` parameter in the graph file can be an absolute path or relative to the `base_path` from `config.json`.
Check the [LLM calculator documentation](../../docs/llm/reference.md) to learn about configuration options.

> **Note:** The parameter `cache_size` in the graph represents KV cache size in GB. Reduce the value if you don't have enough RAM on the host.

## Server configuration
Prepare config.json:
```bash
cat config.json
{
    "model_config_list": [],
    "mediapipe_config_list": [
        {
            "name": "meta-llama/Meta-Llama-3-8B-Instruct",
            "base_path": "Meta-Llama-3-8B-Instruct"
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
```
```json
{
"meta-llama/Meta-Llama-3-8B-Instruct" : 
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
```

## Client code

A single servable exposes both `chat/completions` and `completions` endpoints with and without stream capabilities.
Chat endpoint is expected to be used for scenarios where conversation context should be pasted by the client and the model prompt is created by the server based on the jinja model template.
Completion endpoint should be used to pass the prompt directly by the client and for models without the jinja template.

### Unary:
```bash
curl http://localhost:8000/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "max_tokens":30,
    "stream":false,
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is OpenVINO?"
      }
    ]
  }'| jq .
```
```json
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
  "model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "object": "chat.completion"
}
```

A similar call can be made with a `completion` endpoint:
```bash
curl http://localhost:8000/v3/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "max_tokens":30,
    "stream":false,
    "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is OpenVINO?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
  }'| jq .
```
```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "text": "\n\nOpenVINO is an open-source software library for deep learning inference that is designed to optimize and run deep learning models on a variety"
    }
  ],
  "created": 1716825108,
  "model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "object": "text_completions"
}
```

### Streaming:

The endpoints `chat/completions` are compatible with OpenAI client so it can be easily used to generate code also in streaming mode:

Install the client library:
```bash
pip3 install openai
```
```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

stream = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

Output:
```
It looks like you're testing me!
```

A similar code can be applied for the completion endpoint:
```bash
pip3 install openai
```
```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

stream = client.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt="<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nSay this is a test.<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].text is not None:
        print(chunk.choices[0].text, end="", flush=True)
```

Output:
```
It looks like you're testing me!
```


## Benchmarking text generation with high concurrency

OpenVINO Model Server employs efficient parallelization for text generation. It can be used to generate text also in high concurrency in the environment shared by multiple clients.
It can be demonstrated using benchmarking app from vLLM repository:
```bash
git clone https://github.com/vllm-project/vllm
cd vllm
pip3 install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
cd benchmarks
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json  # sample dataset
python benchmark_serving.py --host localhost --port 8000 --endpoint /v3/chat/completions --backend openai-chat --model meta-llama/Meta-Llama-3-8B-Instruct --dataset ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000 --request-rate inf

Namespace(backend='openai-chat', version='N/A', base_url=None, host='localhost', port=8000, endpoint='/v3/chat/completions', dataset='ShareGPT_V3_unfiltered_cleaned_split.json', model='meta-llama/Meta-Llama-3-8B-Instruct', tokenizer=None, best_of=1, use_beam_search=False, num_prompts=1000, request_rate=inf.0, seed=0, trust_remote_code=False, disable_tqdm=False, save_result=False)
Traffic request rate: inf
100%|██████████████████████████████████████████████████| 1000/1000 [17:17<00:00,  1.04s/it]
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  447.62
Total input tokens:                      215201
Total generated tokens:                  198588
Request throughput (req/s):              2.23
Input token throughput (tok/s):          480.76
Output token throughput (tok/s):         443.65
---------------Time to First Token----------------
Mean TTFT (ms):                          171999.94
Median TTFT (ms):                        170699.21
P99 TTFT (ms):                           360941.40
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          211.31
Median TPOT (ms):                        223.79
P99 TPOT (ms):                           246.48
==================================================
```

## RAG with Model Server

The service deployed above can be used in RAG chain using `langchain` library with OpenAI endpoint as the LLM engine.

Check the example in the [RAG notebook](https://github.com/openvinotoolkit/model_server/blob/releases/2024/3/demos/continuous_batching/rag/rag_demo.ipynb)

