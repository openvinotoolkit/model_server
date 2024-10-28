# How to serve LLM models with Continuous Batching via OpenAI API {#ovms_demos_continuous_batching}
This demo shows how to deploy LLM models in the OpenVINO Model Server using continuous batching and paged attention algorithms.
Text generation use case is exposed via OpenAI API `chat/completions` and `completions` endpoints.
That makes it easy to use and efficient especially on on Intel® Xeon® processors.

> **Note:** This demo was tested on Intel® Xeon® processors Gen4 and Gen5 and Intel dGPU ARC and Flex models on Ubuntu22/24 and RedHat8/9.

## Get the docker image

Pull public image with CPU only support or including also GPU support.
```bash
docker pull openvino/model_server:latest-gpu
docker pull openvino/model_server:latest
```
or build the image from source to try the latest enhancements in this feature.
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make release_image GPU=1
```
It will create an image called `openvino/model_server:latest`.
> **Note:** This operation might take 40min or more depending on your build host.
> **Note:** `GPU` parameter in image build command is needed to include dependencies for GPU device.

## Model preparation
> **Note** Python 3.9 or higher is need for that step
Here, the original Pytorch LLM model and the tokenizer will be converted to IR format and optionally quantized.
That ensures faster initialization time, better performance and lower memory consumption.
LLM engine parameters will be defined inside the `graph.pbtxt` file.

Install python dependencies for the conversion script:
```bash
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
pip3 install optimum-intel@git+https://github.com/huggingface/optimum-intel.git  openvino-tokenizers[transformers]==2024.4.* openvino==2024.4.* nncf>=2.11.0 "transformers<4.45"
```

Run optimum-cli to download and quantize the model:
```bash
cd demos/continuous_batching
convert_tokenizer -o Meta-Llama-3-8B-Instruct --utf8_replace_mode replace --with-detokenizer --skip-special-tokens --streaming-detokenizer meta-llama/Meta-Llama-3-8B-Instruct
optimum-cli export openvino --disable-convert-tokenizer --model meta-llama/Meta-Llama-3-8B-Instruct --weight-format fp16 Meta-Llama-3-8B-Instruct
```
> **Note** Change the `--weight-format` to quantize the model to `int8` or `int4` precision to reduce memory consumption and improve performance.
> **Note:** Before downloading the model, access must be requested. Follow the instructions on the [HuggingFace model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B) to request access. When access is granted, create an authentication token in the HuggingFace account -> Settings -> Access Tokens page. Issue the following command and enter the authentication token. Authenticate via `huggingface-cli login`.
> **Note** You can change the model used in the demo out of any topology [tested](https://github.com/openvinotoolkit/openvino.genai/blob/master/tests/python_tests/models/real_models) with OpenVINO.

Copy the [graph.pbtxt](./graph.pbtxt) or [graph_gpu.pbtxt](./graph_gpu.pbtxt) to the model folder. 
```bash
cp graph.pbtxt Meta-Llama-3-8B-Instruct/
```
or `cp graph_gpu.pbtxt Meta-Llama-3-8B-Instruct/graph.pbtxt`.

You should have a model folder like below:
```bash
tree Meta-Llama-3-8B-Instruct/
Meta-Llama-3-8B-Instruct/
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
└── tokenizer.json
```

The default configuration of the `LLMExecutor` should work in most cases but the parameters can be tuned inside the `node_options` section in the `graph.pbtxt` file. 
Note that the `models_path` parameter in the graph file can be an absolute path or relative to the `base_path` from `config.json`.
Check the [LLM calculator documentation](../../docs/llm/reference.md) to learn about configuration options.

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
docker run -d --rm -p 8000:8000 -v $(pwd)/:/workspace:ro openvino/model_server:latest --port 9000 --rest_port 8000 --config_path /workspace/config.json
```
In case you want to use GPU device to run the generation, add extra docker parameters `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)` 
to `docker run` command, use the image with GPU support and make sure you copy the graph.pbtxt tuned for GPU device. 
Also make sure the export model quantization level and cache size fit to the GPU memory.
```


Wait for the model to load. You can check the status with a simple command:
```bash
curl http://localhost:8000/v1/config
```
```json
{
    "meta-llama/Meta-Llama-3-8B-Instruct": {
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
      "finish_reason": "length",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "OpenVINO is an open-source software framework developed by Intel for optimizing and deploying computer vision, machine learning, and deep learning models on various devices,",
        "role": "assistant"
      }
    }
  ],
  "created": 1724405301,
  "model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 27,
    "completion_tokens": 30,
    "total_tokens": 57
  }
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
      "finish_reason": "length",
      "index": 0,
      "logprobs": null,
      "text": "\n\nOpenVINO is an open-source computer vision platform developed by Intel for deploying and optimizing computer vision, machine learning, and autonomous driving applications. It"
    }
  ],
  "created": 1724405354,
  "model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "object": "text_completion",
  "usage": {
    "prompt_tokens": 23,
    "completion_tokens": 30,
    "total_tokens": 53
  }
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
git clone --branch v0.6.0 --depth 1 https://github.com/vllm-project/vllm
cd vllm
pip3 install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
cd benchmarks
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json  # sample dataset
python benchmark_serving.py --host localhost --port 8000 --endpoint /v3/chat/completions --backend openai-chat --model meta-llama/Meta-Llama-3-8B-Instruct --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000 --request-rate inf

Namespace(backend='openai-chat', base_url=None, host='localhost', port=8000, endpoint='/v3/chat/completions', dataset=None, dataset_name='sharegpt', dataset_path='ShareGPT_V3_unfiltered_cleaned_split.json', model='meta-llama/Meta-Llama-3-8B-Instruct', tokenizer=None, best_of=1, use_beam_search=False, num_prompts=1000, sharegpt_output_len=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, request_rate=inf, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, metadata=None, result_dir=None, result_filename=None, percentile_metrics='ttft,tpot,itl', metric_percentiles='99')
Traffic request rate: inf
100%|██████████████████████████████████████████████████| 1000/1000 [17:17<00:00,  1.04s/it]
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  447.62
Total input tokens:                      215201
Total generated tokens:                  198588
Request throughput (req/s):              2.23
Output token throughput (tok/s):         443.65
Total Token throughput (tok/s):          924.41
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

Check the example in the [RAG notebook](https://github.com/openvinotoolkit/model_server/blob/main/demos/continuous_batching/rag/rag_demo.ipynb)

## Scaling the Model Server

Check this simple [text generation scaling demo](https://github.com/openvinotoolkit/model_server/blob/main/demos/continuous_batching/scaling/README.md).


## Testing the model accuracy over serving API

Check the [guide of using lm-evaluation-harness](https://github.com/openvinotoolkit/model_server/blob/main/demos/continuous_batching/accuracy/README.md)