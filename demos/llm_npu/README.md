# Text generation serving with NPU acceleration {#ovms_demos_llm_npu}


This demo shows how to deploy LLM models in the OpenVINO Model Server with NPU acceleration.
From the client perspective it is very similar to the [generative model deployment with continuous batching](../continuous_batching/README.md)
Likewise it exposes the models via OpenAI API `chat/completions` and `completions` endpoints.
The difference is that it doesn't support request batching. They can be sent concurrently but they are processed sequentially.
It is targeted on client machines equipped with NPU accelerator.

> **Note:** This demo was tested on MeteorLake, LunarLake, ArrowLake platforms on Windows11 and Ubuntu24.

## Prerequisites

**OVMS 2025.1 or higher**

**Model preparation**: Python 3.9 or higher with pip and HuggingFace account

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../docs/deploying_server_baremetal.md)

**(Optional) Client**: git and Python for using OpenAI client package and vLLM benchmark app


## Model preparation
Here, the original Pytorch LLM model and the tokenizer will be converted to IR format and optionally quantized.
That ensures faster initialization time, better performance and lower memory consumption.
LLM engine parameters will be defined inside the `graph.pbtxt` file.

Download export script, install it's dependencies and create directory for the models:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/2/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/2/demos/common/export_models/requirements.txt
mkdir models
```

Run `export_model.py` script to download and quantize the model:

> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.

**LLM**
```console
python export_model.py text_generation --source_model meta-llama/Llama-3.1-8B-Instruct --target_device NPU --config_file_path models/config.json --ov_cache_dir ./models/.ov_cache --model_repository_path models  --overwrite_models
```
**Note:** The parameter `--ov_cache` stores the model compilation cache to speedup initialization time for sequential startup. Drop this parameter if you don't want to store the compilation cache.

Below is a list of tested models:
- meta-llama/Meta-Llama-3-8B-Instruct
- meta-llama/Llama-3.1-8B
- microsoft/Phi-3-mini-4k-instruct
- Qwen/Qwen2-7B
- mistralai/Mistral-7B-Instruct-v0.2
- openbmb/MiniCPM-1B-sft-bf16
- TinyLlama/TinyLlama-1.1B-Chat-v1.0
- TheBloke/Llama-2-7B-Chat-GPTQ
- Qwen/Qwen2-7B-Instruct-GPTQ-Int4

You should have a model folder like below:
```
tree models
models
├── config.json
└── mistralai
    └── Mistral-7B-Instruct-v0.2
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

The default configuration should work in most cases but the parameters can be tuned via `export_model.py` script arguments. 
Note that by default, NPU sets limitation on the prompt length to 1024 tokens. You can modify that limit by using `--max_prompt_len` parameter.
Run the script with `--help` argument to check available parameters and see the [LLM calculator documentation](../../docs/llm/reference.md) to learn more about configuration options.

## Server Deployment

:::{dropdown} **Deploying with Docker**


Running this command starts the container with NPU enabled:
```bash
docker run -d --rm --device /dev/accel --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
-p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --rest_port 8000 --config_path /models/config.json
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

## Readiness Check

Wait for the model to load. You can check the status with a simple command:
```console
curl http://localhost:8000/v1/config
```
```json
{
    "meta-llama/Llama-3.1-8B-Instruct": {
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

## Request Generation

A single servable exposes both `chat/completions` and `completions` endpoints with and without stream capabilities.
Chat endpoint is expected to be used for scenarios where conversation context should be pasted by the client and the model prompt is created by the server based on the jinja model template.
Completion endpoint should be used to pass the prompt directly by the client and for models without the jinja template.

:::{dropdown} **Unary call with cURL**
```console
curl http://localhost:8000/v3/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"meta-llama/Llama-3.1-8B-Instruct\", \"max_tokens\":30,\"stream\":false, \"messages\": [{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},{\"role\": \"user\",\"content\": \"What is OpenVINO?\"}]}"
```
```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "OpenVINO (Open Visual Inference and Optimization for computational resources) is an open-source toolkit that automates neural network model computations across various platforms and",
        "role": "assistant"
      }
    }
  ],
  "created": 1742944805,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 47,
    "completion_tokens": 30,
    "total_tokens": 77
  }
}
```

A similar call can be made with a `completion` endpoint:
```console
curl http://localhost:8000/v3/completions -H "Content-Type: application/json" -d "{\"model\": \"meta-llama/Llama-3.1-8B-Instruct\",\"max_tokens\":30,\"stream\":false,\"prompt\": \"You are a helpful assistant. What is OpenVINO? \"}"
```
```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "text": " Introduction\nOpenVINO can be used in automation of various business processes, which brings timely assistance in operations with these models. Additionally OpenVINO simpl"
    }
  ],
  "created": 1742944929,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "object": "text_completion",
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 30,
    "total_tokens": 44
  }
}
```

:::

:::{dropdown} **Unary call with OpenAI Python package**

The endpoints `chat/completions` are compatible with OpenAI client so it can be easily used to generate code also in streaming mode:

Install the client library:
```console
pip3 install openai
```
```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=False,
)
print(response.choices[0].message.content)
```

Output:
```
This is only a test.
```

A similar code can be applied for the completion endpoint:
```console
pip3 install openai
```
```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="Say this is a test.",
    stream=False,
)
print(response.choices[0].text)
```

Output:
```
This is only a test.
```
:::

:::{dropdown} **Streaming call with OpenAI Python package**

The endpoints `chat/completions` are compatible with OpenAI client so it can be easily used to generate code also in streaming mode:

Install the client library:
```console
pip3 install openai
```
```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

Output:
```
This is only a test.
```

A similar code can be applied for the completion endpoint:
```console
pip3 install openai
```
```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

stream = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="Say this is a test.",
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].text is not None:
        print(chunk.choices[0].text, end="", flush=True)
```

Output:
```
This is only a test.
```
:::

## Benchmarking text generation with high concurrency

OpenVINO Model Server employs efficient parallelization for text generation. It can be used to generate text also in high concurrency in the environment shared by multiple clients.
It can be demonstrated using benchmarking app from vLLM repository:
```console
git clone --branch v0.7.3 --depth 1 https://github.com/vllm-project/vllm
cd vllm
pip3 install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
cd benchmarks
curl -L https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -o ShareGPT_V3_unfiltered_cleaned_split.json # sample dataset
python benchmark_serving.py --host localhost --port 8000 --endpoint /v3/chat/completions --backend openai-chat --model meta-llama/Llama-3.1-8B-Instruct --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 30 --max-concurrency 1
Maximum request concurrency: 1

============ Serving Benchmark Result ============
Successful requests:                     30
Benchmark duration (s):                  480.20
Total input tokens:                      6434
Total generated tokens:                  6113
Request throughput (req/s):              0.06
Output token throughput (tok/s):         12.73
Total Token throughput (tok/s):          26.13
---------------Time to First Token----------------
Mean TTFT (ms):                          1922.09
Median TTFT (ms):                        1920.85
P99 TTFT (ms):                           1952.11
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          65.74
Median TPOT (ms):                        68.95
P99 TPOT (ms):                           70.40
---------------Inter-token Latency----------------
Mean ITL (ms):                           83.65
Median ITL (ms):                         70.11
P99 ITL (ms):                            212.48
==================================================
```

## Testing the model accuracy over serving API

Check the [guide of using lm-evaluation-harness](https://github.com/openvinotoolkit/model_server/blob/main/demos/continuous_batching/accuracy/README.md)

> **Note:** Text generation on NPU is not returning the log_probs which are required to calculate some of the metrics. Only the tasks of type `generate_until` can be used.
For example `--tasks leaderboard_ifeval`.


## Limitations

- beam_search algorithm is not supported with NPU. Greedy search and multinomial algorithms are supported.
- models must be exported with INT4 precision and `--sym --ratio 1.0 --group-size -1` params. This is enforced in the export_model.py script when the target_device in NPU.
- log_probs are not supported
- finish reason is always set to "stop".
- only a single response can be returned. Parameter `n` is not supported.

## References
- [Chat Completions API](../../docs/model_server_rest_api_chat.md)
- [Completions API](../../docs/model_server_rest_api_completions.md)
- [Writing client code](../../docs/clients_genai.md)
- [LLM calculator reference](../../docs/llm/reference.md)
