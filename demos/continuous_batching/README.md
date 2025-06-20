# How to serve LLM models with Continuous Batching via OpenAI API {#ovms_demos_continuous_batching}

```{toctree}
---
maxdepth: 1
hidden:
---
ovms_demos_continuous_batching_accuracy
ovms_demos_continuous_batching_rag
ovms_demos_continuous_batching_scaling
ovms_demos_continuous_batching_speculative_decoding
```

This demo shows how to deploy LLM models in the OpenVINO Model Server using continuous batching and paged attention algorithms.
Text generation use case is exposed via OpenAI API `chat/completions` and `completions` endpoints.
That makes it easy to use and efficient especially on on Intel® Xeon® processors and ARC GPUs.

> **Note:** This demo was tested on 4th - 6th generation Intel® Xeon® Scalable Processors, Intel® Arc™ GPU Series and Intel® Data Center GPU Series on Ubuntu22/24, RedHat8/9 and Windows11.

## Prerequisites

**Model preparation**: Python 3.9 or higher with pip and HuggingFace account

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../docs/deploying_server_baremetal.md)

**(Optional) Client**: git and Python for using OpenAI client package and vLLM benchmark app


## Model preparation
Here, the original Pytorch LLM model and the tokenizer will be converted to IR format and optionally quantized.
That ensures faster initialization time, better performance and lower memory consumption.
LLM engine parameters will be defined inside the `graph.pbtxt` file.

Download export script, install it's dependencies and create directory for the models:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025.2/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025.2/demos/common/export_models/requirements.txt
mkdir models
```

Run `export_model.py` script to download and quantize the model:

> **Note:** Before downloading the model, access must be requested. Follow the instructions on the [HuggingFace model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B) to request access. When access is granted, create an authentication token in the HuggingFace account -> Settings -> Access Tokens page. Issue the following command and enter the authentication token. Authenticate via `huggingface-cli login`. 
> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.

**CPU**
```console
python export_model.py text_generation --source_model meta-llama/Meta-Llama-3-8B-Instruct --weight-format fp16 --kv_cache_precision u8 --config_file_path models/config.json --model_repository_path models  --overwrite_models
```

**GPU**
```console
python export_model.py text_generation --source_model meta-llama/Meta-Llama-3-8B-Instruct --weight-format int4 --target_device GPU --cache_size 2 --config_file_path models/config.json --model_repository_path models --overwrite_models
```

> **Note:** Change the `--weight-format` to quantize the model to `int8` or `int4` precision to reduce memory consumption and improve performance.

> **Note:** You can change the model used in the demo out of any topology [tested](https://github.com/openvinotoolkit/openvino.genai/blob/master/tests/python_tests/models/real_models) with OpenVINO.

You should have a model folder like below:
```
tree models
models
├── config.json
└── meta-llama
    └── Meta-Llama-3-8B-Instruct
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

The default configuration should work in most cases but the parameters can be tuned via `export_model.py` script arguments. Run the script with `--help` argument to check available parameters and see the [LLM calculator documentation](../../docs/llm/reference.md) to learn more about configuration options.

## Server Deployment

:::{dropdown} **Deploying with Docker**

Select deployment option depending on how you prepared models in the previous step.

**CPU**

Running this command starts the container with CPU only target device:
```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/workspace:ro openvino/model_server:latest --rest_port 8000 --config_path /workspace/config.json
```
**GPU**

In case you want to use GPU device to run the generation, add extra docker parameters `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)`
to `docker run` command, use the image with GPU support. Export the models with precision matching the GPU capacity and adjust pipeline configuration.
It can be applied using the commands below:
```bash
docker run -d --rm -p 8000:8000 --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -v $(pwd)/models:/workspace:ro openvino/model_server:latest-gpu --rest_port 8000 --config_path /workspace/config.json
```
:::

:::{dropdown} **Deploying on Bare Metal**

Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.

Depending on how you prepared models in the first step of this demo, they are deployed to either CPU or GPU (it's defined in `graph.pbtxt`). If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

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

## Request Generation

A single servable exposes both `chat/completions` and `completions` endpoints with and without stream capabilities.
Chat endpoint is expected to be used for scenarios where conversation context should be pasted by the client and the model prompt is created by the server based on the jinja model template.
Completion endpoint should be used to pass the prompt directly by the client and for models without the jinja template.

### Unary calls to chat/completions endpoint using cURL 

::::{tab-set}

:::{tab-item} Linux
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
:::

:::{tab-item} Windows
Windows Powershell
```powershell
(Invoke-WebRequest -Uri "http://localhost:8000/v3/chat/completions" `
 -Method POST `
 -Headers @{ "Content-Type" = "application/json" } `
 -Body '{"model": "meta-llama/Meta-Llama-3-8B-Instruct", "max_tokens": 30, "temperature": 0, "stream": false, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What are the 3 main tourist attractions in Paris?"}]}').Content
```

Windows Command Prompt
```bat
curl -s http://localhost:8000/v3/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"meta-llama/Meta-Llama-3-8B-Instruct\", \"max_tokens\": 30, \"temperature\": 0, \"stream\": false, \"messages\": [{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"}, {\"role\": \"user\", \"content\": \"What are the 3 main tourist attractions in Paris?\"}]}"
```
:::

::::

:::{dropdown} Expected Response
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
:::
### Unary calls to completions endpoint using cURL 
A similar call can be made with a `completion` endpoint:
::::{tab-set}

:::{tab-item} Linux
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
:::

:::{tab-item} Windows
Windows Powershell
```powershell
(Invoke-WebRequest -Uri "http://localhost:8000/v3/completions" `
 -Method POST `
 -Headers @{ "Content-Type" = "application/json" } `
 -Body '{"model": "meta-llama/Meta-Llama-3-8B-Instruct", "max_tokens": 30, "temperature": 0, "stream": false, "prompt":"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is OpenVINO?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"}').Content
```

Windows Command Prompt
```bat
curl -s http://localhost:8000/v3/completions -H "Content-Type: application/json" -d "{\"model\": \"meta-llama/Meta-Llama-3-8B-Instruct\", \"max_tokens\": 30, \"temperature\": 0, \"stream\": false, \"prompt\":\"^<^|begin_of_text^|^>^<^|start_header_id^|^>system^<^|end_header_id^|^>\n\nYou are assistant^<^|eot_id^|^>^<^|start_header_id^|^>user^<^|end_header_id^|^>\n\nWhat is OpenVINO?^<^|eot_id^|^>^<^|start_header_id^|^>assistant^<^|end_header_id^|^>\"}"
```
::: 

::::

:::{dropdown} Expected Response
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
:::

### Streaming call with OpenAI Python package

The endpoints `chat/completions` and `completions` are compatible with OpenAI client so it can be easily used to generate code also in streaming mode:

Install the client library:
```console
pip3 install openai
```

::::{tab-set}

:::{tab-item} Chat completions
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

:::

:::{tab-item} Completions

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
:::

::::

## Benchmarking text generation with high concurrency

OpenVINO Model Server employs efficient parallelization for text generation. It can be used to generate text also in high concurrency in the environment shared by multiple clients.
It can be demonstrated using benchmarking app from vLLM repository:
```console
git clone --branch v0.7.3 --depth 1 https://github.com/vllm-project/vllm
cd vllm
pip3 install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
cd benchmarks
curl -L https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -o ShareGPT_V3_unfiltered_cleaned_split.json # sample dataset
python benchmark_serving.py --host localhost --port 8000 --endpoint /v3/chat/completions --backend openai-chat --model meta-llama/Meta-Llama-3-8B-Instruct --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000 --request-rate inf

Namespace(backend='openai-chat', base_url=None, host='localhost', port=8000, endpoint='/v3/chat/completions', dataset=None, dataset_name='sharegpt', dataset_path='ShareGPT_V3_unfiltered_cleaned_split.json', max_concurrency=None, model='meta-llama/Meta-Llama-3-8B-Instruct', tokenizer=None, best_of=1, use_beam_search=False, num_prompts=1000, logprobs=None, request_rate=inf, burstiness=1.0, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_output_len=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None)

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

## Use Speculative Decoding

Check the [guide for speculative decoding](./speculative_decoding/README.md)

## Check how to use text generation with visual language models

Check the demo [text generation with visual model](./vlm/README.md)

## Check how to use AI agents with MCP servers and language models

Check the demo [AI agent with MCP server and OpenVINO acceleration](./agentic_ai/README.md)

## References
- [Chat Completions API](../../docs/model_server_rest_api_chat.md)
- [Completions API](../../docs/model_server_rest_api_completions.md)
- [Writing client code](../../docs/clients_genai.md)
- [LLM calculator reference](../../docs/llm/reference.md)
