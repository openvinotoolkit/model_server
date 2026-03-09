# LLM models via OpenAI API {#ovms_demos_continuous_batching}

```{toctree}
---
maxdepth: 1
hidden:
---
ovms_demos_continuous_batching_agent
ovms_demos_continuous_batching_rag
ovms_demos_continuous_batching_scaling
ovms_demos_continuous_batching_speculative_decoding
ovms_structured_output
ovms_demo_long_context
ovms_demos_llm_npu
ovms_demos_continuous_batching_accuracy
```

This demo shows how to deploy LLM models in the OpenVINO Model Server using continuous batching and paged attention algorithms.
Text generation use case is exposed via OpenAI API `chat/completions` and `completions` endpoints.
That makes it easy to use and efficient especially on on Intel® Xeon® processors and ARC GPUs.

> **Note:** This demo was tested on 4th - 6th generation Intel® Xeon® Scalable Processors, and Intel® Core Ultra Series on Ubuntu24 and Windows11.

## Prerequisites

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../docs/deploying_server_baremetal.md)

**(Optional) Client**: Git and Python for using OpenAI client package and vLLM benchmark app


## Server Deployment

**Container on Linux and CPU target device**

Running this command starts the container with CPU only target device:
```bash
docker run -it -p 8000:8000 --rm -e MOE_USE_MICRO_GEMM_PREFILL=0 --user $(id -u):$(id -g) -v $(pwd)/models:/models/:rw openvino/model_server:weekly --model_repository_path /models --source_model OpenVINO/Qwen3-30B-A3B-Instruct-2507-int4-ov --task text_generation --target_device CPU --tool_parser hermes3 --rest_port 8000 --model_name Qwen3-30B-A3B-Instruct-2507-int4-ov
```
> **Note:** In case you want to use GPU target device, add extra docker parameters `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)`
to `docker run` command. The parameter `--target_device` should be also updated to `GPU`. 


**Binary package on Windows 11 with GPU target device**

After ovms is installed according to steps from [baremetal deployment guide](../../docs/deploying_server_baremetal.md), run the following command:

```bat
set MOE_USE_MICRO_GEMM_PREFILL=0
ovms.exe --model_repository_path c:\models --source_model OpenVINO/Qwen3-30B-A3B-Instruct-2507-int4-ov --task text_generation --target_device GPU --tool_parser hermes3 --rest_port 8000 --model_name Qwen3-30B-A3B-Instruct-2507-int4-ov
```


## Readiness Check

Wait for the model to load. You can check the status with a simple command:
```console
curl http://localhost:8000/v3/models
```
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3-30B-A3B-Instruct-2507-int4-ov",
      "object": "model",
      "created": 1772928358,
      "owned_by": "OVMS"
    }
  ]
}
```

## Request Generation

Model exposes both `chat/completions` and `completions` endpoints with and without stream capabilities.
Chat endpoint is expected to be used for scenarios where conversation context should be pasted by the client and the model prompt is created by the server based on the jinja model template.
Completion endpoint should be used to pass the prompt directly by the client and for models without the jinja template. Here is demonstrated model `Qwen/Qwen3-30B-A3B-Instruct-2507` in int4 precision. It has chat capability so `chat/completions` endpoint will be employed:

### Unary calls to chat/completions endpoint using cURL 

::::{tab-set}

:::{tab-item} Linux
```bash
curl http://localhost:8000/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-30B-A3B-Instruct-2507-int4-ov",
    "max_completion_tokens":300,
    "stream":false,
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "If 1=3 2=3 3=5 4=4 5=4 Then, 6=?"
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
 -Body '{"model": "Qwen3-30B-A3B-Instruct-2507-int4-ov", "max_tokens": 30, "temperature": 0, "stream": false, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "If 1=3 2=3 3=5 4=4 5=4 Then, 6=?"}]}').Content
```

Windows Command Prompt
```bat
curl -s http://localhost:8000/v3/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"Qwen3-30B-A3B-Instruct-2507-int4-ov\", \"max_tokens\": 30, \"temperature\": 0, \"stream\": false, \"messages\": [{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"}, {\"role\": \"user\", \"content\": \"If 1=3 2=3 3=5 4=4 5=4 Then, 6=?\"}]}"
```
:::

::::

:::{dropdown} Expected Response
```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "We are given a pattern:\n\n- 1 = 3  \n- 2 = 3  \n- 3 = 5  \n- 4 = 4  \n- 5 = 4  \n- 6 = ?\n\nWe need to find what **6** equals based on this pattern.\n\nLet’s analyze the pattern.\n\nAt first glance, it's not a mathematical operation like addition or multiplication. Let's look at the **number of letters** in the **English word** for each number.\n\nTry that:\n\n- 1 → \"one\" → 3 letters → matches 1 = 3 ✅  \n- 2 → \"two\" → 3 letters → matches 2 = 3 ✅  \n- 3 → \"three\" → 5 letters → matches 3 = 5 ✅  \n- 4 → \"four\" → 4 letters → matches 4 = 4 ✅  \n- 5 → \"five\" → 4 letters → matches 5 = 4 ✅  \n- 6 → \"six\" → 3 letters → So, 6 = 3?\n\nWait — let’s double-check:\n\n- \"six\" has 3 letters → so 6 = 3?\n\nBut let's confirm the pattern again.\n\nYes! The pattern is:  \n**The number on the left equals the number of letters in the English word for that number.**\n\nSo:\n\n| Number | Word     | Letters |\n|--------|----------|---------|\n| 1      | one      | 3       |\n| 2      | two      | 3       |\n| 3      | three    | 5       |\n| 4      | four     | 4       |\n| 5      | five     | 4       |\n| 6      | six      | 3       |\n\nSo, **6 = 3**\n\n### ✅ Final Answer: **3**",
        "role": "assistant",
        "tool_calls": []
      }
    }
  ],
  "created": 1772929186,
  "model": "ovms-model",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 394,
    "total_tokens": 439
  }
}
```
:::


### OpenAI Python package

The endpoints `chat/completions` and `completions` are compatible with OpenAI client so it can be easily used to generate code also in streaming mode:

Install the client library:
```console
pip3 install openai
```

::::{tab-set}

:::{tab-item} Chat completions with streaming
```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

stream = client.chat.completions.create(
    model="Qwen3-30B-A3B-Instruct-2507-int4-ov",
    messages=[{"role": "user", "content": "If 1=3 2=3 3=5 4=4 5=4 Then, 6=?"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

Output:
```
We are given a pattern:

- 1 = 3  
- 2 = 3  
- 3 = 5  
- 4 = 4  
- 5 = 4  
- 6 = ?

We need to find the value for 6.

Let’s look at the pattern. The numbers on the left are integers, and the values on the right seem to represent something about the number itself.

Let’s consider: **the number of letters in the English word for the number.**

Check:

- **1** → "one" → 3 letters → matches 3 ✅  
- **2** → "two" → 3 letters → matches 3 ✅  
- **3** → "three" → 5 letters → matches 5 ✅  
- **4** → "four" → 4 letters → matches 4 ✅  
- **5** → "five" → 4 letters → matches 4 ✅  
- **6** → "six" → 3 letters → so **6 = 3**

### ✅ Answer: **3**
```

:::

:::{tab-item} Chat completions with unary response

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
    model="Qwen3-30B-A3B-Instruct-2507-int4-ov",
    messages=[{"role": "user", "content": "If 1=3 2=3 3=5 4=4 5=4 Then, 6=?"}],
    stream=False,
)
print(response.choices[0].message.content)
```

Output:
```
We are given a pattern:

- 1 = 3  
- 2 = 3  
- 3 = 5  
- 4 = 4  
- 5 = 4  
- 6 = ?

We need to find the value for 6.

Let’s look at the pattern. The numbers on the left are integers, and the values on the right seem to represent something about the **number of letters** when the number is written out in English.

Let’s check:

- **1** = "one" → 3 letters → matches 3 ✅  
- **2** = "two" → 3 letters → matches 3 ✅  
- **3** = "three" → 5 letters → matches 5 ✅  
- **4** = "four" → 4 letters → matches 4 ✅  
- **5** = "five" → 4 letters → matches 4 ✅  
- **6** = "six" → 3 letters → so 6 = **3**

### ✅ Answer: **3**

So, **6 = 3**.
```
:::

::::

## Check how to use AI agents with MCP servers and language models

Check the demo [AI agent with MCP server and OpenVINO acceleration](./agentic_ai/README.md)

## RAG with Model Server

The service deployed above can be used in RAG chain using `langchain` library with OpenAI endpoint as the LLM engine.

Check the example in the [RAG notebook](https://github.com/openvinotoolkit/model_server/blob/main/demos/continuous_batching/rag/rag_demo.ipynb)

## Scaling the Model Server

Check this simple [text generation scaling demo](https://github.com/openvinotoolkit/model_server/blob/main/demos/continuous_batching/scaling/README.md).

## Use Speculative Decoding

Check the [guide for speculative decoding](./speculative_decoding/README.md)

## Check how to use text generation with visual language models

Check the demo [text generation with visual model](./vlm/README.md)

## Use structured output with json schema guided generation

Check the demo [structured output](./structured_output/README.md)

## Testing the model accuracy over serving API

Check the [guide of using lm-evaluation-harness](./accuracy/README.md)

## References
- [Export models to OpenVINO format](../common/export_models/README.md)
- [Supported LLM models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#large-language-models-llms)
- [Official OpenVINO LLM models in HuggingFace](https://huggingface.co/collections/OpenVINO/llm)
- [Chat Completions API](../../docs/model_server_rest_api_chat.md)
- [Completions API](../../docs/model_server_rest_api_completions.md)
- [Writing client code](../../docs/clients_genai.md)
- [LLM calculator reference](../../docs/llm/reference.md)
