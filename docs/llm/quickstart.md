# QuickStart - LLM models {#ovms_docs_llm_quickstart}

Let's deploy [OpenVINO/Qwen3-8B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-8B-int4-ov) model on Intel iGPU or ARC GPU.
It is [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) quantized to INT4 precision and converted to IR format.
You can use another model from [OpenVINO organization on HuggingFace](https://huggingface.co/OpenVINO) if you find one that better suits your needs and hardware configuration.

## Requirements
- Linux or Windows 11
- Docker Engine or `ovms` binary package [installed](../deploying_server_baremetal.md)
- Intel iGPU or ARC GPU

## Deployment Steps

### 1. Deploy the Model
::::{tab-set}

:::{tab-item} With Docker
**Required:** Docker Engine installed

```bash
mkdir models
# in case GPU is available
export GPU_ARGS=$(ls /dev/dri/render* >/dev/null 2>&1 && echo "--device /dev/dri --group-add $(stat -c '%g' /dev/dri/render* | head -n1)")

docker run --user $(id -u):$(id -g) -d $GPU_ARGS --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --source_model OpenVINO/Qwen3-8B-int4-ov --model_repository_path models --rest_port 8000
```
:::

:::{tab-item} On Baremetal Host
**Required:** OpenVINO Model Server package - see [deployment instructions](../deploying_server_baremetal.md) for details.

```bat
mkdir c:\models
ovms.exe --source_model OpenVINO/Qwen3-8B-int4-ov --model_repository_path c:\models --rest_port 8000
```
:::
::::

First run of the command will download the https://huggingface.co/OpenVINO/Qwen3-8B-int4-ov to models/OpenVINO/Qwen3-8B-int4-ov directory and start serving it with ovms.
The consecutive run of the command will check that the model exists and start serving it.

### 2. Check Model Readiness

Wait for the model to load. You can check the status with a simple command:

```console
curl http://localhost:8000/v1/models
```

:::{dropdown} Expected Response
```json
{
  "data": [
    {
      "id": "OpenVINO/Qwen3-8B-int4-ov",
      "object": "model",
      "created": 1784334119,
      "owned_by": "OVMS"
    }
  ],
  "object": "list"
}
```
:::

### 3. Run Generation

::::{tab-set}

:::{tab-item} Linux
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenVINO/Qwen3-8B-int4-ov",
    "max_tokens": 30,
    "temperature": 0,
    "stream": false,
    "messages": [
      { "role": "system", "content": "You are a helpful assistant." },
      { "role": "user", "content": "What are the 3 main tourist attractions in Paris?" }
    ]
  }' | jq .
```
:::

:::{tab-item} Windows

Windows Powershell
```powershell
(Invoke-WebRequest -Uri "http://localhost:8000/v1/chat/completions" `
 -Method POST `
 -Headers @{ "Content-Type" = "application/json" } `
 -Body '{"model": "OpenVINO/Qwen3-8B-int4-ov", "max_tokens": 30, "temperature": 0, "stream": false, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What are the 3 main tourist attractions in Paris?"}]}').Content
```

Windows Command Prompt
```bat
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"OpenVINO/Qwen3-8B-int4-ov\", \"max_tokens\": 30, \"temperature\": 0, \"stream\": false, \"messages\": [{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"}, {\"role\": \"user\", \"content\": \"What are the 3 main tourist attractions in Paris?\"}]}"
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
        "content": "Paris, the charming City of Light, is renowned for its rich history, iconic landmarks, architectural splendor, and artistic",
        "role": "assistant"
      }
    }
  ],
  "created": 1744716414,
  "model": "OpenVINO/Qwen3-8B-int4-ov",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 30,
    "total_tokens": 54
  }
}
```
:::

#### Using OpenAI Python Client:

First, install the openai client library:
```console
pip3 install openai
```
Then run the following Python code:

```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v1",
  api_key="unused"
)

stream = client.chat.completions.create(
    model="OpenVINO/Qwen3-8B-int4-ov",
    messages=[{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": "What are the 3 main tourist attractions in Paris?"}
    ],
    max_tokens=30,
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

Expected output:
```
Paris, the charming City of Light, is renowned for its rich history, iconic landmarks, architectural splendor, and artistic
```



## References
- [Efficient LLM Serving - reference](reference.md)
- [Exporting GEN AI Models](../../demos/common/export_models/README.md)
- [Chat Completions API](../model_server_rest_api_chat.md)
- [Completions API](../model_server_rest_api_completions.md)
- [Demo with Llama3 serving](../../demos/continuous_batching/README.md)
