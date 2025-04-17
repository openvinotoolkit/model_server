# QuickStart - LLM models {#ovms_docs_llm_quickstart}

Let's deploy [OpenVINO/Phi-3.5-mini-instruct-int4-ov](https://huggingface.co/OpenVINO/Phi-3.5-mini-instruct-int4-ov) model on Intel iGPU or ARC GPU.
It is [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) quantized to INT4 precision and converted to IR format.

## Requirements
- Linux or Windows 11
- Docker Engine or `ovms` binary package [installed](../deploying_server_baremetal.md)
- Intel iGPU or ARC GPU 

## Deployment Steps

### 1. Install Python dependencies:
```bash
pip3 install huggingface_hub jinja2
```

### 2. Download and Prepare the Model:
Using `export_model.py` script, download the OpenVINO model and prepare models repository including all configuration required for deployment with OpenVINO Model Server. For details, see [Exporting GEN AI Models](../../demos/common/export_models/README.md).

```bash
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/simpler-quick-start-llm/demos/common/export_models/export_model.py -o export_model.py
mkdir models
python export_model.py text_generation --source_model OpenVINO/Phi-3.5-mini-instruct-int4-ov --model_repository_path models --target_device GPU --cache 2
```
LLM engine parameters will be defined inside the `graph.pbtxt` file.

> **Note:** The users in China need to set environment variable `HF_ENDPOINT="https://hf-mirror.com"` before running the export script to connect to the HF Hub.
> **Note:** If you want to export models outside of the `OpenVINO` organization in HuggingFace, you need to install additional Python dependencies:
> ```console
> pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/1/demos/common/export_models/requirements.txt
> ```
 
### 3. Deploy the Model
::::{tab-set}

:::{tab-item} With Docker
**Required:** Docker Engine installed

```bash
docker run -d --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render*) --rm -p 8000:8000 -v $(pwd)/models:/models:ro openvino/model_server:latest-gpu --rest_port 8000 --model_name Phi-3.5-mini-instruct --model_path /models/OpenVINO/Phi-3.5-mini-instruct-int4-ov
```
:::

:::{tab-item} On Baremetal Host
**Required:** OpenVINO Model Server package - see [deployment instructions](../deploying_server_baremetal.md) for details.

```bash
ovms --rest_port 8000 --model_name Phi-3.5-mini-instruct --model_path /models/OpenVINO/Phi-3.5-mini-instruct-int4-ov
```
:::
::::

### 4. Check Model Readiness

Wait for the model to load. You can check the status with a simple command:

```bash
curl http://localhost:8000/v1/config
```

:::{dropdown} Expected Response
```json
{
  "Phi-3.5-mini-instruct": {
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
:::

### 5. Run Generation

::::{tab-set}

:::{tab-item} Linux
```bash
curl -s http://localhost:8000/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Phi-3.5-mini-instruct",
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
```bat
(Invoke-WebRequest -Uri "http://localhost:8000/v3/chat/completions" `
 -Method POST `
 -Headers @{ "Content-Type" = "application/json" } `
 -Body '{"model": "Phi-3.5-mini-instruct", "max_tokens": 30, "temperature": 0, "stream": false, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What are the 3 main tourist attractions in Paris?"}]}').Content
```

Windows Command Prompt
```bat
curl -s http://localhost:8000/v3/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"Phi-3.5-mini-instruct\", \"max_tokens\": 30, \"temperature\": 0, \"stream\": false, \"messages\": [{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"}, {\"role\": \"user\", \"content\": \"What are the 3 main tourist attractions in Paris?\"}]}") 
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
  "model": "Phi-3.5-mini-instruct",
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
```bash
pip3 install openai
```
Then run the following Python code:

```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

stream = client.chat.completions.create(
    model="Phi-3.5-mini-instruct",
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
