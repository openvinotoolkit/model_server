# QuickStart - LLM models {#ovms_docs_llm_quickstart}

Let's deploy [OpenVINO/Phi-3.5-mini-instruct-int4-ov](https://huggingface.co/OpenVINO/Phi-3.5-mini-instruct-int4-ov) model on Intel iGPU or ARC GPU.
It is [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) quantized to INT4 precision and converted to IR format.
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
docker run --user $(id -u):$(id -g) -d --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --source_model OpenVINO/Phi-3.5-mini-instruct-int4-ov --model_repository_path models --task text_generation --rest_port 8000 --target_device GPU --cache_size 2
```
:::

:::{tab-item} On Baremetal Host
**Required:** OpenVINO Model Server package - see [deployment instructions](../deploying_server_baremetal.md) for details.

```bat
ovms.exe --source_model OpenVINO/Phi-3.5-mini-instruct-int4-ov --model_repository_path models --rest_port 8000 --task text_generation --target_device GPU --cache_size 2
```
:::
::::

First run of the command will download the https://huggingface.co/OpenVINO/Phi-3.5-mini-instruct-int4-ov to models/OpenVINO/Phi-3.5-mini-instruct-int4-ov directory and start serving it with ovms.
The consecutive run of the command will check that the model exists and start serving it.

### 2. Check Model Readiness

Wait for the model to load. You can check the status with a simple command:

```console
curl http://localhost:8000/v1/config
```

:::{dropdown} Expected Response
```json
{
  "OpenVINO/Phi-3.5-mini-instruct-int4-ov": {
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

### 3. Run Generation

::::{tab-set}

:::{tab-item} Linux
```bash
curl -s http://localhost:8000/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenVINO/Phi-3.5-mini-instruct-int4-ov",
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
(Invoke-WebRequest -Uri "http://localhost:8000/v3/chat/completions" `
 -Method POST `
 -Headers @{ "Content-Type" = "application/json" } `
 -Body '{"model": "OpenVINO/Phi-3.5-mini-instruct-int4-ov", "max_tokens": 30, "temperature": 0, "stream": false, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What are the 3 main tourist attractions in Paris?"}]}').Content
```

Windows Command Prompt
```bat
curl -s http://localhost:8000/v3/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"OpenVINO/Phi-3.5-mini-instruct-int4-ov\", \"max_tokens\": 30, \"temperature\": 0, \"stream\": false, \"messages\": [{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"}, {\"role\": \"user\", \"content\": \"What are the 3 main tourist attractions in Paris?\"}]}"
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
  "model": "OpenVINO/Phi-3.5-mini-instruct-int4-ov",
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
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

stream = client.chat.completions.create(
    model="OpenVINO/Phi-3.5-mini-instruct-int4-ov",
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

## Tokenization (release 2025.4.1 or weekly)

The `tokenize` endpoint provides a simple API for tokenizing input text using the same tokenizer as the deployed LLM. This allows you to see how your text will be split into tokens before feature extraction or inference. The endpoint accepts a string or list of strings and returns the corresponding token IDs.

Example usage:
```console
curl http://localhost:8000/v3/tokenize -H "Content-Type: application/json" -d "{ \"model\": \"Qwen/Qwen3-8B\", \"text\": \"hello world\"}"
```
Response:
```json
{
  "tokens":[14990,1879]
}
```

It's possible to use additional parameters:
 - `pad_to_max_length` - whether to pad the sequence to the maximum length. Default is False. 
 - `max_length` - maximum length of the sequence. If specified, it truncates the tokens to the provided number.
 - `padding_side` - side to pad the sequence, can be `left` or `right`. Default is `right`.
 - `add_special_tokens` - whether to add special tokens like BOS, EOS, PAD. Default is True. 

 Example usage:
```console
curl http://localhost:8000/v3/tokenize -H "Content-Type: application/json" -d "{ \"model\": \"Qwen/Qwen3-8B\", \"text\": \"hello world\", \"max_length\": 5, \"pad_to_max_length\": true, \"padding_side\": \"left\", \"add_special_tokens\": true }"
```

Response:
```json
{
  "tokens": [151643,151643,151643,14990,1879]
}
```
> **Note:** Additional parameters are working only with the latest models. That means it's required to export it manually with [exoport model script](../../demos/common/export_models/README.md)

## References
- [Efficient LLM Serving - reference](reference.md)
- [Exporting GEN AI Models](../../demos/common/export_models/README.md)
- [Chat Completions API](../model_server_rest_api_chat.md)
- [Completions API](../model_server_rest_api_completions.md)
- [Demo with Llama3 serving](../../demos/continuous_batching/README.md)
