# Omni models via OpenAI API {#ovms_demos_omni}

This demo shows how to deploy Omni multimodal models in the OpenVINO Model Server.
Omni models accept any combination of text, audio, and image inputs and can produce both text and audio outputs.
Generation is exposed via OpenAI API `chat/completions` and `responses` endpoints.

> **Note:** This demo was tested with Qwen3-Omni-30B-A3B. At least 22GB RAM is needed to deploy the model with int4 quantization. The model is not yet publicly available on HuggingFace — sections below will be updated once the model is published.

## Prerequisites

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../docs/deploying_server_baremetal.md)

**(Optional) Client**: git and Python for using OpenAI client package

```console
pip3 install openai numpy sounddevice
```

## Deploying the Model

### Option A: Pull from HuggingFace Hub

> **Note:** The pre-optimized OpenVINO model is not yet available on HuggingFace. The commands below are placeholders and will work once the model is published to [OpenVINO HuggingFace organization](https://huggingface.co/OpenVINO).

:::{dropdown} **Deploying with Docker**

```bash
mkdir -p models
# in case GPU is available
export GPU_ARGS=$(if ls /dev/dri/render* >/dev/null 2>&1; then echo "--device /dev/dri --group-add $(stat -c '%g' /dev/dri/render* | head -n1)"; fi)
docker run -d ${GPU_ARGS} -u $(id -u):$(id -g) --rm -p 8000:8000 -v ${HOME}/models:/models:rw openvino/model_server:weekly --rest_port 8000 --source_model OpenVINO/Qwen3-Omni-30B-A3B-int4-ov --model_repository_path /models
```
:::

:::{dropdown} **Deploying on Bare Metal**

If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

```bat
mkdir c:\models
ovms --rest_port 8000 --source_model OpenVINO/Qwen3-Omni-30B-A3B-int4-ov --model_repository_path c:\models
```
:::

### Option B: Export with export_model.py

> **Note:** Export support for Qwen3-Omni is not yet available. The commands below are placeholders and will work once the model and export pipeline are finalized.

Use the [export_model.py](../common/export_models/README.md) script to convert the model from HuggingFace format to OpenVINO IR:
```console
cd demos/common/export_models
pip install -q -r requirements.txt
mkdir -p models

python export_model.py text_generation \
    --source_model Qwen/Qwen3-Omni-30B-A3B \
    --weight-format int4 \
    --kv_cache_precision u8 \
    --config_file_path models/config_all.json \
    --model_repository_path models
```

Then start the server pointing at the exported models directory:

:::{dropdown} **Deploying with Docker**

```bash
export GPU_ARGS=$(if ls /dev/dri/render* >/dev/null 2>&1; then echo "--device /dev/dri --group-add $(stat -c '%g' /dev/dri/render* | head -n1)"; fi)
docker run -d ${GPU_ARGS} -u $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:weekly --rest_port 8000 --config_path /models/config_all.json
```
:::

:::{dropdown} **Deploying on Bare Metal**

```bat
ovms --rest_port 8000 --config_path models\config_all.json
```
:::

## Readiness Check

Wait for the model to load. You can check the status with a simple command:
```console
curl http://localhost:8000/v1/models
```
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-Omni-30B-A3B",
      "object": "model",
      "created": 1772928358,
      "owned_by": "OVMS"
    }
  ]
}
```

## Request Generation

Omni models support multimodal inputs (text, audio, image) and can generate text and audio outputs.

### Text Input

:::{dropdown} **Text-only request with curl**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B",
    "messages": [{"role": "user", "content": "What is OpenVINO?"}],
    "max_completion_tokens": 100
  }'
```
```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "OpenVINO is an open-source toolkit developed by Intel for optimizing and deploying AI inference. It enables high-performance deep learning inference across Intel hardware including CPUs, GPUs, and NPUs.",
        "role": "assistant"
      }
    }
  ],
  "created": 1772928358,
  "model": "Qwen/Qwen3-Omni-30B-A3B",
  "object": "chat.completion"
}
```
:::

:::{dropdown} **Text-only request via Responses API**

```bash
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B",
    "input": [{"role": "user", "content": [{"type": "input_text", "text": "What is OpenVINO?"}]}],
    "max_output_tokens": 100
  }'
```
:::

### Text + Audio Output

:::{dropdown} **Request text and audio response with curl**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B",
    "modalities": ["text", "audio"],
    "audio": {"voice": "m36", "format": "wav"},
    "messages": [{"role": "user", "content": "Say 3 sentences about France."}],
    "max_completion_tokens": 100
  }'
```
The response includes both text in `message.content` and base64-encoded audio in `message.audio.data`.
:::

:::{dropdown} **Using the chat_completions.py client script**

```console
# Text → text
python3 chat_completions.py --prompt "What is OpenVINO?"

# Text → text + audio (saved to file)
python3 chat_completions.py --prompt "Say 3 sentences about France." --audio-output --voice m36 --save output.wav
```
:::

### Audio Input

:::{dropdown} **Audio input using the client scripts**

```console
# Audio → text
python3 chat_completions.py --audio recording.wav --prompt "What is in the recording?"

# Audio → text via Responses API
python3 responses.py --audio recording.wav --prompt "What is in the recording?"

# Audio → text + audio (streaming with real-time playback)
python3 responses.py --audio recording.wav --prompt "What is in the recording?" --audio-output --stream --voice m02 --save output.wav
```
:::

### Image + Audio + Text (Full Multimodal)

:::{dropdown} **All modalities combined**

```console
# Image + text → text + audio
python3 chat_completions.py \
    --image ../common/static/images/gorilla.jpeg \
    --prompt "Describe what is in the image." \
    --audio-output --voice m02 --save output.wav

# Audio + image + text → text + audio
python3 chat_completions.py \
    --audio recording.wav \
    --image ../common/static/images/gorilla.jpeg \
    --prompt "Detect which animal is in the image and tell me what is described in the audio." \
    --audio-output --voice m02 --save output.wav

# Image + text → text + audio (streaming via Responses API)
python3 responses.py \
    --image ../common/static/images/gorilla.jpeg \
    --prompt "Explain what is in the image." \
    --audio-output --stream --voice m02 --save output.wav
```
:::

### Streaming with OpenAI Client

:::{dropdown} **Streaming text via chat/completions**

```python
from openai import OpenAI

client = OpenAI(api_key='unused', base_url='http://localhost:8000/v1')

stream = client.chat.completions.create(
    model="Qwen/Qwen3-Omni-30B-A3B",
    messages=[{"role": "user", "content": "What is OpenVINO?"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```
:::

:::{dropdown} **Streaming text via Responses API**

```python
from openai import OpenAI

client = OpenAI(api_key='unused', base_url='http://localhost:8000/v1')

stream = client.responses.create(
    model="Qwen/Qwen3-Omni-30B-A3B",
    input=[{"role": "user", "content": [{"type": "input_text", "text": "What is OpenVINO?"}]}],
    stream=True,
)
for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
```
:::

## Available Voices

The following voices are available (model-dependent): `f04`, `f245`, `f37`, `m02`, `m31`, `m36`, `br_f019`

## References

- [Chat Completions API](../../docs/model_server_rest_api_chat.md)
- [Responses API](../../docs/model_server_rest_api_responses.md)
- [Responses Streaming API (OpenAI docs)](https://developers.openai.com/api/reference/resources/responses/streaming-events#response.audio.delta)
- [Export models to OpenVINO format](../common/export_models/README.md)
- [Writing client code](../../docs/clients_genai.md)
- [LLM calculator reference](../../docs/llm/reference.md)
