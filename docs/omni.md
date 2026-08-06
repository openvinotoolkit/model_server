# Omni Models Serving {#ovms_docs_omni}

## Overview

OpenVINO Model Server supports **Omni** models — multimodal models that accept text, images, and audio as input and can produce both text and speech as output. Unlike standalone text-to-speech or speech-to-text pipelines, Omni models combine a Visual Language Model (VLM) with a speech synthesis component (Talker) in a single pipeline.

Text generation with audio input and output is exposed via the existing [chat/completions](./model_server_rest_api_chat.md) and [responses](./model_server_rest_api_responses.md) endpoints — no separate API is needed.

## Supported Models

| Model | Quantization | Status |
|-------|-------------|--------|
| Qwen3-Omni Dense | INT4 / INT8 / FP16 / FP32 | Supported |
| Qwen3-Omni MoE | INT4 / INT8 / FP16 / FP32 | **Not yet supported** |

## Pipeline Detection

Omni pipeline type is auto-detected based on the model directory contents. When the directory contains both VLM model files (`openvino_model.xml`, `openvino_text_embeddings_model.xml`) **and** `openvino_talker_model.xml`, the server selects `OMNI` pipeline type automatically.

The pipeline type can also be set explicitly using `pipeline_type: OMNI` in the LLM calculator `node_options` or `--pipeline_type OMNI` command line parameter.

## Models Directory Structure

The model directory must contain the standard VLM files plus additional Talker-related models:

```
models/qwen3-omni/
├── openvino_model.xml                  # thinker (VLM)
├── openvino_model.bin
├── openvino_talker_model.xml           # talker (speech synthesis)
├── openvino_talker_model.bin
├── openvino_code_predictor_model.xml   # code predictor
├── openvino_code_predictor_model.bin
├── openvino_code2wav_model.xml         # code-to-waveform converter
├── openvino_code2wav_model.bin
├── openvino_text_embeddings_model.xml  # text embeddings (VLM)
├── openvino_text_embeddings_model.bin
├── openvino_vision_embeddings_model.xml
├── openvino_vision_embeddings_model.bin
├── openvino_tokenizer.xml
├── openvino_tokenizer.bin
├── openvino_detokenizer.xml
├── openvino_detokenizer.bin
├── config.json
├── generation_config.json
├── tokenizer_config.json
├── chat_template.jinja
└── ...
```

We recommend using the [export script](../demos/common/export_models/README.md) to prepare this directory structure.

## Configuration

Omni models use the same [LLM calculator](./llm/reference.md#llm-calculator) as text generation and VLM models. The graph configuration follows the standard `graph.pbtxt` format:

```protobuf
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
```

With `pipeline_type` set to `AUTO` (the default), the server auto-detects the Omni pipeline from the model directory contents. See [LLM calculator reference](./llm/reference.md) for the full list of supported `node_options`.

## Audio Input

Audio is sent in messages using `input_audio` content parts, supported by both `chat/completions` and `responses` endpoints.

### Request format

::::{tab-set}
:::{tab-item} Chat Completions
:sync: chat

```json
{
  "model": "qwen3-omni",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_audio",
          "input_audio": {
            "data": "<base64-encoded-audio>",
            "format": "wav"
          }
        },
        {
          "type": "text",
          "text": "What is said in this recording?"
        }
      ]
    }
  ]
}
```
:::

:::{tab-item} Responses
:sync: responses

```json
{
  "model": "qwen3-omni",
  "input": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_audio",
          "input_audio": {
            "data": "<base64-encoded-audio>",
            "format": "wav"
          }
        },
        {
          "type": "input_text",
          "text": "What is said in this recording?"
        }
      ]
    }
  ]
}
```
:::
::::

### Input audio specifications

| Property | Value |
|----------|-------|
| Supported formats | `wav`, `mp3` |
| Channels | mono or stereo (auto-converted to mono) |
| Sample rate | any (handled internally by the model) |
| Encoding | base64 |

## Audio Output

Request speech generation by including `"audio"` in `modalities`.

### Request parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `modalities` | array of strings | Include `"audio"` alongside `"text"` to enable speech output. Default: text only. |
| `audio.voice` | string | Speaker name. See [Available Voices](#available-voices). |
| `audio.format` | string | `"pcm16"` (raw signed 16-bit LE @ 24 kHz) or `"wav"` (RIFF WAV with float32 samples @ 24 kHz). |

### Example request

```json
{
  "model": "qwen3-omni",
  "modalities": ["text", "audio"],
  "audio": {
    "voice": "f04",
    "format": "pcm16"
  },
  "messages": [
    {"role": "user", "content": "Say hello."}
  ]
}
```

### Output audio specifications

| Property | Value |
|----------|-------|
| Sample rate | 24000 Hz |
| Channels | mono |
| `pcm16` format | raw signed 16-bit little-endian samples, no header |
| `wav` format | RIFF/WAV header + 32-bit float samples |

### Endpoint support

| Feature | Chat Completions | Responses API |
|---------|:----------------:|:-------------:|
| Unary audio output | ✅ | ✅ |
| Streaming audio | — | ✅ |

## Response Format

### Unary response with audio (Chat Completions)

When audio output is requested, the response includes an `audio` object alongside the text content:

```json
{
  "choices": [{
    "message": {
      "content": "Hello! How can I help you?",
      "audio": {
        "data": "<base64-encoded-audio>",
        "transcript": "Hello! How can I help you?"
      }
    }
  }]
}
```

### Streaming response with audio (Responses API)

Audio streaming is available only via the Responses API. Text tokens are streamed first, followed by audio chunks:

```
event: response.output_text.delta   ← text tokens
event: response.output_text.done
event: response.audio.delta         ← audio chunks (base64)
event: response.audio.delta
...
event: response.audio.done
```

> **Note:** Audio streaming starts after text generation completes. The model does not interleave text and audio tokens. This is a current OpenVINO GenAI limitation.

### Example: streaming with OpenAI client

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v3", api_key="unused")

with client.responses.stream(
    model="qwen3-omni",
    input=[{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
    max_output_tokens=256,
    extra_body={
        "modalities": ["text", "audio"],
        "audio": {"voice": "f04", "format": "pcm16"},
    },
) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="")
        elif event.type == "response.audio.delta":
            audio_bytes = base64.b64decode(event.delta)
            # Play or save audio_bytes (pcm16 @ 24kHz mono)
```

## Available Voices

Voices are model-dependent. Check `config.json` → `talker_config.speaker_id` for the list of available voices for a given model.

**Qwen3-Omni Dense:**

| Voice | Description |
|-------|-------------|
| `f04` | Female |
| `f245` | Female |
| `f37` | Female |
| `m02` | Male |
| `m31` | Male |
| `m36` | Male |
| `br_f019` | Female (Brazilian Portuguese) |

**Qwen3-Omni MoE:** `Ethan`, `Chelsie`, `Aiden`, `Cherry`

When `voice` is omitted, the model's default speaker is used.

## Limitations

- **Sequential request processing** — Omni pipeline processes requests one at a time
- **Speech generation starts after text generation** — audio and text tokens are not interleaved
- **Audio input position is not preserved** — `input_audio` is always placed at the beginning of the prompt regardless of its position in the message, which may affect multi-turn applications
- **Streaming audio only via Responses API** — Chat Completions API does not support streaming audio output
- **Performance** — the pipeline is under active development and may be slower than expected

## References

- [Chat Completions API](./model_server_rest_api_chat.md)
- [Responses API](./model_server_rest_api_responses.md)
- [LLM calculator reference](./llm/reference.md)
- [Export models to OpenVINO format](../demos/common/export_models/README.md)
- [Omni demo with example clients](../demos/omni/README.md)