# Omni Model Support

OpenVINO Model Server supports **Qwen3-Omni** models — multimodal models that accept text, images, and audio as input and produce text and speech as output.

## Supported Models

- Qwen3-Omni Dense (INT4/INT8/FP16)
- Qwen3-Omni MoE (INT4/INT8/FP16) - **does not work yet**

## Quick Start

### Model Setup

The model directory must contain `openvino_talker_model.xml` — OVMS auto-detects it as an Omni pipeline.

```
models/qwen3-omni/
├── openvino_model.xml          # thinker (VLM)
├── openvino_talker_model.xml   # talker (speech)
├── openvino_code_predictor_model.xml
├── openvino_code2wav_model.xml
├── config.json
├── generation_config.json
├── tokenizer.xml / tokenizer.json
└── ...
```

### Graph Configuration
It should be inside model folder, or simply run OVMS with `--task text_generation` param so it will automatically create one in memory.

```protobuf
node {
  name: "LLMExecutor"
  calculator: "HttpLLMCalculator"
  node_options: {
    [type.googleapis.com/mediapipe.LLMCalculatorOptions]: {
      models_path: "./"
      # pipeline_type: OMNI  # optional — auto-detected from openvino_talker_model.xml
    }
  }
}
```

## API Support

### Audio Input

Send audio in messages using `input_audio` content parts.

| API | Endpoint | Supported |
|-----|----------|-----------|
| Chat Completions | `/v3/chat/completions` | ✅ |
| Responses | `/v3/responses` | ✅ |

**Request format:**
```json
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
```

**Input audio specs:**
- Formats: `wav`, `mp3`
- Channels: mono or stereo (auto-converted to mono)
- Sample rate: any (no resampling — the model handles it internally)
- Encoding: base64

### Audio Output

Request speech generation by including `"audio"` in `modalities`.

| Feature | Chat Completions | Responses API |
|---------|-----------------|---------------|
| Unary audio output | ✅ | ✅ |
| Streaming audio (`response.audio.delta`) | — | ✅ |

**Request:**
```json
{
  "model": "qwen3-omni",
  "modalities": ["text", "audio"],
  "audio": {
    "voice": "f04",
    "format": "pcm16"
  },
  "messages": [...]
}
```

**Parameters:**
- `modalities`: include `"audio"` to enable speech output (default: text only)
- `audio.voice`: speaker name (see [Available Voices](#available-voices))
- `audio.format`: `"pcm16"` (raw signed 16-bit LE @ 24kHz) or `"wav"` (RIFF float32 @ 24kHz)

**Output audio specs:**
- Sample rate: 24000 Hz (fixed)
- Channels: mono
- `pcm16`: raw signed 16-bit little-endian samples, no header
- `wav`: RIFF/WAV header + 32-bit float samples

### Chat Completions Response (unary)

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

### Responses API Streaming

> Current OpenVINO GenAI limitation is that audio streaming generation starts after text generation completes. The model does not interleave text and audio tokens.

> Chat Completions API does not support streaming audio output — use Responses API for that.

Audio is streamed via SSE after text generation completes:

```
event: response.output_text.delta   ← text tokens
event: response.output_text.done
event: response.audio.delta         ← audio chunks (base64 pcm16)
event: response.audio.delta
...
event: response.audio.done
```

**Python client example:**
```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9000/v3", api_key="unused")

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

More clients are available in `demos/omni/` (see [README](demos/omni/README.md)).

## Available Voices

Voices are model-dependent. Check `config.json` → `talker_config.speaker_id`.

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

## Current Limitations

- **Speech generation starts after text generation** — not interleaved
- **Performance** — it is still under ongoing development and may be slower than expected
- **Audio is always placed at the beginning of the prompt** — the user-specified position of input_audio relative to text is not preserved, multi-turn applications are affected

## Demo Clients

Example clients are in `demos/omni/` (see [README](demos/omni/README.md)).