# Omni Model Support

OpenVINO Model Server supports **Qwen3-Omni** models — multimodal models that accept text, images, and audio as input and produce text and speech as output.

## Supported Models

- Qwen3-Omni Dense (INT4/INT8/FP16)
- Qwen3-Omni MoE (INT4/INT8/FP16)

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

```protobuf
node {
  name: "LLMExecutor"
  calculator: "HttpLLMCalculator"
  node_options: {
    [type.googleapis.com/mediapipe.LLMCalculatorOptions]: {
      models_path: "/models/qwen3-omni"
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

## Limitations

- **No streaming for Chat Completions audio output** — use Responses API for streaming audio
- **Speech generation is sequential** — audio starts after text generation completes (not interleaved)
- **Real-time factor** — speech generation speed depends on hardware; typically 1.8-2x slower than real-time on SPR-36 CPU
- **No input transcription** — the model processes audio internally but doesn't return a transcript of user speech
- **No voice cloning** — available voices are fixed per model checkpoint
- **Audio input size limit** — decoded audio cannot exceed 1 GB (configurable via `OVMS_AUDIO_MAX_FILE_SIZE_BYTES` environment variable). Crafted files with spoofed headers are rejected.
- **Audio input formats** — only WAV and MP3 are supported; mono or stereo only (stereo is downmixed to mono)
- **Audio input channels** — files with more than 2 channels are rejected

## Issues
- **Audio has slowly decaying volume**
- **Audio is always positioned at the beginning** — known issue (sgonorov)

## Demo Clients

Example clients are in `demos/omni/`:

```bash
# Chat Completions: audio+text → text
python3 demos/omni/chat_completions_unary_at_t.py audio.wav

# Chat Completions: text → audio+text
python3 demos/omni/chat_completions_unary_t_at.py --prompt "Hello" --voice m02

# Chat Completions: audio+text+image → audio+text
python3 demos/omni/chat_completions_unary_ati_at.py --audio audio.wav --image photo.jpg

# Responses: audio+text → text
python3 demos/omni/responses_unary_at_t.py audio.wav

# Responses: text → audio+text (streaming with playback)
python3 demos/omni/responses_stream_t_at.py --prompt "Tell me a story" --voice f04

# Responses: multi-turn voice chat (mic recording + streaming playback)
python3 demos/omni/responses_multiturn_voice_chat.py --voice f04
```

Requirements: `pip install openai numpy sounddevice`
