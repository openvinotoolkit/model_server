# Audio Output Support for Omni Pipeline

## GenAI Audio Output API

The `OmniPipeline::generate()` returns `OmniDecodedResults` which extends `VLMDecodedResults`:

```cpp
class OmniDecodedResults : public VLMDecodedResults {
public:
    TalkerResults speech_result;
};

struct TalkerResults {
    std::vector<ov::Tensor> waveforms;      // float32 PCM @ 24kHz, shape {N_samples}
    TalkerPerfMetrics perf_metrics;
};
```

Speech generation is controlled by `OmniTalkerSpeechConfig`:
- `return_audio = true` — enables speech generation
- `speaker` — voice identity (string name or embedding tensor)
- `audio_chunk_frames` — streaming chunk size (each frame = 80ms @ 24kHz)
- `rng_seed` — sampling seed for talker
- `talker_temperature`, `talker_top_k`, `talker_repetition_penalty` — talker sampling
- `cp_temperature`, `cp_top_k`, `cp_repetition_penalty` — code predictor sampling

Output waveform is **float32 PCM at 24kHz**, stored in `speech_result.waveforms[0]`.

## OpenAI-Compatible API Design

Following the OpenAI Chat Completions audio output spec:

### Request

```json
{
  "model": "omni-model",
  "modalities": ["text", "audio"],
  "audio": {
    "voice": "default",
    "format": "wav"
  },
  "messages": [...]
}
```

- `modalities`: when `["text", "audio"]` → enable `return_audio = true`
- `audio.voice`: maps to `OmniTalkerSpeechConfig::speaker` (string name)
- `audio.format`: output encoding — `"wav"` (WAV header + PCM float32 @ 24kHz) or `"pcm16"` (raw s16le @ 24kHz)

When `modalities` is absent or `["text"]` only, speech generation is disabled (existing behavior).

### Response

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The text transcript...",
      "audio": {
        "data": "<base64-encoded-audio>",
        "transcript": "The text transcript..."
      }
    }
  }]
}
```

The `audio` field is only present when speech was requested and generated.

## Implementation Steps

### Step 1: Parse `modalities` and `audio` from request

**File:** `src/llm/apis/openai_completions.cpp`

In the common request parsing (`parseCommonPart` or top-level parse):
- Parse `modalities` array → set `request.audioOutputRequested = true` if `"audio"` present
- Parse `audio` object → extract `voice` (string) and `format` (string, default "wav")
- Store in existing `OpenAIRequest` struct

### Step 2: Add audio output fields to OpenAIRequest

**File:** `src/llm/apis/openai_api_handler.hpp`

```cpp
struct OpenAIRequest {
    ...
    bool audioOutputRequested = false;
    std::string audioVoice;         // maps to OmniTalkerSpeechConfig::speaker
    std::string audioFormat = "wav"; // "wav" or "pcm16"
};
```

### Step 3: Pass audio config to executor

**File:** `src/llm/omni_model/legacy/servable.cpp` and `servable.hpp`

Store `audioOutputRequested`, `audioVoice`, `audioFormat` on the execution context so the executor can configure `OmniTalkerSpeechConfig`.

### Step 4: Enable speech generation in executor

**File:** `src/llm/omni_model/legacy/legacy_executor.cpp`

```cpp
speechConfig.return_audio = requestExecutionContext->audioOutputRequested;
if (!requestExecutionContext->audioVoice.empty()) {
    speechConfig.speaker = requestExecutionContext->audioVoice;
}
```

### Step 5: Encode speech output in response

**File:** `src/llm/omni_model/legacy/servable.cpp` (in `prepareCompleteResponse`)

When `audioOutputRequested` and results have waveforms:
1. Get `speech_result.waveforms[0]` (float32 PCM tensor @ 24kHz)
2. If format is "wav": use `audio_utils::prepareAudioOutput()` to create WAV buffer
3. If format is "pcm16": convert float32 → int16 samples
4. Base64-encode the buffer
5. Add `"audio": {"data": "<b64>", "transcript": "<text>"}` to the response message

### Step 6: Streaming audio (future)

For streaming responses, speech arrives after text generation completes (the talker
runs on hidden states from the full text decode). Options:
- Deliver audio as a final SSE chunk after `[DONE]` → non-standard
- Deliver in a separate field in the last delta → `{"delta":{"audio":{"data":"..."}}}`
- **Phase 1: unary only** — only include audio in non-streamed responses

## Output Format Details

### WAV format (default)
- RIFF/WAV header
- 24000 Hz sample rate, 32-bit float, mono
- Uses existing `audio_utils::prepareAudioOutput()`

### PCM16 format
- Raw signed 16-bit little-endian samples
- 24000 Hz, mono
- No header

## Omni pipeline output sample rate

The Qwen3-Omni talker generates audio at **24 kHz**. This is fixed by the model architecture.
The sample rate is not exposed in `TalkerResults` — it's a known constant.

---

## GenAI Audio Streaming Deep Dive

### Two-Stage Architecture

```
┌─────────────────────────┐     ┌─────────────────────────┐
│  VLM Thinker            │     │  Talker + Code2Wav      │
│  (text generation)      │ ──► │  (speech generation)    │
│                         │     │                         │
│  StreamerVariant ───────┼──►  │  OmniSpeechStreamer ────┼──► audio chunks
│  (token-by-token text)  │     │  (chunk-by-chunk audio) │
└─────────────────────────┘     └─────────────────────────┘
```

Speech generation is **sequential after text** — the talker consumes hidden states accumulated during the full text decode. There is no interleaved text+audio streaming.

### C++ Speech Streaming Example

```cpp
#include "openvino/genai/omni/pipeline.hpp"
#include "openvino/genai/omni/speech_streamer_base.hpp"

// Option 1: Lambda callback
auto speech_callback = [](const ov::Tensor& audio_chunk) -> ov::genai::StreamingStatus {
    // audio_chunk: float32, shape {N_samples}, 24kHz mono
    // N_samples = audio_chunk_frames * 1920 (each frame = 80ms)
    write_to_audio_device(audio_chunk.data<float>(), audio_chunk.get_size());
    return ov::genai::StreamingStatus::RUNNING;
};

// Option 2: Subclass OmniSpeechStreamerBase
class MyAudioStreamer : public ov::genai::OmniSpeechStreamerBase {
public:
    StreamingStatus write(const ov::Tensor& audio_chunk) override {
        buffer.insert(buffer.end(),
            audio_chunk.data<float>(),
            audio_chunk.data<float>() + audio_chunk.get_size());
        return StreamingStatus::RUNNING;
    }
    void end() override { flush_buffer(); }
private:
    std::vector<float> buffer;
};

auto streamer = std::make_shared<MyAudioStreamer>();

ov::genai::OmniTalkerSpeechConfig speech_config;
speech_config.return_audio = true;
speech_config.audio_chunk_frames = 3;  // ~240ms chunks

auto results = pipe.generate(prompt, images, videos, metadata, audios,
                             text_config, speech_config,
                             text_streamer,    // StreamerVariant for text
                             streamer);        // OmniSpeechStreamerVariant for audio
```

### Python Speech Streaming Example

```python
import openvino_genai
from openvino import Tensor

def text_streamer(token: str) -> openvino_genai.StreamingStatus:
    print(token, end="", flush=True)
    return openvino_genai.StreamingStatus.RUNNING

audio_chunks = []

def speech_streamer(audio_chunk: Tensor) -> openvino_genai.StreamingStatus:
    # audio_chunk.data is float32 numpy array @ 24kHz
    audio_chunks.append(audio_chunk.data.copy())
    return openvino_genai.StreamingStatus.RUNNING

pipe = openvino_genai.OmniPipeline(model_dir, "CPU")

text_config = openvino_genai.GenerationConfig()
text_config.max_new_tokens = 256

speech_config = openvino_genai.OmniTalkerSpeechConfig(model_dir)
speech_config.return_audio = True
speech_config.audio_chunk_frames = 1  # minimum latency

results = pipe.generate(
    history,
    images=images,
    audios=audios,
    text_config=text_config,
    talker_speech_config=speech_config,
    streamer=text_streamer,
    speech_streamer=speech_streamer,
)

# audio_chunks contains list of numpy arrays, each ~80ms of audio
import numpy as np
full_audio = np.concatenate(audio_chunks)  # float32 @ 24kHz
```

### Streaming Control Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `audio_chunk_frames` | 1 | Frames per streaming callback. Each frame = 80ms (1920 samples @ 24kHz). Higher = fewer callbacks, more latency |
| `max_new_tokens` | unlimited | Cap on talker AR steps (independent of text max_new_tokens) |
| `rng_seed` | 0 | Seed for talker + CodePredictor sampling randomness |

### Streaming Return Values

The speech streamer callback returns `StreamingStatus`:
- `RUNNING` — continue generating audio
- `STOP` — stop gracefully, keep generated audio
- `CANCEL` — cancel, generated audio may be discarded

`end()` is always called on `OmniSpeechStreamerBase` subclasses, even on STOP/CANCEL.

### OVMS Streaming Audio Strategy

**Current implementation:** Responses API streaming with `response.audio.delta` events (Option 3 below).

---

## Audio Streaming Options Comparison

### Option 1: Realtime API via WebRTC

**Reference:** https://developers.openai.com/api/docs/guides/realtime-webrtc

- **Transport:** WebRTC peer connection (UDP media + data channels)
- **Latency:** Lowest (~100ms round-trip)
- **Use case:** Browser-based voice agents, live conversations
- **Audio flow:** Bidirectional — client streams mic audio in, server streams model audio out simultaneously
- **Events:** `response.audio.delta`, `input_audio_buffer.speech_started`, etc.
- **Complexity:** High — requires STUN/TURN, ICE negotiation, media track handling
- **Client:** Browser with `getUserMedia()` or native WebRTC library

**Not suitable for OVMS:** Requires full WebRTC stack, NAT traversal infrastructure, and a fundamentally different server architecture (persistent session per client). Overkill for request-based inference serving.

### Option 2: Realtime API via WebSocket

**Reference:** https://developers.openai.com/api/docs/guides/realtime-websocket

- **Transport:** Persistent WebSocket connection
- **Latency:** Low (~200-500ms)
- **Use case:** Server-side voice agents, telephony integration (SIP), real-time translation
- **Audio flow:** Bidirectional JSON events over WebSocket
- **Events:** `response.audio.delta`, `response.audio.done`, `conversation.item.created`, etc.
- **Complexity:** Medium-high — persistent connections, session state management, VAD
- **Client:** Any WebSocket client

**Possible for OVMS in future:** Would require a new WebSocket endpoint, session management, and connection lifecycle handling. Significant new infrastructure.

### Option 3: Responses API Streaming via SSE (Implemented)

**Reference:** https://developers.openai.com/api/reference/resources/responses/streaming-events

- **Transport:** Server-Sent Events (SSE) over HTTP — same as text streaming
- **Latency:** Medium (~500ms for first audio chunk after text completes)
- **Use case:** Extending existing text streaming with audio output
- **Audio flow:** Unidirectional — server streams audio chunks after text generation
- **Events:** `response.audio.delta`, `response.audio.done`
- **Complexity:** Low — extends existing SSE streaming infrastructure
- **Client:** Any HTTP client with SSE support

**Chosen for OVMS:** Natural fit because:
1. GenAI's `speech_streamer` delivers chunks sequentially after text
2. Existing Responses API handler already emits SSE events
3. No new transport or connection infrastructure needed
4. Standard OpenAI SDK support: `client.responses.stream()`

#### SSE Event Sequence

```
event: response.created
event: response.in_progress
event: response.output_item.added
event: response.content_part.added
event: response.output_text.delta      ← text tokens (during VLM generation)
event: response.output_text.delta
...
event: response.output_text.done
event: response.content_part.done
event: response.output_item.done
event: response.audio.delta            ← audio chunks (during talker generation)
event: response.audio.delta
...
event: response.audio.done
event: response.completed
```

#### Audio Delta Event Format

```json
{
  "type": "response.audio.delta",
  "delta": "<base64-encoded-pcm16-chunk>",
  "sequence_number": 42
}
```

#### Client Usage

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11338/v3", api_key="unused")

with client.responses.stream(
    model="ovms-model",
    input="Tell me a short story",
    modalities=["text", "audio"],
    audio={"voice": "f04", "format": "pcm16"},
) as stream:
    for event in stream:
        if event.type == "response.audio.delta":
            # event.delta is base64 pcm16 audio chunk
            play_audio_chunk(base64.b64decode(event.delta))
        elif event.type == "response.audio.done":
            break
```

#### Streaming Format

- Only `pcm16` format is supported for streaming (raw s16le @ 24kHz)
- `wav` format is available only for unary responses (Chat Completions `message.audio.data`)
- Each `response.audio.delta` contains ~80ms of audio (1920 samples × 2 bytes = 3840 bytes before base64)
