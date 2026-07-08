# Omni Model Implementation Plan

## Overview

The `OmniPipeline` from OpenVINO GenAI composes a VLM pipeline (text generation with hidden-state collection) with a Talker speech pipeline (Qwen3-Omni architecture). It accepts **text + images + videos + audio** as input and produces **text + speech waveforms** as output.

We follow the same pattern as `visual_language_model/legacy/` (servable + executor + initializer).

### API References

- **OpenAI Audio Guide:** https://developers.openai.com/api/docs/guides/audio
- **OpenAI Chat Completions — Create:** https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
  - `modalities` parameter: `["text", "audio"]`
  - `audio` parameter: `ChatCompletionAudioParam { format, voice }`
  - `ChatCompletionContentPartInputAudio`: `{"type": "input_audio", "input_audio": {"data": "<b64>", "format": "wav"}}`
  - `ChatCompletionAudio` (response): `{"id", "data", "expires_at", "transcript"}`
- **OpenAI Responses API:** https://developers.openai.com/api/reference/responses/overview
  - Per OpenAI docs: *"The Responses API docs currently describe text and image inputs with text outputs. For this audio-chat pattern, use Chat Completions with an audio-capable model."*

---

## Phase 1: Text-Only Omni (Audio Input, Text Output)

Goal: Get omni model loaded and serving text responses that leverage multimodal understanding (images + audio input). Speech output disabled (`return_audio=false`).

### Step 1.1: Proto & Enum — Add `OMNI` Pipeline Type

**Files:**
- `src/llm/llm_calculator.proto` — Add `OMNI = 5;` to `PipelineType` enum
- `src/llm/servable_initializer.hpp` — Add `OMNI` to C++ `PipelineType` enum

### Step 1.2: Create `src/llm/omni_model/legacy/` Directory

Follows the same 3-file pattern as VLM legacy (`src/llm/visual_language_model/legacy/`).

| File | Purpose |
|------|---------|
| `servable.hpp` | `OmniModelLegacyServable` + `ExecutionContext` + `Properties` |
| `servable.cpp` | Request lifecycle methods |
| `legacy_executor.hpp` | `OmniModelLegacyExecutor` + wrapper thread |
| `legacy_executor.cpp` | Calls `OmniPipeline::generate()` |
| `servable_initializer.hpp` | `OmniModelLegacyServableInitializer` |
| `servable_initializer.cpp` | Constructs `ov::genai::OmniPipeline` |

**Key design** (see `~/openvino.genai/src/cpp/include/openvino/genai/omni/pipeline.hpp` for API):
- `Properties` holds `std::shared_ptr<ov::genai::OmniPipeline>` (not VLMPipeline)
- `ExecutionContext` stores `ov::genai::OmniDecodedResults` (extends `VLMDecodedResults` with `TalkerResults`)
- Executor calls: `pipe->generate(prompt, images, videos, audios, text_config, talker_speech_config, streamer)`
- Phase 1 hardcodes `talker_speech_config.return_audio = false`

### Step 1.3: Model Detection & Routing

**File:** `src/llm/servable_initializer.cpp`

Detection logic in `determinePipelineType()`:
```cpp
bool isOmni = std::filesystem::exists(parsedModelsPathFs / "openvino_talker_model.xml");
```
- `isOmni` takes priority over `isVLM` (omni models also contain VLM components)
- Add routing in `initializeGenAiServable()`: `PipelineType::OMNI` → `OmniModelLegacyServableInitializer`

### Step 1.4: Input Processing Config

**File:** `src/llm/io_processing/input_processing_config.hpp`

Add `bool isOmni = false;` — implies VLM-like image handling plus audio decoding.

### Step 1.5: Build Integration

**File:** `src/llm/BUILD` — Add new source files to Bazel target.

### Step 1.6: Tests

- Detection test: omni model dir → `PipelineType::OMNI`
- Initialization test: valid model loads
- Text generation test: produces text output

---

## Phase 2: Audio Input Support

Goal: Parse audio from API requests and pass to the omni pipeline.

### Audio Input Format (OpenAI-Compatible)

**Chat Completions API** content part:
```json
{
  "type": "input_audio",
  "input_audio": {
    "data": "<base64-encoded-audio>",
    "format": "wav"
  }
}
```

**Responses API** content item:
```json
{
  "type": "input_audio",
  "input_audio": {
    "data": "<base64-encoded-audio>",
    "format": "wav"
  }
}
```

### Step 2.1: Chat Completions — Parse `input_audio` Content Type

**File:** `src/llm/apis/openai_completions.cpp`

In the content array validation loop (around line 210), add handling after `image_url`:
```cpp
} else if (entryType == "input_audio") {
    if (!entry.HasMember("input_audio") || !entry["input_audio"].IsObject()) {
        return absl::InvalidArgumentError("Invalid message structure - input_audio missing");
    }
    const auto inputAudio = entry["input_audio"].GetObject();
    if (!inputAudio.HasMember("data") || !inputAudio["data"].IsString()) {
        return absl::InvalidArgumentError("Invalid message structure - input_audio data missing");
    }
    // format is optional, defaults to "wav"
}
```

### Step 2.2: Responses API — Parse `input_audio` Content Item

**File:** `src/llm/apis/openai_responses.cpp`

In the content item type dispatch (around line 430), add:
```cpp
} else if (type == "input_audio") {
    auto status = buildAudioEntry(contentObj);
    if (!status.ok()) return status;
}
```

### Step 2.3: Audio Decoding Processor

**New files:** `src/llm/io_processing/input_processors/audio_decoding_processor.hpp/cpp`

Analogous to `image_decoding_processor`:
- Extracts base64-encoded audio data from the chat history content arrays
- Decodes base64 → raw bytes
- Converts WAV/MP3 bytes → `ov::Tensor` (PCM float32, expected by GenAI)
- Populates `InputRequest::inputAudios`

Supported formats (Phase 2): `wav` only (simplest — raw PCM extraction from WAV header).
Later: `mp3`, `flac`, `ogg` via a lightweight decoder.

### Step 2.4: Register Audio Processor in Pipeline

**File:** `src/llm/io_processing/input_processor.cpp`

Add `AudioDecodingProcessor` to the processor chain when `config.isOmni == true`.

### Step 2.5: Both APIs Supported

Both Chat Completions (`/v3/chat/completions`) and Responses (`/v3/responses`) will support `input_audio`. This is consistent with how image input is handled today (both APIs support `image_url`/`input_image`).

---

## Phase 3: Audio Output Support

Goal: Return generated speech in the API response.

### API Design Decision

**Use Chat Completions API** for audio output — this matches the OpenAI API design where audio output is a Chat Completions feature (their Responses API docs explicitly state audio output uses Chat Completions).

### Audio Output Request Format

Request fields (top-level in the request body):
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

- `modalities`: Array containing `"text"` and/or `"audio"`. When `"audio"` is present, speech generation is enabled.
- `audio.voice`: Speaker name (string) — maps to `OmniTalkerSpeechConfig::speaker`
- `audio.format`: Output format — `"wav"` (24kHz), `"pcm16"` (raw PCM s16le)

### Audio Output Response Format

**Non-streaming response:**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The text transcript...",
      "audio": {
        "id": "audio_abc123",
        "data": "<base64-encoded-wav>",
        "transcript": "The text transcript..."
      }
    }
  }]
}
```

**Streaming response:**
Audio chunks are delivered as SSE events. Two options:

Option A — Inline in delta (matches OpenAI):
```json
{"choices":[{"delta":{"audio":{"data":"<base64-chunk>"}}}]}
```
Final chunk:
```json
{"choices":[{"delta":{"audio":{"id":"audio_abc123","transcript":"..."}}}]}
```

Option B — Simpler: deliver full audio only in the final non-streamed response (text streams normally, audio arrives at end). **Recommended for Phase 3** due to simplicity.

### Step 3.1: Parse `modalities` and `audio` Request Fields

**File:** `src/llm/apis/openai_completions.cpp`

Parse:
- `modalities` array → check for `"audio"` presence
- `audio` object → extract `voice` and `format`

Store in a new `AudioOutputConfig` struct on the request.

### Step 3.2: Map to OmniTalkerSpeechConfig

**File:** `src/llm/omni_model/legacy/servable.cpp` (in `parseRequest` or `scheduleExecution`)

```cpp
ov::genai::OmniTalkerSpeechConfig speechConfig(properties->modelsPath);
speechConfig.return_audio = request.audioOutputRequested;
if (!request.audioVoice.empty()) {
    speechConfig.speaker = request.audioVoice;
}
```

### Step 3.3: Encode Speech Output in Response

**File:** `src/llm/omni_model/legacy/servable.cpp` (in `prepareCompleteResponse`)

- Take `OmniDecodedResults::speech_result.speech_outputs[0]` (ov::Tensor, float32 PCM @ 24kHz)
- Convert to WAV format (add WAV header) or raw PCM16
- Base64-encode
- Add `"audio"` field to response JSON

### Step 3.4: Streaming Audio (Phase 3b — Future)

For streaming, use GenAI's `OmniSpeechStreamerBase` callback:
- Each audio chunk arrives via the speech streamer
- Encode chunk as base64
- Push as SSE delta event with `"audio": {"data": "..."}`

This requires a dual-streamer setup in the executor:
- Text streamer → text deltas (existing `OVMSTextStreamer`)
- Speech streamer → audio deltas (new `OVMSSpeechStreamer`)

### Step 3.5: Responses API Audio Output (Phase 3c — Future)

If/when needed, add audio output to Responses API following OpenAI's eventual spec. For now, audio output is Chat Completions only.

---

## Phase 4: Advanced Features (Future)

| Feature | Description |
|---------|-------------|
| Voice blending | Expose speaker embedding blending via request params |
| Multi-turn audio | Reference previous audio with `audio.id` in follow-up messages |
| Video input | Parse video content parts, pass as `inputVideos` |
| Continuous batching | `omni_model/continuous_batching/` — when GenAI supports CB for omni |
| TalkerSpeechConfig tuning | Expose `talker_temperature`, `talker_top_k`, etc. via request params |

---

## Summary of API Support

| Feature | Chat Completions | Responses API |
|---------|-----------------|---------------|
| Audio input (`input_audio`) | Phase 2 ✓ | Phase 2 ✓ |
| Image input | Existing ✓ | Existing ✓ |
| Text output | Phase 1 ✓ | Phase 1 ✓ |
| Audio output (`modalities: ["audio"]`) | Phase 3 ✓ | Phase 3c (future) |
| Streaming text | Phase 1 ✓ | Phase 1 ✓ |
| Streaming audio | Phase 3b (future) | — |

---

## File Structure After Implementation

```
src/llm/
├── omni_model/
│   └── legacy/
│       ├── servable.hpp
│       ├── servable.cpp
│       ├── legacy_executor.hpp
│       ├── legacy_executor.cpp
│       ├── servable_initializer.hpp
│       └── servable_initializer.cpp
├── io_processing/
│   └── input_processors/
│       ├── audio_decoding_processor.hpp    (new)
│       └── audio_decoding_processor.cpp    (new)
└── ...
```

---

## Execution Order

1. **Phase 1** — Omni model loading + text generation (no audio I/O)
2. **Phase 2** — Audio input parsing (both APIs)
3. **Phase 3** — Audio output via Chat Completions (unary first, then streaming)
4. **Phase 4** — Advanced features
