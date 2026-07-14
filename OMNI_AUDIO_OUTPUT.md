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
