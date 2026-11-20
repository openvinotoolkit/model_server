# OpenAI API text to speech endpoints {#ovms_docs_rest_api_t2s}

## API Reference
OpenVINO Model Server includes now the `audio/speech` endpoint using OpenAI API.
It is used to execute [text to speech](https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/cpp/speech_generation) task with OpenVINO GenAI pipeline.
Please see the [OpenAI API Create Speech Reference](https://platform.openai.com/docs/api-reference/audio/createSpeech) for more information on the API.
The endpoint is exposed via a path:

<b>http://server_name:port/v3/audio/speech</b>

Request body must be in JSON format, and the request must have `Content-Type: application/json` header.

### Qwen3 TTS compatibility

OVMS supports the Qwen3 TTS variants that fit the OpenAI-compatible request schema used by `/v1/audio/speech`:

- `Qwen3-TTS-*-CustomVoice`: supported via `voice` and optional `instructions`
- `Qwen3-TTS-*-VoiceDesign`: supported via `instructions`

`Qwen3-TTS-*-Base` is not currently supported through this endpoint, because it requires reference-audio inputs such as `ref_audio` and optionally `ref_text`, which are outside the standard OpenAI `audio/speech` request shape.

### Example request

```
curl http://localhost:8000/v3/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Kokoro-82M-int8-ov",
    "voice": "af_alloy",
    "input": "The quick brown fox jumped over the lazy dog."
  }' \
  -o speech.wav
```

### Example response

`speech.wav` - audio file in wav format.

### Language handling

For Kokoro-based speech generation, OVMS forwards the optional `language` field to the OpenVINO GenAI speech generation pipeline. The currently supported GenAI phonemization values are:

- `en-us`
- `en-gb`
- `es`
- `fr-fr`
- `it`
- `pt-br`
- `hi`

If `language` is omitted, OVMS defaults to `en-us`. For non-English text, set `language` explicitly. Voice selection does not infer the phonemization language.


## Request

| Param | OpenVINO Model Server | OpenAI /audio//speech API | Type | Description |
|-----|----------|----------|---------|-----|
| model | ✅ | ✅ | string (required) | Name of the model to use. Name assigned to a MediaPipe graph configured to schedule generation using desired embedding model. **Note**: This can also be omitted to fall back to URI based routing. Read more on routing topic **TODO** |
| input | ✅ | ✅ | string (required) | The text to generate audio for. |
| voice | ✅ | ✅ | string | The voice to use when generating the audio. OVMS first checks graph-configured or model-local voice embeddings, then forwards the value as a model-native speaker name when the backend supports it. If not provided, model default voice will be used. |
| language | ✅ | ❌ | string | Optional phonemization language passed to the OpenVINO GenAI speech generation pipeline. Supported values for Kokoro are `en-us`, `en-gb`, `es`, `fr-fr`, `it`, `pt-br`, `hi`. Defaults to `en-us`. |
| instructions | ✅ | ✅ | string | Additional generation instructions forwarded to the OpenVINO GenAI speech generation pipeline for models that support prompt-based voice control. |
| response_format | ❌ | ✅ | string | The format to audio in. |
| speed | ❌ | ✅ | number | The speed of the generated audio. |
| stream_format | ❌ | ✅ | string | The format to stream the audio in. |

## Error handling
Endpoint can raise an error related to incorrect request in the following conditions:
- Incorrect format of any of the fields based on the schema

## References

[End to end demo with speech generation endpoint](../demos/audio/README.md#speech-generation)

[Code snippets](./clients_genai.md)

[Speech Generation calculator configuration and limitations](speech_generation/reference.md)
