# OpenAI API speech to text endpoints {#ovms_docs_rest_api_s2t}

## API Reference
OpenVINO Model Server includes now the `audio/transcriptions` and `audio/translations` endpoints using OpenAI API.
It is used to execute [speech to text](https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/cpp/automatic_speech_recognition) task with OpenVINO GenAI pipeline.
Please see the [OpenAI API Transcription Reference](https://platform.openai.com/docs/api-reference/audio/createTranscription) and [OpenAI API Translation Reference](https://platform.openai.com/docs/api-reference/audio/createTranslation) for more information on the API.

The are two endpoints exposed:

<b>http://server_name:port/v3/audio/transcriptions</b>
<b>http://server_name:port/v3/audio/translations</b>

Request body must be in `multipart/form-data` format.

### Example request
#### Transcription
```
curl -X POST http://localhost:8000/v3/audio/transcriptions \
  -F "model=OpenVINO/whisper-large-v3-fp16-ov" \
  -F "file=@speech_english.wav"
```

#### Translations
```
curl -X POST http://localhost:8000/v3/audio/translations \
  -F "model=OpenVINO/whisper-large-v3-fp16-ov" \
  -F "file=@speech_spanish.wav"
```

### Example response

```json
{"text":"..."}
```

## Request
### Transcription
| Param | OpenVINO Model Server | OpenAI /audio/transcriptions API | Type | Description |
|-----|----------|----------|---------|-----|
| model | ✅ | ✅ | string (required) | Name of the model to use. **Note**: This can also be omitted to fall back to URI based routing. Read more on routing topic **TODO** |
| file | ⚠️ | ✅ | file (required) | The audio file object to transcribe, in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm. (⚠️**Note**: For now supported formats are mp3 and wav) |
| language | ✅ | ✅ | string | The language of the input audio in ISO-639-1. Providing language for multilanguage model may improve accuracy and performance. |
| chunking_strategy | ❌ | ✅ | "auto" or object | Controls how the audio is cut into chunks. |
| include | ❌ | ✅ | array | Additional information to include in the transcription response. |
| known_speaker_names | ❌ | ✅ | array | List of speaker names corresponding to the audio samples |
| known_speaker_references | ❌ | ✅ | array | Optional list of audio samples with known speaker references matching known_speaker_names |
| prompt | ❌ | ✅ | string | An optional text to guide the model's style or continue a previous audio segment. |
| response_format | ❌ | ✅ | string | The format of the output. |
| stream | ✅ | ✅ | boolean | Generate the response in streaming mode. **Note**: Streaming is not supported for translations endpoint. |
| temperature | ✅ | ✅ | float (default: `1.0`) | The sampling temperature, cannot be negative. |
| timestamp_granularities | ⚠️ | ✅ | array | The timestamp granularities to populate for this transcription. Supported values: "word" and "segment" (⚠️**Note**: To enable word timestamps `enable_word_timestamps: true` need to be set in graph.pbtxt) |


### Translation
| Param | OpenVINO Model Server | OpenAI /audio/transcriptions API | Type | Description |
|-----|----------|----------|---------|-----|
| model | ✅ | ✅ | string (required) | Name of the model to use. **Note**: This can also be omitted to fall back to URI based routing. Read more on routing topic **TODO** |
| file | ⚠️ | ✅ | file (required) | The audio file object to transcribe, in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm. (⚠️**Note**: For now supported formats are mp3 and wav) |
| prompt | ❌ | ✅ | string | An optional text to guide the model's style or continue a previous audio segment. |
| response_format | ❌ | ✅ | string | The format of the output. |
| temperature |  ✅ | ✅ | float (default: `1.0`) | The sampling temperature, cannot be negative. |

## Response
### Transcription
| Param | OpenVINO Model Server | OpenAI /audio/transcriptions API | Type | Description |
|-----|----------|----------|---------|-----|
| text | ✅ | ✅ | string | The transcribed text. |
| logprobs | ❌ | ✅ |  array | The log probabilities of the tokens in the transcription. |
| usage| ❌ | ✅ | object | Token usage statistics for the request. |
### Translation
| Param | OpenVINO Model Server | OpenAI /audio/transcriptions API | Type | Description |
|-----|----------|----------|---------|-----|
| text | ✅ | ✅ | string | The translated text. |

## Error handling
Endpoint can raise an error related to incorrect request in the following conditions:
- Incorrect format of any of the fields based on the schema

## Realtime WebRTC transcription

`POST /v1/realtime/calls` accepts either of the OpenAI Realtime signaling formats:

- `Content-Type: application/sdp`: the request body is the SDP offer. OVMS selects
  the sole available speech-to-text graph; non-STT graphs are ignored. If no STT
  graph or multiple STT graphs are available, the request returns HTTP 400 with a
  descriptive JSON error. Use multipart signaling to select a model explicitly.
- `Content-Type: multipart/form-data`: the `sdp` field contains the offer and the
  `session` field contains a JSON object with `audio.input.transcription.model`
  (or top-level `model` as a fallback) naming the served STT graph.

Both formats return an `application/sdp` answer with embedded ICE candidates and
a `Location: /v1/realtime/calls/<id>` header. To end the call, send
`POST /v1/realtime/calls/<id>/hangup`. Transcription events use the `oai-events`
WebRTC data channel.

When API-key authentication is enabled, both call creation and hangup require
`Authorization: Bearer <configured-api-key>`. OVMS does not issue OpenAI ephemeral
tokens or implement `/v1/realtime/client_secrets`; plain SDP uses the configured
API key, not a token carrying session configuration. No model query parameter or
custom header is required.

```bash
curl http://localhost:8000/v1/realtime/calls \
  -H 'Content-Type: application/sdp' \
  --data-binary @offer.sdp
```

Malformed signaling requests return HTTP 400 with a descriptive JSON error,
rather than a JSON-file parsing error.

## References

[End to end demo with transcription endpoint](../demos/audio/README.md#transcription)
[End to end demo with translation endpoint](../demos/audio/README.md#translation)

[Code snippets](./clients_genai.md)

[Speech Recognition calculator configuration and limitations](speech_recognition/reference.md)
