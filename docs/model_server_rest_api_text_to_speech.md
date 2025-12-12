# OpenAI API text to speech endpoints {#ovms_docs_rest_api_t2s}

## API Reference
OpenVINO Model Server includes now the `audio/speech` endpoint using OpenAI API.
It is used to execute [text to speech](https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/cpp/speech_generation) task with OpenVINO GenAI pipeline.
Please see the [OpenAI API Create Speech Reference](https://platform.openai.com/docs/api-reference/audio/createSpeech) for more information on the API.
The endpoint is exposed via a path:

<b>http://server_name:port/v3/audio/speech</b>

Request body must be in JSON format, and the request must have `Content-Type: application/json` header.

### Example request

```
curl http://localhost:8000/v3/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/speecht5_tts",
    "input": "The quick brown fox jumped over the lazy dog.",
  }' \
  -o speech.wav
```

### Example response

`speech.wav` - audio file in wav format.


## Request

| Param | OpenVINO Model Server | OpenAI /audio//speech API | Type | Description |
|-----|----------|----------|---------|-----|
| model | ✅ | ✅ | string (required) | Name of the model to use. Name assigned to a MediaPipe graph configured to schedule generation using desired embedding model. **Note**: This can also be omitted to fall back to URI based routing. Read more on routing topic **TODO** |
| input | ✅ | ✅ | string (required) | The text to generate audio for. |
| voice | ❌ | ✅ | string (required) | The voice to use when generating the audio. |
| instructions | ❌ | ✅ | string | Control the voice of your generated audio with additional instructions. |
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