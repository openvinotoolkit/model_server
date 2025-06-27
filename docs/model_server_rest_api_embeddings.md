# OpenAI API embeddings endpoint {#ovms_docs_rest_api_embeddings}

## API Reference
OpenVINO Model Server includes now the `embeddings` endpoint using OpenAI API.
Please see the [OpenAI API Reference](https://platform.openai.com/docs/api-reference/embeddings) for more information on the API.
The endpoint is exposed via a path:

<b>http://server_name:port/v3/embeddings</b>

### Example request

```
curl http://localhost/v3/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gte-large",
    "input": ["This is a test"],
    "encoding_format": "float"
  }'
```

### Example response

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [
        -0.03440694510936737,
        -0.02553200162947178,
        -0.010130723007023335,
        -0.013917984440922737,
...
        0.02722850814461708,
        -0.017527244985103607,
        -0.0053995149210095406
      ],
      "index": 0
    }
  ],
  "usage":{"prompt_tokens":6,"total_tokens":6}
}
```


### Request

#### Generic

| Param | OpenVINO Model Server | OpenAI /completions API | Type | Description |
|-----|----------|----------|---------|-----|
| model | ✅ | ✅ | string (required) | Name of the model to use. Name assigned to a MediaPipe graph configured to schedule generation using desired embedding model.  |
| input | ✅ | ✅ | string/list of strings (required) | Input text to embed, encoded as a string or a list of strings  |
| encoding_format | ✅ | ✅ | float or base64 (default: `float`) | The format to return the embeddings in |

#### Unsupported params from OpenAI service:
- user
- dimensions

## Response

| Param | OpenVINO Model Server | OpenAI /completions API | Type | Description |
|-----|----------|----------|---------|-----|
| data | ✅ | ✅ | array | A list of responses for each string |
| data.embedding | ✅ | ✅ | array of float or base64 string | Vector of embeddings for a string. |
| data.index | ✅ | ✅ | integer | Response index |
| model | ✅ | ✅ | string |  Model name |
| usage | ✅ | ✅ | dictionary |  Info about assessed tokens |

## Error handling
Endpoint can raise an error related to incorrect request in the following conditions:
- Incorrect format of any of the fields based on the schema
- Any tokenized input text exceeds the maximum length of the model context. Make sure input documents are chunked to fit the model
- The number of input documents exceeds allowed configured value - default 500


## References

[End to end demo with embeddings endpoint](../demos/embeddings/README.md)

[Code snippets](./clients_genai.md)
