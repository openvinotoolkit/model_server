# OpenAI API {#ovms_docs_rest_api_completion}

**Note**: This endpoint works only with [LLM graphs](./llm/reference.md).

## API Reference
OpenVINO Model Server includes now the `completions` endpoint using OpenAI API.
Please see the [OpenAI API Reference](https://platform.openai.com/docs/api-reference/completions) for more information on the API.
The endpoint is exposed via a path:

<b>http://server_name:port/v3/completions</b>

### Example request

```bash
curl http://localhost/v3/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "prompt": "This is a test",
    "stream": false
  }'
```

### Example response

```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "text": "You are testing me!"
    }
  ],
  "created": 1716825108,
  "model": "llama3",
  "object": "text_completion"
}
```


### Request parameters
We support now the following parameters:
- model - this name is assigning the request to a graph of the same name
- messages - user content type is supported as text. Currently it is not allowed to send the image_url in the user message content
- max_tokens
- stream
- temperature
- top_p


Extra parameters:
- best_of - This is treated as the beam width 
- use_beam_search - Whether to use beam search instead of sampling.
- top_k - Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.
- repetition_penalty 

Not supported parameters:
- tools
- tools_choice
- functions
- function_call
- seed
- n
- logit_bias
- logprobs
- top_logprobs
- frequency_penalty
- presence_penalty
- response_format 

### Response parameters
Supported parameters:
- choices - that includes `finish_reason` ((only null and stop reasons, others unsupported)), `message.content`, `message.role`, index (always 0 for now)
- created - time when the request was received
- model - model name like specified in the config
- object - "chat.completion" or "chat.completion.chunk"

Not supported now:
- id
- system_fingerprint
- usage


## References

[End to end demo with LLM model serving over OpenAI API](../demos/continuous_batching/README.md)

[Code snippets](./clients_openai.md)

[LLM calculator](./llm_calculator.md)

[Developer guide for writing custom calculators with REST API extension](./mediapipe.md)
