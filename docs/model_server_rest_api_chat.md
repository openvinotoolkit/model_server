# OpenAI API {#ovms_docs_rest_api_chat}

**Note**: This endpoint works only with [LLM graphs](./llm/reference.md).

## API Reference
OpenVINO Model Server includes now the `chat/completions` endpoint using OpenAI API.
Please see the [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat) for more information on the API.
The endpoint is exposed via a path:

<b>http://server_name:port/v3/chat/completions</b>

### Example request

```bash
curl http://localhost/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "hello"
      }
    ],
    stream: false
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
      "message": {
        "content": "\n\nHow can I help you?",
        "role": "assistant"
      }
    }
  ],
  "created": 1716825108,
  "model": "llama3",
  "object": "chat.completion"
}
```


### Request

#### Generic

| Param | OpenVINO Model Server | OpenAI /chat/completions API | vLLM Serving Sampling Params | Type | Description |
|-----|----------|----------|----------|---------|-----|
| model | ✅ | ✅ | ✅ | string (required) | Name of the model to use. From administrator point of view it is the name assigned to a MediaPipe graph configured to schedule generation using desired model.  |
| stream | ✅ | ✅ | ✅ | bool (optional, default: `false`) | If set to true, partial message deltas will be sent to the client. The generation chunks will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format) as they become available, with the stream terminated by a `data: [DONE]` message. [Example Python code](clients_openai.md) |
| messages | ✅ | ✅ | ✅ | array (required) | A list of messages comprising the conversation so far. Each object in the list should contain `role` and `content` - both of string type. [Example Python code](clients_openai.md) |
| max_tokens | ✅ | ✅ | ✅ | integer | The maximum number of tokens that can be generated. If not set, the generation will stop once `EOS` token is generated. |
| ignore_eos | ✅ | ❌ | ✅ | bool (default: `false`) | Whether to ignore the `EOS` token and continue generating tokens after the `EOS` token is generated. If set to `true`, the maximum allowed `max_tokens` value is `4000`. |

#### Beam search sampling specific
| Param | OpenVINO Model Server | OpenAI /chat/completions API | vLLM Serving Sampling Params | Type | Description |
|-------|----------|----------|----------|---------|-----|
| n | ✅ | ❌ | ✅ | integer (default: `1`) | Number of output sequences to return for the given prompt. This value must be between `1 <= N <= BEST_OF`. |
| best_of | ✅ | ❌ | ✅ | integer (default: `1`) | Number of output sequences that are generated from the prompt. From these _best_of_ sequences, the top _n_ sequences are returned. _best_of_ must be greater than or equal to _n_. This is treated as the beam width for beam search sampling.  |
| diversity_penalty | ✅ | ❌ | ❌ | float (default: `1.0`) | This value is subtracted from a beam's score if it generates the same token as any beam from other group at a particular time. See [arXiv 1909.05858](https://arxiv.org/pdf/1909.05858). |
| length_penalty | ✅ | ❌ | ✅ | float (default: `1.0`) | Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while `length_penalty` < 0.0 encourages shorter sequences. |

#### Multinomial sampling specific
| Param | OpenVINO Model Server | OpenAI /chat/completions API | vLLM Serving Sampling Params | Type | Description |
|-------|----------|----------|----------|---------|-----|
| temperature | ✅ | ✅ | ✅ | float (default: `0.0`) | The value is used to modulate token probabilities for multinomial sampling. It enables multinomial sampling when set to `> 0.0`. |
| top_p | ✅ | ✅ | ✅ | float (default: `1.0`) | Controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens. |
| top_k | ✅ | ❌ | ✅ | int (default: `0`) | Controls the number of top tokens to consider. Set to 0 to consider all tokens. |
| repetition_penalty | ✅ | ❌ | ✅ | float (default: `1.0`) | Penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat tokens. `1.0` means no penalty. |
| seed | ✅ | ✅ | ✅ | integer (default: `0`) | Random seed to use for the generation. |

#### Unsupported params from OpenAI service:
- frequency_penalty
- logit_bias
- logprobs
- top_logprobs
- presence_penalty
- response_format
- seed
- stop
- stream_options
- tools
- tool_choice
- user
- function_call
- functions

#### Unsupported params from vLLM:
- presence_penalty 
- frequency_penalty
- min_p
- use_beam_search (**In OpenVINO Model Server just simply increase _best_of_ param to enable beam search**)
- early_stopping
- stop
- stop_token_ids
- include_stop_str_in_output
- min_tokens
- logprobs
- prompt_logprobs
- detokenize
- skip_special_tokens
- spaces_between_special_tokens
- logits_processors
- truncate_prompt_tokens

## Response

| Param | OpenVINO Model Server | OpenAI /chat/completions API | Type | Description |
|-----|----------|----------|---------|-----|
| choices | ✅ | ✅ | array | A list of chat completion choices. Can be more than one if `n` is greater than 1 (beam search or multinomial samplings). |
| choices.index | ✅ | ✅ | integer | The index of the choice in the list of choices. |
| choices.message | ✅ | ✅ | object | A chat completion message generated by the model. **When streaming, the field name is `delta` instead of `message`.** |
| choices.message.role | ⚠️ | ✅ | string | The role of the author of this message. **_Currently hardcoded as `assistant`_** |
| choices.message.content | ✅ | ✅ | string | The contents of the message. |
| choices.finish_reason | ⚠️ | ✅ | string or null | The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence, `length` if the maximum number of tokens specified in the request was reached, or `null` when generation continues (streaming). **_However, in current version `length` is not supported_** |
| choices.logprobs | ❌ | ✅ | object or null | Log probability information for the choice. **_In current version, the logprobs is always null._** |
| created | ✅ | ✅ | string | The Unix timestamp (in seconds) of when the chat completion was created.  |
| model | ✅ | ✅ | string | The model used for the chat completion. |
| object | ✅ | ✅ | string | `chat.completion` for unary requests and `chat.completion.chunk` for streaming responses |

#### Unsupported params from OpenAI service:

- id
- system_fingerprint
- usage
- choices.message.tool_calls
- choices.message.function_call
- choices.logprobs.content


## References

[End to end demo with LLM model serving over OpenAI API](../demos/continuous_batching/README.md)

[Code snippets](./clients_openai.md)

[LLM calculator](./llm_calculator.md)

[Developer guide for writing custom calculators with REST API extension](./mediapipe.md)
