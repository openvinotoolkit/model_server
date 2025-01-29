# OpenAI API completions endpoint {#ovms_docs_rest_api_completion}

**Note**: This endpoint works only with [LLM graphs](./llm/reference.md).

## API Reference
OpenVINO Model Server includes now the `completions` endpoint using OpenAI API.
Please see the [OpenAI API Reference](https://platform.openai.com/docs/api-reference/completions) for more information on the API.
The endpoint is exposed via a path:

<b>http://server_name:port/v3/completions</b>

### Example request

```
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
  "object": "text_completion",
  "usage": {
        "completion_tokens": 14,
        "prompt_tokens": 17,
        "total_tokens": 31
  }
}
```


### Request

#### Generic

| Param | OpenVINO Model Server | OpenAI /completions API | vLLM Serving Sampling Params | Type | Description |
|-----|----------|----------|----------|---------|-----|
| model | ✅ | ✅ | ✅ | string (required) | Name of the model to use. From administrator point of view it is the name assigned to a MediaPipe graph configured to schedule generation using desired model.  |
| stop | ✅ | ✅ | ✅ | string/array of strings (optional) | Up to 4 sequences where the API will stop generating further tokens. If `stream` is set to `false` matched stop string **is not** included in the output by default. If `stream` is set to `true` matched stop string **is** included in the output by default. It can be changed with `include_stop_str_in_output` parameter, but for `stream=true` setting `include_stop_str_in_output=false` is invalid. |
| stream | ✅ | ✅ | ✅ | bool (optional, default: `false`) | If set to true, partial message deltas will be sent to the client. The generation chunks will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format) as they become available, with the stream terminated by a `data: [DONE]` message. [Example Python code](clients_genai.md) |
| stream_options | ✅ | ✅ | ✅ | object (optional) | Options for streaming response. Only set this when you set stream: true |
| stream_options.include_usage | ✅ | ✅ | ✅ | bool (optional) | Streaming option. If set, an additional chunk will be streamed before the data: [DONE] message. The usage field in this chunk shows the token usage statistics for the entire request, and the choices field will always be an empty array. All other chunks will also include a usage field, but with a null value. |
| prompt | ⚠️ | ✅ | ✅ | string or array (required) | The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays. **_Limitations: only single string prompt is currently supported._** |
| max_tokens | ✅ | ✅ | ✅ | integer | The maximum number of tokens that can be generated. If not set, the generation will stop once `EOS` token is generated. |
| ignore_eos | ✅ | ❌ | ✅ | bool (default: `false`) | Whether to ignore the `EOS` token and continue generating tokens after the `EOS` token is generated. If set to `true`, the maximum allowed `max_tokens` value is `4000`. |
| include_stop_str_in_output | ✅ | ❌ | ✅ | bool (default: `false` if `stream=false`, `true` if `stream=true`) | Whether to include matched stop string in output. Setting it to false when `stream=true` is invalid configuration and will result in error. |
| logprobs | ⚠️ | ✅ | ✅ | integer (optional) | Include the log probabilities on the logprob of the returned output token. **_ in stream mode logprobs are not returned. Only value 1 is accepted which returns logarithm or the chosen token _** |
| echo | ✅ | ✅ | ✅ | boolean (optional) | Echo back the prompt in addition to the completion |

#### Beam search sampling specific
| Param | OpenVINO Model Server | OpenAI /completions API | vLLM Serving Sampling Params | Type | Description |
|-------|----------|----------|----------|---------|-----|
| n | ✅ | ✅ | ✅ | integer (default: `1`) | Number of output sequences to return for the given prompt. This value must be between `1 <= N <= BEST_OF`. |
| best_of | ✅ | ✅ | ✅ | integer (default: `1`) | Number of output sequences that are generated from the prompt. From these _best_of_ sequences, the top _n_ sequences are returned. _best_of_ must be greater than or equal to _n_. This is treated as the beam width for beam search sampling.  |
| diversity_penalty | ✅ | ❌ | ❌ | float (default: `1.0`) | This value is subtracted from a beam's score if it generates the same token as any beam from other group at a particular time. See [arXiv 1909.05858](https://arxiv.org/pdf/1909.05858). |
| length_penalty | ✅ | ❌ | ✅ | float (default: `1.0`) | Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while `length_penalty` < 0.0 encourages shorter sequences. |

#### Multinomial sampling specific
| Param | OpenVINO Model Server | OpenAI /completions API | vLLM Serving Sampling Params | Type | Description |
|-------|----------|----------|----------|---------|-----|
| temperature | ✅ | ✅ | ✅ | float (default: `1.0`) | The value is used to modulate token probabilities for multinomial sampling. It enables multinomial sampling when set to `> 0.0`. |
| top_p | ✅ | ✅ | ✅ | float (default: `1.0`) | Controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens. |
| top_k | ✅ | ❌ | ✅ | int (default: all tokens) | Controls the number of top tokens to consider. Set to empty or -1 to consider all tokens. |
| repetition_penalty | ✅ | ❌ | ✅ | float (default: `1.0`) | Penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat tokens. `1.0` means no penalty. |
| frequency_penalty | ✅ | ✅ | ✅ | float (default: `0.0`) | Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. |
| presence_penalty | ✅ | ✅ | ✅ | float (default: `0.0`) | Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. |
| seed | ✅ | ✅ | ✅ | integer (default: `0`) | Random seed to use for the generation. |

#### Speculative decoding specific

| Param | OpenVINO Model Server | OpenAI /completions API | vLLM Serving Sampling Params | Type | Description |
|-------|----------|----------|----------|---------|-----|
| num_assistant_tokens | ✅ | ❌ | ⚠️ | int | This value defines how many tokens should a draft model generate before main model validates them. Equivalent of `num_speculative_tokens` in vLLM. Cannot be used with `assistant_confidence_threshold`. |
| assistant_confidence_threshold | ✅ | ❌ | ❌ | float | This parameter determines confidence level for continuing generation. If draft model generates token with confidence below that threshold, it stops generation for the current cycle and main model starts validation. Cannot be used with `num_assistant_tokens`. |

#### Unsupported params from OpenAI service:
- logit_bias
- suffix


#### Unsupported params from vLLM:
- min_p
- use_beam_search (**In OpenVINO Model Server just simply increase _best_of_ param to enable beam search**)
- early_stopping
- stop_token_ids
- min_tokens
- prompt_logprobs
- detokenize
- skip_special_tokens
- spaces_between_special_tokens
- logits_processors
- truncate_prompt_tokens

## Response

| Param | OpenVINO Model Server | OpenAI /completions API | Type | Description |
|-----|----------|----------|---------|-----|
| choices | ✅ | ✅ | array | A list of chat completion choices. Can be more than one if `n` is greater than 1 (beam search or multinomial samplings). |
| choices.index | ✅ | ✅ | integer | The index of the choice in the list of choices. |
| choices.text | ✅ | ✅ | string | A chat completion text generated by the model. |
| choices.finish_reason | ✅ | ✅ | string or null | The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence, `length` if the maximum number of tokens specified in the request was reached, or `null` when generation continues (streaming). |
| choices.logprobs | ⚠️ | ✅ | object or null | Log probability information for the choice. **_In current version, only one logprob per token can be returned _** |
| created | ✅ | ✅ | string | The Unix timestamp (in seconds) of when the chat completion was created.  |
| model | ✅ | ✅ | string | The model used for the chat completion. |
| object | ✅ | ✅ | string | always `text_completion` |
| usage | ✅ | ✅ | object | Usage statistics for the completion request. Consists of three integer fields: `completion_tokens`, `prompt_tokens` and `total_tokens` that inform how many tokens have been generated in a completion, number of tokens in a prompt and the sum of both |

#### Unsupported params from OpenAI service:

- id
- system_fingerprint

> **NOTE**:
OpenAI python client supports a limited list of parameters. Those native to OpenVINO Model Server, can be passed inside a generic container parameter `extra_body`. Below is an example how to encapsulated `top_k` value.
```{code} python
response = client.completions.create(
    model=model,
    prompt="hello",
    max_tokens=100,
    extra_body={"top_k" : 1},
    stream=False
)
```

## References

[LLM quick start guide](./llm/quickstart.md)

[End to end demo with LLM model serving over OpenAI API](../demos/continuous_batching/README.md)

[Code snippets](./clients_genai.md)

[LLM calculator](./llm/reference.md#llm-calculator)

