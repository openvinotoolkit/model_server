# OpenAI API responses endpoint {#ovms_docs_rest_api_responses}

**Note**: This endpoint works only with [LLM graphs](./llm/reference.md).

## API Reference
OpenVINO Model Server includes now the `responses` endpoint using OpenAI API.
Please see the [OpenAI API Reference](https://developers.openai.com/api/reference/resources/responses/methods/create) for more information on the API.
The endpoint is exposed via a path:

<b>http://server_name:port/v3/responses</b>

### Example request

```
curl http://localhost/v3/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "input": "What is OpenVINO?"
  }'
```

### Example response

```json
{
  "id": "resp-1716825108",
  "object": "response",
  "created_at": 1716825108,
  "completed_at": 1716825110,
  "error": null,
  "model": "llama3",
  "status": "completed",
  "parallel_tool_calls": true,
  "store": true,
  "text": { "format": { "type": "text" } },
  "tool_choice": "auto",
  "tools": [],
  "truncation": "disabled",
  "metadata": {},
  "output": [
    {
      "id": "msg-0",
      "type": "message",
      "role": "assistant",
      "status": "completed",
      "content": [
        {
          "type": "output_text",
          "text": "OpenVINO is an open-source toolkit ...",
          "annotations": []
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 5,
    "output_tokens": 42,
    "total_tokens": 47
  }
}
```

In case of VLM models, the request can include images:
```
curl http://localhost/v3/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llava",
    "input": [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "What is on the picture?"
                },
                {
                    "type": "input_image",
                    "image_url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBD ..."
                }
            ]
        }
    ],
    "max_output_tokens": 128
}'
```

### Request

#### Generic

| Param | OpenVINO Model Server | OpenAI /responses API | Type | Description |
|-----|----------|----------|---------|-----|
| model | ✅ | ✅ | string (required) | Name of the model to use. From administrator point of view it is the name assigned to a MediaPipe graph configured to schedule generation using desired model. |
| input | ✅ | ✅ | string or array (required) | The input to generate a response for. Accepts a plain string or an array of message items with `input_text` / `input_image` types. |
| stream | ✅ | ✅ | bool (optional, default: `false`) | If set to true, partial message deltas will be sent to the client as [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format) as they become available, with the stream terminated by a `data: [DONE]` message. See [Streaming events](#streaming-events) section for details. |
| max_output_tokens | ✅ | ✅ | integer (optional) | An upper bound for the number of tokens that can be generated. If not set, the generation will stop once `EOS` token is generated. If `max_tokens_limit` is set in `graph.pbtxt` it will be the default value. |
| stop | ✅ | ❌ | string/array of strings (optional) | Up to 4 sequences where the API will stop generating further tokens. If `stream` is set to `false` matched stop string **is not** included in the output by default. If `stream` is set to `true` matched stop string **is** included in the output by default. It can be changed with `include_stop_str_in_output` parameter, but for `stream=true` setting `include_stop_str_in_output=false` is invalid. |
| ignore_eos | ✅ | ❌ | bool (default: `false`) | Whether to ignore the `EOS` token and continue generating tokens after the `EOS` token is generated. |
| include_stop_str_in_output | ✅ | ❌ | bool (default: `false` if `stream=false`, `true` if `stream=true`) | Whether to include matched stop string in output. Setting it to false when `stream=true` is invalid configuration and will result in error. |
| logprobs | ⚠️ | ❌ | bool (default: `false`) | Include the log probabilities on the logprob of the returned output token. **_In stream mode logprobs are not supported._** |
| response_format | ✅ | ❌ | object (optional) | An object specifying the format that the model must output. Setting to `{ "type": "json_schema", "json_schema": {...} }` enables Structured Outputs. Additionally accepts [XGrammar structural tags format](https://github.com/mlc-ai/xgrammar/blob/main/docs/tutorials/structural_tag.md#format-types). OpenAI Responses API uses `text.format` instead (not supported in OVMS). |
| tools | ⚠️ | ✅ | array (optional) | A list of tools the model may call. Currently, only **function** tools are supported. OpenAI also supports built-in tools (web_search, file_search, code_interpreter, etc.) and MCP tools. OVMS additionally accepts a flat `{type, name, parameters}` format alongside the nested `{type, function: {name, parameters}}` format. See [OpenAI API reference](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools) for more details. |
| tool_choice | ✅ | ✅ | string or object (optional) | Controls which (if any) tool is called by the model. `none` means the model will not call any tool and instead generates a message. `auto` means the model can pick between generating a message or calling one or more tools. `required` means that model should call at least one tool. Specifying a particular function via `{"type": "function", "function": {"name": "my_function"}}` forces the model to call that tool. |
| reasoning | ⚠️ | ✅ | object (optional) | Configuration for reasoning/thinking mode. The `effort` field accepts `"low"`, `"medium"`, or `"high"` — any value enables thinking mode (`enable_thinking: true` is injected into chat template kwargs). The `summary` field is accepted but ignored. |
| chat_template_kwargs | ✅ | ❌ | object (optional) | Additional keyword arguments passed to the chat template. When `reasoning` is also provided, `enable_thinking: true` is merged into these kwargs. |
| stream_options | ❌ | ❌ | | Not supported in Responses API. Usage statistics are always included in the `response.completed` event. |

#### Beam search sampling specific
| Param | OpenVINO Model Server | OpenAI /responses API | Type | Description |
|-------|----------|----------|---------|-----|
| n | ✅ | ❌ | integer (default: `1`) | Number of output sequences to return for the given prompt. This value must be between `1 <= N <= BEST_OF`. For Responses API streaming, only `n=1` is supported. |
| best_of | ✅ | ❌ | integer (default: `1`) | Number of output sequences that are generated from the prompt. From these _best_of_ sequences, the top _n_ sequences are returned. _best_of_ must be greater than or equal to _n_. This is treated as the beam width for beam search sampling. |
| length_penalty | ✅ | ❌ | float (default: `1.0`) | Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while `length_penalty` < 0.0 encourages shorter sequences. |

#### Multinomial sampling specific
| Param | OpenVINO Model Server | OpenAI /responses API | Type | Description |
|-------|----------|----------|---------|-----|
| temperature | ✅ | ✅ | float (default: `1.0`) | The value is used to modulate token probabilities for multinomial sampling. It enables multinomial sampling when set to `> 0.0`. |
| top_p | ✅ | ✅ | float (default: `1.0`) | Controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens. |
| top_k | ✅ | ❌ | int (default: all tokens) | Controls the number of top tokens to consider. Set to empty or -1 to consider all tokens. |
| repetition_penalty | ✅ | ❌ | float (default: `1.0`) | Penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat tokens. `1.0` means no penalty. |
| frequency_penalty | ✅ | ❌ | float (default: `0.0`) | Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. |
| presence_penalty | ✅ | ❌ | float (default: `0.0`) | Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. |
| seed | ✅ | ❌ | integer (default: `0`) | Random seed to use for the generation. |

#### Speculative decoding specific

Note that below parameters are valid only for speculative pipeline. See [speculative decoding demo](../demos/continuous_batching/speculative_decoding/README.md) for details on how to prepare and serve such pipeline.

| Param | OpenVINO Model Server | OpenAI /responses API | Type | Description |
|-------|----------|----------|---------|-----|
| num_assistant_tokens | ✅ | ❌ | int | This value defines how many tokens should a draft model generate before main model validates them. Cannot be used with `assistant_confidence_threshold`. |
| assistant_confidence_threshold | ✅ | ❌ | float | This parameter determines confidence level for continuing generation. If draft model generates token with confidence below that threshold, it stops generation for the current cycle and main model starts validation. Cannot be used with `num_assistant_tokens`. |

#### Prompt lookup decoding specific

Note that below parameters are valid only for prompt lookup pipeline. Add `"prompt_lookup": true` to `plugin_config` in your graph config node options to serve it.

| Param | OpenVINO Model Server | OpenAI /responses API | Type | Description |
|-------|----------|----------|---------|-----|
| num_assistant_tokens | ✅ | ❌ | int | Number of candidate tokens proposed after ngram match is found |
| max_ngram_size | ✅ | ❌ | int | The maximum ngram to use when looking for matches in the prompt |

#### Unsupported params from OpenAI Responses API:
- instructions
- previous_response_id
- conversation
- context_management
- text
- truncation
- top_logprobs
- include
- store
- metadata
- parallel_tool_calls
- max_tool_calls
- background
- prompt
- prompt_cache_key
- prompt_cache_retention
- service_tier
- safety_identifier
- user

## Response

| Param | OpenVINO Model Server | OpenAI /responses API | Type | Description |
|-----|----------|----------|---------|-----|
| id | ✅ | ✅ | string | A unique identifier for the response. OVMS uses timestamp-based IDs (e.g. `resp-1716825108`). |
| object | ✅ | ✅ | string | Always `response`. |
| created_at | ✅ | ✅ | integer | The Unix timestamp (in seconds) of when the response was created. |
| completed_at | ✅ | ✅ | integer | The Unix timestamp (in seconds) of when the response was completed. Only present when `status` is `completed`. |
| incomplete_details | ✅ | ✅ | object or null | Details about why the response is incomplete. Contains `{"reason": "max_tokens"}` when generation was truncated due to token limit. `null` otherwise. |
| error | ✅ | ✅ | object or null | Error information. `null` when no error occurred. |
| model | ✅ | ✅ | string | The model used for the response. |
| status | ✅ | ✅ | string | `completed` or `incomplete` for unary requests; transitions from `in_progress` to `completed`/`incomplete` during streaming. |
| output | ✅ | ✅ | array | A list of output items. May include items of type `message`, `function_call`, or `reasoning`. See [Output item types](#output-item-types) below. |
| output[].content[].text | ✅ | ✅ | string | The generated text content (for `message` type items). |
| output[].content[].annotations | ✅ | ✅ | array | Always an empty array (annotations not yet supported). |
| usage | ✅ | ✅ | object | Usage statistics: `input_tokens`, `output_tokens`, `total_tokens`. |
| tool_choice | ✅ | ✅ | string or object | Echoed back from the request. |
| tools | ✅ | ✅ | array | Echoed back from the request. |
| max_output_tokens | ✅ | ✅ | integer | Echoed back from the request (if set). |
| parallel_tool_calls | ⚠️ | ✅ | bool | Hardcoded to `true` in OVMS. |
| store | ⚠️ | ✅ | bool | Hardcoded to `true` in OVMS. |
| temperature | ✅ | ✅ | float | Echoed back from the request. Only included when explicitly provided. |
| text | ⚠️ | ✅ | object | Hardcoded to `{"format": {"type": "text"}}` in OVMS. |
| top_p | ✅ | ✅ | float | Echoed back from the request. Only included when explicitly provided. |
| truncation | ⚠️ | ✅ | string | Hardcoded to `"disabled"` in OVMS. |
| metadata | ⚠️ | ✅ | object | Hardcoded to `{}` in OVMS. |

### Output item types

The `output` array may contain the following item types:

| Type | Description |
|------|-------------|
| `message` | A text message from the assistant. Contains `id`, `type`, `role`, `status`, and `content` array with `output_text` entries. |
| `function_call` | A tool/function call. Contains `id`, `type`, `status`, `call_id`, `name`, and `arguments`. Emitted when the model invokes a tool. |
| `reasoning` | Reasoning output (for models with thinking/reasoning enabled via `chat_template_kwargs`). Contains `id`, `type`, and `summary` array with `summary_text` entries. |

#### Unsupported response fields from OpenAI service:

- instructions (echoed back)
- output_text (convenience field)

## Streaming events

When `stream` is set to `true`, the server emits [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format) in the following order:

### Standard text generation events

| Event | When emitted | Description |
|-------|-------------|-------------|
| `response.created` | After execution is scheduled | Contains the full response object with `status: "in_progress"`. |
| `response.in_progress` | When the model starts producing tokens | Signals that the response is actively being processed. Emitted as part of the first streaming chunk. |
| `response.output_item.added` | After `response.in_progress` | A new output item (message) has been initialized. Contains `output_index` and the item object. |
| `response.content_part.added` | After `response.output_item.added` | A new content part (output_text) has been initialized. Contains `output_index`, `content_index`, `item_id` and the part object. |
| `response.output_text.delta` | For each text chunk during generation | Contains the text `delta`, `output_index`, `content_index`, and `item_id`. May be emitted many times. |
| `response.output_text.done` | When text generation is finalized | Contains the full accumulated `text`. |
| `response.content_part.done` | After `response.output_text.done` | The content part is complete. Contains the final part object with full text. |
| `response.output_item.done` | After `response.content_part.done` | The output item is complete. Contains the final item object with `status: "completed"`. |
| `response.completed` | Last event before `[DONE]` | Contains the full response object with `status: "completed"` and `usage` statistics. |
| `response.incomplete` | Last event before `[DONE]` (when truncated) | Emitted instead of `response.completed` when generation was stopped due to `max_output_tokens` limit. Contains the response object with `status: "incomplete"` and `incomplete_details`. |
| `response.failed` | On error during generation | Contains the response object with `status: "failed"` and error details. |

### Reasoning events (for models with thinking enabled)

When using models that support reasoning (e.g., via `chat_template_kwargs: {"enable_thinking": true}`), the following additional events may be emitted before the standard message events:

| Event | When emitted | Description |
|-------|-------------|-------------|
| `response.output_item.added` | When reasoning begins | A reasoning output item (`type: "reasoning"`) is added at `output_index: 0`. |
| `response.reasoning_summary_part.added` | After reasoning item added | A reasoning summary part has been initialized. Contains `output_index`, `summary_index`, and `item_id`. |
| `response.reasoning_summary_text.delta` | For each reasoning text chunk | Contains the reasoning text `delta`. |
| `response.reasoning_summary_text.done` | When reasoning is finalized | Contains the full accumulated reasoning text. |
| `response.reasoning_summary_part.done` | After reasoning text done | The reasoning summary part is complete. |
| `response.output_item.done` | After reasoning part done | The reasoning output item is complete. |

When reasoning is present, the subsequent message output item will have `output_index: 1` instead of `0`.

### Function call events (for tool calling)

When the model generates tool/function calls, the following events are emitted (after reasoning events if present, before or instead of message events):

| Event | When emitted | Description |
|-------|-------------|-------------|
| `response.output_item.added` | When a function call begins | A function call output item (`type: "function_call"`) is added. Contains output_index and the item object with `call_id`, `name`, and empty `arguments`. |
| `response.function_call_arguments.delta` | For each arguments chunk | Contains the arguments text `delta`, `item_id`, `output_index`, and `call_id`. |
| `response.function_call_arguments.done` | When arguments are complete | Contains the full accumulated `arguments`. |
| `response.output_item.done` | After arguments done | The function call output item is complete. |

All events include a monotonically increasing `sequence_number` field.

The stream is terminated by a `data: [DONE]` message.

> **NOTE**:
OpenAI python client supports a limited list of parameters. Those native to OpenVINO Model Server, can be passed inside a generic container parameter `extra_body`. Below is an example how to encapsulate `top_k` value.
```{code} python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v3", api_key="unused")
response = client.responses.create(
    model="llama3",
    input="What is OpenVINO?",
    max_output_tokens=100,
    extra_body={"top_k": 1},
    stream=False
)
```

## References

[LLM quick start guide](./llm/quickstart.md)

[End to end demo with LLM model serving over OpenAI API](../demos/continuous_batching/README.md)

[Code snippets](./clients_genai.md)

[LLM calculator](./llm/reference.md#llm-calculator)
