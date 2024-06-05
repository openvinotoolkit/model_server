# OpenAI API {#ovms_docs_rest_api_chat}




## API Reference
Please see the [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat) for more information on the API. 

### Request parameters
We support now the following parameters:
- model - this name is assigning the request to a graph of the same name
- messages - user content type is supported as text. Currently it is not allowed to send the image_url in the user message content
- max_tokens
- response_format - only `text` is implemented
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

### Response parameters
Supported parameters:
- choices - that includes `finish_reason`, `message.content`, `message.role`, index (always 0 for now)
- created - time when the request was received
- model - model name like specified in the config
- object - "chat.completion" or "chat.completion.chunk"

Not supported now:
- id
- system_fingerprint
- usage


## Parameter comparision

### Generic

| Param | OpenVINO Model Server | OpenAI /chat/completions API | vLLM Serving Sampling Params | Type | Description |
|-----|----------|----------|----------|---------|-----|
| model | ❎ | ❎ | ❎ | string (required) | Name of the model to use. From administrator point of view it is the name assigned to a MediaPipe graph configured to schedule generation using desired model.  |
| stream | ❎ | ❎ | ❎ | bool (optional, default: `false`) | If set to true, partial message deltas will be sent to the client. The generation chunks will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format) as they become available, with the stream terminated by a `data: [DONE]` message. [Example Python code](clients_openai.md) |
| messages | ❎ | ❎ | ❎ | array (required) | A list of messages comprising the conversation so far. Each object in the list should contain `role` and `content` - both of string type. [Example Python code](clients_openai.md) |
| max_tokens | ❎ | ❎ | ❎ | integer | The maximum number of tokens that can be generated. If not set, the generation will stop once `EOS` token is generated. **_TODO: there is upper limit model can handle without hallucinating (context length), CB lib has default 30 - to be changed in CB lib_** |
| ignore_eos | ❎ | ❌ | ❎ | bool (default: `false`) | Whether to ignore the `EOS` token and continue generating tokens after the `EOS` token is generated. |

### Beam search sampling specific
| Param | OpenVINO Model Server | OpenAI /chat/completions API | vLLM Serving Sampling Params | Type | Description |
|-------|----------|----------|----------|---------|-----|
| n | ❎ | ❌ | ❎ | int (default: `1`) | Number of output sequences to return for the given prompt. This value must be between `1 <= N <= BEST_OF`. **_TODO: still not implemented handling of this parameter in CB lib_** |
| best_of | ❎ | ❌ | ❎ | int (default: `1`) | Number of output sequences that are generated from the prompt. From these _best_of_ sequences, the top _n_ sequences are returned. _best_of_ must be greater than or equal to _n_. This is treated as the beam width for beam search sampling.  |
| diversity_penalty | ❎ | ❌ | ❌ | float (default: `1.0`) | **_TODO: explain_** |
| repetition_penalty | ❎ | ❌ | ❎ | float (default: `1.0`) | Penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat tokens. |
| length_penalty | ❎ | ❌ | ❎ | float (default: `1.0`) | Penalizes sequences based on their length. |

### Multinomial sampling specific
| Param | OpenVINO Model Server | OpenAI /chat/completions API | vLLM Serving Sampling Params | Type | Description |
|-------|----------|----------|----------|---------|-----|
| temperature | ❎ | ❎ | ❎ | float (default: `0.0`) | Penalizes sequences based on their length. **_TODO: other servings default is 1.0 instead of 0.0, should we change it? If we do, then multinomial sampling will be used by default (instead of greedy)_** |
| top_p | ❎ | ❎* | ❎ | float (default: `1.0`) | Controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens. |
| top_k | ❎ | ❌ | ❎ | int (default: `0`) | Controls the number of top tokens to consider. Set to 0 to consider all tokens. **_TODO: in vLLM it is default -1 which considers all tokens, in CB lib (and HF) it is 0_** |

\* In OpenAI service `top_p` refers to _nucleus sampling_, in OpenVINO Model Server it refers to _multinomial sampling_.

### Unsupported params from OpenAI service:
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

### Unsupported params from vLLM:
- presence_penalty 
- frequency_penalty
- min_p
- seed
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

## References

[End to end demo with LLM model serving over OpenAI API](../demos/continuous_batching/README.md)

[Code snippets](./clients_openai.md)

[Developer guide for writing custom calculators with REST API extension](./mediapipe.md)
