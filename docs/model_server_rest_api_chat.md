# OpenAI API {#ovms_docs_rest_api_chat}




## API Reference
Please see the [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat) for more information on the API. 

### Request parameters
We support now the following parameters:
- model - this name is assigning the request to a graph of the same name
- messages - user content type is supported as text. Currently it is not allowed to send the image_url in the user message content
- max_tokens
- response_format - only json is implemented
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
- created - data and time when the request was received
- model - model name like specified in the config
- object - "chat.completion" 

Not supported now:
- id
- system_fingerprint
- usage


## References

[End to end demo with LLM model serving over OpenAI API](../demos/continuous_batching/)

[Code snippets](./clients_openai.md)

[Developer guide for writing custom calculators with REST API extension](TBD)
