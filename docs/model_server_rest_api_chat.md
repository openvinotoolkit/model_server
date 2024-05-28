# OpenAI API {#ovms_docs_rest_api_chat}




## API Reference
Please see the [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat) for more information on the API. 

### Request parameters
We support now the following parameters:
- model - this name is assigning the request to a graph of the same name
- messages - user content type is supported as text. Currently it is not allowed to send the image_url in the user message content
- frequency_penalty
- max_tokens
- presence_penalty
- response_format - only json is implementation
- stream
- temperature
- top_p


Extra parameters:
- best_of - This is treated as the beam width 
- use_beam_search - Whether to use beam search instead of sampling.
- top_k - Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.


Not supported parameters:
- tools
- tools_choice
- functions
- funcation_call
- seed
- n
- logit_bias
- logprobs
- top_logprobs

### Response parameters
Supported parameters:
- choices - that includes `finish_reason`, `message.content`, `mesage.role`, index (always 0 for now)
- created
- model
- object - always "chat.completion" 

Not supported now:
- id
- system_fingerprint
- usage


## References

[End to end demo with LLM model serving over OpenAI API](../demos/continuous_batching/)

Code snippets

[Developer guide for writing custom calcultors with REST API extention](TBD)