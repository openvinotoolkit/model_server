# Tokenization {#ovms_docs_rest_api_tokenize}

The `tokenize` endpoint provides a simple API for tokenizing input text using the same tokenizer as the deployed LLM, VLM or embedding. This allows you to see how your text will be split into tokens before feature extraction or inference. The endpoint accepts a string or list of strings and returns the corresponding token IDs.

Example usage:

Deploy OVMS with LLM, VLM or embedding model:
```bash
mkdir models
# in case GPU is available
export GPU_ARGS=$(ls /dev/dri/render* >/dev/null 2>&1 && echo "--device /dev/dri --group-add $(stat -c '%g' /dev/dri/render* | head -n1)")

docker run --user $(id -u):$(id -g) -d $GPU_ARGS --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --source_model OpenVINO/Qwen3-8B-int4-ov --model_repository_path /models --rest_port 8000
```

Run the client:

```bash
curl http://localhost:8000/v1/tokenize -H "Content-Type: application/json" -d { "model": "Qwen/Qwen3-8B", "text": "hello world"}"
```
Response:
```json
{
  "tokens":[14990,1879]
}
```

It's possible to use additional parameters:
 - `pad_to_max_length` - whether to pad the sequence to the maximum length. Default is False. 
 - `max_length` - maximum length of the sequence. If specified, it truncates the tokens to the provided number.
 - `padding_side` - side to pad the sequence, can be `left` or `right`. Default is `right`.
 - `add_special_tokens` - whether to add special tokens like BOS, EOS, PAD. Default is True. 

 Example usage:
```bash
curl http://localhost:8000/v1/tokenize -H "Content-Type: application/json" -d '{ "model": "Qwen/Qwen3-8B", "text": "hello world", "max_length": 5, "pad_to_max_length": true, "padding_side": "left", "add_special_tokens": true }"
```

Response:
```json
{
  "tokens": [151643,151643,151643,14990,1879]
}
```