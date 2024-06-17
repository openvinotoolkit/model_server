# Efficient LLM Serving - quickstart {#ovms_docs_llm_quickstart}

Let's deploy [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model and request generation.

1. Install python dependencies for the conversion script:
```bash
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu https://storage.openvinotoolkit.org/simple/wheels/nightly"

pip3 install --pre "optimum-intel[nncf,openvino]"@git+https://github.com/huggingface/optimum-intel.git openvino-tokenizers
```

2. Run optimum-cli to download and quantize the model:
```bash
mkdir workspace && cd workspace

optimum-cli export openvino --disable-convert-tokenizer --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weight-format int8 TinyLlama-1.1B-Chat-v1.0

convert_tokenizer -o TinyLlama-1.1B-Chat-v1.0 --with-detokenizer --skip-special-tokens --streaming-detokenizer --not-add-special-tokens TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

3. Create `graph.pbtxt` file in a model directory: 
```bash
echo '
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "LLMExecutor"
  calculator: "HttpLLMCalculator"
  input_stream: "LOOPBACK:loopback"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "LLM_NODE_RESOURCES:llm"
  output_stream: "LOOPBACK:loopback"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  input_stream_info: {
    tag_index: "LOOPBACK:0",
    back_edge: true
  }
  node_options: {
      [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
          models_path: "./"
      }
  }
  input_stream_handler {
    input_stream_handler: "SyncSetInputStreamHandler",
    options {
      [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
        sync_set {
          tag_index: "LOOPBACK:0"
        }
      }
    }
  }
}
' >> TinyLlama-1.1B-Chat-v1.0/graph.pbtxt
```

4. Create server `config.json` file:
```bash
echo '
{
    "model_config_list": [],
    "mediapipe_config_list": [
        {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "base_path": "TinyLlama-1.1B-Chat-v1.0"
        }
    ]
}
' >> config.json
```
5. Deploy:

```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/:/workspace:ro openvino/model_server --rest_port 8000 --config_path /workspace/config.json
```
Wait for the model to load. You can check the status with a simple command:
```bash
curl http://localhost:8000/v1/config
{
"TinyLlama/TinyLlama-1.1B-Chat-v1.0" : 
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": "OK"
   }
  }
 ]
}
```
6. Run generation
```bash
curl -s http://localhost:8000/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "max_tokens":30,
    "stream":false,
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is OpenVINO?"
      }
    ]
  }'| jq .
```
```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "OpenVINO is a software toolkit developed by Intel that enables developers to accelerate the training and deployment of deep learning models on Intel hardware.",
        "role": "assistant"
      }
    }
  ],
  "created": 1718607923,
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "object": "chat.completion"
}
```
**Note:** If you want to get the response chunks streamed back as they are generated change `stream` parameter in the request to `true`.


## References
- [Efficient LLM Serving - reference](reference.md)
- [Chat Completions API](../model_server_rest_api_chat.md)
- [Completions API](../model_server_rest_api_completions.md)
- [Demo with Llama3 serving](../../demos/continuous_batching/README.md)
