# Efficient LLM Serving - quickstart {#ovms_docs_llm_quickstart}

Let's deploy [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model and request generation.

1. Install python dependencies for the conversion script:
```console
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/requirements.txt
```

2. Run optimum-cli to download and quantize the model:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/export_model.py -o export_model.py
mkdir models
python export_model.py text_generation --source_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weight-format int8 --kv_cache_precision u8 --config_file_path models/config.json --model_repository_path models 
```

3. Deploy:

:::{dropdown} With Docker

> Required: Docker Engine installed

```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/workspace:ro openvino/model_server --rest_port 8000 --config_path /workspace/config.json
```
:::

:::{dropdown} On Baremetal Host

> Required: OpenVINO Model Server package - see [deployment instruction](../deploying_server_baremetal.md) for details.

```bat
ovms --rest_port 8000 --config_path ./models/config.json
```
:::

4. Check readiness
Wait for the model to load. You can check the status with a simple command:
```console
curl http://localhost:8000/v1/config
```
```json
{
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
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
}
```

5. Run generation
```console
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
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 23,
    "completion_tokens": 30,
    "total_tokens": 53
  }
}
```
**Note:** If you want to get the response chunks streamed back as they are generated change `stream` parameter in the request to `true`.


## References
- [Efficient LLM Serving - reference](reference.md)
- [Chat Completions API](../model_server_rest_api_chat.md)
- [Completions API](../model_server_rest_api_completions.md)
- [Demo with Llama3 serving](../../demos/continuous_batching/README.md)
