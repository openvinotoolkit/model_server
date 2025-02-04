# QuickStart - LLM models {#ovms_docs_llm_quickstart}

Let's deploy [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) model and request generation on Intel iGPU or ARC GPU.

Requirements:
- Linux or Windows11
- Docker Engine or `ovms` binary package [installed]((../deploying_server_baremetal.md) )
- Intel iGPU or ARC GPU 


1. Install python dependencies for the conversion script:
```console
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/requirements.txt
```

2. Run optimum-cli to download and quantize the model:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/export_model.py -o export_model.py
mkdir models
python export_model.py text_generation --source_model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --weight-format int4 --config_file_path models/config.json --model_repository_path models --target_device GPU --cache 2
```

3. Deploy:

:::{dropdown} With Docker

> Required: Docker Engine installed

```bash
docker run -d --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render*) --rm -p 8000:8000 -v $(pwd)/models:/workspace:ro openvino/model_server:latest-gpu --rest_port 8000 --config_path /workspace/config.json
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
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
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
curl -s http://localhost:8000/v3/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "max_tokens":30, "temperature":0,
    "stream":false,
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What are the 3 main tourist attractions in Paris?"
      }
    ]
  }'| jq .

```
```json
{
  "choices": [
    {
      "finish_reason": "length",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "The three main tourist attractions in Paris are the Eiffel Tower, the Louvre Museum, and the Paris RATP Metro.<｜User｜>",
        "role": "assistant"
      }
    }
  ],
  "created": 1738656445,
  "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 37,
    "completion_tokens": 30,
    "total_tokens": 67
  }
}

```
**Note:** If you want to get the response chunks streamed back as they are generated change `stream` parameter in the request to `true`.


## References
- [Efficient LLM Serving - reference](reference.md)
- [Chat Completions API](../model_server_rest_api_chat.md)
- [Completions API](../model_server_rest_api_completions.md)
- [Demo with Llama3 serving](../../demos/continuous_batching/README.md)
