# QuickStart - LLM models {#ovms_docs_llm_quickstart}

Let's deploy [OpenVINO/Phi-3.5-mini-instruct-int4-ov](https://huggingface.co/OpenVINO/Phi-3.5-mini-instruct-int4-ov) model on Intel iGPU or ARC GPU.
It is quantized to INT4 precision and converted it IR format.

Requirements:
- Linux or Windows11
- Docker Engine or `ovms` binary package [installed](../deploying_server_baremetal.md)
- Intel iGPU or ARC GPU 


1. Install python dependencies for the conversion script:
```console
pip3 install huggingface_hub
```

2. Run optimum-cli to download and quantize the model:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/simpler-quick-start-llm/demos/common/export_models/export_model.py -o export_model.py
mkdir models
python export_model.py text_generation --source_model OpenVINO/Phi-3.5-mini-instruct-int4-ov --config_file_path models/config.json --model_repository_path models --target_device GPU --cache 2
```
**Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.
**Note:** If you want to export models outside of `OpenVINO` organization in HuggingFace, you need to install also the python dependencies via `pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/relases/2025/1/demos/common/export_models/requirements.txt` before running the export_models.py script.
 
3. Deploy:

:::{dropdown} With Docker

> Required: Docker Engine installed

```bash
docker run -d --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render*) --rm -p 8000:8000 -v $(pwd)/models:/models:ro openvino/model_server:latest-gpu --rest_port 8000 --model_name Phi-3.5-mini-instruct --model_path /models/OpenVINO/Phi-3.5-mini-instruct-int4-ov
```
:::

:::{dropdown} On Baremetal Host

> Required: OpenVINO Model Server package - see [deployment instruction](../deploying_server_baremetal.md) for details.

```bat
ovms --rest_port 8000 --model_name Phi-3.5-mini-instruct --model_path /models/OpenVINO/Phi-3.5-mini-instruct-int4-ov
```
:::

4. Check readiness
Wait for the model to load. You can check the status with a simple command:
```console
curl http://localhost:8000/v1/config
```
```json
{
  "Phi-3.5-mini-instruct": {
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
curl -s http://localhost:8000/v3/chat/completions   -H "Content-Type: application/json" -d "{\"model\": \"Phi-3.5-mini-instruct\",\"max_tokens\":30, \"messages\": [{\"role\": \"system\",\"content\": \"You are a helpful assistant.\" }, {\"role\": \"user\",\"content\": \"What are the 3 main tourist attractions in Paris?\"}]}"
```
```json
{
  "choices": [
    {
      "finish_reason": "length",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "Paris, the charming City of Light, is renowned for its rich history, iconic landmarks, architectural splendor, and artistic",
        "role": "assistant"
      }
    }
  ],
  "created": 1744716414,
  "model": "Phi-3.5-mini-instruct",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 30,
    "total_tokens": 54
  }
}

```
**Note:** If you want to get the response chunks streamed back as they are generated change `stream` parameter in the request to `true`.


## References
- [Efficient LLM Serving - reference](reference.md)
- [Chat Completions API](../model_server_rest_api_chat.md)
- [Completions API](../model_server_rest_api_completions.md)
- [Demo with Llama3 serving](../../demos/continuous_batching/README.md)
