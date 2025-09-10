# Loading GGUF models in OVMS {#ovms_demos_gguf}

This demo shows how to deploy  model with the OpenVINO Model Server.

> **NOTE**: This is experimental feature and issues in accuracy of models may be observed.

If the model already exists locally, it will skip the downloading and immediately start the serving.

> **NOTE:** Optionally, to only download the model and omit the serving part, use `--pull` parameter.

Start with deploying the model:

::::{tab-set}
:::{tab-item} Docker (Linux)
:sync: docker
Start docker container:
```bash
mkdir models
docker run -d --rm --user $(id -u):$(id -g) -p 8000:8000 -v $(pwd)/models:/models/:rw \
  -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy \
  openvino/model_server:latest \
    --rest_port 8000 \
    --model_repository_path /models/ \
    --task text_generation \
    --source_model "Qwen/Qwen2.5-3B-Instruct-GGUF" \
    --gguf_filename qwen2.5-3b-instruct-q4_k_m.gguf \
    --model_name LLM
```
:::

:::{tab-item} Bare metal (Windows)
:sync: bare-metal
```bat
mkdir models
ovms --rest_port 8000 ^
  --model_repository_path /models/ ^
  --task text_generation ^
  --source_model "Qwen/Qwen2.5-3B-Instruct-GGUF" ^
  --gguf_filename qwen2.5-3b-instruct-q4_k_m.gguf ^
  --model_name LLM
```
:::
::::

Then send a request to the model:

```text
curl http://localhost:8000/v3/chat/completions -H "Content-Type: application/json" \
                                               -d '{"model": "LLM", \
                                                    "max_tokens":300, \
                                                    "stream":false, \
                                                    "messages": [{"role": "system","content": "You are a helpful assistant."}, \
                                                                 {"role": "user","content": "What is the capital of France in one word?"}]}'| jq .
```

Example response would be:

```text
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   456  100   260  100   196    411    309 --:--:-- --:--:-- --:--:--   720
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "Paris",
        "role": "assistant",
        "tool_calls": []
      }
    }
  ],
  "created": 1756986130,
  "model": "LLM",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 54,
    "completion_tokens": 1,
    "total_tokens": 55
  }
}
```

> **NOTE:** Model downloading feature is described in depth in separate documentation page: [Pulling HuggingFaces Models](../../docs/pull_hf_models.md).

