# Loading GGUF models in OVMS {#ovms_demos_gguf}

This demo shows how to deploy GGUF model with the OpenVINO Model Server.

Currently supported models are DeepSeek-R1-Distill-Qwen (1.5B, 7B), Qwen2.5 Instruct (1.5B, 3B, 7B), llama-3.2 Instruct (1B, 3B) & llama-3.1-8B.
Check the [list of supported models](https://blog.openvino.ai/blog-posts/openvino-genai-supports-gguf-models) for more details.

If the model already exists locally, it will skip the downloading and immediately start the serving.

> **NOTE:** Optionally, to only download the model and omit the serving part, use `--pull` parameter and remove `--rest_port`.


Deploy the model:

::::{tab-set}
:::{tab-item} Docker (Linux)
:sync: docker
Start docker container:
```bash
mkdir models
docker run -d --rm --user $(id -u):$(id -g) -p 8000:8000 -v $(pwd)/models:/models/:rw \
  -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy \
  openvino/model_server:weekly \
    --rest_port 8000 \
    --model_repository_path /models/ \
    --task text_generation \
    --source_model "Qwen/Qwen2.5-3B-Instruct-GGUF" \
    --gguf_filename qwen2.5-3b-instruct-q4_k_m.gguf \
    --model_name Qwen/Qwen2.5-3B-Instruct
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
  --model_name Qwen/Qwen2.5-3B-Instruct
```
:::
::::

> **NOTE:** If you want to use model that is splitted into several `.gguf` files, you should specify the filename of the first part only, e.g. `--gguf_filename model-name-00001-of-00002.gguf`.

Then send a request to the model:

```text
curl http://localhost:8000/v3/chat/completions -s -H "Content-Type: application/json" \
                                                  -d '{"model": "Qwen/Qwen2.5-3B-Instruct", "stream":false, "messages": [{"role": "system","content": "You are a helpful assistant."}, {"role": "user","content": "What is the capital of France in one word?"}]}' \
| jq .
```

Example response would be:

```text
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
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 54,
    "completion_tokens": 1,
    "total_tokens": 55
  }
}
```

> **NOTE:** Model downloading feature is described in depth in separate documentation page: [Pulling HuggingFaces Models](../../docs/pull_hf_models.md).

