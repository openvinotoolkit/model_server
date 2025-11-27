# OpenVINO Model Server service integration with llama_swap


In scenario when OVMS is installed on a client platform, it might be common that the host doesn't have capacity to load all the desired models at the same time.

[Llama_swap](https://github.com/mostlygeek/llama-swap) provides capabilities to load the models on-demand and unload them when not needed.

While this tool was implemented for llama-cpp project, it can be easily enabled also for OpenVINO Model Server.


## Prerequisites

- OVMS installed as a [windows service](../../docs/windows_service.md)

## Pull the models needed for the deployment

```bat
ovms pull --task embeddings --model_name OpenVINO/Qwen3-Embedding-0.6B-int8-ov --target_device GPU --cache_dir .ov_cache --pooling LAST
ovms pull --task text_generation --model_name OpenVINO/Qwen3-4B-int4-ov --target_device GPU --cache_dir .ov_cache --tool_parser hermes3
ovms pull --task text_generation --model_name OpenVINO/InternVL2-2B-int4-ov --target_device GPU --cache_dir .ov_cache 
ovms pull --task text_generation --model_name OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov --target_device GPU --cache_dir .ov_cache --tool_parser mistral
```

## Configure config.yaml for llama_swap

Follow the [installation steps](https://github.com/mostlygeek/llama-swap/tree/main?tab=readme-ov-file#installation). Recommended is using windows binary package.

The important elements for OVMS integrations are for each model are:
```
    cmd: |
      powershell -NoProfile -Command "ovms.exe --add_to_config --model_name ${MODEL_ID}; Start-Sleep -Seconds 999999"
    cmdStop: |
      powershell -NoProfile -Command "ovms.exe --remove_from_config  --model_name ${MODEL_ID}"
    proxy: ${base_url}
    checkEndpoint: models/${MODEL_ID}
    name: ${MODEL_ID}
```

This configuration adds and removes a model on demand from OVMS config.json. That automatically loads or unloads the model from the service serving.
Thanks to cache_dir which stored model compilation result, reloading of the model is faster.

Here is an example of a complete [config.yaml](./config.yaml)

Models which should act together in a workflow, should be grouped to minimize impact from model loading time. Check llama_swap documentation about it. Be aware that model reloading is clearing KV cache.

## Connect from the client

Start llama_swap proxy as:
```
llama-swap.exe -listen 127.0.0.1:8080 -watch-config
```

On the OpenAI client connect using `base_url=http://127.0.0.1:8080/v1`.

For example:
```python
from openai import OpenAI

client = OpenAI(
  base_url="http://127.0.0.1:8080/v1",
  api_key="unused"
)

stream = client.chat.completions.create(
    model="OpenVINO/Qwen3-4B-int4-ov",
    messages=[{"role": "user", "content": "Hello."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```


## Limitations

Currently, llama_cpp doesn't support `image` and `rerank` endpoints. It can be used for `chat/completions` `embeddings` and `audio` endpoints.

