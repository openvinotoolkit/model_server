# Loading GGUF models in OVMS {#ovms_demos_gguf}

This demo shows how to deploy GGUF model with the OpenVINO Model Server.

OVMS selects the OpenVINO GGUF frontend automatically for `.gguf` models; no reader-selection parameter is required.

Before selecting a pipeline, OVMS reads GGUF v2/v3 metadata without loading tensor data. Metadata inspection is limited to 64 MiB and requires a nonempty string `general.architecture`. Unknown architectures are passed to GenAI, which validates model support during loading.

GGUF language models use `LM_CB` in `AUTO` mode, or `LM` on NPU. Explicit `LM` and `LM_CB` choices are preserved. A `clip` vision projector cannot be served as a language model. GGUF `VLM`, `VLM_CB`, and `OMNI` selections are rejected because companion-projector loading is not supported. Sibling GGUF filenames containing `mmproj` are reported as projector candidates but are not loaded; their presence does not enable vision support.

Currently recommended model is Ternary-Bonsai-2-27B PQ2_0.
The model is recommended for GPU and the gguf support is in preview mode, so other models and hardware may not give satisfactory performance for 2026.5 release.
Check the [list of supported models](https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/gguf/docs/supported_models.md) for more details.

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
    --source_model "prism-ml/Ternary-Bonsai-2-27B-gguf" \
    --gguf_filename Ternary-Bonsai-2-27B-PQ2_0.gguf \
    --model_name prism-ml/Ternary-Bonsai-2-27B
```
:::

:::{tab-item} Bare metal (Windows)
:sync: bare-metal
```bat
mkdir models
ovms --rest_port 8000 ^
  --model_repository_path ./models/ ^
  --task text_generation ^
  --source_model "prism-ml/Ternary-Bonsai-2-27B-gguf" ^
  --gguf_filename Ternary-Bonsai-2-27B-PQ2_0.gguf ^
  --model_name prism-ml/Ternary-Bonsai-2-27B
```
:::
::::

> **NOTE:** If you want to use model that is split into several `.gguf` files, you should specify the filename of the first part only, e.g. `--gguf_filename model-name-00001-of-00002.gguf`.

Then send a request to the model:

```text
curl http://localhost:8000/v3/chat/completions -s -H "Content-Type: application/json" \
                                                  -d '{"model": "prism-ml/Ternary-Bonsai-2-27B", "stream":false, "messages": [{"role": "system","content": "You are a helpful assistant."}, {"role": "user","content": "What is the capital of France in one word?"}]}' \
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
  "model": "prism-ml/Ternary-Bonsai-2-27B",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 54,
    "completion_tokens": 1,
    "total_tokens": 55
  }
}
```

> **NOTE:** Model downloading feature is described in depth in separate documentation page: [Pulling HuggingFaces Models](../../docs/pull_hf_models.md).

