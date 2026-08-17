# VLM models via OpenAI API {#ovms_demos_continuous_batching_vlm}

```{toctree}
---
maxdepth: 1
hidden:
---
ovms_demos_vlm_npu
```

This demo shows how to deploy Vision Language Models in the OpenVINO Model Server.
Text generation use case is exposed via OpenAI API `chat/completions` and `responses` endpoints.

> **Note:** This demo was tested on 4th - 6th generation Intel® Xeon® Scalable Processors, Intel® Arc™ Xe2 GPU platforms on Ubuntu24 and Windows11. At least 22GB VRAM is needed to deploy model Qwen3.6-35B-A3B used in the demo. On platforms with less memory, use smaller model like [OpenVINO/Qwen3.5-4B-int4-ov](https://huggingface.co/OpenVINO/Qwen3.5-4B-int4-ov).

## Prerequisites

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../../docs/deploying_server_baremetal.md)

**(Optional) Client**: git and Python for using OpenAI client package and vLLM benchmark app


## Fast deployment with OpenVINO models pulled directly from HuggingFace Hub
VLM models can be deployed in a single command by using pre-configured models from [OpenVINO HuggingFace organization](https://huggingface.co/OpenVINO)
Here is an example of `Qwen3.6-35B-A3B-int4` deployment:

:::{dropdown} **Deploying with Docker**

Running this command starts the container:
```bash
mkdir -p models
# in case GPU is available
export GPU_ARGS=$(if ls /dev/dri/render* >/dev/null 2>&1; then echo "--device /dev/dri --group-add $(stat -c '%g' /dev/dri/render* | head -n1)"; fi)
docker run -d ${GPU_ARGS} -u $(id -u):$(id -g) --rm -p 8000:8000 -v ${HOME}/models:/models:rw openvino/model_server:weekly --rest_port 8000 --source_model OpenVINO/Qwen3.6-35B-A3B-int4-ov --model_repository_path /models --allowed_media_domains raw.githubusercontent.com
```
:::

:::{dropdown} **Deploying on Bare Metal**

If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

```bat
mkdir c:\models
ovms --rest_port 8000 --source_model OpenVINO/Qwen3.6-35B-A3B-int4-ov --model_repository_path c:\models --allowed_media_domains raw.githubusercontent.com
```

:::

## Readiness Check

Wait for the model to load. You can check the status with a simple command:
```console
curl http://localhost:8000/v1/models
```
```json
{
  "object": "list",
  "data": [
    {
      "id": "OpenVINO/Qwen3.6-35B-A3B-int4-ov",
      "object": "model",
      "created": 1772928358,
      "owned_by": "OVMS"
    }
  ]
}
```

## Request Generation

Let's send a request with text an image in the messages context.
![zebra](../../../demos/common/static/images/zebra.jpeg) 

:::{dropdown} **Unary call with curl using image url**
**Note**: using urls in request requires `--allowed_media_domains` parameter described [here](../../../docs/parameters.md)

```bash
curl http://localhost:8000/v1/chat/completions  -H "Content-Type: application/json" -d "{ \"model\": \"OpenVINO/Qwen3.6-35B-A3B-int4-ov\", \"messages\":[{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"Describe what is on the picture.\"},{\"type\": \"image_url\", \"image_url\": {\"url\": \"http://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/static/images/zebra.jpeg\"}}]}], \"max_completion_tokens\": 100}"
```
```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "The picture features a zebra standing in a grassy plain. Zebras are known for their distinctive black and white striped patterns, which help them blend in for camouflage purposes. The zebra pictured is standing on a green field with patches of grass, indicating it may be in its natural habitat. Zebras are typically social animals and are often found in savannahs and grasslands.",
        "role": "assistant"
      }
    }
  ],
  "created": 1741731554,
  "model": "OpenVINO/Qwen3.6-35B-A3B-int4-ov",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 19,
    "completion_tokens": 83,
    "total_tokens": 102
  }
}
```
:::

:::{dropdown} **Unary call with cURL using Responses API**
**Note**: Using urls in request requires `--allowed_media_domains` parameter described [here](../../../docs/parameters.md)

```bash
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenVINO/Qwen3.6-35B-A3B-int4-ov",
    "input": [
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": "Describe what is on the picture."
          },
          {
            "type": "input_image",
            "image_url": "http://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/static/images/zebra.jpeg"
          }
        ]
      }
    ],
    "max_output_tokens": 100
  }'
```
```json
{
  "id": "resp-1741731554",
  "object": "response",
  "created_at": 1741731554,
  "model": "OpenVINO/Qwen3.6-35B-A3B-int4-ov",
  "status": "completed",
  "output": [
    {
      "id": "msg-0",
      "type": "message",
      "role": "assistant",
      "status": "completed",
      "content": [
        {
          "type": "output_text",
          "text": "The picture features a zebra standing in a grassy plain. Zebras are known for their distinctive black and white striped patterns, which help them blend in for camouflage purposes.",
          "annotations": []
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 19,
    "input_tokens_details": { "cached_tokens": 0 },
    "output_tokens": 83,
    "total_tokens": 102
  }
}
```
:::

:::{dropdown} **Unary call with python requests library**

```console
pip3 install requests
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/static/images/zebra.jpeg -o zebra.jpeg
```
```python
import requests
import base64
base_url='http://127.0.0.1:8000/v1'
model_name = "OpenVINO/Qwen3.6-35B-A3B-int4-ov"

def convert_image(Image):
    with open(Image,'rb' ) as file:
        base64_image = base64.b64encode(file.read()).decode("utf-8")
    return base64_image

payload = {"model": model_name,
    "messages": [
        {
            "role": "user",
            "content": [
              {"type": "text", "text": "Describe what is on the picture."},
              {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{convert_image('zebra.jpeg')}"}}
            ]
        }
        ],
    "max_completion_tokens": 100
}
headers = {"Content-Type": "application/json", "Authorization": "not used"}
response = requests.post(base_url + "/chat/completions", json=payload, headers=headers)
print(response.text)
```
```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "The picture features a zebra standing in a grassy plain. Zebras are known for their distinctive black and white striped patterns, which help them blend in for camouflage purposes. The zebra pictured is standing on a green field with patches of grass, indicating it may be in its natural habitat. Zebras are typically social animals and are often found in savannahs and grasslands.",
        "role": "assistant"
      }
    }
  ],
  "created": 1741731554,
  "model": "OpenVINO/Qwen3.6-35B-A3B-int4-ov",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 19,
    "completion_tokens": 83,
    "total_tokens": 102
  }
}
```
:::
:::{dropdown} **Streaming request with OpenAI client using chat/completions**

The endpoints `chat/completions` and `responses` are compatible with OpenAI client so it can be easily used to generate code also in streaming mode:

Install the client library:
```console
pip3 install openai
```
```python
from openai import OpenAI
import base64
base_url='http://localhost:8000/v1'
model_name = "OpenVINO/Qwen3.6-35B-A3B-int4-ov"

client = OpenAI(api_key='unused', base_url=base_url)

def convert_image(Image):
    with open(Image,'rb' ) as file:
        base64_image = base64.b64encode(file.read()).decode("utf-8")
    return base64_image

stream = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": [
              {"type": "text", "text": "Describe what is on the picture."},
              {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{convert_image('zebra.jpeg')}"}}
            ]
        }
        ],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

Output:
```
The picture features a zebra standing in a grassy area. The zebra is characterized by its distinctive black and white striped pattern, which covers its entire body, including its legs, neck, and head. Zebras have small, rounded ears and a long, flowing tail. The background appears to be a natural grassy habitat, typical of a savanna or plain.
```

:::

:::{dropdown} **Streaming request with OpenAI client via Responses API**

```console
pip3 install openai
```
```python
from openai import OpenAI
import base64
base_url='http://localhost:8000/v1'
model_name = "OpenVINO/Qwen3.6-35B-A3B-int4-ov"

client = OpenAI(api_key='unused', base_url=base_url)

def convert_image(Image):
    with open(Image,'rb' ) as file:
        base64_image = base64.b64encode(file.read()).decode("utf-8")
    return base64_image

stream = client.responses.create(
    model=model_name,
    input=[
        {
            "role": "user",
            "content": [
              {"type": "input_text", "text": "Describe what is on the picture."},
              {"type": "input_image", "image_url": f"data:image/jpeg;base64,{convert_image('zebra.jpeg')}"}
            ]
        }
        ],
    stream=True,
)
for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
```

Output:
```
The picture features a zebra standing in a grassy area. The zebra is characterized by its distinctive black and white striped pattern, which covers its entire body, including its legs, neck, and head. Zebras have small, rounded ears and a long, flowing tail. The background appears to be a natural grassy habitat, typical of a savanna or plain.
```

:::

## Testing the model accuracy over serving API

Check the [guide of using lm-evaluation-harness](../accuracy/README.md)



## References
- [Official OpenVINO LLM models in HuggingFace](https://huggingface.co/collections/OpenVINO/llm)
- [Supported VLM models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#visual-language-models-vlms)
- [Export models to OpenVINO format](../../../demos/common/export_models/README.md)
- [Chat Completions API](../../../docs/model_server_rest_api_chat.md)
- [Responses API](../../../docs/model_server_rest_api_responses.md)
- [Writing client code](../../../docs/clients_genai.md)
- [LLM calculator reference](../../../docs/llm/reference.md)
