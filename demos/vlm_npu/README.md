# Serving for Text generation with Visual Language Models with NPU acceleration {#ovms_demos_vlm_npu}


This demo shows how to deploy VLM models in the OpenVINO Model Server with NPU acceleration.
From the client perspective it is very similar to the [generative model deployment with continuous batching](../continuous_batching/vlm/README.md)
Likewise it exposes the models via OpenAI API `chat/completions`.
The difference is that it doesn't support request batching. They can be sent concurrently but they are processed sequentially.
It is targeted on client machines equipped with NPU accelerator.

> **Note:** This demo was tested on MeteorLake, LunarLake, ArrowLake platforms on Windows11 and Ubuntu24.

## Prerequisites

**OVMS 2025.1 or higher**

**Model preparation**: Python 3.9 or higher with pip and HuggingFace account

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../docs/deploying_server_baremetal.md)

**(Optional) Client**: git and Python for using OpenAI client package and vLLM benchmark app


## Model preparation
Here, the original Pytorch LLM model and the tokenizer will be converted to IR format and optionally quantized.
That ensures faster initialization time, better performance and lower memory consumption.
LLM engine parameters will be defined inside the `graph.pbtxt` file.

Download export script, install it's dependencies and create directory for the models:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/2/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/2/demos/common/export_models/requirements.txt
mkdir models
```

Run `export_model.py` script to download and quantize the model:

> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.

**LLM**
```console
python export_model.py text_generation --source_model microsoft/Phi-3.5-vision-instruct --target_device NPU --config_file_path models/config.json --model_repository_path models  --overwrite_models
```

Note that by default, NPU sets limitation on the prompt length (which in VLM also include image tokens) to 1024 tokens. You can modify that limit by using `--max_prompt_len` parameter.

> **Note:** You can change the model used in the demo out of any topology [tested](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#visual-language-models-vlms) with OpenVINO.

You should have a model folder like below:
```
tree models
models
├── config.json
└── microsoft
    └── Phi-3.5-vision-instruct
        ├── config.json
        ├── generation_config.json
        ├── graph.pbtxt
        ├── openvino_detokenizer.bin
        ├── openvino_detokenizer.xml
        ├── openvino_model.bin
        ├── openvino_model.xml
        ├── openvino_tokenizer.bin
        ├── openvino_tokenizer.xml
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        └── tokenizer.json
```

The default configuration should work in most cases but the parameters can be tuned via `export_model.py` script arguments. Run the script with `--help` argument to check available parameters and see the [LLM calculator documentation](../../docs/llm/reference.md) to learn more about configuration options.

## Server Deployment

:::{dropdown} **Deploying with Docker**


Running this command starts the container with NPU enabled:
```bash
docker run -d --rm --device /dev/accel --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
-p 8000:8000 -v $(pwd)/models:/workspace:ro openvino/model_server:latest-gpu --rest_port 8000 --config_path /workspace/config.json
```
:::

:::{dropdown} **Deploying on Bare Metal**

Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.

Depending on how you prepared models in the first step of this demo, they are deployed to either CPU or GPU (it's defined in `config.json`). If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

```bat
ovms --rest_port 8000 --config_path ./models/config.json
```
:::

## Readiness Check

Wait for the model to load. You can check the status with a simple command:
```console
curl http://localhost:8000/v1/config
```
```json
{
    "microsoft/Phi-3.5-vision-instruct": {
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

## Request Generation

```console
pip3 install requests
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/static/images/zebra.jpeg -o zebra.jpeg
```
![zebra](https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/static/images/zebra.jpeg)

:::{dropdown} **Unary call with curl using image from local filesystem**

Referring to local filesystem images in requests requires passing additional parameter `--allowed_local_media_path` (described in [Model Server Parameters](../../docs/parameters.md) section) when starting docker container: 
```bash
docker run -d --rm --device /dev/accel --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
-p 8000:8000 -v $(pwd)/models:/workspace:ro -v $(pwd):/images:ro openvino/model_server:latest-gpu --rest_port 8000 --config_path /workspace/config.json --allowed_local_media_path /images
```

```bash
curl http://localhost:8000/v3/chat/completions  -H "Content-Type: application/json" -d "{ \"model\": \"microsoft/Phi-3.5-vision-instruct\", \"messages\":[{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"Describe what is one the picture.\"},{\"type\": \"image_url\", \"image_url\": {\"url\": \"/images/zebra.jpeg\"}}]}], \"max_completion_tokens\": 100}"
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
  "model": "microsoft/Phi-3.5-vision-instruct",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 19,
    "completion_tokens": 83,
    "total_tokens": 102
  }
}
```
:::

:::{dropdown} **Unary call with python requests library**

```python
import requests
import base64
base_url='http://localhost:8000/v3'
model_name = "microsoft/Phi-3.5-vision-instruct"

def convert_image(Image):
    with open(Image,'rb' ) as file:
        base64_image = base64.b64encode(file.read()).decode("utf-8")
    return base64_image

import requests
payload = {
    "model": model_name, 
    "messages": [
        {
            "role": "user",
            "content": [
              {"type": "text", "text": "Describe what is one the picture."},
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
  "model": "microsoft/Phi-3.5-vision-instruct",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 19,
    "completion_tokens": 83,
    "total_tokens": 102
  }
}
```
:::

:::{dropdown} **Unary call with OpenAI Python package**

The endpoints `chat/completions` are compatible with OpenAI client so it can be easily used to generate code also in streaming mode:

Install the client library:
The endpoints `chat/completions` are compatible with OpenAI client so it can be easily used to generate code also in streaming mode:

Install the client library:
```console
pip3 install openai
```
```python
from openai import OpenAI
import base64
base_url='http://localhost:8000/v3'
model_name = "microsoft/Phi-3.5-vision-instruct"

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
              {"type": "text", "text": "Describe what is one the picture."},
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

## Benchmarking text generation with high concurrency

OpenVINO Model Server with NPU acceleration process the requests sequentially. For that reason, benchmarking should be performed in max_concurrency set to 1.
Parallel requests will be accepted but they will wait for a free execution slot.
Benchmarking can be demonstrated using benchmarking app from vLLM repository:
```console
git clone --branch v0.7.3 --depth 1 https://github.com/vllm-project/vllm
cd vllm
pip3 install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
cd benchmarks
curl -L https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -o ShareGPT_V3_unfiltered_cleaned_split.json # sample dataset
python benchmark_serving.py --backend openai-chat --dataset-name hf --dataset-path lmarena-ai/vision-arena-bench-v0.1 --hf-split train --host localhost --port 8000 --model microsoft/Phi-3.5-vision-instruct --endpoint /v3/chat/completions --num-prompts 10 --trust-remote-code --max-concurrency 1

```

## Testing the model accuracy over serving API

Check the [guide of using lm-evaluation-harness](https://github.com/openvinotoolkit/model_server/blob/main/demos/continuous_batching/accuracy/README.md)

## Limitations

- requests MUST include one and only one image in the messages context. Other request will be rejected. 
- beam_search algorithm is not supported with NPU. Greedy search and multinomial algorithms are supported.
- models must be exported with INT4 precision and `--sym --ratio 1.0 --group-size -1` params. This is enforced in the export_model.py script when the target_device in NPU.
- log_probs are not supported
- finish reason is always set to "stop".
- only a single response can be returned. Parameter `n` is not supported.

## References
- [Chat Completions API](../../docs/model_server_rest_api_chat.md)
- [Completions API](../../docs/model_server_rest_api_completions.md)
- [Writing client code](../../docs/clients_genai.md)
- [LLM calculator reference](../../docs/llm/reference.md)
