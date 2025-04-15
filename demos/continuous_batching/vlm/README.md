# How to serve VLM models with Continuous Batching via OpenAI API {#ovms_demos_continuous_batching_vlm}

This demo shows how to deploy Vision Language Models in the OpenVINO Model Server using continuous batching and paged attention algorithms.
Text generation use case is exposed via OpenAI API `chat/completions` endpoint.
That makes it easy to use and efficient especially on Intel® Xeon® processors and ARC GPUs.

> **Note:** This demo was tested on 4th - 6th generation Intel® Xeon® Scalable Processors, Intel® Arc™ GPU Series and Intel® Data Center GPU Series on Ubuntu22/24, RedHat8/9 and Windows11.

## Prerequisites

**OVMS version 2025.1** This demo require version 2025.1. Till it is published, it should be [built from source](../../../docs/build_from_source.md)

**Model preparation**: Python 3.9 or higher with pip and HuggingFace account

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../docs/deploying_server_baremetal.md)

**(Optional) Client**: git and Python for using OpenAI client package and vLLM benchmark app


## Model preparation
Here, the original VLM model and its auxiliary models (tokenizer, vision encoder, embeddings model etc.) will be converted to IR format and optionally quantized.
That ensures faster initialization time, better performance and lower memory consumption.
Execution parameters will be defined inside the `graph.pbtxt` file.

Download export script, install it's dependencies and create directory for the models:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/common/export_models/requirements.txt
mkdir models
```

Run `export_model.py` script to download and quantize the model:

> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.

**CPU**
```console
python export_model.py text_generation --source_model OpenGVLab/InternVL2_5-8B --weight-format int8 --config_file_path models/config.json --model_repository_path models  --overwrite_models
```

**GPU**
```console
python export_model.py text_generation --source_model OpenGVLab/InternVL2_5-8B --weight-format int4 --target_device GPU --cache_size 2 --config_file_path models/config.json --model_repository_path models --overwrite_models
```

> **Note:** Change the `--weight-format` to quantize the model to `int8` or `int4` precision to reduce memory consumption and improve performance.

> **Note:** You can change the model used in the demo out of any topology [tested](https://github.com/openvinotoolkit/openvino.genai/blob/master/SUPPORTED_MODELS.md#visual-language-models) with OpenVINO.
Be aware that QwenVL models executed on GPU might experience execution errors with very high resolution images. In case of such behavior, it is recommended to reduce the parameter `max_pixels` in `preprocessor_config.json`.

You should have a model folder like below:
```
models/
├── config.json
└── OpenGVLab
    └── InternVL2_5-8B
        ├── added_tokens.json
        ├── config.json
        ├── configuration_internlm2.py
        ├── configuration_intern_vit.py
        ├── configuration_internvl_chat.py
        ├── generation_config.json
        ├── graph.pbtxt
        ├── openvino_config.json
        ├── openvino_detokenizer.bin
        ├── openvino_detokenizer.xml
        ├── openvino_language_model.bin
        ├── openvino_language_model.xml
        ├── openvino_text_embeddings_model.bin
        ├── openvino_text_embeddings_model.xml
        ├── openvino_tokenizer.bin
        ├── openvino_tokenizer.xml
        ├── openvino_vision_embeddings_model.bin
        ├── openvino_vision_embeddings_model.xml
        ├── preprocessor_config.json
        ├── special_tokens_map.json
        ├── tokenization_internlm2.py
        ├── tokenizer_config.json
        └── tokenizer.model
```

The default configuration should work in most cases but the parameters can be tuned via `export_model.py` script arguments. Run the script with `--help` argument to check available parameters and see the [LLM calculator documentation](../../docs/llm/reference.md) to learn more about configuration options.

## Server Deployment

:::{dropdown} **Deploying with Docker**

Select deployment option depending on how you prepared models in the previous step.

**CPU**

Running this command starts the container with CPU only target device:
```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/workspace:ro openvino/model_server:latest --rest_port 8000 --config_path /workspace/config.json
```
**GPU**

In case you want to use GPU device to run the generation, add extra docker parameters `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)`
to `docker run` command, use the image with GPU support. Export the models with precision matching the GPU capacity and adjust pipeline configuration.
It can be applied using the commands below:
```bash
docker run -d --rm -p 8000:8000 --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -v $(pwd)/models:/workspace:ro openvino/model_server:latest-gpu --rest_port 8000 --config_path /workspace/config.json
```
:::

:::{dropdown} **Deploying on Bare Metal**

Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.

Depending on how you prepared models in the first step of this demo, they are deployed to either CPU or GPU (it's defined in `graph.pbtxt`). If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

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
    "OpenGVLab/InternVL2_5-8B": {
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

Let's send a request with text an image in the messages context.
![zebra](../../../demos/common/static/images/zebra.jpeg) 

:::{dropdown} **Unary call with python requests library**

```console
pip3 install requests
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/static/images/zebra.jpeg -o zebra.jpeg
```
```python
import requests
import base64
base_url='http://localhost:8000/v3'
model_name = "OpenGVLab/InternVL2_5-8B"

def convert_image(Image):
    with open(Image,'rb' ) as file:
        base64_image = base64.b64encode(file.read()).decode("utf-8")
    return base64_image

import requests
payload = {"model": "OpenGVLab/InternVL2_5-8B", 
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
  "model": "OpenGVLab/InternVL2_5-8B",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 19,
    "completion_tokens": 83,
    "total_tokens": 102
  }
}
```
:::
:::{dropdown} **Streaming request with OpenAI client**

The endpoints `chat/completions` is compatible with OpenAI client so it can be easily used to generate code also in streaming mode:

Install the client library:
```console
pip3 install openai
```
```python
from openai import OpenAI
import base64
base_url='http://localhost:8080/v3'
model_name = "OpenGVLab/InternVL2_5-8B"

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

OpenVINO Model Server employs efficient parallelization for text generation. It can be used to generate text also in high concurrency in the environment shared by multiple clients.
It can be demonstrated using benchmarking app from vLLM repository:
```console
git clone --branch v0.7.3 --depth 1 https://github.com/vllm-project/vllm
cd vllm
pip3 install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
cd benchmarks
python benchmark_serving.py --backend openai-chat --dataset-name hf --dataset-path lmarena-ai/vision-arena-bench-v0.1 --hf-split train --host localhost --port 8000 --model OpenGVLab/InternVL2_5-8B --endpoint /v3/chat/completions  --request-rate inf --num-prompts 100 --trust-remote-code

Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: None
============ Serving Benchmark Result ============
Successful requests:                     100
Benchmark duration (s):                  328.26
Total input tokens:                      15381
Total generated tokens:                  5231
Request throughput (req/s):              0.30
Output token throughput (tok/s):         15.94
Total Token throughput (tok/s):          62.79
---------------Time to First Token----------------
Mean TTFT (ms):                          166209.84
Median TTFT (ms):                        185027.54
P99 TTFT (ms):                           317956.52
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          217.28
Median TPOT (ms):                        243.84
P99 TPOT (ms):                           473.48
---------------Inter-token Latency----------------
Mean ITL (ms):                           394.46
Median ITL (ms):                         311.98
P99 ITL (ms):                            1500.04
```

## RAG with visual language model and Model Server

TBD

## Testing the model accuracy over serving API

Check the [guide of using lm-evaluation-harness](../accuracy/README.md)


## References
- [Chat Completions API](../../docs/model_server_rest_api_chat.md)
- [Writing client code](../../docs/clients_genai.md)
- [LLM calculator reference](../../docs/llm/reference.md)
