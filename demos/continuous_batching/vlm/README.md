# How to serve VLM models via OpenAI API {#ovms_demos_continuous_batching_vlm}

This demo shows how to deploy Vision Language Models in the OpenVINO Model Server.
Text generation use case is exposed via OpenAI API `chat/completions` endpoint.

> **Note:** This demo was tested on 4th - 6th generation Intel® Xeon® Scalable Processors, Intel® Arc™ GPU Series and Intel® Data Center GPU Series on Ubuntu22/24, RedHat8/9 and Windows11.

## Prerequisites

**OVMS version 2025.1** This demo require version 2025.1 or newer.

**Model preparation**: Python 3.9 or higher with pip and HuggingFace account

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../../docs/deploying_server_baremetal.md)

**(Optional) Client**: git and Python for using OpenAI client package and vLLM benchmark app


## Fast deployment with OpenVINO models pulled directly from HuggingFace Hub
VLM models can be deployed in a single command by using pre-configured models from [OpenVINO HuggingFace organization](https://huggingface.co/OpenVINO)
For other models go to the model preparation step and deployment for converted models.
Here is an example of OpenVINO/InternVL2-2B-int4-ov deployment:

:::{dropdown} **Deploying with Docker**

Select deployment option depending on how you prepared models in the previous step.

**CPU**

Running this command starts the container with CPU only target device:
```bash
mkdir -p models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest --rest_port 8000 --source_model OpenVINO/InternVL2-2B-int4-ov --model_repository_path /models --model_name OpenGVLab/InternVL2-2B --task text_generation --pipeline_type VLM
```
**GPU**

In case you want to use GPU device to run the generation, add extra docker parameters `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)`
to `docker run` command, use the image with GPU support.
It can be applied using the commands below:
```bash
mkdir -p models
docker run -d -u $(id -u):$(id -g) --rm -p 8000:8000 --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --rest_port 8000 --source_model OpenVINO/InternVL2-2B-int4-ov --model_repository_path models --model_name OpenGVLab/InternVL2-2B --task text_generation --target_device GPU --pipeline_type VLM
```
:::

:::{dropdown} **Deploying on Bare Metal**

If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

```bat
mkdir models
ovms --rest_port 8000 --source_model OpenVINO/InternVL2-2B-int4-ov --model_repository_path models --model_name OpenGVLab/InternVL2-2B --task text_generation --pipeline_type VLM --target_device CPU
```
or
```bat
ovms --rest_port 8000 --source_model OpenVINO/InternVL2-2B-int4-ov --model_repository_path models --model_name OpenGVLab/InternVL2-2B --task text_generation --pipeline_type VLM --target_device GPU
```
:::



## Model preparation
Use this step for models outside of OpenVINO organization.

Specific OVMS pull mode example for models outside of OpenVINO organization is described in section `## Pulling models outside of OpenVINO organization` in the [Ovms pull mode](https://github.com/openvinotoolkit/model_server/blob/main/docs/pull_hf_models.md)

Or you can use the python export_model.py script described below.

Here, the original VLM model and its auxiliary models (tokenizer, vision encoder, embeddings model etc.) will be converted to IR format and optionally quantized.
That ensures faster initialization time, better performance and lower memory consumption.
Execution parameters will be defined inside the `graph.pbtxt` file.

Download export script, install it's dependencies and create directory for the models:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/3/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/3/demos/common/export_models/requirements.txt
mkdir models
```

Run `export_model.py` script to download and quantize the model:

> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.

**CPU**
```console
python export_model.py text_generation --source_model OpenGVLab/InternVL2-2B --weight-format int4 --pipeline_type VLM --model_name OpenGVLab/InternVL2-2B --config_file_path models/config.json --model_repository_path models  --overwrite_models
```

**GPU**
```console
python export_model.py text_generation --source_model OpenGVLab/InternVL2-2B --weight-format int4 --pipeline_type VLM --model_name OpenGVLab/InternVL2-2B --config_file_path models/config.json --model_repository_path models --overwrite_models --target_device GPU
```

> **Note:** Change the `--weight-format` to quantize the model to `fp16` or `int8` precision to reduce memory consumption and improve performance.

> **Note:** You can change the model used in the demo out of any topology [tested](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#visual-language-models-vlms) with OpenVINO.
Be aware that QwenVL models executed on GPU might experience execution errors with very high resolution images. In case of such behavior, it is recommended to reduce the parameter `max_pixels` in `preprocessor_config.json`.


You should have a model folder like below:
```
models/
├── config.json
└── OpenGVLab
    └── InternVL2
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

The default configuration should work in most cases but the parameters can be tuned via `export_model.py` script arguments. Run the script with `--help` argument to check available parameters and see the [LLM calculator documentation](../../../docs/llm/reference.md) to learn more about configuration options.

## Server Deployment with locally stored models

:::{dropdown} **Deploying with Docker**

Select deployment option depending on how you prepared models in the previous step.

**CPU**

Running this command starts the container with CPU only target device:
```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/models:ro openvino/model_server:latest --rest_port 8000 --model_name OpenGVLab/InternVL2-2B --model_path /models/OpenGVLab/InternVL2-2B
```
**GPU**

In case you want to use GPU device to run the generation, add extra docker parameters `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)`
to `docker run` command, use the image with GPU support. Export the models with precision matching the GPU capacity and adjust pipeline configuration.
It can be applied using the commands below:
```bash
docker run -d --rm -p 8000:8000 --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -v $(pwd)/models:/models:ro openvino/model_server:latest-gpu --rest_port 8000 --model_name OpenGVLab/InternVL2-2B --model_path /models/OpenGVLab/InternVL2-2B
```
:::

:::{dropdown} **Deploying on Bare Metal**

Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.

Depending on how you prepared models in the first step of this demo, they are deployed to either CPU or GPU (it's defined in `graph.pbtxt`). If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

```bat
ovms --rest_port 8000 --model_name OpenGVLab/InternVL2-2B --model_path /models/OpenGVLab/InternVL2-2B
```
:::

> **Note:** VLM models can be enabled also with continuous batching pipeline. It that case the export_model.py or the ovms deployment from HuggingFace source model should have parameter `--pipeline_type VLM_CB`. It has, however, a defect related to accuracy for models Qwen2, Qwen2.5 and Phi3.5. 
The pipeline with continuous batching will give better throughput especially if there are many requests with text only in the requests. 

## Readiness Check

Wait for the model to load. You can check the status with a simple command:
```console
curl http://localhost:8000/v1/config
```
```json
{
    "OpenGVLab/InternVL2-2B": {
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

:::{dropdown} **Unary call with curl using image url**


```bash
curl http://localhost:8000/v3/chat/completions  -H "Content-Type: application/json" -d "{ \"model\": \"OpenGVLab/InternVL2-2B\", \"messages\":[{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"Describe what is one the picture.\"},{\"type\": \"image_url\", \"image_url\": {\"url\": \"http://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/3/demos/common/static/images/zebra.jpeg\"}}]}], \"max_completion_tokens\": 100}"
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
  "model": "OpenGVLab/InternVL2-2B",
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

```console
pip3 install requests
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/static/images/zebra.jpeg -o zebra.jpeg
```
```python
import requests
import base64
base_url='http://localhost:8000/v3'
model_name = "OpenGVLab/InternVL2-2B"

def convert_image(Image):
    with open(Image,'rb' ) as file:
        base64_image = base64.b64encode(file.read()).decode("utf-8")
    return base64_image

import requests
payload = {"model": "OpenGVLab/InternVL2-2B",
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
  "model": "OpenGVLab/InternVL2-2B",
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
model_name = "OpenGVLab/InternVL2-2B"

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
python benchmark_serving.py --backend openai-chat --dataset-name hf --dataset-path lmarena-ai/vision-arena-bench-v0.1 --hf-split train --host localhost --port 8000 --model OpenGVLab/InternVL2-2B --endpoint /v3/chat/completions --max-concurrency 1 --num-prompts 100 --trust-remote-code

Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: None
============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  287.81    
Total input tokens:                      15381     
Total generated tokens:                  20109     
Request throughput (req/s):              0.35       
Output token throughput (tok/s):         69.87     
Total Token throughput (tok/s):          123.31    
---------------Time to First Token----------------
Mean TTFT (ms):                          1513.96   
Median TTFT (ms):                        1368.93   
P99 TTFT (ms):                           2647.45   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          6.68      
Median TPOT (ms):                        6.68      
P99 TPOT (ms):                           8.02      
```

## Testing the model accuracy over serving API

Check the [guide of using lm-evaluation-harness](../accuracy/README.md)


## References
- [Chat Completions API](../../../docs/model_server_rest_api_chat.md)
- [Writing client code](../../../docs/clients_genai.md)
- [LLM calculator reference](../../../docs/llm/reference.md)
