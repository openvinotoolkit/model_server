# Text generation serving with NPU acceleration {#ovms_demos_llm_npu}


This demo shows how to deploy LLM models in the OpenVINO Model Server with NPU acceleration.
From the client perspective it is very similar to the [generative model deployment with continuous batching](../continuous_batching/README.md)
Likewise it exposes the models via OpenAI API `chat/completions` and `completions` endpoints.
The difference is that it doesn't support request batching. They can be sent concurrently but they are processed sequentially.
It is targeted on client machines equipped with NPU accelerator.

> **Note:** This demo was tested on MeteorLake, LunarLake, ArrowLake platforms on Windows11 and Ubuntu24.

## Prerequisites

**Model preparation**: Python 3.12 or higher with pip and HuggingFace account

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../docs/deploying_server_baremetal.md)

**(Optional) Client**: git and Python for using OpenAI client package

## Model preparation - using ovms
Another way to downlaod models is using `--pull` parameter with ovms command.

There are multiple [OpenVINO models](https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu) recommended to use with NPU:
- OpenVINO/Qwen3-8B-int4-cw-ov
- OpenVINO/Phi-3.5-mini-instruct-int4-cw-ov
- OpenVINO/Mistral-7B-Instruct-v0.2-int4-cw-ov
- OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov
- OpenVINO/falcon-7b-instruct-int4-cw-ov
- OpenVINO/gpt-j-6b-int4-cw-ov


### Pulling model

:::{tab-set}
::{tab-item} Linux
:sync: Linux
```bash
docker run -d --rm -u $(id -u):$(id -g) -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --pull --source_model OpenVINO/Qwen3-8B-int4-cw-ov --model_repository_path /models --target_device NPU --task text_generation --tool_parser hermes3 --cache_dir .ov_cache --enable_prefix_caching true --max_prompt_len 4000
docker run -d --rm -u $(id -u):$(id -g) -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --add_to_config --config_path /models/config.json --model_name OpenVINO/Qwen3-8B-int4-cw-ov --model_path /models/OpenVINO/Qwen3-8B-int4-cw-ov
```
:: 
::{tab-item} Windows
:sync: Windows
```bat
ovms.exe --pull --source_model OpenVINO/Qwen3-8B-int4-cw-ov --model_repository_path models --target_device NPU --task text_generation --tool_parser hermes3 --cache_dir .ov_cache --enable_prefix_caching true --max_prompt_len 4000 
ovms.exe --add_to_config --config_path models\config.json --model_name OpenVINO/Qwen3-8B-int4-cw-ov --model_path OpenVINO\Qwen3-8B-int4-cw-ov
```
::
:::

## Model preparation - using export model script
Here, the original Pytorch LLM model and the tokenizer will be converted to IR format and optionally quantized.
That ensures faster initialization time, better performance and lower memory consumption.
LLM engine parameters will be defined inside the `graph.pbtxt` file.

Download export script, install it's dependencies and create directory for the models:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/requirements.txt
mkdir models
```

Run `export_model.py` script to download and quantize the model:

> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.

**LLM**
```console
python export_model.py text_generation --source_model meta-llama/Llama-3.1-8B-Instruct --target_device NPU --config_file_path models/config.json --cache_dir ./models/.ov_cache --model_repository_path models --overwrite_models
```
**Note:** The parameter `--cache_dir` stores the model compilation cache to speedup initialization time for sequential startup. Drop this parameter if you don't want to store the compilation cache.

Below is a list of tested models:
- Qwen/Qwen3-8B
- meta/LLama3.1-8B-instruct
- mistralai/Mistral-7B-instruct-v0.3
- NousResearch/Hermes-3-Llama-3.1-8B
- microsoft/Phi-3-mini-4k-instruct

The default configuration should work in most cases but the parameters can be tuned via `export_model.py` script arguments. 
Note that by default, NPU sets limitation on the prompt length to 1024 tokens. You can modify that limit by using `--max_prompt_len` parameter.
Run the script with `--help` argument to check available parameters and see the [LLM calculator documentation](../../docs/llm/reference.md) to learn more about configuration options.

## Server Deployment

:::{dropdown} **Deploying with Docker**

Running this command starts the container with NPU enabled:
```bash
docker run -d --rm --device /dev/accel --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
-p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --rest_port 8000 --config_path /models/config.json
```
:::

:::{dropdown} **Deploying on Bare Metal**

Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.

Depending on how you prepared models in the first step of this demo, they are deployed to either CPU or GPU (it's defined in `config.json`). If you run on GPU make sure to have appropriate drivers installed, so the device is accessible for the model server.

```bat
ovms --rest_port 8000 --config_path models\config.json
```
:::

## Readiness Check

Wait for the model to load. You can check the status with a simple command:
```console
curl http://localhost:8000/v1/config
```
```json
{
  "OpenVINO/Qwen3-8B-int4-cw-ov": {
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

A single servable exposes both `chat/completions` and `completions` endpoints with and without stream capabilities.
Chat endpoint is expected to be used for scenarios where conversation context should be pasted by the client and the model prompt is created by the server based on the jinja model template.
Completion endpoint should be used to pass the prompt directly by the client and for models without the jinja template.

:::{dropdown} **Unary call with cURL**
```console
curl http://localhost:8000/v3/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"OpenVINO/Qwen3-8B-int4-cw-ov\", \"max_tokens\":50, \"stream\":false, \"chat_template_kwargs\":{\"enable_thinking\":false}, \"messages\": [{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},{\"role\": \"user\",\"content\": \"What is OpenVINO Model Server?\"}]}"
```
```json
{
   "choices":[
      {
         "finish_reason":"stop",
         "index":0,
         "message":{
            "content":"**OpenVINO Model Server** (also known as **Model Server** or **OVMS**) is a high-performance, open-source inference server that allows you to deploy and serve deep learning models as RESTful or gRPC endpoints. It is part",
            "role":"assistant",
            "tool_calls":[
               
            ]
         }
      }
   ],
   "created":1763718082,
   "model":"OpenVINO/Qwen3-8B-int4-cw-ov",
   "object":"chat.completion",
   "usage":{
      "prompt_tokens":31,
      "completion_tokens":50,
      "total_tokens":81
   }
}
```

A similar call can be made with a `completion` endpoint:
```console
curl http://localhost:8000/v3/completions -H "Content-Type: application/json" -d "{\"model\": \"OpenVINO/Qwen3-8B-int4-cw-ov\", \"max_tokens\":50, \"stream\":false, \"prompt\": \"What are the 3 main tourist attractions in Paris?\"}"
```
```json
{
   "choices":[
      {
         "finish_reason":"stop",
         "index":0,
         "text":" The three main tourist attractions in Paris are the Eiffel Tower, the Louvre, and the Notre-Dame de Paris. The Eiffel Tower is one of the most iconic landmarks in Paris and is a must-see for most visitors."
      }
   ],
   "created":1763976213,
   "model":"OpenVINO/Qwen3-8B-int4-cw-ov",
   "object":"text_completion",
   "usage":{
      "prompt_tokens":11,
      "completion_tokens":50,
      "total_tokens":61
   }
}
```

:::

:::{dropdown} **Unary call with OpenAI Python package**

The endpoints `chat/completions` are compatible with OpenAI client so it can be easily used to generate code also in streaming mode:

Install the client library:
```console
pip3 install openai
```
```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

response = client.chat.completions.create(
    model="OpenVINO/Qwen3-8B-int4-cw-ov",
    messages=[{"role": "user", "content": "What is OpenVINO Model Server?"}],
    max_tokens=100,
    stream=False,
    extra_body={"chat_template_kwargs":{"enable_thinking": False}}
)
print(response.choices[0].message.content)
```

Output:
```
**OpenVINO™ Model Server** is a high-performance, open-source inference server that allows you to deploy and serve deep learning models as a RESTful API. It is part of the **Intel® OpenVINO™ toolkit**, which is a comprehensive development toolkit for optimizing and deploying deep learning models on Intel®-based hardware.

---

## ✅ What is OpenVINO Model Server?

The **OpenVINO Model Server** is a **lightweight**, **highly optimized** and ...
```

A similar code can be applied for the completion endpoint:
```console
pip3 install openai
```
```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

response = client.completions.create(
    model="OpenVINO/Qwen3-8B-int4-cw-ov",
    prompt="What are the 3 main tourist attractions in Paris?",
    max_tokens=100,
    stream=False,
)
print(response.choices[0].text)
```

Output:
```
The three main tourist attractions in Paris are the Eiffel Tower, the Louvre Museum, and the Notre-Dame de Paris. The Eiffel Tower is a symbol of Paris and one of the most visited landmarks in the world. The Louvre Museum is home to the Mona Lisa and other famous artworks. The Notre-Dame de Paris is a famous cathedral and a symbol of the city's rich history and architecture. These three attractions are the most popular among tourists visiting Paris.
```
:::

:::{dropdown} **Streaming call with OpenAI Python package**

The endpoints `chat/completions` are compatible with OpenAI client so it can be easily used to generate code also in streaming mode:

Install the client library:
```console
pip3 install openai
```
```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

stream = client.chat.completions.create(
    model="OpenVINO/Qwen3-8B-int4-cw-ov",
    messages=[{"role": "user", "content": "What is OpenVINO Model Server?"}],
    max_tokens=100,
    stream=True,
    extra_body={"chat_template_kwargs":{"enable_thinking": False}}
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

Output:
```
**OpenVINO™ Model Server** (formerly known as **OpenVINO™ Toolkit Model Server**) is a high-performance, open-source server that allows you to deploy and serve deep learning models in a production environment. It is part of the **Intel® OpenVINO™ Toolkit**, which is designed to optimize and deploy deep learning models for inference on Intel hardware.

---

## 📌 What is OpenVINO Model Server?

The **OpenVINO Model Server** is a **lightweight**...
```

A similar code can be applied for the completion endpoint:
```console
pip3 install openai
```
```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

stream = client.completions.create(
    model="OpenVINO/Qwen3-8B-int4-cw-ov",
    prompt="What are the 3 main tourist attractions in Paris?",
    max_tokens=100,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].text is not None:
        print(chunk.choices[0].text, end="", flush=True)
```

Output:
```
The three main tourist attractions in Paris are the Eiffel Tower, the Louvre, and the Notre-Dame de Paris. The Eiffel Tower is the most iconic landmark and offers a great view of the city. The Louvre is a world-famous art museum that houses the Mona Lisa and other famous artworks. The Notre-Dame de Paris is a stunning example of French Gothic architecture and is the cathedral of the city. These three attractions are the most visited and most famous in Paris,
```
:::

## Testing the model accuracy over serving API

Check the [guide of using lm-evaluation-harness](https://github.com/openvinotoolkit/model_server/blob/main/demos/continuous_batching/accuracy/README.md)

> **Note:** Text generation on NPU is not returning the log_probs which are required to calculate some of the metrics. Only the tasks of type `generate_until` can be used.
For example `--tasks leaderboard_ifeval`.


## Limitations

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
