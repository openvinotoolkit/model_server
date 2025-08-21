# Agentic AI with OpenVINO Model Server {#ovms_demos_continuous_batching_agent}

This demo version requires OVMS version 2025.3. Build it from [source](../../../docs/build_from_source.md) before it is published.

OpenVINO Model Server can be used to serve language models for AI Agents. It supports the usage of tools in the context of content generation.
It can be integrated with MCP servers and AI agent frameworks. 
You can learn more about [tools calling based on OpenAI API](https://platform.openai.com/docs/guides/function-calling?api-mode=responses)

Here are presented required steps to deploy language models trained for tools support. The diagram depicting the demo setup is below:
![picture](./agent.png)

The application employing OpenAI agent SDK is using MCP server. It is equipped with a set of tools to providing context for the content generation.
The tools can also be used for automation purposes based on input in text format.  



## Export LLM model
Currently supported models:
- Qwen/Qwen3-8B
- Qwen/Qwen3-4B
- meta-llama/Llama-3.1-8B-Instruct
- meta-llama/Llama-3.2-3B-Instruct
- NousResearch/Hermes-3-Llama-3.1-8B
- microsoft/Phi-4-mini-instruct
- mistralai/Mistral-7B-Instruct-v0.3

### Export using python script

Use those steps to convert the model from HugginFace Hub to OpenVINO format and export it to a local storage.

```console
# Download export script, install its dependencies and create directory for the models
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/2/demos/common/export_models/requirements.txt
mkdir models
```
Run `export_model.py` script to download and quantize the model:

> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" or "https://www.modelscope.cn/models" before running the export script to connect to the HF Hub.

::::{tab-set}
:::{tab-item} Qwen3-8B
:sync: Qwen3-8B
```console
python export_model.py text_generation --source_model Qwen/Qwen3-8B --weight-format int4 --config_file_path models/config.json --model_repository_path models --tool_parser hermes3 --cache_size 2
```
:::
:::{tab-item} Qwen3-4B
:sync: Qwen3-4B
```console
python export_model.py text_generation --source_model Qwen/Qwen3-4B --weight-format int4 --config_file_path models/config.json --model_repository_path models --tool_parser hermes3 --cache_size 2
```
:::
:::{tab-item} Llama-3.1-8B-Instruct
:sync: Llama-3.1-8B-Instruct
```console
python export_model.py text_generation --source_model meta-llama/Llama-3.1-8B-Instruct --weight-format int4 --config_file_path models/config.json --model_repository_path models --tool_parser llama3 --cache_size 2
curl -L -o models/meta-llama/Llama-3.1-8B-Instruct/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_llama3.1_json.jinja
```
:::
:::{tab-item} Llama-3.2-3B-Instruct
:sync: Llama-3.2-3B-Instruct
```console
python export_model.py text_generation --source_model meta-llama/Llama-3.2-3B-Instruct --weight-format int4 --config_file_path models/config.json --model_repository_path models --tool_parser llama3 --cache_size 2
curl -L -o models/meta-llama/Llama-3.2-3B-Instruct/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_llama3.2_json.jinja
```
:::
:::{tab-item} Hermes-3-Llama-3.1-8B
:sync: Hermes-3-Llama-3.1-8B
```console
python export_model.py text_generation --source_model NousResearch/Hermes-3-Llama-3.1-8B --weight-format int4 --config_file_path models/config.json --model_repository_path models --tool_parser hermes3 --cache_size 2
curl -L -o models/NousResearch/Hermes-3-Llama-3.1-8B/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_hermes.jinja
```
:::
:::{tab-item} Mistral-7B-Instruct-v0.3
:sync: Mistral-7B-Instruct-v0.3
```console
python export_model.py text_generation --source_model mistralai/Mistral-7B-Instruct-v0.3 --weight-format int4 --config_file_path models/config.json --model_repository_path models --tool_parser mistral --cache_size 2 --extra_quantization_params "--task text-generation-with-past"
curl -L -o models/mistralai/Mistral-7B-Instruct-v0.3/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_mistral.jinja
```
:::
:::{tab-item} Phi-4-mini-instruct 
:sync: microsoft/Phi-4-mini-instruct 
```console
python export_model.py text_generation --source_model microsoft/Phi-4-mini-instruct --weight-format int4 --config_file_path models/config.json --model_repository_path models --tool_parser phi4 --cache_size 2
curl -L -o models/microsoft/Phi-4-mini-instruct/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_phi4_mini.jinja
```
:::
::::

### Direct pulling of pre-configured HuggingFace models from docker containers

This procedure can be used to pull preconfigured models from OpenVINO organization in HuggingFace Hub
::::{tab-set}
:::{tab-item} Qwen3-8B-int4-ov
:sync: Qwen3-8B-int4-ov
```bash
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/Qwen3-8B-int4-ov --task text_generation --tool_parser hermes3
```
:::
:::{tab-item} Mistral-7B-Instruct-v0.3-int4-ov
:sync: Mistral-7B-Instruct-v0.3-int4-ov
```bash
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov --task text_generation --tool_parser mistral
```
:::
:::{tab-item} Phi-4-mini-instruct-int4-ov
:sync: Phi-4-mini-instruct-int4-ov
```bash
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/Phi-4-mini-instruct-int4-ov --task text_generation --tool_parser phi4
curl -L -o models/OpenVINO/Phi-4-mini-instruct-int4-ov/chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_phi4_mini.jinja
```
:::
::::


### Direct pulling of pre-configured HuggingFace models on Windows

Assuming you have unpacked model server package with python enabled version, make sure to run `setupvars` script
as mentioned in [deployment guide](../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.

::::{tab-set}
:::{tab-item} Qwen3-8B-int4-ov
:sync: Qwen3-8B-int4-ov
```bat
ovms.exe --pull --model_repository_path models --source_model OpenVINO/Qwen3-8B-int4-ov --task text_generation --tool_parser hermes3
```
:::
:::{tab-item} Mistral-7B-Instruct-v0.3-int4-ov
:sync: Mistral-7B-Instruct-v0.3-int4-ov
```bat
ovms.exe --pull --model_repository_path models --source_model OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov --task text_generation --tool_parser mistral
```
:::
:::{tab-item} Phi-4-mini-instruct-int4-ov
:sync: Phi-4-mini-instruct-int4-ov
```bash
ovms.exe --pull --model_repository_path models --source_model OpenVINO/Phi-4-mini-instruct-int4-ov --task text_generation --tool_parser phi4
curl -L -o models\microsoft\Phi-4-mini-instruct\chat_template.jinja https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.9.0/examples/tool_chat_template_phi4_mini.jinja
```
:::
::::

You can use similar commands for different models. Change the source_model and the weights-format. 
> **Note:** Some models give more reliable responses with tuned chat template.

> **Note:** Currently are supported models with tools call format compatible with phi4, llama3, mistral and hermes3.



## Start OVMS

This deployment procedure assumes the model was pulled or exported using the procedure above. The exception are models from OpenVINO organization if they support tools correctly with the default template like "OpenVINO/Qwen3-8B-int4-ov" - they can be deployed in a single command pulling and staring the server.


### Deploying on Windows with GPU
Assuming you have unpacked model server package with python enabled version, make sure to run `setupvars` script
as mentioned in [deployment guide](../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.

::::{tab-set}
:::{tab-item} Qwen3-8B
:sync: Qwen3-8B
```bat
ovms.exe --rest_port 8000 --model_path models/Qwen/Qwen3-8B --model_name Qwen/Qwen3-8B --tool_parser hermes3 --target_device GPU --cache_size 2 --task text_generation
```
:::
:::{tab-item} Qwen3-4B
:sync: Qwen3-4B
```bat
ovms.exe --rest_port 8000 --model_path models/Qwen/Qwen3-4B --model_name Qwen/Qwen3-4B --tool_parser hermes3 --target_device GPU --cache_size 2 --task text_generation
```
:::
:::{tab-item} Llama-3.1-8B-Instruct
:sync: Llama-3.1-8B-Instruct
```bat
ovms.exe --rest_port 8000 --model_path models/meta-llama/Llama-3.1-8B-Instruct --model_name meta-llama/Llama-3.1-8B-Instruct --tool_parser llama3 --target_device GPU --cache_size 2 --task text_generation
```
:::
:::{tab-item} Llama-3.2-3B-Instruct
:sync: Llama-3.2-3B-Instruct
```bat
ovms.exe --rest_port 8000 --model_path models/meta-llama/Llama-3.2-3B-Instruct --model_name meta-llama/Llama-3.2-3B-Instruct --tool_parser llama3 --target_device GPU --cache_size 2 --task text_generation
```
:::
:::{tab-item} Mistral-7B-Instruct-v0.3
:sync: Mistral-7B-Instruct-v0.3
```bat
ovms.exe --rest_port 8000 --model_path models/mistralai/Mistral-7B-Instruct-v0.3 --model_name mistralai/Mistral-7B-Instruct-v0.3 --tool_parser mistral --target_device GPU --cache_size 2 --task text_generation
```
:::
:::{tab-item} Phi-4-mini-instruct
:sync: Phi-4-mini-instruct
```bat
ovms.exe --rest_port 8000 --model_path models/microsoft/Phi-4-mini-instruct --model_name microsoft/Phi-4-mini-instruct --tool_parser phi4 --target_device GPU --cache_size 2 --task text_generation
```
:::
:::{tab-item} Qwen3-8B-int4-ov
:sync: Qwen3-8B-int4-ov
```bat
ovms.exe --rest_port 8000 --model_path models/OpenVINO/Qwen3-8B-int4-ov --model_name OpenVINO/Qwen3-8B-int4-ov --tool_parser hermes3 --target_device GPU --cache_size 2 --task text_generation
```
:::
:::{tab-item} Mistral-7B-Instruct-v0.3-int4-ov
:sync: Mistral-7B-Instruct-v0.3-int4-ov
```bat
ovms.exe --rest_port 8000 --model_path models/OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov --model_name OpenVINO/Phi-4-mini-instruct-int4-ov --tool_parser mistral --target_device GPU --cache_size 2 --task text_generation
```
:::
:::{tab-item} Phi-4-mini-instruct-int4-ov
:sync: Phi-4-mini-instruct-int4-ov
```bat
ovms.exe --rest_port 8000 --model_path models/OpenVINO/Phi-4-mini-instruct-int4-ov --model_name OpenVINO/Phi-4-mini-instruct-int4-ov --tool_parser phi4 --target_device GPU --cache_size 2 --task text_generation
```
:::
::::


### Deploying in a docker container on CPU

::::{tab-set}
:::{tab-item} Qwen3-8B
:sync: Qwen3-8B
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model Qwen/Qwen3-8B --tool_parser hermes3 --cache_size 2 --task text_generation
```
:::
:::{tab-item} Qwen3-8B
:sync: Qwen3-4B
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model Qwen/Qwen3-4B --tool_parser hermes3 --cache_size 2 --task text_generation
```
:::
:::{tab-item} Llama-3.1-8B-Instruct
:sync: Qwen3-4B
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model meta-llama/Llama-3.1-8B-Instruct --tool_parser llama3 --cache_size 2 --task text_generation
```
:::
:::{tab-item} Llama-3.2-3B-Instruct
:sync: Qwen3-4B
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model meta-llama/Llama-3.2-3B-Instruct --tool_parser llama3 --cache_size 2 --task text_generation
```
:::
:::{tab-item} Hermes-3-Llama-3.1-8B
:sync: Hermes-3-Llama-3.1-8B
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model NousResearch/Hermes-3-Llama-3.1-8B --tool_parser hermes3 --cache_size 2 --task text_generation
```
:::
:::{tab-item} Mistral-7B-Instruct-v0.3
:sync: Mistral-7B-Instruct-v0.3
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000  -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model mistralai/Mistral-7B-Instruct-v0.3 --tool_parser mistral --cache_size 2 --task text_generation
```
:::
:::{tab-item} Phi-4-mini-instruct
:sync: Phi-4-mini-instruct
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000  -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model microsoft/Phi-4-mini-instruct --tool_parser phi4 --cache_size 2 --task text_generation
```
:::
:::{tab-item} Qwen3-8B-int4-ov
:sync: Qwen3-8B-int4-ov
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model OpenVINO/Qwen3-8B-int4-ov --tool_parser hermes3 --cache_size 2 --task text_generation
```
:::
:::{tab-item} Mistral-7B-Instruct-v0.3-int4-ov
:sync: Mistral-7B-Instruct-v0.3-int4-ov
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov --tool_parser mistral --cache_size 2 --task text_generation
```
:::
:::{tab-item} Phi-4-mini-instruct-int4-ov
:sync: Phi-4-mini-instruct-int4-ov
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8000 --model_repository_path models --source_model OpenVINO/Phi-4-mini-instruct-int4-ov --tool_parser phi4 --cache_size 2 --task text_generation
```
:::
::::


### Deploying in a docker container on GPU

In case you want to use GPU device to run the generation, add extra docker parameters `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)`
to `docker run` command, use the image with GPU support. Export the models with precision matching the GPU capacity and adjust pipeline configuration.
It can be applied using the commands below:

::::{tab-set}
:::{tab-item} Qwen3-8B
:sync: Qwen3-8B
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) openvino/model_server:latest-gpu \
--rest_port 8000 --model_repository_path models --source_model Qwen/Qwen3-8B --tool_parser hermes3 --target_device GPU --cache_size 2 --task text_generation
```
:::
:::{tab-item} Qwen3-4B
:sync: Qwen3-4B
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) openvino/model_server:latest-gpu \
--rest_port 8000 --model_repository_path models --source_model Qwen/Qwen3-4B --tool_parser hermes3 --target_device GPU --cache_size 2 --task text_generation
```
:::
:::{tab-item} Llama-3.1-8B-Instruct
:sync: Llama-3.1-8B-Instruct
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) openvino/model_server:latest-gpu \
--rest_port 8000 --model_repository_path models --source_model meta-llama/Llama-3.1-8B-Instruct --tool_parser llama3 --target_device GPU --cache_size 2 --task text_generation
```
:::
:::{tab-item} Llama-3.2-3B-Instruct
:sync: Llama-3.2-3B-Instruct
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) openvino/model_server:latest-gpu \
--rest_port 8000 --model_repository_path models --source_model meta-llama/Llama-3.2-3B-Instruct --tool_parser llama3 --target_device GPU --cache_size 2 --task text_generation
```
:::
:::{tab-item} Hermes-3-Llama-3.1-8B
:sync: Hermes-3-Llama-3.1-8B
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) openvino/model_server:latest-gpu \
--rest_port 8000 --model_repository_path models --source_model NousResearch/Hermes-3-Llama-3.1-8B --tool_parser llama3 --target_device GPU --cache_size 2 --task text_generation
```
:::
:::{tab-item} Mistral-7B-Instruct-v0.3
:sync: Mistral-7B-Instruct-v0.3
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000  -v $(pwd)/models:/models --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) openvino/model_server:latest-gpu \
--rest_port 8000 --model_repository_path models --source_model mistralai/Mistral-7B-Instruct-v0.3 --tool_parser mistral --target_device GPU --cache_size 2 --task text_generation
```
:::
:::{tab-item} Phi-4-mini-instruct
:sync: Phi-4-mini-instruct
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000  -v $(pwd)/models:/models --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) openvino/model_server:latest-gpu \
--rest_port 8000 --model_repository_path models --source_model microsoft/Phi-4-mini-instruct --tool_parser phi4 --target_device GPU --cache_size 2 --task text_generation
```
:::
:::{tab-item} Qwen3-8B-int4-ov
:sync: Qwen3-8B-int4-ov
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) openvino/model_server:latest-gpu \
--rest_port 8000 --model_repository_path models --source_model OpenVINO/Qwen3-8B-int4-ov --tool_parser hermes3 --target_device GPU --cache_size 2 --task text_generation
```
:::
:::{tab-item} Mistral-7B-Instruct-v0.3-int4-ov
:sync: Mistral-7B-Instruct-v0.3-int4-ov
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) openvino/model_server:latest-gpu \
--rest_port 8000 --model_repository_path models --source_model OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov --tool_parser phi4 --target_device GPU --cache_size 2 --task text_generation
```
:::
:::{tab-item} Phi-4-mini-instruct-int4-ov
:sync: Phi-4-mini-instruct-int4-ov
```bash
docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000 -v $(pwd)/models:/models --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) openvino/model_server:latest-gpu \
--rest_port 8000 --model_repository_path models --source_model OpenVINO/Phi-4-mini-instruct-int4-ov --tool_parser phi4 --target_device GPU --cache_size 2 --task text_generation
```
:::
::::

### Deploy all models in a single container
Those steps deploy all the models exported earlier. The python script added the models to `models/config.json` so just the remaining models pulled directly from HuggingFace Hub are to be added:
```bash
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --add_to_config /models --model_name OpenVINO/Qwen3-8B-int4-ov --model_path OpenVINO/Qwen3-8B-int4-ov
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --add_to_config /models --model_name OpenVINO/Phi-4-mini-instruct-int4-ov  --model_path OpenVINO/Phi-4-mini-instruct-int4-ov
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --add_to_config /models --model_name OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov  --model_path OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/models:ro openvino/model_server:latest --rest_port 8000 --config_path /models/config.json
```


## Start MCP server with SSE interface

### Linux
```bash
git clone https://github.com/isdaniel/mcp_weather_server
cd mcp_weather_server
docker build . -t mcp_weather_server
docker run -d -v $(pwd)/src/mcp_weather_server:/mcp_weather_server  -p 8080:8080 mcp_weather_server bash -c ". .venv/bin/activate ; python /mcp_weather_server/server-see.py"
```

> **Note:** On Windows the MCP server will be demonstrated as an instance with stdio interface inside the agent application

## Start the agent

Install the application requirements

```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/continuous_batching/agentic_ai/openai_agent.py -o openai_agent.py
pip install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/continuous_batching/agentic_ai/requirements.txt
```
Make sure nodejs and npx are installed. On ubuntu it would require `sudo apt install nodejs npm`. On windows, visit https://nodejs.org/en/download. It is needed for the `file system` MCP server.

Run the agentic application:


::::{tab-set}
:::{tab-item} Qwen3-8B
:sync: Qwen3-8B
```bash
python openai_agent.py --query "What is the current weather in Tokyo?" --model Qwen/Qwen3-8B --base-url http://localhost:8000/v3 --mcp-server-url http://localhost:8080/sse --mcp-server all --stream --enable-thinking
```
```bash
python openai_agent.py --query "List the files in folder /root" --model Qwen/Qwen3-8B --base-url http://localhost:8000/v3 --mcp-server-url http://localhost:8080/sse --mcp-server all
```
:::
:::{tab-item} Qwen3-4B 
:sync: Qwen3-4B
```bash
python openai_agent.py --query "What is the current weather in Tokyo?" --model Qwen/Qwen3-4B --base-url http://localhost:8000/v3 --mcp-server-url http://localhost:8080/sse --mcp-server all --stream
```
```bash
python openai_agent.py --query "List the files in folder /root" --model Qwen/Qwen3-4B --base-url http://localhost:8000/v3 --mcp-server-url http://localhost:8080/sse --mcp-server all
```
:::
:::{tab-item} Llama-3.1-8B-Instruct
:sync: Llama-3.1-8B-Instruct
```bash
python openai_agent.py --query "List the files in folder /root" --model meta-llama/Llama-3.1-8B-Instruct --base-url http://localhost:8000/v3 --mcp-server-url http://localhost:8080/sse --mcp-server all
```
:::
:::{tab-item} Mistral-7B-Instruct-v0.3
:sync: Mistral-7B-Instruct-v0.3
```bash
python openai_agent.py --query "List the files in folder /root" --model mistralai/Mistral-7B-Instruct-v0.3 --base-url http://localhost:8000/v3 --mcp-server-url http://localhost:8080/sse --mcp-server weather
```
:::
:::{tab-item} Llama-3.2-3B-Instruct
:sync: Llama-3.2-3B-Instruct
```bash
python openai_agent.py --query "List the files in folder /root" --model meta-llama/Llama-3.2-3B-Instruct --base-url http://localhost:8000/v3 --mcp-server-url http://localhost:8080/sse --mcp-server weather
```
:::
:::{tab-item} Phi-4-mini-instruct
:sync: Phi-4-mini-instruct
```console
python openai_agent.py --query "What is the current weather in Tokyo?" --model microsoft/Phi-4-mini-instruct --base-url http://localhost:8000/v3 --mcp-server-url http://localhost:8080/sse --mcp-server weather
```
:::
:::{tab-item} Qwen3-8B-int4-ov
:sync: Qwen3-8B-int4-ov
```bash
python openai_agent.py --query "What is the current weather in Tokyo?" --model OpenVINO/Qwen3-8B-int4-ov --base-url http://localhost:8000/v3 --mcp-server-url http://localhost:8080/sse --mcp-server weather
```
:::
:::{tab-item} OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov
:sync: OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov
```bash
python openai_agent.py --query "What is the current weather in Tokyo?" --model OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov --base-url http://localhost:8000/v3 --mcp-server-url http://localhost:8080/sse --mcp-server weather --tool_choice required
```
:::
:::{tab-item} Phi-4-mini-instruct-int4-ov
:sync: Phi-4-mini-instruct-int4-ov
```bash
python openai_agent.py --query "What is the current weather in Tokyo?" --model OpenVINO/Phi-4-mini-instruct-int4-ov --base-url http://localhost:8000/v3 --mcp-server-url http://localhost:8080/sse --mcp-server weather --tool_choice required
```
:::
::::

> **Note:** The tool checking the weather forecast in the demo is making a remote call to a REST API server. Make sure you have internet connection and proxy configured while running the agent. 

> **Note:**  For more interactive mode you can run the application with streaming enabled by providing `--stream` parameter to the script. Currently streaming is enabled models using `hermes3` tool parser.