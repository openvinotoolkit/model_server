# Agentic AI with OVMS

OpenVINO Model Server can be used to serve as a language model for AI Agents. It supports the usage of tools in the context of content generation.
It can be integrated with MCP servers and AI agent frameworks. 
You can learn more about [tools calling based on OpenAI API](https://platform.openai.com/docs/guides/function-calling?api-mode=responses)

Here are presented required steps to deploy language models trained for tools support. The diagram depicting the demo setup is below:
![picture](./agent.png)

The application employing OpenAI agent SDK is using MCP server. It is equiped with a set of tools to providing context for the content generation.
The tools can also be used for automation purposes based on input in text format.  

## Export LLM model
Currently supported models:
- microsoft/Phi-4-mini-instruct
- meta-llama/Llama-3.1-8B-Instruct
- NousResearch/Hermes-3-Llama-3.1-8B
- Qwen/Qwen3-8B
The model chat template defines how the conversation with tools and tools schema should be embedded in the prompt. 
The model response with tool call follow a specific syntax which is process by a response parser. The export tool allows choosing which template and output parset should be applied.

Download export script, install it's dependencies and create directory for the models:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/requirements.txt
mkdir models
```
Run `export_model.py` script to download and quantize the model:

> **Note:** The users in China need to set environment variable HF_ENDPOINT="https://hf-mirror.com" before running the export script to connect to the HF Hub.

::::{tab-set}

:::{tab-item} CPU
```console
python export_model.py text_generation --source_model Qwen/Qwen3-8B --weight-format int8 --config_file_path models/config.json --model_repository_path models --tools_model_type qwen3 --overwrite_models
```
:::

:::{tab-item} GPU
```console
python export_model.py text_generation --source_model Qwen/Qwen3-8B --weight-format int4 --config_file_path models/config.json --model_repository_path models --tools_model_type qwen3 --target_device GPU
```
```
:::

:::{tab-item} NPU
```console
python export_model.py text_generation --source_model Qwen/Qwen3-8B --config_file_path models/config.json --model_repository_path models --tools_model_type qwen3 --overwrite_models --target_device GPU
```
:::

::::

You can use similar commands for different models. Change the source_model and the tools_model_type (note that as of today the following types as available: `[phi4, llama3, qwen3, hermes3]`).
> **Note:** The tuned chat template will be copied to the model folder as template.jinja and the response parser will be set in the graph.pbtxt


## Start OVMS

Starting the model server is identical like with other demos with generative endpoints:
**Deploying with Docker**

Select deployment option depending on how you prepared models in the previous step.

::::{tab-set}

:::{tab-item} CPU

Running this command starts the container with CPU only target device:
```bash
docker run -d --rm -p 8000:8000 -v $(pwd)/models:/models:ro openvino/model_server:2025.2 --rest_port 8000 --model_path /models/Qwen/Qwen3-8B --model_name Qwen/Qwen3-8B
```
:::

:::{tab-item} GPU

In case you want to use GPU device to run the generation, add extra docker parameters `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)`
to `docker run` command, use the image with GPU support. Export the models with precision matching the GPU capacity and adjust pipeline configuration.
It can be applied using the commands below:
```bash
docker run -d --rm -p 8000:8000 --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -v $(pwd)/models:/models:ro openvino/model_server:2025.2-gpu \
--rest_port 8000 --model_path /models/Qwen/Qwen3-8B --model_name Qwen/Qwen3-8B
```
:::
:::{tab-item} NPU

Running this command starts the container with NPU enabled:
```bash
docker run -d --rm --device /dev/accel --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
-p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --rest_port 8000 --model_path /models/Qwen/Qwen3-8B --model_name Qwen/Qwen3-8B
```
:::

::::


**Deploying on Bare Metal**

Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.

Depending on how you prepared models in the first step of this demo, they are deployed to either CPU, GPU or NPU (it's defined in `graph.pbtxt`). If you run on GPU or NPU, make sure to have appropriate drivers installed, so the device is accessible for the model server.

```bat
ovms --rest_port 8000 --model_path models/Qwen/Qwen3-8B --model_name Qwen/Qwen3-8B
```


## Start MCP server with SSE interface

### Linux
```bash
git clone https://github.com/isdaniel/mcp_weather_server
cd mcp_weather_server
docker build . -t mcp_weather_server
docker run -d -v $(pwd)/src/mcp_weather_server:/mcp_weather_server  -p 8080:8080 mcp_weather_server bash -c ". .venv/bin/activate ; python /mcp_weather_server/server-see.py"
```

> **Note:** On Windows the MCP server will be demonstrated as an instance with stdip interface inside the agent application

## Start the agent

Install the application requirements

```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/agentic-demo/demos/continuous_batching/agentic_ai/openai_agent.py -o openai_agent.py
pip install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/agentic-demo/demos/continuous_batching/agentic_ai/requirements.txt
```

Run the agentic application
```console
python openai_agent.py --query "What is the weather now in Tokyo?"

python openai_agent.py --query "What is the content of my local folder?"

```

> **Note:** The tool checking the weather forecast in the demo is making a remote call to a REST API server. Make sure you have internet connection and proxy configured while running the agent. 