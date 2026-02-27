# Visual Studio Code Local Assistant {#ovms_demos_code_completion_vsc}

## Intro
With the rise of AI PC capabilities, hosting own Visual Studio code assistant is at your reach. In this demo, we will showcase how to deploy local LLM serving with OVMS and integrate it with Continue extension. It will employ GPU acceleration.

# Requirements
- Windows (for standalone app) or Linux (using Docker)
- Python installed (for model preparation only)
- Intel Meteor Lake, Lunar Lake, Arrow Lake or Panter Lake. 
- Memory requirements depend on the model size

### Windows: deploying on bare metal

::::{tab-set}
:::{tab-item} OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4
:sync: OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4
```bash
mkdir c:\models
set MOE_USE_MICRO_GEMM_PREFILL=0  # temporary workaround to improve accuracy with long context
ovms --model_repository_path c:\models --source_model OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4 --task text_generation --target_device GPU --tool_parser qwen3coder --rest_port 8000 --cache_dir .ovcache --model_name Qwen3-Coder-30B-A3B-Instruct
```
> **Note:** For deployment, the model requires ~16GB disk space and recommended 19GB+ of VRAM on the GPU.
:::

:::{tab-item} OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int8
:sync: OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int8
```bash
mkdir c:\models
set MOE_USE_MICRO_GEMM_PREFILL=0  # temporary workaround to improve accuracy with long context
ovms --model_repository_path c:\models --source_model OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4 --task text_generation --target_device GPU --tool_parser qwen3coder --rest_port 8000 --cache_dir .ovcache --model_name Qwen3-Coder-30B-A3B-Instruct
```
> **Note:** For deployment, the model requires ~16GB disk space and recommended 34GB+ of VRAM on the GPU.
:::

:::{tab-item} OpenVINO/gpt-oss-20B-int4
:sync: OpenVINO/gpt-oss-20B-int4
```bash
mkdir c:\models
ovms --model_repository_path c:\models --source_model OpenVINO/gpt-oss-20B-int4 --task text_generation --target_device GPU --tool_parser gptoss --reasoning_parser gptoss --rest_port 8000 --cache_dir .ovcache --model_name gpt-oss-20B
```
> **Note:** For deployment, the model requires ~12GB disk space and recommended 16GB+ of VRAM on the GPU.
:::

:::{tab-item} OpenVINO/gpt-oss-20B-int8
:sync: OpenVINO/gpt-oss-20B-int8
```bash
mkdir c:\models
ovms --model_repository_path c:\models --source_model OpenVINO/gpt-oss-20B-int8 --task text_generation --target_device GPU --tool_parser gptoss --reasoning_parser gptoss --rest_port 8000 --cache_dir .ovcache --model_name gpt-oss-20B
```
> **Note:** For deployment, the model requires ~24GB disk space and recommended 28GB+ of VRAM on the GPU.
:::

:::{tab-item} OpenVINO/Qwen3-8B-int4-ov
:sync: OpenVINO/Qwen3-8B-int4-ov
```bash
mkdir c:\models
ovms --model_repository_path c:\models --source_model OpenVINO/Qwen3-8B-int4-ov --task text_generation --target_device GPU --tool_parser hermes3 --reasoning_parser qwen3 --rest_port 8000 --cache_dir .ovcache --model_name Qwen3-8B
```
> **Note:** For deployment, the model requires ~4GB disk space and recommended 6GB+ of VRAM on the GPU.
:::
:::{tab-item} OpenVINO/Qwen3-8B-int4-cw-ov
:sync: OpenVINO/Qwen3-8B-int4-cw-ov
```bash
mkdir c:\models
ovms --model_repository_path c:\models --source_model OpenVINO/Qwen3-8B-int4-cw-ov --task text_generation --target_device NPU --tool_parser hermes3 --rest_port 8000 --max_prompt_len 16384 --plugin_config "{\"NPUW_LLM_PREFILL_ATTENTION_HINT\":\"PYRAMID\"}" --cache_dir .ovcache --model_name Qwen3-8B
```
> **Note:** First model initialization might be long. With the compilation cache, sequential model loading will be fast.
:::
::::

### Linux: via Docker

::::{tab-set}
:::{tab-item} OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4
:sync: OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4
```bash
mkdir -p models
docker run -d -p 8000:8000 --rm -e MOE_USE_MICRO_GEMM_PREFILL=0 --user $(id -u):$(id -g) -v $(pwd)/models:/models/:rw --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
    openvino/model_server:weekly \
    --model_repository_path /models --source_model OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4 --task text_generation --target_device GPU --tool_parser qwen3coder --rest_port 8000 --model_name Qwen3-Coder-30B-A3B-Instruct
```
> **Note:** For deployment, the model requires ~16GB disk space and recommended 19GB+ of VRAM on the GPU.
:::

:::{tab-item} OpenVINO/QwCoder-30B-A3B-Instruct-int8
:sync: OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int8
```bash
mkdir c:\models
docker run -d -p 8000:8000 --rm -e MOE_USE_MICRO_GEMM_PREFILL=0 --user $(id -u):$(id -g) -v $(pwd)/models:/models/:rw --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
    openvino/model_server:weekly \
    --model_repository_path /models --source_model OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int8 --task text_generation --target_device GPU --tool_parser qwen3coder --rest_port 8000 --model_name Qwen3-Coder-30B-A3B-Instruct
```
> **Note:** For deployment, the model requires ~16GB disk space and recommended 34GB+ of VRAM on the GPU.
:::

:::{tab-item} OpenVINO/gpt-oss-20B-int4
:sync: OpenVINO/gpt-oss-20B-int4
```bash
mkdir c:\models
docker run -d -p 8000:8000 --rm -e MOE_USE_MICRO_GEMM_PREFILL=0 --user $(id -u):$(id -g) -v $(pwd)/models:/models/:rw --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
    openvino/model_server:weekly \
    --model_repository_path /models --source_model OpenVINO/gpt-oss-20B-int4 --task text_generation --target_device GPU --tool_parser gptoss --reasoning_parser gptoss --rest_port 8000 --model_name gpt-oss-20B
```
> **Note:** For deployment, the model requires ~12GB disk space and recommended 16GB+ of VRAM on the GPU.
:::

:::{tab-item} OpenVINO/gpt-oss-20B-int8
:sync: OpenVINO/gpt-oss-20B-int8
```bash
mkdir c:\models
docker run -d -p 8000:8000 --rm -e MOE_USE_MICRO_GEMM_PREFILL=0 --user $(id -u):$(id -g) -v $(pwd)/models:/models/:rw --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
    openvino/model_server:weekly \
    --model_repository_path /models --source_model OpenVINO/gpt-oss-20B-int8 --task text_generation --target_device GPU --tool_parser gptoss --reasoning_parser gptoss --rest_port 8000 --model_name gpt-oss-20B
```
> **Note:** For deployment, the model requires ~24GB disk space and recommended 28GB+ of VRAM on the GPU.
:::

:::{tab-item} OpenVINO/Qwen3-8B-int4-ov
:sync: OpenVINO/Qwen3-8B-int4-ov
```bash
mkdir c:\models
docker run -d -p 8000:8000 --rm -e MOE_USE_MICRO_GEMM_PREFILL=0 --user $(id -u):$(id -g) -v $(pwd)/models:/models/:rw --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
    openvino/model_server:weekly \
    --model_repository_path c:\models --source_model OpenVINO/Qwen3-8B-int4-ov --task text_generation --target_device GPU --tool_parser hermes3 --reasoning_parser qwen3 --rest_port 8000  --model_name Qwen3-8B
```
> **Note:** For deployment, the model requires ~4GB disk space and recommended 6GB+ of VRAM on the GPU.
:::
:::{tab-item} OpenVINO/Qwen3-8B-int4-cw-ov
:sync: OpenVINO/Qwen3-8B-int4-cw-ov
```bash
mkdir c:\models
docker run -d -p 8000:8000 --rm --user $(id -u):$(id -g) -v $(pwd)/models:/models/:rw --device /dev/accel --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -u $(id -u):$(id -g) \
    openvino/model_server:weekly \
    --model_repository_path /models --source_model OpenVINO/Qwen3-8B-int4-cw-ov --task text_generation --target_device NPU --tool_parser hermes3 --rest_port 8000 --max_prompt_len 16384 --plugin_config '{"NPUW_LLM_PREFILL_ATTENTION_HINT":"PYRAMID"}' --model_name Qwen3-8B
```
> **Note:** First model initialization might be long. With the compilation cache, sequential model loading will be fast.
:::
::::


## Custom models

Models which are not published in OpenVINO format can be exported and quantized with custom parameters. Below is an example how to export and deploy model Devstral-Small-2507.

```bash
mkdir models
python export_model.py text_generation --source_model unsloth/Devstral-Small-2507 --weight-format int4 --config_file_path models/config_all.json --model_repository_path models --tool_parser devstral --target_device GPU
curl -L -o models/unsloth/Devstral-Small-2507/chat_template.jinja https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/extras/chat_template_examples/chat_template_devstral.jinja

ovms --model_repository_path models --source_model unsloth/Devstral-Small-2507 --task text_generation --target_device GPU --tool_parser devstral --rest_port 8000 --cache_dir .ovcache
```
> **Note:** Exporting models is a one time operation but might consume RAM at least of the model size and might take a lot of time depending on the model size.



## Set Up Visual Studio Code

### Download [Continue plugin](https://www.continue.dev/)

> **Note:** This demo has been tested with Continue plugin version `1.2.11`. While newer versions should work, some configuration options may vary.

![search_continue_plugin](search_continue_plugin.png)

### Setup Local Assistant

We need to point Continue plugin to our OpenVINO Model Server instance.
Open configuration file:

![setup_local_assistant](setup_local_assistant.png)

Prepare a config:

::::{tab-set}
:::{tab-item} Qwen3-Coder-30B-A3B-Instruct
:sync: Qwen3-Coder-30B-A3B-Instruct
```
name: Local Assistant
version: 1.0.0
schema: v1
models:
  - name: OVMS Qwen3-Coder-30B-A3B-Instruct
    provider: openai
    model: Qwen3-Coder-30B-A3B-Instruct
    apiKey: unused
    apiBase: http://localhost:8000/v3
    roles:
      - chat
      - edit
      - apply
      - autocomplete
    capabilities:
      - tool_use
    autocompleteOptions:
      maxPromptTokens: 500
      debounceDelay: 124
      modelTimeout: 400
      onlyMyCode: true
      useCache: true
context:
  - provider: code
  - provider: docs
  - provider: diff
  - provider: terminal
  - provider: problems
  - provider: folder
  - provider: codebase
```
:::

:::{tab-item} gpt-oss-20b
:sync: gpt-oss-20b
```
name: Local Assistant
version: 1.0.0
schema: v1
models:
  - name: OVMS gpt-oss-20b 
    provider: openai
    model: gpt-oss-20b
    apiKey: unused
    apiBase: http://localhost:8000/v3
    roles:
      - chat
      - edit
      - apply
    capabilities:
      - tool_use
  - name: OVMS gpt-oss-20b autocomplete
    provider: openai
    model: gpt-oss-20b
    apiKey: unused
    apiBase: http://localhost:8000/v3
    roles:
      - autocomplete
    capabilities:
      - tool_use
    requestOptions:
      extraBodyProperties:
        reasoning_effort:
          none
    autocompleteOptions:
      maxPromptTokens: 500
      debounceDelay: 124
      useCache: true
      onlyMyCode: true
      modelTimeout: 400
context:
  - provider: code
  - provider: docs
  - provider: diff
  - provider: terminal
  - provider: problems
  - provider: folder
  - provider: codebase
```
:::
:::{tab-item} unsloth/Devstral-Small-2507
:sync: unsloth/Devstral-Small-2507
```
name: Local Assistant
version: 1.0.0
schema: v1
models:
  - name: OVMS unsloth/Devstral-Small-2507
    provider: openai
    model: unsloth/Devstral-Small-2507
    apiKey: unused
    apiBase: http://localhost:8000/v3
    roles:
      - chat
      - edit
      - apply
      - autocomplete
    capabilities:
      - tool_use
    autocompleteOptions:
      maxPromptTokens: 500
      debounceDelay: 124
      useCache: true
      onlyMyCode: true
      modelTimeout: 400
context:
  - provider: code
  - provider: docs
  - provider: diff
  - provider: terminal
  - provider: problems
  - provider: folder
  - provider: codebase
```
:::
:::{tab-item} Qwen3-8B
:sync: Qwen3-8B
```
name: Local Assistant
version: 1.0.0
schema: v1
models:
  - name: OVMS Qwen3-8B
    provider: openai
    model: Qwen3-8B
    apiKey: unused
    apiBase: http://localhost:8000/v3
    roles:
      - chat
      - edit
      - apply
    capabilities:
      - tool_use
    requestOptions:
      extraBodyProperties:
        chat_template_kwargs:
          enable_thinking: false
context:
  - provider: code
  - provider: docs
  - provider: diff
  - provider: terminal
  - provider: problems
  - provider: folder
  - provider: codebase
```
::::

> **Note:** For more information about this config, see [configuration reference](https://docs.continue.dev/reference#models).

## Chatting, code editing and autocompletion in action

- to use chatting feature click continue button on the left sidebar
- use `CTRL+I` to select and include source in chat message
- use `CTRL+L` to select and edit the source via chat request
- simply write code to see code autocompletion (NOTE: this is turned off by default)

![final](final.png)


## AI Agents in action
Continue.dev plugin is shipped with multiple built-in tools. For full list please [visit Continue documentation](https://docs.continue.dev/features/agent/how-it-works#what-tools-are-available-in-plan-mode-read-only).

To use them, select Agent Mode:

![select agent](./select_agent.png)

Select model that support tool calling from model list:

![select model](./select_qwen.png)

Example use cases for tools:

* Run terminal commands

![git log](./using_terminal.png)

* Look up web links

![wikipedia](./wikipedia.png)

* Search files

![glob](./glob.png)

* Extending VRAM allocation to iGPU

![xram](./vram.png)
