# DeepAgents Code Integration Demo

This demo shows how to use DeepAgents Code [TODO link] with OpenVINO Model Server.

---

## Setup

### Prerequisites

- Linux
- Python 3.12+

### Start OVMS with both models

If OVMS is already running with a chat model, skip this step.

```bash
mkdir -p ${HOME}/models
export GPU_ARGS=$(if ls /dev/dri/render* >/dev/null 2>&1; then echo "--device /dev/dri --group-add $(stat -c '%g' /dev/dri/render* | head -n1)"; fi)
docker run --rm ${GPU_ARGS} -u $(id -u):$(id -g) \
  -v ${HOME}/models:/models openvino/model_server:weekly \
  --pull --source_model OpenVINO/gemma-4-26b-a4b-it-int4-ov --model_repository_path /models
docker run --rm ${GPU_ARGS} -u $(id -u):$(id -g) \
  -v ${HOME}/models:/models openvino/model_server:weekly \
  --pull --source_model OpenVINO/LFM2.5-350M-fp16-ov --model_repository_path /models
docker run --rm -u $(id -u):$(id -g) -v ${HOME}/models:/models openvino/model_server:weekly \
  --add_to_config --config_path /models/config.json \
  --model_path OpenVINO/gemma-4-26b-a4b-it-int4-ov --model_name OpenVINO/gemma-4-26b-a4b-it-int4-ov
docker run --rm -u $(id -u):$(id -g) -v ${HOME}/models:/models openvino/model_server:weekly \
  --add_to_config --config_path /models/config.json \
  --model_path OpenVINO/LFM2.5-350M-fp16-ov --model_name OpenVINO/LFM2.5-350M-fp16-ov
docker run -d ${GPU_ARGS} -u $(id -u):$(id -g) \
  -v ${HOME}/models:/models -p 8000:8000 openvino/model_server:weekly \
  --rest_port 8000 --config_path /models/config.json
```

Readiness check:

```console
curl -f http://localhost:8000/v1/models
```

### Configure dcode environment

Use the shell where dcode will be started:

```bash
export OPENAI_API_KEY=not_used
export OPENAI_BASE_URL=http://localhost:8000/v1
export DEEPAGENTS_CODE_PRICES_AUTO_UPDATE=0
```

### Step 3: Install dependencies

```bash
python -m venv .env
source .env/bin/activate
cd demos/integration_with_deepagents_code
python -m pip install --upgrade pip
python -m pip install deepagents-code mcp
```

### Start dcode

```bash
git init
dcode --model openai:OpenVINO/gemma-4-26b-a4b-it-int4-ov --interpreter-tools safe --trust-project-mcp
```
Git init is required because dcode treats the Git root as the project root.


PLACEHOLDER FOR SCREENSHOT

### Optional
#### Limit tool access for smaller context

For open-weight models, limiting available tools can improve reliability by reducing tool-selection overhead.

Use this profile for non-MCP steps (Step 1, Step 3, Step 4):

```bash
dcode --model openai:OpenVINO/gemma-4-26b-a4b-it-int4-ov \
  --interpreter-tools safe \
  --allow-fs-tools read_file,list_dir,grep_search,file_search \
  --no-mcp
```

Use this profile for MCP steps (Step 2 and Step 6):

```bash
dcode --model openai:OpenVINO/gemma-4-26b-a4b-it-int4-ov \
  --interpreter-tools safe \
  --allow-fs-tools read_file,list_dir,grep_search,file_search
```

If a task fails because a tool is missing, restart dcode with a broader tool set.
#### Change mode to AUTO or YOLO

TODO

---

## Demo summary

Prompt:

```text
Summarize demo in current directory.
```

PLACEHOLDER FOR SCREENSHOT

## MCP server creation

As you can see, dcode reports an issue with the MCP tool:

PLACEHOLDER FOR SCREENSHOT

This happens because the MCP server script is missing. We will ask the agent to create it.
To improve reliability, include MCP SDK guidance from demos/integration_with_deepagents_code/.deepagents/skills/python-mcp-sdk-skill/SKILL.md.

Prompt:

```text
/skill:python-mcp-sdk-skill implement a Python MCP stdio server at mcp_server/time_mcp_server.py that provides current UTC time using the Python MCP SDK.
```

PLACEHOLDER FOR SCREENSHOT

## Enhance server capabilities

Prompt:

```text
/goal Extend mcp_server/time_mcp_server.py with date.
```

PLACEHOLDER FOR SCREENSHOT acceptance criteria
PLACEHOLDER FOR SCREENSHOT WIP
PLACEHOLDER FOR SCREENSHOT Grader

## Subagent delegation

This demo includes project-local assets:

- .deepagents/skills/python-mcp-sdk-skill/SKILL.md
- .deepagents/agents/mcp-tester/AGENTS.md

AGENTS.md contains a small MCP validation subagent profile. We can use it to confirm that the server is correct.

Prompt:

```text
Delegate to subagent mcp-tester:
Verify mcp_server/time_mcp_server.py by running it with a short timeout.
Confirm .deepagents/.mcp.json has a matching stdio entry.
```

PLACEHOLDER FOR SCREENSHOT

If everything works, run reload so the current session picks up MCP config changes.
```text
/tools
```
PLACEHOLDER FOR SCREENSHOT mcp unavailable

```text
/reload
```
PLACEHOLDER FOR SCREENSHOT mcp recovered

```text
/tools
```

PLACEHOLDER FOR SCREENSHOT mcp available

## Compact conversation with /offload

Use this before starting the next scenario to reduce conversation size.

Prompt:

```text
/offload
```

PLACEHOLDER FOR SCREENSHOT
before / after

## Tool-use verification

Prompt:

```text
Give me the exact current UTC timestamp down to the current second using your tool.
Also give me current date.
```

PLACEHOLDER FOR SCREENSHOT
