# Built-in Tools Execution with GPT-OSS {#ovms_demos_builtin_tools_execution}

This demo shows how to use **built-in tools** with the [GPT-OSS](https://github.com/openai/gpt-oss) model served by OpenVINO Model Server.

GPT-OSS natively supports a `python` built-in tool. When the model decides it needs to execute Python code (e.g. for calculations), it emits a `tool_call`. A client-side loop catches that call, forwards the code to an **MCP server** for sandboxed execution, and sends the result back to the model so it can produce a final answer.

The diagram below depicts the demo setup:
```
┌────────────┐       ┌──────────────┐       ┌──────────────┐
│  Client     │──1──▶│  OVMS        │       │  MCP Server  │
│  (Python)   │◀──2──│  (GPT-OSS)   │       │  (Python     │
│             │──3──▶│              │       │   executor)  │
│             │      │              │       │              │
│             │──4──▶│              │       │              │
│             │◀──5──│              │       │              │
└─────┬───▲──┘       └──────────────┘       └──────▲───────┘
      │   │                                        │
      └───┼────────────────3a───────────────────────┘
          └────────────────3b───────────────────────┘
```
1. Client sends chat request with `builtin_tools: ["python"]`
2. Model returns a `tool_call` for `python` with generated code
3. Client forwards the code to the MCP server (3a) and receives the result (3b)
4. Client sends tool result back to the model
5. Model produces the final answer

> **Note:** This demo was tested with GPT-OSS-20b on Intel® Arc™ GPU and Intel® Data Center GPU Series on Ubuntu 22/24.

## Prerequisites

- **Docker Engine** with GPU support (`--device /dev/dri`)
- **Python 3.10+** with pip


## Step 1: Export the GPT-OSS Model

GPT-OSS has built-in tool support. Export the model to OpenVINO IR format using the `export_model.py` script:

```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/export_model.py -o export_model.py
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/export_models/requirements.txt
mkdir models
```

Run the export:
```console
python export_model.py text_generation \
    --source_model openai/gpt-oss-20b \
    --weight-format int4 \
    --target_device GPU \
    --config_file_path models/config.json \
    --model_repository_path models \
    --tool_parser gptoss \
    --reasoning_parser gptoss
```

Download the GPT-OSS chat template:
```console
curl -L -o models/openai/gpt-oss-20b/chat_template.jinja \
    https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/extras/chat_template_examples/chat_template_gpt_oss.jinja
```

You should have a model folder like below:
```
models
├── config.json
└── openai
    └── gpt-oss-20b
        ├── chat_template.jinja
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

## Step 2: Start OpenVINO Model Server

Deploy OVMS with the GPU image:

```bash
docker run -d --rm --name ovms-gptoss \
    -p 8000:8000 \
    -v $(pwd)/models:/models:ro \
    --device /dev/dri \
    --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) \
    openvino/model_server:latest-gpu \
    --rest_port 8000 \
    --config_path /models/config.json
```

Wait for the model to load and verify readiness:
```console
curl http://localhost:8000/v1/config
```

:::{dropdown} Expected Response
```json
{
    "openai/gpt-oss-20b": {
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
:::

## Step 3: Set Up the MCP Python Executor

The GPT-OSS repository includes a reference MCP server that executes Python code via a Jupyter kernel. Clone the repository and set up the MCP server:

```console
git clone https://github.com/openai/gpt-oss.git
cd gpt-oss
```

Install the gpt-oss package and MCP server dependencies:
```console
pip install .
cd gpt-oss-mcp-server
pip install "mcp[cli]>=1.12.2" jupyter_client ipykernel
```

The MCP server uses the `dangerously_use_local_jupyter` backend which runs Python code through a local Jupyter kernel instead of Docker containers. To configure the port, patch the `FastMCP` constructor in `python_server.py`:
```console
sed -i 's/mcp = FastMCP(/mcp = FastMCP(port=8080, host="0.0.0.0",/' python_server.py
```

Set the environment variable and start the MCP server:
```console
PYTHON_EXECUTION_BACKEND=dangerously_use_local_jupyter mcp run -t sse python_server.py:mcp
```

> **Note:** The MCP server must remain running in the foreground. Open a new terminal for the next steps.

> **Note:** `dangerously_use_local_jupyter` runs code through a local Jupyter kernel. For production use, consider the Docker-based backend with `PYTHON_EXECUTION_BACKEND=docker` and `docker pull python:3.11`.


## Step 4: Run the Client

Install the client dependencies:
```console
pip install openai mcp
```

Download and run the client script:
```console
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/builtin_tools_execution/client.py -o client.py
python client.py --question "Which day of the week will be for 31 January of 3811? Use python for that."
```

You can pass any question via `--question`. The script follows the flow from the diagram, printing each step:

:::{dropdown} Expected Output
```
==============================================================================
  Built-in Tools Execution Demo (GPT-OSS + MCP Python Executor)
==============================================================================

Model:      openai/gpt-oss-20b
OVMS URL:   http://localhost:8000/v3
MCP URL:    http://127.0.0.1:8080/sse

── Step 1: Sending chat request to OVMS with builtin_tools=["python"] ────────
Question: Which day of the week will be for 31 January of 3811? Use python for that.

── Step 2: Model returned a tool_call for "python" ──────────────────────────
Finish reason: tool_calls
Generated code:
    import datetime
    print(datetime.date(3811, 1, 31).strftime('%A'))

── Step 3: Forwarding code to MCP server for execution ──────────────────────
MCP server: http://127.0.0.1:8080/sse
Execution result: Friday

── Step 4: Sending tool result back to OVMS ──────────────────────────────────

── Step 5: Model produced the final answer ───────────────────────────────────
Content: January 31, 3811 will be a **Friday**.
Finish reason: stop
Usage: 113 prompt / 14 completion / 127 total tokens
```
:::

### Configuration

The script can be configured via environment variables:

| Argument / Variable | Default | Description |
|---------------------|---------|-------------|
| `--question` | *"Which day of the week will be for 31 January of 3811? Use python for that."* | Question to send to the model |
| `--base-url` / `OPENAI_BASE_URL` | `http://localhost:8000/v3` | OVMS REST API base URL |
| `--mcp-server-url` / `MCP_SERVER_URL` | `http://127.0.0.1:8080/sse` | MCP server SSE endpoint |
| `--model` / `OVMS_MODEL` | `openai/gpt-oss-20b` | Model name to use |

Example with custom configuration:
```console
python client.py --question "What is the 50th prime number?" --base-url http://my-server:8000/v3
```

## How It Works

### Built-in Tools in GPT-OSS

GPT-OSS was trained with native support for a `python` built-in tool. To activate it, pass `builtin_tools: ["python"]` in the `chat_template_kwargs` parameter of the request:

```python
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "What is 2**100?"}],
    extra_body={"chat_template_kwargs": {"builtin_tools": ["python"]}},
)
```

When the model decides code execution is needed, it returns a response with `finish_reason: "tool_calls"` and a `tool_calls` array containing the generated Python code.

### Client-Side Tool Execution

Unlike standard function calling where tools are defined in the request, built-in tools are part of the model's training. The client is responsible for:
1. Detecting `tool_calls` in the response
2. Executing the code in a sandboxed environment (via the MCP server)
3. Sending the result back as a `tool` message

This pattern gives the client full control over the execution environment and security boundaries.

### MCP Server

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server provides a standardized interface for tool execution. The gpt-oss reference implementation supports multiple execution backends:
- **`dangerously_use_local_jupyter`** (used in this demo) — runs code through a local Jupyter kernel. Quick to set up, suitable for development and demos.
- **`docker`** — runs code in isolated Docker containers for sandboxed execution. Recommended for production use.

The backend is selected via the `PYTHON_EXECUTION_BACKEND` environment variable.

## References

- [GPT-OSS repository](https://github.com/openai/gpt-oss)
- [GPT-OSS model on HuggingFace](https://huggingface.co/openai/gpt-oss-20b)
- [LLM quick start guide](../../docs/llm/quickstart.md)
- [Agentic AI demo](../continuous_batching/agentic_ai/README.md)
- [Chat completions API reference](../../docs/model_server_rest_api_chat.md)
- [Model Context Protocol](https://modelcontextprotocol.io/)
