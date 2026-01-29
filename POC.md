
## Start up Python MCP Server
```bash
git clone https://github.com/openai/gpt-oss
cd gpt-oss/gpt-oss-mcp-server
python3 -m venv .venv
. .venv/bin/activate
pip install mcp[cli]>=1.12.2 gpt_oss openai-harmony
```

```bash
mcp run -t sse python_server.py:mcp
```

In build container:
```bash
git clone https://github.com/hkr04/cpp-mcp /cpp-mcp && cd /cpp-mcp && cmake -B build && cmake --build build --config Release
```

## BuiltInToolExecutor MCP Integration

The `BuiltInToolExecutor` class now supports real Python code execution via MCP SSE client. When initialized with an MCP server connection, the `code_interpreter` and `python` tools will execute Python code remotely instead of returning mock responses.

### Automatic Initialization via Environment Variable

The MCP client is automatically initialized when a GenAI servable starts, if the `MCP_SERVER_URL` environment variable is set.

**Environment Variables:**
- `MCP_SERVER_URL` - Required. The base URL of the MCP server (e.g., `http://localhost:8000`)
- `MCP_SSE_ENDPOINT` - Optional. The SSE endpoint path (defaults to `/sse`)

**Example: Starting OVMS with MCP support**
```bash
# Start MCP Python server first
cd gpt-oss/gpt-oss-mcp-server
mcp run -t sse python_server.py:mcp

# In another terminal, start OVMS with MCP enabled
export MCP_SERVER_URL=http://localhost:8000
export MCP_SSE_ENDPOINT=/sse  # optional, defaults to /sse
bazel-bin/src/ovms --config_path /path/to/config.json
```

When OVMS starts and loads a GenAI servable, you will see logs like:
```
[INFO] MCP_SERVER_URL environment variable detected: http://localhost:8000, initializing MCP client...
[INFO] MCP client initialized successfully. Built-in tools (python, code_interpreter) will use REAL execution.
```

If the environment variable is not set:
```
[INFO] MCP_SERVER_URL not set. Built-in tools will use MOCK responses. Set MCP_SERVER_URL=http://localhost:8000 to enable real Python execution.
```

### Programmatic Usage (C++)

```cpp
#include "src/llm/servable.hpp"

// GenAiServable has a built-in BuiltInToolExecutor member
// You can also manually initialize/check MCP status:
servable->initializeMcpClient("http://localhost:8000", "/sse");
if (servable->isMcpClientReady()) {
    // Real Python execution enabled
}
```

### Supported Tools

When MCP client is connected:
- `python` - Executes Python code via MCP server
- `code_interpreter` - Alias for Python execution
- `code_interpreter.run` - Alias for Python execution

When MCP client is not connected (fallback to mock):
- All tools return mock responses for testing/development

## MCP Test Client

```bash
cd /ovms && bazel build --config=linux --strip=never --config=mp_on_py_on --//:distro=ubuntu //src:mcp_test_client
bazel-bin/src/mcp_test_client
```

```
root@ov-spr-36:/ovms# bazel-bin/src/mcp_test_client
MCP SSE Client Test Program
MCP Version: 2024-11-05
Created MCP SSE client for http://localhost:8000/sse
2025-10-16 08:19:38 [INFO] Initializing MCP client...
2025-10-16 08:19:38 [INFO] Opening SSE connection...
2025-10-16 08:19:38 [INFO] Attempting to establish SSE connection: Base URL: http://localhost:8000, SSE Endpoint: /sse
2025-10-16 08:19:38 [INFO] SSE thread: Attempting to connect to /sse
2025-10-16 08:19:38 [INFO] Message endpoint set, stopping wait
2025-10-16 08:19:38 [INFO] Successfully got message endpoint: /messages/?session_id=0f24f28890dd481084ba3db58c997680
Client initialized successfully
Ping result: SUCCESS
Available tools: 1
Tool: python - 
Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing python code to python, it will be executed in a stateless docker container, and the stdout of that process will be returned to you.
    
Executing Python code: print('hello')
Python execution result: {
  "content": [
    {
      "type": "text",
      "text": "hello\n"
    }
  ],
  "structuredContent": {
    "result": "hello\n"
  },
  "isError": false
}
2025-10-16 08:19:39 [INFO] Actively closing SSE connection (normal exit flow)...
2025-10-16 08:19:39 [INFO] Waiting for SSE thread to end...
2025-10-16 08:19:53 [INFO] SSE connection actively closed, no retry needed
2025-10-16 08:19:53 [INFO] SSE thread: Exiting
2025-10-16 08:19:53 [INFO] SSE thread successfully ended
2025-10-16 08:19:53 [INFO] SSE connection successfully closed (normal exit flow)
MCP SSE Client test completed
```