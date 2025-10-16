
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