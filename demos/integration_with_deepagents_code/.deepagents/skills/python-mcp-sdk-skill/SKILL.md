---
name: python-mcp-sdk-skill
description: "Guidance for implementing Python MCP SDK stdio servers with FastMCP, robust validation, and deterministic tool output."
license: MIT
compatibility: designed for deepagents-code
---

# Python MCP SDK Server Builder

Use this skill when the user asks to create or modify an MCP server in Python.

## Goal
Build a minimal, production-safe MCP stdio server using the Python MCP SDK package `mcp`.

## Implementation pattern

1. Create `mcp_server/<name>_mcp_server.py`.
2. Use `FastMCP`.
3. Register one or more `@mcp.tool()` functions.
4. Add explicit docstrings and strict input validation.
5. Start with `mcp.run(transport="stdio")`.

## Required template

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("your-server-name")


@mcp.tool()
def your_tool_name(arg: str) -> str:
    """Describe exactly what this tool returns."""
    value = arg.strip()
    if not value:
        raise ValueError("arg cannot be empty")
    return value


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

## Time tool recipe

When the user needs exact current time, expose:
- `get_current_utc_iso8601()` returning UTC timestamp in ISO 8601 with trailing Z.
- Optional timezone helper for named zones using `zoneinfo`.

## Quality checks

- Avoid third-party dependencies beyond `mcp`.
- Do not use network calls unless required.
- Keep tool outputs deterministic and machine-readable.
