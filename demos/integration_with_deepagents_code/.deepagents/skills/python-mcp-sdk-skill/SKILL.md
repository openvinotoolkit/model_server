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
from mcp.types import ToolAnnotations

mcp = FastMCP("your-server-name")

# Reusable annotations for tools that only read state and never mutate anything.
READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


@mcp.tool(annotations=READ_ONLY)
def your_tool_name(arg: str) -> str:
    """Describe exactly what this tool returns."""
    value = arg.strip()
    if not value:
        raise ValueError("arg cannot be empty")
    return value


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

## Tool annotations — MANDATORY

Every `@mcp.tool()` MUST declare MCP tool annotations. Annotations tell hosts (like dcode) whether a tool is safe to call without user approval. Tools with no annotations are treated as "possibly mutating" and are rejected outright in dcode headless mode (`dcode -n ...`) with:

> "This MCP action requires approval, but the current headless runtime has no approval UI."

dcode's `HeadlessMCPGuardMiddleware` gates any tool whose metadata does not satisfy BOTH:
- `readOnlyHint is True`
- `destructiveHint is not True`

Choose the annotation set that matches the tool's real behavior:

| Tool behavior                                     | Annotations                                                                                                         |
|---------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Pure read (clock, config lookup, HTTP GET)        | `readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False`                                |
| Read that consults an external service (web/API)  | `readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True`                                 |
| Idempotent write (set a value; same input twice = same result) | `readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=<True/False>`             |
| Destructive write (delete, drop, DROP TABLE, rm)  | `readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=<True/False>`                        |

Never claim `readOnlyHint=True` for a tool that mutates state. Lying to skip the approval gate is a security issue.

For tools that are NOT read-only, users must run them from the interactive TUI (approval required); document that limitation in the tool's docstring.

## Time tool recipe

When the user needs exact current time, expose:
- `get_current_utc_iso8601()` returning UTC timestamp in ISO 8601 with trailing Z.
- Optional timezone helper for named zones using `zoneinfo`.

Both are pure reads → annotate with `READ_ONLY` (see template above).

## Quality checks

- Avoid third-party dependencies beyond `mcp`.
- Do not use network calls unless required.
- Keep tool outputs deterministic and machine-readable.
- **Every tool must have `ToolAnnotations` matching its real behavior** — see the table above. Verify by running the tool from `dcode -n` after implementation; if it fails with the "approval, but the current headless runtime has no approval UI" message, the annotations are missing or wrong.
