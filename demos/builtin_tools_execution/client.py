#!/usr/bin/env python3
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Client script demonstrating built-in Python tool execution with GPT-OSS
served via OpenVINO Model Server.

When the model returns a python tool_call, the code is forwarded to an MCP
server (python executor) for execution, and the result is sent back to the
model so it can produce a final answer.

Requirements:
    pip install openai mcp
"""

import argparse
import asyncio
import json
import os

from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import OpenAI

BANNER_WIDTH = 78


def step_header(step: int, text: str) -> None:
    """Print a visually distinct step header matching the demo diagram."""
    tag = f"Step {step}: {text} "
    print(f"\n\u2500\u2500 {tag}" + "\u2500" * max(0, BANNER_WIDTH - len(tag) - 3))


def format_usage(usage) -> str:
    if usage is None:
        return ""
    return (f"{usage.prompt_tokens} prompt / "
            f"{usage.completion_tokens} completion / "
            f"{usage.total_tokens} total tokens")


# ── MCP python executor ─────────────────────────────────────────────────────
async def run_python_via_mcp(code: str, mcp_url: str) -> str:
    """Connect to the MCP SSE server and call the 'python' tool."""
    async with sse_client(url=mcp_url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool("python", arguments={"code": code})
            parts = []
            for item in result.content:
                if hasattr(item, "text"):
                    parts.append(item.text)
            return "\n".join(parts) if parts else "(no output)"


def execute_python(code: str, mcp_url: str) -> str:
    """Synchronous wrapper around the async MCP call."""
    return asyncio.run(run_python_via_mcp(code, mcp_url))


# ── Main flow ────────────────────────────────────────────────────────────────
def chat_with_python(question: str, *, base_url: str, model: str, mcp_url: str):
    """
    Execute the full built-in tool flow:
      1. Send chat request with builtin_tools=["python"]
      2. Model returns a tool_call with generated code
      3. Forward code to MCP server for execution
      4. Send tool result back to the model
      5. Model produces the final answer
    """
    client = OpenAI(base_url=base_url, api_key="unused")
    messages = [{"role": "user", "content": question}]

    # ── Step 1 ───────────────────────────────────────────────────────────────
    step_header(1, 'Sending chat request to OVMS with builtin_tools=["python"]')
    print(f"Question: {question}")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={"chat_template_kwargs": {"builtin_tools": ["python"]}},
    )
    message = response.choices[0].message

    # If the model answered directly (no tool call), print and return
    if not message.tool_calls:
        step_header(5, "Model produced the final answer (no tool needed)")
        print(f"Content: {message.content}")
        print(f"Finish reason: {response.choices[0].finish_reason}")
        print(f"Usage: {format_usage(response.usage)}")
        return response

    # ── Step 2 ───────────────────────────────────────────────────────────────
    step_header(2, f'Model returned a tool_call for "{message.tool_calls[0].function.name}"')
    print(f"Finish reason: {response.choices[0].finish_reason}")

    if hasattr(message, "reasoning_content") and message.reasoning_content:
        print(f"Reasoning: {message.reasoning_content}")

    messages.append({
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [tc.model_dump() for tc in message.tool_calls],
    })

    for tc in message.tool_calls:
        if tc.function.name != "python":
            continue

        code = tc.function.arguments
        try:
            parsed = json.loads(code)
            if isinstance(parsed, dict) and "code" in parsed:
                code = parsed["code"]
        except (json.JSONDecodeError, TypeError):
            pass

        print("Generated code:")
        for line in code.splitlines():
            print(f"    {line}")

        # ── Step 3 ───────────────────────────────────────────────────────────
        step_header(3, "Forwarding code to MCP server for execution")
        print(f"MCP server: {mcp_url}")
        tool_result = execute_python(code, mcp_url)
        print(f"Execution result: {tool_result}")

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": "python",
            "content": tool_result,
        })

    # ── Step 4 ───────────────────────────────────────────────────────────────
    step_header(4, "Sending tool result back to OVMS")

    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={"chat_template_kwargs": {"builtin_tools": ["python"]}},
    )

    # ── Step 5 ───────────────────────────────────────────────────────────────
    final_message = final_response.choices[0].message
    step_header(5, "Model produced the final answer")
    print(f"Content: {final_message.content}")
    print(f"Finish reason: {final_response.choices[0].finish_reason}")
    print(f"Usage: {format_usage(final_response.usage)}")

    return final_response


def parse_args():
    parser = argparse.ArgumentParser(
        description="Built-in tools execution demo — GPT-OSS + MCP Python executor")
    parser.add_argument(
        "--question", "-q",
        default="Which day of the week will be for 31 January of 3811? Use python for that.",
        help="Question to send to the model")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v3"),
        help="OVMS REST API base URL (default: http://localhost:8000/v3)")
    parser.add_argument(
        "--mcp-server-url",
        default=os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8080/sse"),
        help="MCP server SSE endpoint (default: http://127.0.0.1:8080/sse)")
    parser.add_argument(
        "--model",
        default=os.getenv("OVMS_MODEL", "openai/gpt-oss-20b"),
        help="Model name (default: openai/gpt-oss-20b)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * BANNER_WIDTH)
    print("  Built-in Tools Execution Demo (GPT-OSS + MCP Python Executor)")
    print("=" * BANNER_WIDTH)
    print(f"\nModel:      {args.model}")
    print(f"OVMS URL:   {args.base_url}")
    print(f"MCP URL:    {args.mcp_server_url}")

    try:
        chat_with_python(
            args.question,
            base_url=args.base_url,
            model=args.model,
            mcp_url=args.mcp_server_url,
        )
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
