#
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

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
import platform
import sys

from openai import AsyncOpenAI
from agents import Agent, Runner, RunConfig
from agents.mcp import MCPServerSse, MCPServerStdio
from agents.model_settings import ModelSettings
import argparse

from openai.types.responses import ResponseTextDeltaEvent
from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
)

API_KEY = "not_used"
os.environ["PYTHONUTF8"] = "1"
env_proxy = {}
http_proxy = os.environ.get("http_proxy")
https_proxy = os.environ.get("https_proxy")
if http_proxy:
    env_proxy["http_proxy"] = http_proxy
if https_proxy:
    env_proxy["https_proxy"] = https_proxy

RunConfig.tracing_disabled = False  # Enable tracing for this example


def _image_url_from_path(path: str) -> str:
    """Return a data-URI for a local file or pass through an HTTP(S) URL."""
    if path.startswith(("http://", "https://")):
        return path
    mime_type = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"


def build_multimodal_input(query: str, image_paths: list[str]) -> list[dict]:
    """Build a Responses-API-style multimodal user message with text and images.

    The OpenAI Agents SDK expects content parts typed as ``input_text`` /
    ``input_image`` (Responses API format), *not* the Chat Completions
    ``text`` / ``image_url`` format.
    """
    content: list[dict] = [{"type": "input_text", "text": query}]
    for img in image_paths:
        content.append({
            "type": "input_image",
            "image_url": _image_url_from_path(img),
        })
    return [{"role": "user", "content": content}]


def check_if_tool_calls_present(result) -> bool:
    if hasattr(result, 'new_items') and result.new_items:
        for item in result.new_items:
            if hasattr(item, 'type') and item.type == "tool_call_item":
                return True
    return False

async def run(query, agent, OVMS_MODEL_PROVIDER, stream: bool = False):
    for server in agent.mcp_servers:
        await server.connect()
    print(f"\n\nRunning: {query}")
    if stream:
        result = Runner.run_streamed(starting_agent=agent, input=query, run_config=RunConfig(model_provider=OVMS_MODEL_PROVIDER, tracing_disabled=True))
        print("=== Stream run starting ===")

        async for event in result.stream_events():
            # Print text deltas as they come in
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
            # When the agent updates, print that
            elif event.type == "agent_updated_stream_event":
                print(f"Agent updated: {event.new_agent.name}")
                continue
            # When tool events occur, print details
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    print(f"\n-- Tool '{event.item.raw_item.name}' was called with arguments: {event.item.raw_item.arguments}. Call id: {event.item.raw_item.call_id}")
                elif event.item.type == "tool_call_output_item":
                    print(f"\n-- Tool output:\n {event.item.raw_item}\n\n")
                else:
                    pass  # Ignore other event types

        print("\n=== Stream run complete ===")
    else: 
        result = await Runner.run(starting_agent=agent, input=query, run_config=RunConfig(model_provider=OVMS_MODEL_PROVIDER, tracing_disabled=True))
        print(result.final_output)
        
    return check_if_tool_calls_present(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI Agent with optional query.")
    parser.add_argument("--query", type=str, default="List files in `/root` directory", help="Query to pass to the agent")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model name to use")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v3", help="Base URL for the OpenAI API")
    parser.add_argument("--mcp-server-url", type=str, default="http://localhost:8080/sse", help="URL for the MCP server (if using SSE)")
    parser.add_argument("--stream", action="store_true", help="Stream output from the agent")
    parser.add_argument("--mcp-server", type=str, choices=["all", "weather", "fs"], default="all", help="Which MCP server(s) to use: all, weather, or fs")
    parser.add_argument("--tool-choice", type=str, default="auto", choices=["auto", "required"], help="Tool choice for the agent")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable agent thinking (default: False)")
    parser.add_argument("--image", type=str, nargs="+", default=[], metavar="PATH_OR_URL",
                        help="One or more image file paths or URLs to include with the prompt")
    args = parser.parse_args()
    mcp_servers = []
    if args.mcp_server in ["all", "weather"]:
        if platform.system() == "Windows":
            weather_server = MCPServerStdio(
                name="Weather MCP Server",
                client_session_timeout_seconds=300,
                params={"command": "python", "args": ["-m", "mcp_weather_server"], "env": env_proxy}
            )
        else:
            print("Using SSE weather MCP server")
            weather_server = MCPServerSse(
                name="SSE Python Server",
                params={"url": args.mcp_server_url}
            )
        mcp_servers.append(weather_server)

    if args.mcp_server in ["all", "fs"]:
        fs_server = MCPServerStdio(
            client_session_timeout_seconds=30,
            name="FS MCP Server",
            params={"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"], "env": env_proxy}
        )
        mcp_servers.append(fs_server)
    client = AsyncOpenAI(base_url=args.base_url, api_key=API_KEY)

    class OVMSModelProvider(ModelProvider):
        def get_model(self, _) -> Model:
            return OpenAIChatCompletionsModel(model=args.model, openai_client=client)

    OVMS_MODEL_PROVIDER = OVMSModelProvider()

    agent = Agent(
        name="Assistant",
        mcp_servers=mcp_servers,
        model_settings=ModelSettings(tool_choice=args.tool_choice, temperature=0.0, max_tokens=1000, extra_body={"chat_template_kwargs": {"enable_thinking": args.enable_thinking}}),
    )
    loop = asyncio.new_event_loop()

    if args.image:
        agent_input = build_multimodal_input(args.query, args.image)
    else:
        agent_input = args.query

    is_tool_call_present = loop.run_until_complete(run(agent_input, agent, OVMS_MODEL_PROVIDER, args.stream))
    
    # for testing purposes, exit codes are dependent on whether a tool call was present in the agent's reasoning process
    if is_tool_call_present:
        sys.exit(0)
    else:
        sys.exit(1)