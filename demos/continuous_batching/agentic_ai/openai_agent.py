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
import os
import platform
from typing import List

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
    ItemHelpers,
)

API_KEY = "not_used"
env_proxy = {"http_proxy": os.environ.get("http_proxy"), "https_proxy": os.environ.get("https_proxy")}
RunConfig.tracing_disabled = True  # Disable tracing for this example

async def run(query, agent, OVMS_MODEL_PROVIDER, streaming: bool = False):
    await fs_server.connect()
    await weather_server.connect()
    print(f"\n\nRunning: {query}")
    if streaming:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI Agent with optional query.")
    parser.add_argument("--query", type=str, default="List files in `/tmp/model_server` directory", help="Query to pass to the agent")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model name to use")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v3", help="Base URL for the OpenAI API")
    parser.add_argument("--mcp-server-url", type=str, default="http://localhost:8080/sse", help="URL for the MCP server (if using SSE)")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode for the agent")
    args = parser.parse_args()
    weather_server = None
    if platform.system() == "Windows":
        weather_server = MCPServerStdio(
            name="Weather MCP Server",
            client_session_timeout_seconds=300,
            params={"command": "python", "args": ["-m", "mcp_weather_server"],"env":env_proxy},
        )
    else:
        print("Using SSE weather MCP server")
        weather_server = MCPServerSse(
            name="SSE Python Server",
            params={ "url": args.mcp_server_url}
        )
    fs_server = MCPServerStdio(
            client_session_timeout_seconds=30,
            name="FS MCP Server",
            params={"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"], "env": env_proxy,}
    )
    client = AsyncOpenAI(base_url=args.base_url, api_key=API_KEY)

    class OVMSModelProvider(ModelProvider):
        def get_model(self, _) -> Model:
            return OpenAIChatCompletionsModel(model=args.model, openai_client=client)

    OVMS_MODEL_PROVIDER = OVMSModelProvider()

    agent = Agent(
        name="Assistant",
        mcp_servers=[fs_server, weather_server],
        model_settings=ModelSettings(tool_choice="auto", temperature=0.0),
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(run(args.query, agent, OVMS_MODEL_PROVIDER, args.streaming))
