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
from agents.mcp import MCPServer, MCPServerSse, MCPServerStdio
from agents.model_settings import ModelSettings
import argparse

from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
)

API_KEY = "not_used"
env_proxy = {"http_proxy": os.environ.get("http_proxy"), "https_proxy": os.environ.get("https_proxy")}
RunConfig.tracing_disabled = True  # Disable tracing for this example



async def main(query, model_name: str, base_url: str, mcp_server_url: str):
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
            params={ "url": mcp_server_url}  # URL of the MCP SSE server
        )
    fs_server = MCPServerStdio(
            client_session_timeout_seconds=30,
            name="FS MCP Server",
            params={"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"], "env": env_proxy,}
    )
    async with weather_server, fs_server:
        await run([weather_server, fs_server], query, model_name, base_url)

async def run(mcp_servers: List[MCPServer], message: str, model_name: str, base_url: str):

    client = AsyncOpenAI(base_url=base_url, api_key=API_KEY)

    class OVMSModelProvider(ModelProvider):
        def get_model(self, _) -> Model:
            return OpenAIChatCompletionsModel(model=model_name, openai_client=client)

    OVMS_MODEL_PROVIDER = OVMSModelProvider()

    agent = Agent(
        name="Assistant",
        mcp_servers=mcp_servers,
        model_settings=ModelSettings(tool_choice="auto", temperature=0.0),
    )
    print(f"\n\nRunning: {message}")
    result = await Runner.run(starting_agent=agent, input=message, run_config=RunConfig(model_provider=OVMS_MODEL_PROVIDER, tracing_disabled=True))
    print(result.final_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI Agent with optional query.")
    parser.add_argument("--query", type=str, default="List files in `/tmp/model_server` directory", help="Query to pass to the agent")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model name to use")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v3", help="Base URL for the OpenAI API")
    parser.add_argument("--mcp-server-url", type=str, default="http://localhost:8080/sse", help="URL for the MCP server (if using SSE)")
    args = parser.parse_args()
    asyncio.run(main(args.query, args.model, args.base_url, args.mcp_server_url))
