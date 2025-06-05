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

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/v3")  # Example OVMS server BASE URL
API_KEY = "not_used"
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B") # Example model name, replace with your model

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set EXAMPLE_BASE_URL, EXAMPLE_API_KEY, EXAMPLE_MODEL_NAME via env var or code."
    )
env_proxy = {"http_proxy": os.environ.get("http_proxy"), "https_proxy": os.environ.get("https_proxy")}

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
RunConfig.tracing_disabled = True  # Disable tracing for this example

weather_server = None
if platform.system() == "Windows":
    weather_server = MCPServerStdio(
        name="Weather MCP Server",
        client_session_timeout_seconds=3000,
        params={"command": "python", "args": ["-m", "mcp_weather_server"],"env":env_proxy},
    )
else:
    print("Using SSE server")
    weather_server = MCPServerSse(
        name="SSE Python Server",
        params={
            "url": "http://localhost:8080/sse",  # URL of the MCP SSE server
        }
    )
fs_server = MCPServerStdio(
        client_session_timeout_seconds=30,
        name="FS MCP Server",
        params={"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"], "env": env_proxy,}
)

class OVMSModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(model=model_name or MODEL_NAME, openai_client=client)

OVMS_MODEL_PROVIDER = OVMSModelProvider()

async def main(query):
    async with weather_server, fs_server:
        await run([weather_server, fs_server], query)

async def run(mcp_servers: List[MCPServer], message: str):
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
    args = parser.parse_args()
    asyncio.run(main(args.query))
