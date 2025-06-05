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

from openai import AsyncOpenAI
from agents import Agent, Runner
from agents.mcp import MCPServer, MCPServerSse
from agents.model_settings import ModelSettings

from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    set_tracing_disabled,
)

BASE_URL = "http://localhost:8000/v3" # Example OVMS server URL
API_KEY = "not_used"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct" # Example model name, replace with your model

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set EXAMPLE_BASE_URL, EXAMPLE_API_KEY, EXAMPLE_MODEL_NAME via env var or code."
    )


client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_tracing_disabled(disabled=True)


class OVMSModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(model=model_name or MODEL_NAME, openai_client=client)

OVMS_MODEL_PROVIDER = OVMSModelProvider()

async def run(mcp_server: MCPServer):
    agent = Agent(
        name="Assistant",
        mcp_servers=[mcp_server],
        model_settings=ModelSettings(tool_choice="auto", temperature=0.0),
    )

    # Run the `get_weather` tool
    message = "What's the weather in ?"
    print(f"\n\nRunning: {message}")
    result = await Runner.run(starting_agent=agent, input=message, run_config=RunConfig(model_provider=OVMS_MODEL_PROVIDER))
    print(result.final_output)


async def main():
    async with MCPServerSse(
        name="SSE Python Server",
        params={
            "url": "http://localhost:8080/sse",  # URL of the MCP SSE server
        },
    ) as server:
        await run(server)

if __name__ == "__main__":

    asyncio.run(main())
