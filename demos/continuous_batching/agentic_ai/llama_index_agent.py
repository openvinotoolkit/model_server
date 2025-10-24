#
# Copyright (c) 2024 Intel Corporation
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

from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCallResult,
    AgentStream,
)
import os
import argparse
import platform

parser = argparse.ArgumentParser(description="Run LlamaIndex agent with configurable parameters.")
parser.add_argument("--base-url", type=str, default="http://localhost:8000/v3", help="Base URL for the LLM API (e.g., http://localhost:8000/v3)")
parser.add_argument("--model", type=str, required=True, help="Model name for the LLM (e.g., Phi-4-mini-instruct-int4-ov)")
parser.add_argument("--query", type=str, required=True, help="Query to pass to the agent.")
parser.add_argument("--mcp-server-url", type=str, default="http://127.0.0.1:8080/sse", help="MCP server endpoint URL (e.g., http://127.0.0.1:8080/sse)")
parser.add_argument("--stream", default=False, action="store_true", help="Enable streaming responses from the LLM.")
parser.add_argument("--enable-thinking", action="store_true", help="Enable 'thinking' in the model.")
parser.add_argument("--mcp-server", type=str, choices=["all", "weather", "fs"], default="all", help="Which MCP server(s) to use: all, weather, or fs")
args = parser.parse_args()

API_KEY = "not_used"
env_proxy = {}
http_proxy = os.environ.get("http_proxy")
https_proxy = os.environ.get("https_proxy")
if http_proxy:
    env_proxy["http_proxy"] = http_proxy
if https_proxy:
    env_proxy["https_proxy"] = https_proxy

tools = []
if args.mcp_server in ["all", "weather"]:
        if platform.system() == "Windows":
                weather_server = BasicMCPClient("python", args=["-m", "mcp_weather_server"], env=env_proxy) 
        else:
                print("Using SSE weather MCP server")
                weather_server = BasicMCPClient(args.mcp_server_url)
        tools.extend(McpToolSpec(client=weather_server).to_tool_list())

if args.mcp_server in ["all", "fs"]:
        fs_server = BasicMCPClient("npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"], env=env_proxy)
        tools.extend(McpToolSpec(client=fs_server).to_tool_list())

llm = OpenAILike(
    model=args.model,
    api_key=API_KEY,
    api_base=args.base_url,
    is_chat_model=True,
    is_function_calling_model=True,
    additional_kwargs={
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": args.enable_thinking
            }
        }
    }
)
agent = FunctionAgent(llm=llm, tools=tools, system_prompt="You are a helpful assistant.", streaming=args.stream)

import asyncio

async def main():

    if args.stream is False:
        response = await agent.run(args.query)
        print(response)

    else:
        handler = agent.run(user_msg=args.query)

        # handle streaming output
        async for event in handler.stream_events():
                if isinstance(event, AgentStream):
                        print(event.delta, end="", flush=True)
                elif isinstance(event, AgentInput):
                        print("Agent input: ", event.input)  # the current input messages
                        print("Agent name:", event.current_agent_name)  # the current agent name
                elif isinstance(event, AgentOutput):
                        print("Agent output: ", event.response)  # the current full response
                        print("Tool calls made: ", event.tool_calls)  # the selected tool calls, if any
                        print("Raw LLM response: ", event.raw)  # the raw llm api response
                elif isinstance(event, ToolCallResult):
                        print("Tool called: ", event.tool_name)  # the tool name
                        print("Arguments to the tool: ", event.tool_kwargs)  # the tool kwargs
                        print("Tool output: ", event.tool_output)  # the tool output            

        # print final output
        print(str(await handler))

if __name__ == "__main__":
    asyncio.run(main())
