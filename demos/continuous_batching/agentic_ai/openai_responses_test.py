#!/usr/bin/env python3
# Sends a Responses API request that exercises the full reasoning +
# function_call + function_call_output + follow-up turn buffering path
# (the same scenario the failing curl was hitting).
#
# Uses the openai SDK directly (the AsyncOpenAI Responses client), which is
# what `openai-agents` is built on top of. We don't need the Agents runtime
# here because we're hand-crafting prior-turn items rather than letting an
# agent loop generate them.
#
# Usage:
#   pip install openai
#   python openai_responses_test.py \
#       --base-url http://ov-ptl-23.sclab.intel.com:8000/v1 \
#       --model ovms-model

import argparse
import asyncio
import json
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)
from openai import AsyncOpenAI


def build_input():
    # Mirrors the failing curl payload: a user turn, a model reasoning step,
    # a function_call, its function_call_output, then a follow-up user turn.
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What's the weather in Paris?"}
            ],
        },
        {
            "type": "reasoning",
            "content": [
                {
                    "type": "summary_text",
                    "text": "User wants weather. I should call get_weather.",
                }
            ],
        },
        {
            "type": "function_call",
            "id": "call_abc",
            "call_id": "call_abc",
            "name": "get_weather",
            "arguments": json.dumps({"city": "Paris"}),
        },
        {
            "type": "function_call_output",
            "call_id": "call_abc",
            "output": json.dumps({"temp_c": 17, "sky": "cloudy"}),
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Thanks, and tomorrow?"}
            ],
        },
    ]


def build_tools():
    return [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v3")
    parser.add_argument("--model", default="ovms-model")
    parser.add_argument("--api-key", default="unused")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument(
        "--skip-index",
        type=int,
        action="append",
        default=[],
        help="Drop the input item at this index (repeatable). Useful to bisect which item the server rejects.",
    )
    parser.add_argument(
        "--add-message-type",
        action="store_true",
        help="Add 'type':'message' to role-bearing input items (some servers require it).",
    )
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)

    items = build_input()
    if args.add_message_type:
        for it in items:
            if "role" in it and "type" not in it:
                it["type"] = "message"
    if args.skip_index:
        items = [it for i, it in enumerate(items) if i not in set(args.skip_index)]

    request = {
        "model": args.model,
        "input": items,
        "tools": build_tools(),
    }

    print("--- request body (as constructed) ---")
    print(json.dumps(request, indent=2))
    print("-------------------------------------")

    if args.stream:
        async with client.responses.stream(**request) as stream:
            async for event in stream:
                print(event)
            final = await stream.get_final_response()
            print("\n--- final response ---")
            print(final.model_dump_json(indent=2))
    else:
        response = await client.responses.create(**request)
        print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
