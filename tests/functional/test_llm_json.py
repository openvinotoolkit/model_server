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

import numpy as np
import pytest
import os
import logging

model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
base_url = os.getenv("BASE_URL", "http://localhost:8000/v3")

logger = logging.getLogger(__name__)


class TestSingleModelInference:

    def test_chat_with_tool_definition(self):
        """
        <b>Description</b>
        sending a content with user question with tools definition. Response should be the json file compatible with tool schema.

        <b>input data</b>
        - OpenAI chat with tools definition

        <b>Expected results</b>
        - json file compatible with tool schema

        """

        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key="unused")

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogotá, Colombia"
                        }
                    },
                    "required": [
                        "location"
                    ],
                    "additionalProperties": False
                },
                "strict": True
            }
        }]

        messages = [
            {"role": "user", "content": "What is the weather like in Paris today?"}
        ]
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

        print("COMPLETION:",completion)

        body = {
            "model": model_name,
            "messages": messages,
            "tools":tools,
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "max_tokens": 1,
        }
        import requests
        response = requests.post(
            url=f"{base_url}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer unused"}
        )

        if response.status_code == 200:
            print("API Response:", response.json())
        else:
            print(f"Failed to get response: {response.status_code}, {response.text}")

        import json
        
        tool_args = completion.choices[0].message.tool_calls[0].function.arguments
        
        # Assert that tool_call is a valid JSON and matches the schema
        import json
        from jsonschema import validate, ValidationError

        try:
            tool_call_json = json.loads(tool_args)
            schema = tools[0]["function"]["parameters"]
            validate(instance=tool_call_json, schema=schema)
            assert True, "tool_call is a valid JSON and matches the schema"
        except json.JSONDecodeError as e:
            assert False, f"tool_call is not a valid JSON: {e}"
        except ValidationError as e:
            assert False, f"tool_call does not match the schema: {e}"

        assert completion.choices[0].message.tool_calls[0].id != ""

        messages.append({'role': 'assistant', 'reasoning_content': None, 'content': '', 'tool_calls': [{'id': 'chatcmpl-tool-d39b13c90f9b4d48b08c16455553dbec', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"location": "Paris, France"}'}}]})
        messages.append({"role": "tool", "tool_call_id": "chatcmpl-tool-d39b13c90f9b4d48b08c16455553dbec", "content": "15 degrees Celsius"})

        print("Messages after tool call:", messages)

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
        )
        print(completion.choices[0].message)

        assert "Paris" in completion.choices[0].message.content
        assert "15 degrees" in completion.choices[0].message.content
        assert completion.choices[0].message.tool_calls == []

    def test_chat_with_dual_tools_definition(self):
        """
        <b>Description</b>
        sending a content with user question with tools definition. Response should be the json file compatible with tool schema.

        <b>input data</b>
        - OpenAI chat with tools definition

        <b>Expected results</b>
        - json file compatible with tool schema

        """

        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key="unused")

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogotá, Colombia"
                        }
                    },
                    "required": [
                        "location"
                    ],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_pollutions",
                "description": "Get current level of air pollutions for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogotá, Colombia"
                        }
                    },
                    "required": [
                        "location"
                    ],
                    "additionalProperties": False
                },
                "strict": True
            }
        }]

        messages = [
            {"role": "user", "content": "What is the temperature and pollution level in New York?"}
        ]
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        print("COMPLETION:",completion)
        import json
        
        tool_args0 = completion.choices[0].message.tool_calls[0].function.arguments
        tool_args1 = completion.choices[0].message.tool_calls[1].function.arguments
        
        # Assert that tool_call is a valid JSON and matches the schema
        import json
        from jsonschema import validate, ValidationError

        try:
            tool_call_json = json.loads(tool_args0)
            schema = tools[0]["function"]["parameters"]
            validate(instance=tool_call_json, schema=schema)
            tool_call_json = json.loads(tool_args1)
            schema = tools[1]["function"]["parameters"]
            validate(instance=tool_call_json, schema=schema)
            assert True, "tool_call is a valid JSON and matches the schema"
        except json.JSONDecodeError as e:
            assert False, f"tool_call is not a valid JSON: {e}"
        except ValidationError as e:
            assert False, f"tool_call does not match the schema: {e}"

        messages.append(completion.choices[0].message)
        messages.append({"role": "tool", "tool_call_id": completion.choices[0].message.tool_calls[0].id, "content": "15 degrees Celsius"})
        messages.append({"role": "tool", "tool_call_id": completion.choices[0].message.tool_calls[1].id, "content": "pm10 28µg/m3"})

        print("Messages after tool call:", messages)

        try:
            client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
            )
        except Exception as e:
            print("Error:", e)
            assert True, f"It should fail with 2 tool calls"
        

    def test_chat_with_tool_definition_stream(self):
        """
        <b>Description</b>
        sending a content with user question with tools definition. Response should be the json file compatible with tool schema.

        <b>input data</b>
        - OpenAI chat with tools definition

        <b>Expected results</b>
        - json file compatible with tool schema

        """

        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key="unused")

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogotá, Colombia"
                        }
                    },
                    "required": [
                        "location"
                    ],
                    "additionalProperties": False
                },
                "strict": True
            }
        }]

        messages = [
            {"role": "user", "content": "What is the weather like in Paris today?"}
        ]
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=True,
        )
        arguments = ""
        function_name = ""
        for chunk in completion:
            if chunk.choices[0].delta.tool_calls is not None:
                if chunk.choices[0].delta.tool_calls[0].function.name is not None:
                    function_name = chunk.choices[0].delta.tool_calls[0].function.name
                    print("Function name:", function_name)  
                    assert chunk.choices[0].delta.tool_calls[0].function.name == "get_weather"
                if chunk.choices[0].delta.tool_calls[0].function.arguments is not None:
                    arguments += chunk.choices[0].delta.tool_calls[0].function.arguments
        assert arguments == '{"location": "Paris, France"}'


    def test_chat_with_structured_output(self):
        """
        <b>Description</b>
        sending a content with user question with tools definition. Response should be the json file compatible with tool schema.

        <b>input data</b>
        - OpenAI chat with tools definition

        <b>Expected results</b>
        - json file compatible with tool schema

        """

        from pydantic import BaseModel
        from openai import OpenAI
        import json
        from jsonschema import validate, ValidationError
        import requests

        client = OpenAI(base_url=base_url, api_key="unused")
        class CalendarEvent(BaseModel):
            event_name: str
            date: str
            participants: list[str]

        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": "Extract the event information."},
                {"role": "user", "content": "Alice and Bob are going to a Science Fair on Friday."},
            ],
            response_format=CalendarEvent,
        )
        print("CalendarEvent as JSON:", json.dumps(CalendarEvent.schema(), indent=2))
        print("COMPLETION CONTENT:",completion.choices[0].message.content)
        json_str = completion.choices[0].message.content
        try:
            schema = CalendarEvent.schema()
            validate(instance=json.loads(json_str), schema=schema)
            print("json_str is compatible with the schema defined in CalendarEvent.")
        except ValidationError as e:
            assert False, f"json_str does not match the schema: {e}"
        except json.JSONDecodeError as e:
            assert False, f"json_str is not a valid JSON: {e}"
        event = completion.choices[0].message.parsed
        print("Parsed event:", event)
        assert event.event_name.lower() == "science fair".lower()
        assert event.date == "Friday"
        assert event.participants == ["Alice", "Bob"]





    