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

import asyncio
from datasets import load_dataset
import re
import os
from openai import AsyncOpenAI
import json
from jsonschema import validate, ValidationError
import logging
import ast
import httpx

dataset = load_dataset('isaiahbjork/json-mode-agentic', split='train')
dataset = dataset.select(range(1000))

model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
base_url = os.getenv("BASE_URL", "http://ov-spr-28.sclab.intel.com:9001/v3")
openai_client = AsyncOpenAI(base_url=base_url, api_key="unused", timeout=httpx.Timeout(600, connect=3.0, read=20.0, write=2.0))
logging.basicConfig(filename='responses.log', level=logging.INFO, format='%(asctime)s %(message)s')
print(dataset)


async def get_response(input, instruction, schema, openai_client, model_name=""):
    # try:
        
    schema_dict = ast.literal_eval(schema)
    #     name, schema_object = next(iter(schema_dict.items()))
    # except Exception as e:
    #     return None

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "schema": schema_dict,
            "name": "schema_name",
            "strict": True
        }
    }
    response = await openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": input}
        ],
        temperature=0,
        response_format=response_format,
        max_tokens=1000,
    )
    return response.choices[0].message.content

def evaluate_response(response, schema, expected_output):
    try:
        response_json = json.loads(response)
        expected_json = json.loads(expected_output)
        # _, expected_object = next(iter(expected_json.items()))

        a, b = json.dumps(response_json, sort_keys=True), json.dumps(expected_json, sort_keys=True)
        exact_match = a == b
        if exact_match is False:
            print(f"Response JSON: {response_json}\nExpected JSON: {expected_json}")

    except json.JSONDecodeError as e:
        print(f"Response JSON: {response_json}\nExpected JSON: {expected_json} \nError: {e}")
        return False, False
    try:
        schema_json = ast.literal_eval(schema)
        validate(instance=response_json, schema=schema_json)
        schema_match = True
    except Exception as e:
        schema_match = False

    return exact_match, schema_match

async def main():
    requests = 0
    successes = 0
    schema_matches = 0
    exact_matches = 0
    invalid_inputs = 0

    async def process_item(item):
        nonlocal requests, successes, schema_matches, exact_matches, invalid_inputs

        instruction = item.get('instruction')
        schema_match = re.search(r"<schema>(.*?)</schema>", instruction, re.DOTALL)
        print(schema_match.group(1))
        if schema_match:
            try:
                schema_match_str = schema_match.group(1)
                _ = ast.literal_eval(schema_match_str)
            except Exception as e:
                print(f"Failed to parse schema_match as JSON: {e}")
                schema_match_str = None
        else:
            schema_match_str = None
        
        # if schema_match_json is None:
        #     logging.info(f"Invalid schema match for item: {item}")
        #     invalid_inputs += 1
        #     return  # Skip this item if schema match is invalid
        
        # If schema_match_json doesn't have 'type' element, skip this item
        # if 'type' not in schema_match_json and 'properties' not in schema_match_json:
        #     name, schema_object = next(iter(schema_match_json.items()))
        #     if 'type' not in schema_object and 'properties' not in schema_object:
        #         invalid_inputs += 1
        #         return
        # else:
        #     name = "schema unknown"
        #     schema_object = schema_match_json

        input_text = item.get('input')
        output = item.get('output')
        print("instruction:", instruction)
        print("schema_substring:", schema_match_str)
        print("input_text:", input_text)
        requests += 1
        try:
            response = await get_response(input_text, instruction, schema_match_str, openai_client, model_name=model_name)
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                response = json_match.group(1).strip()

            successes += 1
            a, b = evaluate_response(response, schema_match_str, output)
            print("Response:", response)
            print("Output:", output)
            
            if a:
                exact_matches += 1
            if b:
                schema_matches += 1
        except Exception as e:
            logging.info(f"Failed response for input: {input_text}\nSchema: {schema_match_str}\nExpected Output: {output}\nError: {e}\n")
        print(f"Requests: {requests}, Successes: {successes}, Exact matches: {exact_matches}, Schema matches: {schema_matches}", "Invalid inputs:", invalid_inputs)

    # Run all items concurrently, but limit concurrency to avoid overloading the server
    semaphore = asyncio.Semaphore(50)  # adjust concurrency as needed

    async def sem_task(item):
        async with semaphore:
            await process_item(item)

    await asyncio.gather(*(sem_task(item) for item in dataset))

if __name__ == "__main__":
    asyncio.run(main())










