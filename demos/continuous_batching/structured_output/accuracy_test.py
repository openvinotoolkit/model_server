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
import argparse
from tqdm.asyncio import tqdm
from typing import Optional

dataset = None
model_name = None
base_url = None
openai_client = None
concurrency = None
enable_response_format = None
logging.basicConfig(filename='responses.log', level=logging.INFO, format='%(asctime)s %(message)s')
pbar: Optional[tqdm] = None

async def get_response(input, instruction, schema, openai_client, model_name=""):
    
    schema_dict = ast.literal_eval(schema)

    if enable_response_format:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "schema": schema_dict,
                "name": "schema_name",
                "strict": True
            }
        }
    else:
        response_format = None

    response = await openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": input}
        ],
        temperature=0,
        response_format=response_format,
        max_tokens=1000, # limit just in case of endless loop
    )
    pbar.update(1)
    return response.choices[0].message.content

def evaluate_response(response, schema, expected_output):
    try:
        response_json = json.loads(response)
        expected_json = json.loads(expected_output)

        a, b = json.dumps(response_json, sort_keys=True), json.dumps(expected_json, sort_keys=True)
        exact_match = a == b

    except json.JSONDecodeError as e:
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
        if schema_match:
            try:
                schema_match_str = schema_match.group(1)
                _ = ast.literal_eval(schema_match_str)
            except Exception as e:
                print(f"Failed to parse schema_match as JSON: {e}")
                schema_match_str = None
        else:
            schema_match_str = None

        input_text = item.get('input')
        output = item.get('output')
        requests += 1
        try:
            response = await get_response(input_text, instruction, schema_match_str, openai_client, model_name=model_name)
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                response = json_match.group(1).strip()

            successes += 1
            exact_match, schema_match = evaluate_response(response, schema_match_str, output)
            if exact_match:
                exact_matches += 1
            if schema_match:
                schema_matches += 1
            if not exact_match or not schema_match:
                logging.info(f"Exact match: {exact_match}; schema_match {schema_match}\nInput: {input_text}\nInstruction: {instruction}\nSchema: {schema_match_str}\nResponse: {response}\nExpected Output: {output}\n\n\n")

        except Exception as e:
            logging.error(f"Failed response for input: {input_text}\nSchema: {schema_match_str}\nExpected Output: {output}\nError: {e}\n")

    # Run all items concurrently, but limit concurrency to avoid overloading the server
    semaphore = asyncio.Semaphore(concurrency)  # adjust concurrency as needed

    async def sem_task(item):
        async with semaphore:
            await process_item(item)

    await asyncio.gather(*(sem_task(item) for item in dataset))

    pbar.close()
    print(f"Requests: {requests}, Successful responses: {successes}, Exact matches: {exact_matches}, Schema matches: {schema_matches}", "Invalid inputs:", invalid_inputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structured Output Accuracy Test")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v3", help="Base URL for the OpenAI API")
    parser.add_argument("--model", type=str, default="OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov", help="Model name to use")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--enable_response_format", action="store_true", help="Enable response_format in requests")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of requests to process")
    args = parser.parse_args()

    base_url = args.base_url
    model_name = args.model
    concurrency = args.concurrency
    enable_response_format = args.enable_response_format
    limit = args.limit
    dataset = load_dataset('isaiahbjork/json-mode-agentic', split='train')
    dataset = dataset.select(range(limit)) if limit else dataset
    print(dataset)
    pbar = tqdm(total=len(dataset), desc="Processing items")

    openai_client = AsyncOpenAI(base_url=base_url, api_key="unused", timeout=httpx.Timeout(600))

    asyncio.run(main())
