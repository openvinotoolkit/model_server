import asyncio
from datasets import load_dataset
import re
import os
from openai import AsyncOpenAI
import json
from jsonschema import validate, ValidationError
import logging

dataset = load_dataset('isaiahbjork/json-mode-agentic', split='train')
model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
base_url = os.getenv("BASE_URL", "http://localhost:10000/v3")
openai_client = AsyncOpenAI(base_url=base_url, api_key="unused")
logging.basicConfig(filename='responses.log', level=logging.INFO, format='%(asctime)s %(message)s')
print(dataset)

async def get_response(input, schema, openai_client, model_name=""):
    try:
        import ast
        schema_dict = ast.literal_eval(schema)
        name, schema_object = next(iter(schema_dict.items()))
    except Exception as e:
        return None

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "schema": schema_object,
            "name": name,
            "strict": True
        }
    }
    response = await openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Generate a JSON object that matches the following JSON schema."},
            {"role": "user", "content": input}
        ],
        temperature=0,
        response_format=response_format
    )
    return response.choices[0].message.content

def evaluate_response(response, schema, expected_output):
    try:
        response_json = json.loads(response)
        expected_json = json.loads(expected_output)
        _, expected_object = next(iter(expected_json.items()))
        exact_match = response_json == expected_object
        if exact_match is False:
            logging.info(f"Response JSON: {response_json}\nExpected JSON: {expected_object}")

    except json.JSONDecodeError as e:
        logging.info(f"Response JSON: {response_json}\nExpected JSON: {expected_object} \nError: {e}")
        return False, False
    try:
        schema_json = json.loads(schema.replace("'", '"'))
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

    async def process_item(item):
        nonlocal requests, successes, schema_matches, exact_matches
        
        instruction = item.get('instruction')
        schema_match = re.search(r"<schema>(.*?)</schema>", instruction, re.DOTALL)
        schema_substring = schema_match.group(1) if schema_match else None
        input_text = item.get('input')
        output = item.get('output')
        try:
            response = await get_response(input_text, schema_substring, openai_client, model_name=model_name)
            a, b = evaluate_response(response, schema_substring, output)
            requests += 1
            if a:
                exact_matches += 1
            if b:
                schema_matches += 1
            successes += 1
            if requests > 100:
                exit(0)
        except Exception as e:
            logging.info(f"Failed response for input: {input_text}\nSchema: {schema_substring}\nExpected Output: {output}\nError: {e}\n")
        print(f"Requests: {requests}, Successes: {successes}, Exact matches: {exact_matches}, Schema matches: {schema_matches}")

    # Run all items concurrently, but limit concurrency to avoid overloading the server
    semaphore = asyncio.Semaphore(50)  # adjust concurrency as needed

    async def sem_task(item):
        async with semaphore:
            await process_item(item)

    await asyncio.gather(*(sem_task(item) for item in dataset))

if __name__ == "__main__":
    asyncio.run(main())










