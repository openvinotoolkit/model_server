# Structured response in LLM models {#ovms_structured_output}

OpenVINO Model Server can enforce the LLM models to generate the output with a specific structure for example as JSON object.
That functionality can be applied in automation tasks where content needs to be created based on the text passed in the request.
JSON format is a standard for communication and data exchange between applications and microservices, but structured format could also be multi choice (when we want the model to generate content from predefined subset), regex or another format [among available ones](https://github.com/mlc-ai/xgrammar/blob/v0.1.26/docs/tutorials/structural_tag.md#format-types).

Below are a few examples of using structured output to get result in desired format.

## Deploy LLM model

There are no extra steps needed to use structured output. Whole behavior is triggered based on the client request.

::::{tab-set}
:::{tab-item} With Docker on GPU
**Required:** Docker Engine installed

```bash
mkdir models
docker run --user $(id -u):$(id -g) -d --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render*  | head -1) --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --source_model OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov --model_repository_path models --task text_generation --rest_port 8000 --target_device GPU --cache_size 2
```
:::
:::{tab-item} With Docker on NPU
**Required:** Docker Engine installed

```bash
mkdir models
docker run --user $(id -u):$(id -g) -d --device /dev/accel --group-add=$(stat -c "%g" /dev/dri/render*  | head -1) --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --source_model OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov --model_repository_path models --task text_generation --rest_port 8000 --target_device NPU --cache_size 2
```
:::
:::{tab-item} With Docker on CPU
**Required:** Docker Engine installed

```bash
mkdir models
docker run --user $(id -u):$(id -g) -d --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest --source_model OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov --model_repository_path models --task text_generation --rest_port 8000 --target_device CPU --cache_size 2
```
:::
:::{tab-item} On Baremetal Host and GPU
**Required:** OpenVINO Model Server package - see [deployment instructions](../../../docs/deploying_server_baremetal.md) for details.

```bat
ovms.exe --source_model OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov --model_repository_path models --rest_port 8000 --target_device GPU --cache_size 2 --task text_generation
```
:::
:::{tab-item} On Baremetal Host and NPU
**Required:** OpenVINO Model Server package - see [deployment instructions](../../../docs/deploying_server_baremetal.md) for details.

```bat
ovms.exe --source_model OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov --model_repository_path models --rest_port 8000 --target_device NPU --cache_size 2 --task text_generation
```
:::
:::{tab-item} On Baremetal Host and CPU
**Required:** OpenVINO Model Server package - see [deployment instructions](../../../docs/deploying_server_baremetal.md) for details.

```bat
ovms.exe --source_model OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov --model_repository_path models --rest_port 8000 --target_device CPU --cache_size 2 --task text_generation
```
:::
::::


## Request output in JSON format

::::{tab-set}
:::{tab-item} With Python Requests Library

```console
pip install requests
```

```python
import requests
payload = {
   "model":"OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov",
   "messages":[
      {
         "role":"system",
         "content":"Extract the event information into json format."
      },
      {
         "role":"user",
         "content":"Alice and Bob are going to a science fair on Friday."
      }
   ],
   "response_format": {
      "type":"json_schema",
      "json_schema":{
         "schema":{
            "properties":{
               "event_name":{
                  "title":"Event Name",
                  "type":"string"
               },
               "date":{
                  "title":"Date",
                  "type":"string"
               },
               "participants":{
                  "items":{
                     "type":"string"
                  },
                  "title":"Participants",
                  "type":"array"
               }
            },
            "required":[
               "event_name",
               "date",
               "participants"
            ],
            "title":"CalendarEvent",
            "type":"object"
         }
      }
   }
}

headers = {"Content-Type": "application/json", "Authorization": "not used"}
response = requests.post("http://127.0.0.1:8000/v3/chat/completions", json=payload, headers=headers)
json_response = response.json()

print(json_response["choices"][0]["message"]["content"])
```
```
{"event_name":"Science Fair","date":"Friday","participants":["Alice","Bob"]}
```
:::
:::{tab-item} With Python OpenAI Library

```console
pip install openai
```

```python
from openai import OpenAI
from pydantic import BaseModel
base_url = "http://127.0.0.1:8000/v3"
model_name = "OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov"
client = OpenAI(base_url=base_url, api_key="unused")
class CalendarEvent(BaseModel):
    event_name: str
    date: str
    participants: list[str]

completion = client.beta.chat.completions.parse(
    model=model_name,
    messages=[
        {"role": "system", "content": "Extract the event information into json format."},
        {"role": "user", "content": "Alice and Bob are going to a Science Fair on Friday."},
    ],
    temperature=0.0,
    max_tokens=100,
    response_format=CalendarEvent,
)
print(completion.choices[0].message.content)
```
```
{"event_name":"Science Fair","date":"Friday","participants":["Alice","Bob"]}
```
:::
::::

## Request output in specified subset (choice)

::::{tab-set}
:::{tab-item} With Python Requests Library

```console
pip install requests
```

```python
import requests
payload = {
   "model":"OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov",
   "messages":[
      {
         "role": "system", 
         "content": "Classify sentiments of given prompts"
      },
      {
         "role":"user",
         "content":"OVMS is fantastic!"
      }
   ],
   "response_format": {
      "type":"or",
      "elements": [
         {
            "type": "const_string",
            "value": "positive"
         },
         {
            "type": "const_string",
            "value": "negative"
         }
      ]
   }
}

headers = {"Content-Type": "application/json", "Authorization": "not used"}
response = requests.post("http://127.0.0.1:8000/v3/chat/completions", json=payload, headers=headers)
json_response = response.json()

print(json_response["choices"][0]["message"]["content"])
```
```
positive
```
:::
:::{tab-item} With Python OpenAI Library

```console
pip install openai
```

```python
from openai import OpenAI
from pydantic import BaseModel
base_url = "http://127.0.0.1:8000/v3"
model_name = "OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov"
client = OpenAI(base_url=base_url, api_key="unused")

completion = client.beta.chat.completions.parse(
    model=model_name,
    messages=[
         {"role": "system", "content": "Classify sentiments of given prompts"},
         {"role": "user", "content": "OVMS is fantastic!"},
    ],
    temperature=0.0,
    max_tokens=100,
    response_format={
      "type":"or",
      "elements": [
         {
            "type": "const_string",
            "value": "positive"
         },
         {
            "type": "const_string",
            "value": "negative"
         }
      ]
   }
)
print(completion.choices[0].message.content)
```
```
positive
```
:::
::::


## Request output in matching RegEx

::::{tab-set}
:::{tab-item} With Python Requests Library

```console
pip install requests
```

```python
import requests
payload = {
   "model":"OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov",
   "messages":[
      {
         "role": "system", 
         "content": "You are responsible for generating email address for new employees. Address should have a format '<first_name>.<last_name>@company.com'. Replace <first_name> and <last_name> with data from user prompt."
      },
      {
         "role":"user",
         "content":"Generate email address for Jane Doe."
      }
   ],
   "response_format": {
      "type": "regex",
      "pattern": "\\w+\\.\\w+@company\\.com"
   }
}

headers = {"Content-Type": "application/json", "Authorization": "not used"}
response = requests.post("http://127.0.0.1:8000/v3/chat/completions", json=payload, headers=headers)
json_response = response.json()

print(json_response["choices"][0]["message"]["content"])
```
```
jane.doe@company.com
```
:::
:::{tab-item} With Python OpenAI Library

```console
pip install openai
```

```python
from openai import OpenAI
from pydantic import BaseModel
base_url = "http://127.0.0.1:8000/v3"
model_name = "OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov"
client = OpenAI(base_url=base_url, api_key="unused")

completion = client.beta.chat.completions.parse(
    model=model_name,
    messages=[
      {"role": "system", "content": "You are responsible for generating email address for new empoyees. Address should have a format '<first_name>.<last_name>@company.com'. Replace <first_name> and <last_name> with data from user prompt."},
      {"role":"user", "content":"Generate email address for Jane Doe."}
    ],
    temperature=0.0,
    max_tokens=100,
    response_format={
      "type": "regex",
      "pattern": "\\w+\\.\\w+@company\\.com"
   }
)
print(completion.choices[0].message.content)
```
```
jane.doe@company.com
```
:::
::::
