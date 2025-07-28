# Structured response in LLM models {#ovms_structured_output}

OpenVINO Model Server can enforce the LLM models to generate the output according to a specific json schema.
That functionality can be applied in automation tasks where json content needs to be created based on the text passed in the request.
Json format is a standard for communication and data exchange between applications and microservices.

Below is an example how this capability can be used with an testing procedure to show accuracy gain.

<b>Requirements: OVMS version 2025.3 built from main branch </b>

## Deploy LLM model

There are no extra steps needed to use structured output. Whole behavior is triggered based on the client request.


### 1. Deploy the Model
::::{tab-set}

:::{tab-item} With Docker on GPU
**Required:** Docker Engine installed

```bash
mkdir models
docker run --user $(id -u):$(id -g) -d --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render*) --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --source_model OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov --model_repository_path models --rest_port 8000 --target_device GPU --cache_size 2
```
:::

:::{tab-item} With Docker on NPU
**Required:** Docker Engine installed

```bash
mkdir models
docker run --user $(id -u):$(id -g) -d --device /dev/accel --group-add=$(stat -c "%g" /dev/dri/render*) --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --source_model OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov --model_repository_path models --rest_port 8000 --target_device NPU --cache_size 2
```
:::

:::{tab-item} With Docker on CPU
**Required:** Docker Engine installed

```bash
mkdir models
docker run --user $(id -u):$(id -g) -d --rm -p 8000:8000 -v $(pwd)/models:/models:rw openvino/model_server:latest --source_model OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov --model_repository_path models --rest_port 8000 --target_device CPU --cache_size 2
```
:::

:::{tab-item} On Baremetal Host and GPU
**Required:** OpenVINO Model Server package - see [deployment instructions](../deploying_server_baremetal.md) for details.

```bat
ovms.exe --source_model OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov --model_repository_path models --rest_port 8000 --target_device GPU --cache_size 2
```
:::
:::{tab-item} On Baremetal Host and NPU
**Required:** OpenVINO Model Server package - see [deployment instructions](../deploying_server_baremetal.md) for details.

```bat
ovms.exe --source_model OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov --model_repository_path models --rest_port 8000 --target_device NPU --cache_size 2
```
:::
:::{tab-item} On Baremetal Host and CPU
**Required:** OpenVINO Model Server package - see [deployment instructions](../deploying_server_baremetal.md) for details.

```bat
ovms.exe --source_model OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov --model_repository_path models --rest_port 8000 --target_device CPU --cache_size 2
```
:::
::::


## Client usage

::::{tab-set}

:::{tab-item} With python requests library

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
   "response_format":{
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
response = requests.post("http://localhost:8000/v3/chat/completions", json=payload, headers=headers)
json_response = response.json()

print(json_response["choices"][0]["message"]["content"])
```
```
{"event_name":"Science Fair","date":"Friday","participants":["Alice","Bob"]}
```
:::

:::{tab-item} With python openai library

```console
pip install openai
```

```python
from openai import OpenAI
from pydantic import BaseModel
base_url = "http://localhost:8000/v3"
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

## Testing accuracy impact

The script accuracy_test.py is using the dataset [isaiahbjork/json-mode-agentic](https://huggingface.co/datasets/isaiahbjork/json-mode-agentic)
to assess model response and its compliance with the expected schema and also the json response content with the expected output. 

It will be executed with the response_format request field including the schema and with the schema passed in the system message.

```console
pip install datasets tqdm openai jsonschema
python accuracy_test.py --base_url http://localhost:8000/v3 --model_name OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov --concurrency 50 --limit 1000
```
```
Requests: 1000, Successful responses: 1000, Exact matches: 135, Schema matches: 435 Invalid inputs: 0
```

```console
python accuracy_test.py --base_url http://localhost:8000/v3 --model_name OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov --enable_response_format --concurrency 50 --limit 1000
```
```
Requests: 1000, Successful responses: 1000, Exact matches: 217, Schema matches: 828 Invalid inputs: 0
```
Generally the quality of the responses depend on the model size and topology. The results above proves that the accuracy can be increased even without changing the model via adding the mechanism of guided generation and using the field `response_format`