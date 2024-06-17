# OpenAI API Clients {#ovms_docs_clients_openai}

```{toctree}
---
maxdepth: 1
hidden:
---

Chat completion API <ovms_docs_rest_api_chat>
Completions API <ovms_docs_rest_api_completion>
Demo <ovms_demos_continuous_batching>
LLM calculator <ovms_docs_llm_caclulator>
```
## Introduction
Beside Tensorflow Serving API and KServe API frontends, the model server has now option to delegate the REST input deserialization and output serialization to a MediaPipe graph. A custom calculator can implement any form of REST API including streaming based on [Server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events).

That way we are introducing a preview of OpenAI compatible endpoint [chat/completions](./model_server_rest_api_chat.md). More endpoints are planned for the implementation.


## Python Client

When creating a Python-based client application, you can use OpenAI client library - [openai](https://pypi.org/project/openai/).

Alternatively, it is possible to use just a `curl` command or `requests` python library.

### Install the Package

```bash
pip3 install openai
pip3 install requests
```


### Request chat completions with unary calls

::::{tab-set}
:::{tab-item} python [OpenAI] 
:sync: python-openai
```{code} python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v3", api_key="unused")
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=False,
)

print(response.choices[0].message)
```
:::
:::{tab-item} python [requests]
:sync: python-requests
```{code} python
import requests
payload = {"model": "meta-llama/Llama-2-7b-chat-hf", "messages": [ {"role": "user","content": "Say this is a test" }]}
headers = {"Content-Type": "application/json", "Authorization": "not used"}
response = requests.post("http://localhost:8000/v3/chat/completions", json=payload, headers=headers)
print(response.text)
```
:::
:::{tab-item} curl
:sync: curl
```{code} bash
curl http://localhost:8000/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-2-7b-chat-hf", "messages": [ {"role": "user","content": "Say this is a test" }]}'
```
:::
::::

### Request completions with unary calls

::::{tab-set}
:::{tab-item} python [OpenAI] 
:sync: python-openai
```{code} python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v3", api_key="unused")
response = client.completions.create(
    model="meta-llama/Llama-2-7b",
    prompt="Say this is a test",
    stream=False,
)

print(response.choices[0].text)
```
:::
:::{tab-item} python [requests]
:sync: python-requests
```{code} python
import requests
payload = {"model": "meta-llama/Llama-2-7b", "prompt": "Say this is a test"}
headers = {"Content-Type": "application/json", "Authorization": "not used"}
response = requests.post("http://localhost:8000/v3/completions", json=payload, headers=headers)
print(response.text)
```
:::
:::{tab-item} curl
:sync: curl
```{code} bash
curl http://localhost:8000/v3/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-2-7b", "prompt": "Say this is a test"}'
```
:::
::::


### Request chat completions with streaming

::::{tab-set}
:::{tab-item} python [OpenAI] 
:sync: python-openai
```{code} python
from openai import OpenAI
client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```
:::
::::

### Request completions with streaming

::::{tab-set}
:::{tab-item} python [OpenAI] 
:sync: python-openai
```{code} python
from openai import OpenAI
client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)

stream = client.completions.create(
    model="meta-llama/Llama-2-7b",
    prompt="Say this is a test",
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].text is not None:
        print(chunk.choices[0].text, end="")
```
:::
::::
