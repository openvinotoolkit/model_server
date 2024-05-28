# OpenAI API Clients {#ovms_docs_clients_openai}

```{toctree}
---
maxdepth: 1
hidden:
---

Chat API <ovms_docs_rest_api_chat>
demo <https://github.com/openvinotoolkit/model_server/tree/main/demos/continuous_batching/>
LLM calculator <ovms_docs_llm_caclulator>
```
## Introduction
Beside Tensorflow Serving API and KServe API frontends, the model server has now option to delegate the REST input deserialization and output serialization to a MediaPipe graph. A custom calculator can implement any form of REST API including streaming based on [Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html#server-sent-events).

That way we are introducing a preview of OpenAI compatible endpoint [chat/completions](./model_server_rest_api_chat.md). More endpoints are planned for the implementation.


## Python Client

When creating a Python-based client application, you can use OpenAI client library - [openai](https://pypi.org/project/openai/).

Alternatively, it is possible to use just a `curl` command or `requests` python library.

### Install the Package

```bash
pip3 install openai
pip3 install requests
```


### Request chat completion with unary calls

::::{tab-set}
:::{tab-item} python [OpenAI] 
:sync: python-openai
```{code} python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v3", api_key="unused")
stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=False,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
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


### Request chat completion with streaming

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
    model="llama",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```
:::
::::
