# OpenAI API Clients {#ovms_docs_clients_openai}

```{toctree}
---
maxdepth: 1
hidden:
---

Chat completion API <ovms_docs_rest_api_chat>
Completions API <ovms_docs_rest_api_completion>
Demo - text generation<ovms_demos_continuous_batching>
Embeddings API <ovms_docs_rest_api_embeddings>
Demo - text embeddings <ovms_demos_embeddings>
```
## Introduction
Beside Tensorflow Serving API and KServe API frontends, the model server has now option to delegate the REST input deserialization and output serialization to a MediaPipe graph. A custom calculator can implement any form of REST API including streaming based on [Server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events).

We are introducing OpenAI compatible endpoints:
- [chat/completions](./model_server_rest_api_chat.md)
- [completions](./model_server_rest_api_completions.md).
- [embeddings](./model_server_rest_api_embeddings.md)


## Python Client

When creating a Python-based client application, you can use OpenAI client library - [openai](https://pypi.org/project/openai/).

Alternatively, it is possible to use just a `curl` command or `requests` python library.

Along with the prompt, you can send parameters described here [for chat completions endpoint](./model_server_rest_api_chat.md#Request) and here [for completions endpoint](./model_server_rest_api_completions.md#Request).
> **NOTE**:
OpenAI python client supports a limited list of parameters. Those native to OpenVINO Model Server, can be passed inside a generic container parameter `extra_body`. Below is an example how to encapsulated `top_k` value.
```{code} bash
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "hello"}],
    max_tokens=100,
    extra_body={"top_k" : 1},
    stream=False
)
```

### Install the Package

```{code} bash
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

### Text embeddings

::::{tab-set}
:::{tab-item} python [OpenAI] 
:sync: python-openai
```{code} python
from openai import OpenAI
client = OpenAI(
  base_url="http://localhost:8000/v3",
  api_key="unused"
)
responses = client.embeddings.create(input=[hello world], model='Alibaba-NLP/gte-large-en-v1.5')
for data in responses.data:
    print(data.embedding)
```
:::
:::{tab-item} python [requests]
:sync: python-requests
```{code} python
import requests
payload = {"model": "Alibaba-NLP/gte-large-en-v1.5", "input": "hello world"}
headers = {"Content-Type": "application/json", "Authorization": "not used"}
response = requests.post("http://localhost:8000/v3/embeddings", json=payload, headers=headers)
print(response.text)
```
:::
:::{tab-item} curl
:sync: curl
```{code} bash
curl http://localhost:8000/v3/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "Alibaba-NLP/gte-large-en-v1.5", "input": "hello world"}'
```
:::
::::