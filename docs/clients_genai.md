# Generative AI Use Cases {#ovms_docs_clients_genai}

```{toctree}
---
maxdepth: 1
hidden:
---

Chat completion API <ovms_docs_rest_api_chat>
Completions API <ovms_docs_rest_api_completion>
Embeddings API <ovms_docs_rest_api_embeddings>
Reranking API <ovms_docs_rest_api_rerank>
```
## Introduction
Beside Tensorflow Serving API (`/v1`) and KServe API (`/v2`) frontends, the model server supports a range of endpoints for generative use cases (`v3`). They are extendible using MediaPipe graphs.
Currently supported endpoints are:

OpenAI compatible endpoints:
- [chat/completions](./model_server_rest_api_chat.md)
- [completions](./model_server_rest_api_completions.md)
- [embeddings](./model_server_rest_api_embeddings.md)
Cohere Compatible endpoint:
- [rerank](./model_server_rest_api_rerank.md)

## OpenAI API Clients

When creating a Python-based client application, you can use OpenAI client library - [openai](https://pypi.org/project/openai/).

Alternatively, it is possible to use just a `curl` command or `requests` python library.

### Install the Package

```{code} bash
pip3 install openai
pip3 install requests
pip3 install cohere
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

### Request chat completions with unary calls (with image input)

::::{tab-set}
:::{tab-item} python [OpenAI] 
:sync: python-openai
```{code} python
import base64
from openai import OpenAI

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode("utf-8")

image_path = "/path/to/image"
image = encode_image(image_path)

client = OpenAI(base_url="http://localhost:8000/v3", api_key="unused")
response = client.chat.completions.create(
  model="openbmb/MiniCPM-V-2_6",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?",
        },
        {
          "type": "image_url",
          "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        },
      ],
    }
  ],
  stream=False,
)
print(response.choices[0].message)
```
:::
::::

Check [LLM quick start](./llm/quickstart.md) and [end to end demo of text generation](../demos/continuous_batching/README.md).

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
Check [LLM quick start](./llm/quickstart.md) and [end to end demo of text generation](../demos/continuous_batching/README.md).

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

### Request chat completions with streaming (with image input)

::::{tab-set}
:::{tab-item} python [OpenAI] 
:sync: python-openai
```{code} python
import base64
from openai import OpenAI

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode("utf-8")

image_path = "/path/to/image"
image = encode_image(image_path)

client = OpenAI(base_url="http://localhost:8000/v3", api_key="unused")

stream = client.chat.completions.create(
  model="openbmb/MiniCPM-V-2_6",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?",
        },
        {
          "type": "image_url",
          "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        },
      ],
    }
  ],
  stream=True,
)

for chunk in stream:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")
```
:::
::::

Check [LLM quick start](./llm/quickstart.md) and [end to end demo of text generation](../demos/continuous_batching/README.md).

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
Check [LLM quick start](./llm/quickstart.md) and [end to end demo of text generation](../demos/continuous_batching/README.md).

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
responses = client.embeddings.create(input=['hello world'], model='Alibaba-NLP/gte-large-en-v1.5')
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
Check [text embeddings end to end demo](../demos/embeddings/README.md).

## Cohere Python Client

Clients can use rerank endpoint via cohere python package - [cohere](https://pypi.org/project/cohere/).

Just like with openAI endpoints and alternative is in `curl` command or `requests` python library.

### Install the Package

```{code} bash
pip3 install cohere
pip3 install requests
```

### Documents reranking

::::{tab-set}
:::{tab-item} python [Cohere] 
:sync: python-cohere
```{code} python
import cohere
client = cohere.Client(base_url='http://localhost:8000/v3', api_key="not_used")
responses = client.rerank(query="Hello",documents=["Welcome","Farewell"], model='BAAI/bge-reranker-large')
for res in responses.results:
    print(res.index, res.relevance_score)
```
:::
:::{tab-item} python [requests]
:sync: python-requests
```{code} python
import requests
payload = {"model": "BAAI/bge-reranker-large", "query": "Hello", "documents":["Welcome","Farewell"]}
headers = {"Content-Type": "application/json", "Authorization": "not used"}
response = requests.post("http://localhost:8000/v3/rerank", json=payload, headers=headers)
print(response.text)
```
:::
:::{tab-item} curl
:sync: curl
```{code} bash
curl http://localhost:8000/v3/rerank \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-reranker-large", "query": "Hello", "documents":["Welcome","Farewell"]}'
```
:::
::::
Check [documents reranking end to end demo](../demos/rerank/README.md).