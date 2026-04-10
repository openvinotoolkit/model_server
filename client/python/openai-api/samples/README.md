# OpenAI API usage samples

OpenVINO Model Server exposes an [OpenAI-compatible REST API](https://docs.openvino.ai/2026/model-server/ovms_docs_clients_genai.html) for text generation and chat.

This guide shows how to interact with that API from Python using only the standard library. It covers the following topics:

- [http_list_models.py](http_list_models.py)
- [http_chat_completions.py](http_chat_completions.py)
- [http_chat_completions_stream.py](http_chat_completions_stream.py)

## Before you run the samples

### Clone the OpenVINO&trade; Model Server repository

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/client/python/openai-api/samples
```

### Start OVMS with an LLM model

The samples below require an OVMS instance serving a text-generation model. The quickest way is with Docker:

```bash
mkdir -p ${HOME}/models
docker run -it -p 8000:8000 --rm --user $(id -u):$(id -g) \
  -v ${HOME}/models:/models/:rw \
  openvino/model_server:latest \
  --model_repository_path /models \
  --source_model OpenVINO/Phi-3-mini-4k-instruct-int4-ov \
  --task text_generation \
  --rest_port 8000 \
  --model_name Phi-3-mini
```

For bare-metal deployment see the [deployment guide](../../../../docs/deploying_server_baremetal.md).

Wait until the model is ready before running the samples:

```bash
curl -s http://localhost:8000/v2/health/ready && echo "Server ready"
```

---

## List served models

Queries `GET /v3/models` and prints a formatted table of model IDs currently served by OVMS.

### Command

```bash
python ./http_list_models.py --help
usage: http_list_models.py [-h] [--http_address HTTP_ADDRESS] [--http_port HTTP_PORT] [--timeout TIMEOUT]

Lists models available on the OpenVINO Model Server via the OpenAI-compatible GET /v3/models endpoint.

optional arguments:
  -h, --help            show this help message and exit
  --http_address HTTP_ADDRESS
                        Specify url to HTTP service. default: localhost
  --http_port HTTP_PORT
                        Specify port to HTTP service. default: 8000
  --timeout TIMEOUT     HTTP request timeout in seconds. default: 10
```

### Usage Example

```bash
python ./http_list_models.py --http_port 8000 --http_address localhost
MODEL ID                                            OWNED BY
--------------------------------------------------------------
Phi-3-mini                                          OVMS
```

---

## Send a chat request

Posts a message to `POST /v3/chat/completions` and prints the full JSON response. Optionally checks `/v2/health/ready` first.

### Command

```bash
python ./http_chat_completions.py --help
usage: http_chat_completions.py [-h] [--http_address HTTP_ADDRESS] [--http_port HTTP_PORT]
                                --model_name MODEL_NAME [--message MESSAGE]
                                [--system_prompt SYSTEM_PROMPT] [--max_tokens MAX_TOKENS]
                                [--temperature TEMPERATURE] [--timeout TIMEOUT] [--no_health_check]

Sends a request to the OpenVINO Model Server OpenAI-compatible chat/completions endpoint.

optional arguments:
  -h, --help            show this help message and exit
  --http_address HTTP_ADDRESS
                        Specify url to HTTP service. default: localhost
  --http_port HTTP_PORT
                        Specify port to HTTP service. default: 8000
  --model_name MODEL_NAME
                        Model name as configured in OVMS.
  --message MESSAGE     User message to send. default: "What is the capital of France?"
  --system_prompt SYSTEM_PROMPT
                        Optional system prompt.
  --max_tokens MAX_TOKENS
                        Maximum tokens to generate. default: 200
  --temperature TEMPERATURE
                        Sampling temperature. default: 0.0
  --timeout TIMEOUT     HTTP request timeout in seconds. default: 60
  --no_health_check     Skip the /v2/health/ready check before sending the request.
```

### Usage Example

```bash
python ./http_chat_completions.py \
  --http_port 8000 \
  --model_name Phi-3-mini \
  --message "What is the capital of France?"
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "The capital of France is Paris.",
        "role": "assistant",
        "tool_calls": []
      }
    }
  ],
  "created": 1743000000,
  "model": "ovms-model",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 8,
    "prompt_tokens": 15,
    "total_tokens": 23
  }
}
```

---

## Send a streaming chat request

Posts a message to `POST /v3/chat/completions` with `"stream": true` and prints tokens to stdout as they arrive.

### Command

```bash
python ./http_chat_completions_stream.py --help
usage: http_chat_completions_stream.py [-h] [--http_address HTTP_ADDRESS] [--http_port HTTP_PORT]
                                       --model_name MODEL_NAME [--message MESSAGE]
                                       [--system_prompt SYSTEM_PROMPT] [--max_tokens MAX_TOKENS]
                                       [--temperature TEMPERATURE] [--timeout TIMEOUT] [--no_health_check]

Sends a streaming request to the OpenVINO Model Server OpenAI-compatible chat/completions endpoint
and prints generated tokens as they arrive.

optional arguments:
  -h, --help            show this help message and exit
  --http_address HTTP_ADDRESS
                        Specify url to HTTP service. default: localhost
  --http_port HTTP_PORT
                        Specify port to HTTP service. default: 8000
  --model_name MODEL_NAME
                        Model name as configured in OVMS.
  --message MESSAGE     User message to send. default: "What is the capital of France?"
  --system_prompt SYSTEM_PROMPT
                        Optional system prompt.
  --max_tokens MAX_TOKENS
                        Maximum tokens to generate. default: 200
  --temperature TEMPERATURE
                        Sampling temperature. default: 0.0
  --timeout TIMEOUT     HTTP request timeout in seconds. default: 300
  --no_health_check     Skip the /v2/health/ready check before sending the request.
```

### Usage Example

```bash
python ./http_chat_completions_stream.py \
  --http_port 8000 \
  --model_name Phi-3-mini \
  --message "Explain photosynthesis in one sentence."
Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.
```

---

## References

- [GenAI client documentation](https://docs.openvino.ai/2026/model-server/ovms_docs_clients_genai.html)
- [LLM quickstart](https://docs.openvino.ai/2026/model-server/ovms_docs_llm_quickstart.html)
- [Chat Completions API reference](../../../../docs/model_server_rest_api_chat.md)
- [Completions API reference](../../../../docs/model_server_rest_api_completions.md)
