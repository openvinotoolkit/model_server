# How to serve LLM models with Continous Batching via OpenAI API

## Model preparation
Download latest optimum-intel:
```bash
PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" python3 -m pip install "optimum-intel[nncf,openvino]"@git+https://github.com/huggingface/optimum-intel.git openvino-tokenizers
```

Run optimum-cli to download and quantize the model:
```bash
optimum-cli export openvino --model meta-llama/Llama-2-7b-chat-hf --weight-format int8 ov_model
```

## Server configuration
Prepare config.json:
```
{
    "model_config_list": [],
    "mediapipe_config_list": [
        {
            "name": "llama",
            "graph_path": "graph.pbtxt"
        }
    ]
}
```
Prepare graph.pbtxt:
```
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "LLMExecutor"
  calculator: "OpenAIChatCompletionsCalculator"
  input_stream: "LOOPBACK:loopback"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "LLM_NODE_RESOURCES:llm"
  output_stream: "LOOPBACK:loopback"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  input_stream_info: {
    tag_index: 'LOOPBACK:0',
    back_edge: true
  }
  node_options: {
      [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
          workspace_path: "/model"
      }
  }
  input_stream_handler {
    input_stream_handler: "SyncSetInputStreamHandler",
    options {
      [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
        sync_set {
          tag_index: "LOOPBACK:0"
        }
      }
    }
  }
}
```


## Start-up
```
docker run -d --rm -p 8000:8000 -v $(pwd)/ov_model:/model:ro openvino/model_server --port 9000 --rest_port 8000 --config_path /model/config.json
```

## Client code

Both unary and streaming calls should be available via the same servable:

### Unary:
```bash
curl http://localhost:8000/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is OpenVINO?"
      },
      {
        "role": "assistant",
        "content": "OpenVINO is an open-source software library for deep learning inference that is designed to optimize and run deep learning models on Intel hardware."
      },
      {
        "role": "user",
        "content": "How to install?"
      }
    ]
  }'
```

Output:
```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "Open a command prompt terminal window. You can use the keyboard shortcut: Ctrl+Alt+T\nCreate the /opt/intel folder for OpenVINO by using the following command. If the folder already exists, skip this step.",
        "role": "assistant"
      },
      "logprobs": null
    }
  ],
  "created": 1677664795,
  "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
  "model": "llama",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 17,
    "prompt_tokens": 57,
    "total_tokens": 74
  }
}
```

## Streaming:

Partial outputs:
```bash
pip install openai
```

```python
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

Output:
```
This is a test.
```

Refer to [OpenAI streaming documentation](https://platform.openai.com/docs/api-reference/streaming).
