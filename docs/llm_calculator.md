# LLM calculator {#ovms_docs_llm_calculator}

LLM calculator included in the OpenVINO Model Server implements a MediaPipe node for text generation.
It is desinged to run in cycles and return the chunks of reponses to the client.

On the input it expects a HttpPayload struct passed by the Model Server frontend:
```cpp
struct HttpPayload {
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;                 // always
    rapidjson::Document* parsedJson;  // pre-parsed body = null
};
```
The input json content should be compatible with the `chat/completions` API. Read more about the [API](./model_server_rest_api_chat.md).

The input also includes as a side packet with a reference to LLM_NODE_RESOURCE which is a shared object representing an LLM engine. It loads the model, runs the generation cycles and reports the generated results to the LLM calculator via an iterative handler.

On the output the calculator creates an std::string with the json content, which is returned to the client as one response of in chunks with streaming.

In the backend, LLM_NODE_RESOURCE employes algorithms for efficient generation with high concurrency.
Thanks to continuous batching and paged attentions from a [openAI genai lib](https://github.com/mzegla/openvino.genai/tree/request_rate/text_generation/causal_lm/cpp/continuous_batching/library), throughput results are highly optimized.


Here is an example of the MediaPipe graph for chat completions:
```protobuf
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "LLMExecutor"
  calculator: "HttpLLMCalculator"
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
          models_path: "/model/ov_model"
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

The calculator supports the following node_options for tuning the pipeline configuration:
-    required string models_path = 1;
-    optional uint64 max_num_batched_tokens = 2 [default = 256];
-    optional uint64 num_kv_blocks = 3 [default = 364];
-    optional uint64 block_size = 4 [default = 32];
-    optional uint64 max_num_seqs = 5 [default = 256];
-    optional bool dynamic_split_fuse = 7 [default = false];



References:
- [chat API](./model_server_rest_api_chat.md)
- [demo](./../demos/continuous_batching/)


