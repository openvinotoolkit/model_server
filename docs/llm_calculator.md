# LLM calculator {#ovms_docs_llm_calculator}

LLM calculator included in the OpenVINO Model Server implements a MediaPipe node for text generation.
It is designed to run in cycles and return the chunks of reponses to the client.

On the input it expects a HttpPayload struct passed by the Model Server frontend:
```cpp
struct HttpPayload {
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;                 // always
    rapidjson::Document* parsedJson;  // pre-parsed body = null
};
```
The input json content should be compatible with the `chat/completions` API. Read more about the [API](./model_server_rest_api_chat.md).

The input also includes a side packet with a reference to LLM_NODE_RESOURCES which is a shared object representing an LLM engine. It loads the model, runs the generation cycles and reports the generated results to the LLM calculator via a generation handler.

On the output the calculator creates an std::string with the json content, which is returned to the client as one response or in chunks with streaming.

In the backend, LLM engine from LLM_NODE_RESOURCES employs algorithms for efficient generation with high concurrency.
Thanks to continuous batching and paged attention from a [OpenVINO GenAI Library](https://github.com/ilya-lavrenov/openvino.genai/tree/ct-beam-search/text_generation/causal_lm/cpp/continuous_batching/library), throughput results are highly optimized.


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
-    optional uint64 cache_size = 3 [default = 8];
-    optional uint64 block_size = 4 [default = 32];
-    optional uint64 max_num_seqs = 5 [default = 256];
-    optional bool dynamic_split_fuse = 7 [default = true];
-    optional string device = 8 [default = "CPU"]
-    optional string plugin_config = 9 [default = ""]

The value of `cache_size` might have performance  implications. It is used for storing LLM model KV cache data. Adjust it based on the model size and the expected level of concurrency. Set `50` or even more for high load with large models.

References:
- [chat API](./model_server_rest_api_chat.md)
- [demo](./../demos/continuous_batching/)


