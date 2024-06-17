# Efficient LLM Serving {#ovms_docs_llm_reference}

**THIS IS A PREVIEW FEATURE**

## Overview

With rapid development of generative AI, new techniques and algorithms for performance optimization and better resource utilization are introduced to make best use of the hardware and provide best generation performance. OpenVINO implements those state of the art methods in it's [GenAI Library](https://github.com/ilya-lavrenov/openvino.genai/tree/ct-beam-search/text_generation/causal_lm/cpp/continuous_batching/library) like:
  - Continuous Batching
  - Paged Attention
  - Dynamic Split Fuse 
  - *and more...*

It is now integrated into OpenVINO Model Server providing efficient way to run generative workloads.

Check out the [quickstart guide](quickstart.md) for a simple example that shows how to use this feature.

## LLM Calculator
As you can see in the quickstart above, big part of the configuration resides in `graph.pbtxt` file. That's because model server text generation servables are deployed as MediaPipe graphs with dedicated LLM calculator that works with latest [OpenVINO GenAI](https://github.com/ilya-lavrenov/openvino.genai/tree/ct-beam-search/text_generation/causal_lm/cpp/continuous_batching/library) solutions. The calculator is designed to run in cycles and return the chunks of reponses to the client.

On the input it expects a HttpPayload struct passed by the Model Server frontend:
```cpp
struct HttpPayload {
    std::string uri;
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;                 // always
    rapidjson::Document* parsedJson;  // pre-parsed body             = null
};
```
The input json content should be compatible with the [chat completions](./model_server_rest_api_chat.md) or [completions](./model_server_rest_api_completions.md) API.

The input also includes a side packet with a reference to `LLM_NODE_RESOURCES` which is a shared object representing an LLM engine. It loads the model, runs the generation cycles and reports the generated results to the LLM calculator via a generation handler. 

**Every node based on LLM Calculator MUST have exactly that specification of this side packet:**

`input_side_packet: "LLM_NODE_RESOURCES:llm"`

**If it's modified, model server will fail to provide graph with the model**

On the output the calculator creates an std::string with the json content, which is returned to the client as one response or in chunks with streaming.

Let's have a look at the graph from the graph configuration from the quickstart:
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
          models_path: "./"
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

Above node configuration should be used as a template since user is not expected to change most of it's content. Actually only `node_options` requires user attention as it specifies LLM engine parameters. The rest of the configuration can remain unchanged. 

The calculator supports the following `node_options` for tuning the pipeline configuration:
-    `required string models_path` - location of the model directory (can be relative);
-    `optional uint64 max_num_batched_tokens` - max number of tokens processed in a single iteration [default = 256];
-    `optional uint64 cache_size` - memory size in GB for storing KV cache [default = 8];
-    `optional uint64 block_size` - number of tokens which KV is stored in a single block (Paged Attention related) [default = 32];
-    `optional uint64 max_num_seqs` - max number of sequences actively processed by the engine [default = 256];
-    `optional bool dynamic_split_fuse` - use Dynamic Split Fuse token scheduling [default = true];
-    `optional string device` - device to load models to. Supported values: "CPU" [default = "CPU"]
-    `optional string plugin_config` - [OpenVINO device plugin configuration](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes.html). Should be provided in the same format for regular [models configuration](./parameters.md#model-configuration-options) [default = ""]


The value of `cache_size` might have performance  implications. It is used for storing LLM model KV cache data. Adjust it based on your environment capabilities, model size and expected level of concurrency.

## Models Directory

In node configuration we set `models_path` indicating location of the directory with files loaded by LLM engine. It loads following files:

```
├── openvino_detokenizer.bin
├── openvino_detokenizer.xml
├── openvino_model.bin
├── openvino_model.xml
├── openvino_tokenizer.bin
├── openvino_tokenizer.xml
├── tokenizer_config.json
├── template.jinja
```

Main model as well as tokenizer and detokenizer are loaded from `.xml` and `.bin` files and all of them are required. `tokenizer_config.json` and `template.jinja` are loaded to read information required for chat template processing.

### Chat template

Chat template is used only on `/chat/completions` endpoint. Template is not applied for calls to `/completions`, so it doesn't have to exist, if you plan to work only with `/completions`. 

Loading chat template proceeds as follows:
1. If `tokenizer.jinja` is present, try to load template from it.
2. If there is no `tokenizer.jinja` and `tokenizer_config.json` exists, try to read template from its `chat_template` field. If it's not present, use default template.
3. If `tokenizer_config.json` exists try to read `eos_token` and `bos_token` fields. If they are not present, both values are set to empty string. 

**Note**: If both `template.jinja` file and `chat_completion` field from `tokenizer_config.json` are successfully loaded `template.jinja` takes precedence over `tokenizer_config.json`.

If there are errors in loading or reading files or fields (they exist but are wrong) no template is loaded and servable will not respond to `/chat/completions` calls. 

If no chat template has been specified, default template is applied. The template looks as follows:
```
"{% if messages|length > 1 %} {{ raise_exception('This servable accepts only single message requests') }}{% endif %}{{ messages[0]['content'] }}"
```

When default template is loaded, servable accepts `/chat/completions` calls when `messages` list contains only single element (otherwise returns error) and treats `content` value of that single message as an input prompt for the model.


## Limitations

As it's in preview, this feature has set of limitations:

- Limited support for [API parameters](../model_server_rest_api_chat.md#request),
- Only one node with LLM calculator can be deployed at once,
- Metrics related to text generation - they are planned to be added later,
- Improvements in stability and recovery mechanisms are also expected

## References
- [Chat Completions API](../model_server_rest_api_chat.md)
- [Completions API](../model_server_rest_api_completions.md)
- [Demo](../../demos/continuous_batching/README.md)
