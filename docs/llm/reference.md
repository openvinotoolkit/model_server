# Efficient LLM Serving {#ovms_docs_llm_reference}

## Overview

With rapid development of generative AI, new techniques and algorithms for performance optimization and better resource utilization are introduced to make best use of the hardware and provide best generation performance. OpenVINO implements those state of the art methods in it's [GenAI Library](https://github.com/ilya-lavrenov/openvino.genai/tree/ct-beam-search/text_generation/causal_lm/cpp/continuous_batching/library) like:
  - Continuous Batching
  - Paged Attention
  - Dynamic Split Fuse 
  - *and more...*

It is now integrated into OpenVINO Model Server providing efficient way to run generative workloads.

Check out the [quickstart guide](quickstart.md) for a simple example that shows how to use this feature.

## LLM Calculator
As you can see in the quickstart above, big part of the configuration resides in `graph.pbtxt` file. That's because model server text generation servables are deployed as MediaPipe graphs with dedicated LLM calculator that works with latest [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai/tree/master/src/cpp/include/openvino/genai) library. The calculator is designed to run in cycles and return the chunks of responses to the client.

On the input it expects a HttpPayload struct passed by the Model Server frontend:
```cpp
struct HttpPayload {
    std::string uri;
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;                 // always
    rapidjson::Document* parsedJson;  // pre-parsed body             = null
};
```
The input json content should be compatible with the [chat completions](../model_server_rest_api_chat.md) or [completions](../model_server_rest_api_completions.md) API.

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
-    `optional string device` - device to load models to. Supported values: "CPU", "GPU" [default = "CPU"]
-    `optional string plugin_config` - [OpenVINO device plugin configuration](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes.html). Should be provided in the same format for regular [models configuration](../parameters.md#model-configuration-options) [default = "{}"]
-    `optional uint32 best_of_limit` - max value of best_of parameter accepted by endpoint [default = 20];
-    `optional uint32 max_tokens_limit` - max value of max_tokens parameter accepted by endpoint [default = 4096];
-    `optional bool enable_prefix_caching` - enable caching of KV-blocks [default = false];


The value of `cache_size` might have performance and stability implications. It is used for storing LLM model KV cache data. Adjust it based on your environment capabilities, model size and expected level of concurrency.
You can track the actual usage of the cache in the server logs. You can observe in the logs output like below:
```
[2024-07-30 14:28:02.536][624][llm_executor][info][llm_executor.hpp:65] All requests: 50; Scheduled requests: 25; Cache usage 23.9%;
```
Consider increasing the `cache_size` parameter in case the logs report the usage getting close to 100%. When the cache is consumed, some of the running requests might be preempted to free cache for other requests to finish their generations (preemption will likely have negative impact on performance since preempted request cache will need to be recomputed when it gets processed again). When preemption is not possible i.e. `cache size` is very small and there is a single, long running request that consumes it all, then the request gets terminated when no more cache can be assigned to it, even before reaching stopping criteria. 

`enable_prefix_caching` can improve generation performance when the initial prompt content is repeated. That is the case with chat applications which resend the history of the conversations. Thanks to prefix caching, there is no need to reevaluate the same sequence of tokens. Thanks to that, first token will be generated much quicker and the overall
utilization of resource will be lower. Old cache will be cleared automatically but it is recommended to increase cache_size to take bigger performance advantage.

`plugin_config` accepts a json dictionary of tuning parameters for the OpenVINO plugin. It can tune the behavior of the inference runtime. For example you can include there kv cache compression or the group size '{"KV_CACHE_PRECISION": "u8", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32"}'.

The LLM calculator config can also restrict the range of sampling parameters in the client requests. If needed change the default values for  `max_tokens_limit` and `best_of_limit`. It is meant to avoid the result of memory overconsumption by invalid requests.


## Canceling the generation

In order to optimize the usage of compute resources, it is important to stop the text generation when it becomes irrelevant for the client or when the client gets disconnected for any reason. Such capability is implemented via a tight integration between the LLM calculator and the model server frontend. The calculator gets notified about the client session disconnection. When the client application stops or deliberately breaks the session, the generation cycle gets broken and all resources are released. Below is an easy example how the client can initialize stopping the generation:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v3", api_key="unused")
stream = client.completions.create(model="model", prompt="Say this is a test", stream=True)
for chunk in stream:
    if chunk.choices[0].text is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if some_condition:
        stream.close()
        break
```

> NOTE: To leverage LLM graph cancellation upon client disconnection, use `stream=True` parameter.

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

This model directory can be created based on the models from Hugging Face Hub or from the PyTorch model stored on the local filesystem. Exporting the models to Intermediate Representation format is one time operation and can speed up the loading time and reduce the storage volume, if it's combined with quantization and compression.

In your python environment install required dependencies:
```
pip3 install "optimum-intel[nncf,openvino]
```

Because there is very dynamic development in optimum-intel and openvino, it is recommended to use the latest versions of the dependencies:
```
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu https://storage.openvinotoolkit.org/simple/wheels/pre-release"
pip3 install --pre "optimum-intel[nncf,openvino]"@git+https://github.com/huggingface/optimum-intel.git  openvino_tokenizers openvino
```

LLM model can be exported with a command:
```
optimum-cli export openvino --disable-convert-tokenizer --model {LLM model in HF hub or Pytorch model folder} --weight-format {fp32/fp16/int8/int4/int4_sym_g128/int4_asym_g128/int4_sym_g64/int4_asym_g64} {target folder name}
```
Precision parameter is important and can influence performance, accuracy and memory usage. It is recommended to start experiments with `fp16`. The precision `int8` can reduce the memory consumption and improve latency with low impact on accuracy. Try `int4` to minimize memory usage and check various algorithm to achieve optimal results. 

Export the tokenizer model with a command:
```
convert_tokenizer -o {target folder name} --utf8_replace_mode replace --with-detokenizer --skip-special-tokens --streaming-detokenizer --not-add-special-tokens {tokenizer model in HF hub or Pytorch model folder}
```

Check [tested models](https://github.com/openvinotoolkit/openvino.genai/blob/master/tests/python_tests/models/real_models).

### Chat template

Chat template is used only on `/chat/completions` endpoint. Template is not applied for calls to `/completions`, so it doesn't have to exist, if you plan to work only with `/completions`. 

Loading chat template proceeds as follows:
1. If `tokenizer.jinja` is present, try to load template from it.
2. If there is no `tokenizer.jinja` and `tokenizer_config.json` exists, try to read template from its `chat_template` field. If it's not present, use default template.
3. If `tokenizer_config.json` exists try to read `eos_token` and `bos_token` fields. If they are not present, both values are set to empty string. 

**Note**: If both `template.jinja` file and `chat_completion` field from `tokenizer_config.json` are successfully loaded, `template.jinja` takes precedence over `tokenizer_config.json`.

If there are errors in loading or reading files or fields (they exist but are wrong) no template is loaded and servable will not respond to `/chat/completions` calls. 

If no chat template has been specified, default template is applied. The template looks as follows:
```
"{% if messages|length != 1 %} {{ raise_exception('This servable accepts only single message requests') }}{% endif %}{{ messages[0]['content'] }}"
```

When default template is loaded, servable accepts `/chat/completions` calls when `messages` list contains only single element (otherwise returns error) and treats `content` value of that single message as an input prompt for the model.


## Limitations

There are several known limitations which are expected to be addressed in the coming releases:

- Metrics related to text generation are not exposed via `metrics` endpoint. Key metrics from LLM calculators are included in the server logs with information about active requests, scheduled for text generation and KV Cache usage. It is possible to track in the metrics the number of active generation requests using metric called `ovms_current_graphs`. Also tracking statistics for request and responses is possible. [Learn more](../metrics.md) 
- Multi modal models are not supported yet. Images can't be sent now as the context.
- `logprobs` parameter is not supported currently in streaming mode. It includes only a single logprob and do not include values for input tokens.

## References
- [Chat Completions API](../model_server_rest_api_chat.md)
- [Completions API](../model_server_rest_api_completions.md)
- [Demo](../../demos/continuous_batching/README.md)
