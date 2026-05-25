# Model to Parser Mapping {#ovms_docs_llm_model_parser_mapping}

## Overview

When serving models that support **tool calling** or **reasoning** (chain-of-thought), OpenVINO Model Server needs to know how to parse the model output. Different model families use different output formats for tool calls and reasoning content, so the correct parser must be configured for each model.

Parsers are configured via the `tool_parser` and `reasoning_parser` options in the graph configuration or CLI parameters.

## Tool Parsers

Tool parsers extract structured tool/function call information from model output and return it in OpenAI-compatible `tool_calls` format.

```{list-table} Tool Parser to Model Mapping
:header-rows: 1
:widths: 15 35 50

* - Parser Name
  - Compatible Models
  - Notes
* - `hermes3`
  - Qwen3, Hermes 3, NousResearch Hermes models
  - Uses `<tool_call>`/`</tool_call>` XML tags. Also works for Qwen3 models with tool calling.
* - `llama3`
  - Llama 3.x (with tool calling support)
  - Uses `<|python_tag|>` special token to indicate tool call start.
* - `phi4`
  - Phi-4
  - Uses `functools` marker for tool call content.
* - `mistral`
  - Mistral, Mistral Large, Codestral
  - Uses `[TOOL_CALLS]` special token.
* - `devstral`
  - Devstral
  - Uses `[TOOL_CALLS]` and `[ARGS]` special tokens with tool name and schema matching.
* - `qwen3coder`
  - Qwen3-Coder
  - Structured parsing with tool name schema mapping.
* - `gptoss`
  - GPT-OSS / Harmony-format models
  - Uses harmony-format channels (`<|channel|>`, `<|message|>`, `<|end|>`).
* - `lfm2`
  - LFM-2 (Liquid Foundation Models)
  - Uses custom start/end tags with tool-list and tool-args indicators.
```

## Reasoning Parsers

Reasoning parsers extract chain-of-thought / thinking content from model output and present it as reasoning tokens in the response.

```{list-table} Reasoning Parser to Model Mapping
:header-rows: 1
:widths: 15 35 50

* - Parser Name
  - Compatible Models
  - Notes
* - `qwen3`
  - Qwen3 (thinking mode)
  - Uses `<think>`/`</think>` tags to delimit reasoning content.
* - `gptoss`
  - GPT-OSS / Harmony-format models
  - Uses `<|channel|>analysis<|message|>` / `<|end|>` tags for reasoning content.
```

## Configuration Examples

### Graph configuration (graph.pbtxt)

```protobuf
node {
  calculator: "LLMCalculator"
  node_options: {
    [type.googleapis.com/mediapipe.LLMCalculatorOptions]: {
      models_path: "/models/qwen3"
      tool_parser: "hermes3"
      reasoning_parser: "qwen3"
    }
  }
}
```

### CLI deployment

```bash
docker run -d --rm -p 8000:8000 -v /models:/models \
  openvino/model_server:latest \
  --model_name qwen3 \
  --model_path /models/qwen3 \
  --tool_parser hermes3 \
  --reasoning_parser qwen3
```

## Chat Templates

When using tool calling, models may require a chat template different from the default one bundled with the model. We recommend using templates from the [vLLM repository](https://github.com/vllm-project/vllm/tree/main/examples) when available.

To use a custom template, save it as `chat_template.jinja` in the model directory. The model server will automatically pick it up instead of the default one.

## Guided Generation

When `tool_parser` is configured, you can additionally enable `enable_tool_guided_generation` to enforce that the model output conforms to the tool schemas provided in the request. This is supported for the following parsers:

- `hermes3`
- `llama3`
- `phi4`
- `devstral`

```{note}
If `enable_tool_guided_generation` is set but the model server fails to load any tool schema from the request, the request will still be processed with tool guided generation disabled.
```

## See Also

- [LLM Serving Reference](reference.md) — full configuration options for LLM servables
- [CLI Parameters](../parameters.md) — server startup parameters including `--tool_parser` and `--reasoning_parser`
- [Chat Completions API](../model_server_rest_api_chat.md) — REST API reference for chat endpoints with tool support
- [Responses API](../model_server_rest_api_responses.md) — REST API reference for responses endpoint with reasoning
