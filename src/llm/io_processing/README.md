# Output Parsers — Tool Call & Reasoning Extraction

This directory contains **tool parsers** and **reasoning parsers** that extract structured tool calls and reasoning traces from raw LLM output. Each parser targets a specific model family's output format and converts it into the OpenAI-compatible API response format.

## Configuration

Parsers are selected via the `tool_parser` and `reasoning_parser` fields in the model server graph configuration (MediaPipe `LLMCalculatorOptions`). The parser name strings map to implementations as follows:

| Parser Name    | Type      | Class                    | Target Models                     |
|----------------|-----------|--------------------------|-----------------------------------|
| `hermes3`      | Tool      | `Hermes3ToolParser`      | NousResearch Hermes 3 family      |
| `llama3`       | Tool      | `Llama3ToolParser`       | Meta Llama 3.x family             |
| `mistral`      | Tool      | `MistralToolParser`      | Mistral AI models                 |
| `phi4`         | Tool      | `Phi4ToolParser`         | Microsoft Phi-4 family            |
| `gptoss`       | Tool      | `GptOssToolParser`       | GPT-OSS (Harmony format) models   |
| `qwen3coder`   | Tool      | `Qwen3CoderToolParser`   | Qwen3-Coder family                |
| `devstral`     | Tool      | `DevstralToolParser`     | Devstral (Mistral variant) models |
| `qwen3`        | Reasoning | `Qwen3ReasoningParser`   | Qwen3 family (thinking mode)      |
| `gptoss`       | Reasoning | `GptOssReasoningParser`  | GPT-OSS (Harmony format) models   |

A tool parser and reasoning parser can be combined for models that produce both reasoning traces and tool calls (e.g. Qwen3 + `hermes3` tool parser + `qwen3` reasoning parser).

---

## Parsed Output Format

All parsers produce a `ParsedOutput` struct with these fields:

| Field        | Type                      | Description                                    |
|--------------|---------------------------|------------------------------------------------|
| `content`    | `std::string`             | Regular text content (non-tool, non-reasoning)  |
| `reasoning`  | `std::string`             | Extracted reasoning/thinking trace              |
| `toolCalls`  | `std::vector<ToolCall>`   | Extracted tool calls                            |

Each `ToolCall` contains:

| Field       | Type           | Description                              |
|-------------|----------------|------------------------------------------|
| `name`      | `std::string`  | Function name                            |
| `arguments` | `std::string`  | JSON string of function arguments        |
| `id`        | `std::string`  | Randomly generated alphanumeric call ID  |

---

## Tool Parsers

### 1. `hermes3` — Hermes3ToolParser

**Format:** XML-style `<tool_call>` tags wrapping JSON objects.

**Marker tags:** `<tool_call>` (start), `</tool_call>` (end)

**How it works:** The parser scans the model output for `<tool_call>` tags. The content between the opening and closing tags is parsed as a JSON object containing `name` and `arguments` fields. Multiple consecutive `<tool_call>` blocks are supported. Any text before the first `<tool_call>` tag is returned as `content`.

#### Example: Single tool call

**Model output:**
```
<tool_call>{"name": "example_tool", "arguments": {"arg1": "value1", "arg2": 42}}</tool_call>
```

**Parsed result:**
```
content:    ""
reasoning:  ""
toolCalls:
  [0] name:      "example_tool"
      arguments: {"arg1":"value1","arg2":42}
      id:        "a1b2c3d4e" (random)
```

#### Example: Content followed by tool call

**Model output:**
```
This is a content part and next will be a tool call.

<tool_call>{"name": "example_tool", "arguments": {"arg1": "value1", "arg2": 42}}</tool_call>
```

**Parsed result:**
```
content:    "This is a content part and next will be a tool call.\n\n"
reasoning:  ""
toolCalls:
  [0] name:      "example_tool"
      arguments: {"arg1":"value1","arg2":42}
      id:        "a1b2c3d4e" (random)
```

#### Example: Multiple tool calls

**Model output:**
```
<tool_call>{"name": "example_tool", "arguments": {"arg1": "value1", "arg2": 42}}</tool_call><tool_call>{"name": "another_tool", "arguments": {"param1": "data", "param2": true}}</tool_call><tool_call>{"name": "third_tool", "arguments": {"key": "value"}}</tool_call>
```

**Parsed result:**
```
content:    ""
reasoning:  ""
toolCalls:
  [0] name: "example_tool",   arguments: {"arg1":"value1","arg2":42}
  [1] name: "another_tool",   arguments: {"param1":"data","param2":true}
  [2] name: "third_tool",     arguments: {"key":"value"}
```

> **Note:** The closing `</tool_call>` tag is optional — the parser handles outputs where the model stops without emitting it.

---

### 2. `llama3` — Llama3ToolParser

**Format:** JSON objects separated by semicolons, prefixed by a special `<|python_tag|>` token (token ID `128010`).

**Marker:** Special token ID `128010` in the token stream, or text starting with `{` when tools are available.

**How it works:** The parser inspects the generated token IDs for the special `<|python_tag|>` token. Everything before that token is decoded as regular `content`. Everything after is decoded as tool call JSON. Multiple tool calls are separated by `;` (semicolon). The parser accepts both `"parameters"` and `"arguments"` as the key for function arguments (normalizes to `arguments`).

#### Example: Single tool call

**Token stream:** `[128010, <tokens encoding the JSON below>]`

**Model output (after special token):**
```json
{"name": "example_tool", "parameters": {"arg1": "value1", "arg2": 42}}
```

**Parsed result:**
```
content:    ""
reasoning:  ""
toolCalls:
  [0] name:      "example_tool"
      arguments: {"arg1":"value1","arg2":42}
      id:        "a1b2c3d4e" (random)
```

#### Example: Multiple tool calls (semicolon-separated)

**Model output (after special token):**
```
{"name": "example_tool", "parameters": {"arg1": "value1", "arg2": 42}};{"name": "another_tool", "parameters": {"param1": "data", "param2": true}};{"name": "third_tool", "parameters": {"key": "value"}}
```

**Parsed result:**
```
content:    ""
reasoning:  ""
toolCalls:
  [0] name: "example_tool",   arguments: {"arg1":"value1","arg2":42}
  [1] name: "another_tool",   arguments: {"param1":"data","param2":true}
  [2] name: "third_tool",     arguments: {"key":"value"}
```

> **Note:** Requires special tokens in streaming mode (`requiresStreamingWithSpecialTokens() = true`).

---

### 3. `mistral` — MistralToolParser

**Format:** JSON array prefixed by `[TOOL_CALLS]` special token (token ID `5`).

**Marker:** Special token ID `5` or text `[TOOL_CALLS]`.

**How it works:** The parser detects the `[TOOL_CALLS]` special token in the token stream. Everything after it is decoded and parsed as a JSON array of tool call objects. Each object must have `name` and `arguments` fields. The trailing `</s>` end-of-sequence token is stripped before parsing.

#### Example: Single tool call

**Model output:**
```
[TOOL_CALLS][{"name": "example_tool", "arguments": {"arg1": "value1", "arg2": 42}}]</s>
```

**Parsed result:**
```
content:    ""
reasoning:  ""
toolCalls:
  [0] name:      "example_tool"
      arguments: {"arg1":"value1","arg2":42}
      id:        "a1b2c3d4e" (random)
```

#### Example: Tool call with array arguments

**Model output:**
```
[TOOL_CALLS][{"name": "extractLastTransactionId", "arguments": {"filepath": "/var/log/db.log", "status": ["completed", "failed"], "encoding": "utf-8", "processFunction": "processFunction"}}]</s>
```

**Parsed result:**
```
content:    ""
reasoning:  ""
toolCalls:
  [0] name:      "extractLastTransactionId"
      arguments: {"filepath":"/var/log/db.log","status":["completed","failed"],"encoding":"utf-8","processFunction":"processFunction"}
      id:        "a1b2c3d4e" (random)
```

#### Example: Missing `[TOOL_CALLS]` prefix (still works)

**Model output:**
```
[{"name": "example_tool", "arguments": {"arg1": "value1", "arg2": 42}}]</s>
```

**Parsed result:**
```
content:    ""
toolCalls:
  [0] name: "example_tool",  arguments: {"arg1":"value1","arg2":42}
```

> **Note:** Requires special tokens in streaming mode.

---

### 4. `phi4` — Phi4ToolParser

**Format:** JSON array prefixed by the keyword `functools`.

**Marker:** Text `functools` followed by `[`.

**How it works:** The parser scans for the `functools` keyword in the decoded output. Everything after it is expected to be a JSON array of tool call objects (`[{...}, {...}]`). Text before `functools` is returned as `content`. Each object must have `name` and `arguments` fields. If multiple `functools` blocks appear, the output is treated as invalid and no tool calls are extracted.

#### Example: Single tool call

**Model output:**
```
functools[{"name": "example_tool", "arguments": {"arg1": "value1", "arg2": 42}}]
```

**Parsed result:**
```
content:    ""
reasoning:  ""
toolCalls:
  [0] name:      "example_tool"
      arguments: {"arg1":"value1","arg2":42}
      id:        "a1b2c3d4e" (random)
```

#### Example: Content before tool call

**Model output:**
```
This is a content part and next will be a tool call.

functools[{"name": "example_tool", "arguments": {"arg1": "value1", "arg2": 42}}]
```

**Parsed result:**
```
content:    "This is a content part and next will be a tool call.\n\n"
reasoning:  ""
toolCalls:
  [0] name:      "example_tool"
      arguments: {"arg1":"value1","arg2":42}
      id:        "a1b2c3d4e" (random)
```

#### Example: Multiple `functools` blocks (invalid — ignored)

**Model output:**
```
functools[{"name": "tool1", "arguments": {"a": 1}}]

Content in between

functools[{"name": "tool2", "arguments": {"b": 2}}]
```

**Parsed result:**
```
content:    ""
toolCalls:  [] (empty — multiple functools blocks are not supported)
```

> **Streaming state machine:** Uses 4 states: `AWAITING_START_TAG` → `AWAITING_TOOL_CALLS_OPENING_BRACKET` → `AWAITING_TOOL_CALL_OPENING_BRACE` → `PROCESSING_TOOL_CALL`. Tracks brace nesting to detect when each tool call JSON object is complete.

---

### 5. `gptoss` — GptOssToolParser

**Format:** Harmony channel-based format with `<|channel|>`, `<|message|>`, `<|call|>`, and `<|end|>` special tokens.

**Markers:** `<|channel|>analysis<|message|>` (reasoning), `<|channel|>commentary to=functions.{NAME}<|message|>` (tool call), `<|channel|>final<|message|>` (final content).

**How it works:** The parser interprets the Harmony protocol, which uses "channels" to semantically tag different output types. Tool calls are indicated by a `commentary` channel with `to=functions.{function_name}`, followed by JSON arguments between `<|message|>` and `<|end|>` / `<|call|>` / `<|return|>` tags. The function name is extracted from the channel header.

#### Example: Simple tool call

**Model output (using special token IDs):**
```
<|channel|>commentary to=functions.hello<|message|>{"Hello": "world!"}<|end|>
```

**Parsed result:**
```
content:    ""
reasoning:  ""
toolCalls:
  [0] name:      "hello"
      arguments: {"Hello":"world!"}
      id:        "a1b2c3d4e" (random)
```

#### Example: Tool call with constrain tag (ignored)

**Model output:**
```
<|channel|>commentary to=functions.hello<|constrain|>json<|message|>{"Hello": "world!"}<|end|>
```

**Parsed result:**
```
content:    ""
reasoning:  ""
toolCalls:
  [0] name:      "hello"
      arguments: {"Hello":"world!"}
      id:        "a1b2c3d4e" (random)
```

The `<|constrain|>json` tag is consumed but does not affect parsing.

#### Example: Content + reasoning + tool call

**Model output:**
```
<|channel|>analysis<|message|>Let me think about this...<|end|><|channel|>final<|message|>Here is my response<|end|><|channel|>commentary to=functions.get_weather<|message|>{"city": "Paris"}<|call|>
```

**Parsed result:**
```
content:    "Here is my response"
reasoning:  "Let me think about this..."
toolCalls:
  [0] name:      "get_weather"
      arguments: {"city":"Paris"}
```

> **Note:** Requires special tokens in streaming mode. The `analysis` channel maps to reasoning content, `final` channel maps to regular content, and `commentary to=functions.*` maps to tool calls.

---

### 6. `qwen3coder` — Qwen3CoderToolParser

**Format:** Nested XML tags with `<tool_call>`, `<function=...>`, and `<parameter=...>` structure.

**Markers:** `<tool_call>`, `<function=`, `<parameter=`

**How it works:** The parser uses a 7-state finite state machine to parse XML-style nested tags. Function names are extracted from `<function=NAME>` tags and parameters from `<parameter=KEY>VALUE</parameter>` blocks. **This parser is schema-aware** — it uses the provided tool schemas to determine parameter types (string, number, boolean, array, object) and coerces values accordingly.

#### Example: Single string parameter

**Model output:**
```xml
<tool_call>
<function=string_tool>
<parameter=arg1>
value1
</parameter>
</function>
</tool_call>
```

**Parsed result:**
```
content:    ""
reasoning:  ""
toolCalls:
  [0] name:      "string_tool"
      arguments: {"arg1":"value1"}
      id:        "a1b2c3d4e" (random)
```

#### Example: Multiple parameters with different types

**Model output:**
```xml
<tool_call>
<function=string_int_float_tool>
<parameter=arg1>value1</parameter>
<parameter=arg2>42</parameter>
<parameter=arg3>52.32</parameter>
</function>
</tool_call>
```

**Parsed result (types inferred from schema):**
```
content:    ""
reasoning:  ""
toolCalls:
  [0] name:      "string_int_float_tool"
      arguments: {"arg1":"value1","arg2":42,"arg3":52.32}
```

#### Example: Type coercion via schema

When a tool schema declares all parameters as `string` type, numeric and boolean literals are coerced to strings:

**Model output (with `stringx7_tool` schema — all string params):**
```xml
<tool_call>
<function=stringx7_tool>
<parameter=arg1>true</parameter>
<parameter=arg2>-13</parameter>
</function>
</tool_call>
```

**Parsed result:**
```
toolCalls:
  [0] name:      "stringx7_tool"
      arguments: {"arg1":"true","arg2":"-13"}
```

Note: `true` and `-13` are emitted as JSON strings `"true"` and `"-13"` because the schema specifies string type.

#### Example: JSON object parameter

**Model output:**
```xml
<tool_call>
<function=object_tool>
<parameter=arg1>{"a": 1, "b": {"c": "asd"}}</parameter>
</function>
</tool_call>
```

**Parsed result:**
```
toolCalls:
  [0] name:      "object_tool"
      arguments: {"arg1":{"a":1,"b":{"c":"asd"}}}
```

#### Example: Multiline parameter value

**Model output:**
```xml
<tool_call>
<function=string_tool>
<parameter=arg1>
value1line1
value1line2
</parameter>
</function>
</tool_call>
```

**Parsed result:**
```
toolCalls:
  [0] name:      "string_tool"
      arguments: {"arg1":"value1line1\nvalue1line2"}
```

> **Note:** The opening `<tool_call>` tag is optional — the parser can also start parsing from `<function=...>` directly.

---

### 7. `devstral` — DevstralToolParser

**Format:** `[TOOL_CALLS]` and `[ARGS]` special tokens with function name between them and JSON arguments after `[ARGS]`.

**Markers:** `[TOOL_CALLS]` (tool call start), `[ARGS]` (arguments start), `</s>` (end).

**How it works:** The parser detects the `[TOOL_CALLS]` special token in the token stream. The text between `[TOOL_CALLS]` and `[ARGS]` is the function name. The text after `[ARGS]` until `</s>` (or end of output) is the JSON arguments string. Text before `[TOOL_CALLS]` is returned as `content`. **This parser is schema-aware** — it validates tool names against the provided tool schemas.

#### Example: Single tool call

**Model output:**
```
[TOOL_CALLS]example_tool[ARGS]{"arg1":"value1","arg2":42}</s>
```

**Parsed result:**
```
content:    ""
reasoning:  ""
toolCalls:
  [0] name:      "example_tool"
      arguments: {"arg1":"value1","arg2":42}
      id:        "a1b2c3d4e" (random)
```

#### Example: Content before tool call

**Model output:**
```
Reasoning before tool call [TOOL_CALLS]example_tool[ARGS]{"arg1":"value1","arg2":42}</s>
```

**Parsed result:**
```
content:    "Reasoning before tool call "
reasoning:  ""
toolCalls:
  [0] name:      "example_tool"
      arguments: {"arg1":"value1","arg2":42}
      id:        "a1b2c3d4e" (random)
```

#### Example: Empty arguments

**Model output:**
```
Reasoning before tool call [TOOL_CALLS]example_tool[ARGS]</s>
```

**Parsed result:**
```
content:    "Reasoning before tool call "
reasoning:  ""
toolCalls:
  [0] name:      "example_tool"
      arguments: {}
      id:        "a1b2c3d4e" (random)
```

#### Example: Invalid tag order (no tool calls extracted)

**Model output:**
```
Reasoning before tool call [ARGS]example_tool[TOOL_CALLS]{"arg1":"value1","arg2":42}</s>
```

**Parsed result:**
```
content:    "Reasoning before tool call example_tool{\"arg1\":\"value1\",\"arg2\":42}"
toolCalls:  [] (empty — tags in wrong order)
```

#### Example: Missing `[ARGS]` tag (no tool calls extracted)

**Model output:**
```
Some content [TOOL_CALLS]example_tool{"arg1":"value1","arg2":42}</s>
```

**Parsed result:**
```
content:    "Some content example_tool{\"arg1\":\"value1\",\"arg2\":42}"
toolCalls:  [] (empty — [ARGS] tag is required)
```

> **Note:** Both `[TOOL_CALLS]` and `[ARGS]` tags are required and must appear in this order. Requires special tokens in streaming mode.

---

## Reasoning Parsers

### 1. `qwen3` — Qwen3ReasoningParser

**Format:** XML-style `<think>` / `</think>` tags wrapping the reasoning trace.

**Markers:** `<think>` (start), `</think>` (end)

**How it works:** The parser finds `<think>` and `</think>` tags in the model output. The text between them is extracted as `reasoning`. The tags and their content are removed from the final output, and the remaining text is returned as `content`. This parser is commonly combined with a tool parser (e.g. `hermes3`) to handle models that produce both reasoning and tool calls.

#### Example: No thinking

**Model output:**
```
<tool_call>{"name": "example_tool", "arguments": {"arg1": "value1", "arg2": 42}}</tool_call>
```

**Parsed result:**
```
content:    ""
reasoning:  ""
toolCalls:  (handled by the combined tool parser, e.g. hermes3)
```

#### Example: Thinking followed by tool call

**Model output:**
```
<think>Thinking about the tool call</think><tool_call>{"name": "example_tool", "arguments": {"arg1": "value1", "arg2": 42}}</tool_call>
```

**Parsed result:**
```
content:    ""
reasoning:  "Thinking about the tool call"
toolCalls:
  [0] name: "example_tool",  arguments: {"arg1":"value1","arg2":42}
```

#### Example: Thinking followed by multiple tool calls

**Model output:**
```
<think>Thinking text</think><tool_call>{"name": "tool1", "arguments": {"arg1": "v1"}}</tool_call><tool_call>{"name": "tool2", "arguments": {"arg2": "v2"}}</tool_call><tool_call>{"name": "tool3", "arguments": {"arg3": "v3"}}</tool_call>
```

**Parsed result:**
```
content:    ""
reasoning:  "Thinking text"
toolCalls:
  [0] name: "tool1",  arguments: {"arg1":"v1"}
  [1] name: "tool2",  arguments: {"arg2":"v2"}
  [2] name: "tool3",  arguments: {"arg3":"v3"}
```

#### Example: Thinking followed by plain content (no tool calls)

**Model output:**
```
<think>Let me reason about this</think>The answer is 42.
```

**Parsed result:**
```
content:    "The answer is 42."
reasoning:  "Let me reason about this"
toolCalls:  []
```

> **Streaming:** In streaming mode, chunks containing `<think>` or `</think>` tags are suppressed (return `nullopt`). Reasoning chunks are returned with a `reasoning_content` field in the delta, and regular content chunks with a `content` field.

---

### 2. `gptoss` — GptOssReasoningParser

**Format:** Harmony channel-based format — same protocol as the GptOss tool parser.

**Markers:** `<|channel|>analysis<|message|>` (reasoning), `<|channel|>final<|message|>` (content), `<|channel|>commentary<|message|>` (preamble/content). Terminated by `<|end|>` or `<|return|>`.

**How it works:** The parser uses the Harmony protocol's channel semantics to distinguish reasoning from content. The `analysis` channel is treated as reasoning content, while `final` and `commentary` channels are treated as regular content. This parser is typically combined with the `gptoss` tool parser.

#### Example: Simple content (final channel)

**Model output (special tokens):**
```
<|channel|>final<|message|>Hello, world!<|end|>
```

**Parsed result:**
```
content:    "Hello, world!"
reasoning:  ""
```

#### Example: Reasoning + content

**Model output (special tokens):**
```
<|channel|>analysis<|message|>I need to think about this carefully.<|end|><|channel|>final<|message|>The answer is 42.<|end|>
```

**Parsed result:**
```
content:    "The answer is 42."
reasoning:  "I need to think about this carefully."
```

#### Example: Full multi-part output (reasoning + preamble + tool calls + final)

**Model output (special tokens):**
```
<|channel|>analysis<|message|>Analyzing the request...<|end|><|channel|>commentary<|message|>Let me check the weather.<|end|><|channel|>commentary to=functions.get_weather<|message|>{"city": "Paris"}<|call|><|channel|>final<|message|>I've checked the weather for you.<|end|>
```

**Parsed result:**
```
content:    "Let me check the weather.I've checked the weather for you."
reasoning:  "Analyzing the request..."
toolCalls:
  [0] name: "get_weather",  arguments: {"city":"Paris"}
```

> **Streaming states:** `UNKNOWN` → `READING_REASONING` (for analysis channel) or `READING_CONTENT` (for final/commentary channels). Streaming deltas use `reasoning_content` for analysis channel output and `content` for final/commentary channel output.

---

## Streaming Mode

All parsers support streaming via the `parseChunk()` method. The streaming API returns `std::optional<rapidjson::Document>` — `nullopt` means no delta is ready yet (e.g. the parser is still buffering to detect tags).

### Streaming Delta Format — Tool Calls

**First chunk (function name identified):**
```json
{
  "delta": {
    "tool_calls": [{
      "id": "abc123def",
      "type": "function",
      "index": 0,
      "function": { "name": "get_weather" }
    }]
  }
}
```

**Subsequent chunks (arguments streaming):**
```json
{
  "delta": {
    "tool_calls": [{
      "index": 0,
      "function": { "arguments": "{\"city\":" }
    }]
  }
}
```

### Streaming Delta Format — Content

```json
{ "delta": { "content": "text chunk" } }
```

### Streaming Delta Format — Reasoning

```json
{ "delta": { "reasoning_content": "thinking chunk" } }
```

### Special Token Requirements

Some parsers require special tokens to be present in the streaming token stream for correct detection:

| Parser       | Requires Special Tokens |
|--------------|------------------------|
| `hermes3`    | No                     |
| `llama3`     | Yes                    |
| `mistral`    | Yes                    |
| `phi4`       | No                     |
| `gptoss`     | Yes                    |
| `qwen3coder` | No                     |
| `devstral`   | Yes                    |
| `qwen3`      | No                     |

---

## Quick Reference: Model Output → Parser

| Model Output Pattern                                      | Parser to Use |
|-----------------------------------------------------------|---------------|
| `<tool_call>{"name":...}</tool_call>`                     | `hermes3`     |
| `<\|python_tag\|>{"name":..., "parameters":...}`         | `llama3`      |
| `[TOOL_CALLS][{"name":..., "arguments":...}]`            | `mistral`     |
| `functools[{"name":..., "arguments":...}]`                | `phi4`        |
| `<\|channel\|>commentary to=functions.X<\|message\|>...` | `gptoss`      |
| `<tool_call><function=X><parameter=K>V</parameter>...`    | `qwen3coder`  |
| `[TOOL_CALLS]func_name[ARGS]{...}`                        | `devstral`    |
| `<think>...</think>`                                      | `qwen3`       |
| `<\|channel\|>analysis<\|message\|>...`                   | `gptoss`      |
