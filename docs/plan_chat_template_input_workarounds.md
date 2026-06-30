# Plan: Chat Template Input Workarounds & Auto-Detection

## Problem Statement

OVMS currently requires manual configuration of `tool_parser` and `reasoning_parser` via the MediaPipe graph proto. There is no:
- Automatic detection of model/template capabilities
- Input transformation before template application (e.g. Gemma requiring object arguments)
- Auto-detection of which tool/reasoning parser to use based on template content

Both **llama.cpp** and **minja** solve these problems via "dry run" probing and model-specific workarounds. This plan adapts those techniques for OVMS.

---

## Background: How llama.cpp & minja Do It

### llama.cpp approach
1. **Needle-based dry runs** — renders template with probe data, tracks which fields are accessed (via `stats.used`)
2. **String pattern matching** — searches template source for unique markers (`<|tool_call>call:'`, `[TOOL_CALLS]`, `<|channel|>`, etc.) to identify model family
3. **Workarounds applied pre-render** — `func_args_not_string`, `requires_non_null_content`, `system_message_not_supported`, `map_developer_role_to_system`, `convert_tool_responses_gemma4`
4. **Autoparser** — differential analysis (render twice, diff output) to detect reasoning/tool-call format

### minja approach
1. **`try_raw_render` + needles** — renders template with sentinel strings, checks if they appear in output
2. **`<parameter=argument_needle>`** — detects coder-style XML parameter templates (Qwen3-Coder)
3. **Capability struct** — `chat_template_caps` populated at construction time: `supports_tools`, `requires_object_arguments`, `requires_non_null_content`, etc.
4. **Polyfills** — automatic fallbacks when template lacks native support (inject tool definitions into system prompt, merge system into user, etc.)

---

## Current OVMS Architecture (relevant parts)

```
┌─────────────────────────────────────────────────────────────────────┐
│ Request Flow                                                        │
│                                                                     │
│  loadRequest → parseRequest → prepareInputs → scheduleExecution     │
│                     │              │                                 │
│              parseMessages()  applyChatTemplate()                    │
│              parseTools()     (Jinja or GenAI)                       │
│                     │              │                                 │
│              ensureArguments  ─────┼──── NO workarounds today        │
│              InToolCalls()         │                                 │
│                                   ▼                                 │
│                          GenerationConfigBuilder                     │
│                          OutputParser (tool + reasoning)             │
└─────────────────────────────────────────────────────────────────────┘
```

| Servable Type | Template Applicator | Notes |
|---|---|---|
| LM (legacy) | Python Jinja (primary) / GenAI C++ (fallback) | |
| LM_CB | Python Jinja (primary) / GenAI C++ (fallback) | Main production path |
| VLM (legacy) | GenAI C++ only | |
| VLM_CB | GenAI C++ only | |

Key files:
- `src/llm/servable.cpp` — `prepareInputs()` calls template applicator
- `src/llm/py_jinja_template_processor.cpp` — Python Jinja path
- `src/llm/io_processing/output_parser.hpp` — tool/reasoning parsing
- `src/llm/io_processing/generation_config_builder.hpp` — stop strings & guided generation
- `src/llm/language_model/continuous_batching/servable_initializer.cpp` — reads `tool_parser`/`reasoning_parser` from proto

---

## Proposed Design

### New Component: `ChatTemplateAnalyzer`

A singleton-per-servable object created at initialization time that:
1. Reads the chat template source
2. Detects template capabilities (what the template supports/requires)
3. Determines which tool/reasoning parser matches the template
4. Provides workaround functions to transform inputs before template application

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Initialization (servable_initializer)                                     │
│                                                                           │
│  Load chat template source ──► ChatTemplateAnalyzer                       │
│                                   │                                       │
│                                   ├── detectCaps()  (dry-run probing)     │
│                                   ├── detectParsers() (pattern matching)  │
│                                   └── store in GenAiServableProperties    │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ Request time (prepareInputs)                                              │
│                                                                           │
│  messages/tools ──► InputWorkarounds::apply(caps, messages, tools) ───►   │
│                          │                                                │
│                          ├── func_args_to_object()                         │
│                          ├── ensure_non_null_content()                     │
│                          ├── convert_tool_responses_gemma4()               │
│                          ├── convert_typed_content()                       │
│                          └── (future workarounds)                          │
│                                                                           │
│                     ──► applyChatTemplate(modified messages)               │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Design

### Phase 1: Template Capability Detection (`ChatTemplateCaps`)

#### 1.1 Data Structure

```cpp
// src/llm/chat_template_caps.hpp

struct ChatTemplateCaps {
    bool supports_system_role = true;
    bool supports_tools = false;
    bool supports_tool_calls = false;
    bool supports_tool_responses = false;
    bool requires_object_arguments = false;     // Gemma: args as dict not string
    bool requires_non_null_content = false;     // tool_call messages need content=""
};
```

#### 1.2 Detection Strategy (two-tier)

**Tier 1: Pattern matching on template source** (fast, no execution needed)
- Search for known unique strings to identify model family
- Maps directly to a known `ChatTemplateCaps` preset + parser name

| Pattern in template source | Model family | Tool parser | Reasoning parser |
|---|---|---|---|
| `<\|python_tag\|>` | Llama 3.x | `llama3` | — |
| `<tool_call>` + `</tool_call>` | Hermes/Qwen | `hermes3` | — |
| `<\|tool_call\|>` + no `<tool_call>` | Mistral | `mistral` | — |
| `<\|tool_call\|>` + `function_calls` | DeepSeek | (future) | — |
| `<\|channel\|>` | GPT-OSS | `gptoss` | `gptoss` |
| `'<\|tool_call>call:'` | Gemma 4 | `gemma4` | `gemma4` |
| `<parameter=` | Qwen3-Coder | `qwen3coder` | — |
| `[TOOL_CALLS]` + `[SYSTEM_PROMPT]` | Mistral Large / Ministral | `mistral` | — |
| `<\|tool▁call▁begin\|>` | Phi-4 | `phi4` | — |
| `content.split('</think>')` OR `<think>` in generation prompt | Qwen3/DeepSeek-R1-distill | — | `qwen3` |
| `<tool_call>` + `[TOOL_RESULTS]` | Devstral | `devstral` | — |
| `<\|assistant_tool_call\|>` | LFM2 | `lfm2` | — |

**Tier 2: Dry-run probing** (when pattern matching is inconclusive)
- Only for the Python Jinja path (we control execution)
- Render template with needle-containing test messages
- Check output for presence/absence of needles

> **Decision**: For initial implementation, Tier 1 (pattern matching) covers all currently supported parsers. Tier 2 can be added later for unknown templates.

#### 1.3 Integration Points

- **Initialization**: `ChatTemplateAnalyzer` runs in `servable_initializer` after loading the template source
- **Storage**: Results stored in `GenAiServableProperties` as `ChatTemplateCaps caps` + auto-detected `toolParserName`/`reasoningParserName`
- **Override**: If user explicitly sets `tool_parser`/`reasoning_parser` in proto, those take precedence over auto-detection

---

### Phase 2: Input Workarounds (`InputWorkarounds`)

#### 2.1 Workaround Functions

```cpp
// src/llm/input_workarounds.hpp

namespace ovms {
namespace input_workarounds {

// Convert tool_call arguments from JSON string to parsed object
// Triggered by: caps.requires_object_arguments
void funcArgsToObject(rapidjson::Document& doc);

// Ensure tool_call messages have non-null content field
// Triggered by: caps.requires_non_null_content
void ensureNonNullContent(rapidjson::Document& doc);

// Restructure tool response messages for Gemma4 format
// Triggered by: detected model == gemma4
void convertToolResponsesGemma4(rapidjson::Document& doc);

// Apply all relevant workarounds based on caps
void applyAll(const ChatTemplateCaps& caps, const std::string& modelFamily,
              rapidjson::Document& doc);

} // namespace input_workarounds
} // namespace ovms
```

#### 2.2 Where Workarounds Are Applied

Two integration points (mirroring the two applicator paths):

**Python Jinja path** (`PyJinjaTemplateProcessor::applyChatTemplate`):
- Workarounds modify the `processedJson` / `requestBody` JSON **before** it's passed to the Python template renderer
- The JSON already contains `messages` and `tools` arrays

**GenAI C++ path** (`tokenizer.apply_chat_template`):
- Workarounds modify the `ChatHistory` and/or re-serialize tool_calls arguments **before** calling the tokenizer
- May need a helper to serialize/deserialize between `ChatHistory` and a mutable JSON representation

#### 2.3 Call Site

In `GenAiServable::prepareInputs()` (and VLM variants), **after** `parseRequest()` but **before** template application:

```cpp
// After parseRequest populates chatHistory/processedJson:
auto& caps = getProperties()->chatTemplateCaps;
auto& modelFamily = getProperties()->detectedModelFamily;

#if (PYTHON_DISABLE == 0)
    // Modify the JSON document that will be passed to Python Jinja
    input_workarounds::applyAll(caps, modelFamily, executionContext->apiHandler->getMutableProcessedJson());
#else
    // Modify ChatHistory in-place for GenAI path
    input_workarounds::applyAllToHistory(caps, modelFamily, executionContext->apiHandler->getChatHistory());
#endif
```

---

### Phase 3: Auto-Detection of Tool/Reasoning Parsers

#### 3.1 Goal

Eliminate the need for users to manually specify `tool_parser` and `reasoning_parser` in the graph proto for common models.

#### 3.2 Implementation

`ChatTemplateAnalyzer::detectParsers()` returns:
- `std::optional<std::string> detectedToolParser`
- `std::optional<std::string> detectedReasoningParser`

In `servable_initializer`:
```cpp
if (!nodeOptions.has_tool_parser()) {
    // Auto-detect from template
    auto detected = analyzer.detectParsers(templateSource);
    if (detected.toolParser.has_value()) {
        properties->toolParserName = detected.toolParser.value();
        SPDLOG_LOGGER_INFO(logger, "Auto-detected tool_parser: {}", properties->toolParserName);
    }
}
// Same for reasoning_parser
```

#### 3.3 Logging & Transparency

- Log at INFO level when auto-detection fires and what was detected
- Log at WARNING when auto-detection fails (unknown template) and no parser was configured
- Model card / graph node description should still document the manual override

---

### Phase 4: Future — Advanced Probing (Tier 2 Dry-Run)

For templates that don't match any known pattern:

1. **Python path**: Execute the template with needle messages via `PyJinjaTemplateProcessor`, check output
2. **GenAI path**: Use `tokenizer.apply_chat_template()` with probe `ChatHistory`, check output
3. Populate `ChatTemplateCaps` from the probe results
4. Optionally: generate tool-call example by differential rendering (like minja)

This is deferred because:
- All currently supported models have recognizable template patterns
- Dry-run adds initialization latency
- GenAI C++ tokenizer doesn't expose field-access tracking (unlike minja's stat-based approach)

---

## File Structure

```
src/llm/
├── chat_template_caps.hpp              # ChatTemplateCaps struct
├── chat_template_analyzer.hpp          # ChatTemplateAnalyzer class declaration
├── chat_template_analyzer.cpp          # Pattern matching + detection logic
├── input_workarounds.hpp               # Input transformation functions
├── input_workarounds.cpp               # Implementations
└── io_processing/
    └── (existing parsers unchanged)
```

---

## Implementation Order

| Step | Description | Effort | Dependencies |
|------|-------------|--------|--------------|
| 1 | Define `ChatTemplateCaps` struct | S | None |
| 2 | Implement `ChatTemplateAnalyzer` with Tier 1 pattern matching | M | Step 1 |
| 3 | Integrate auto-detection into all 4 servable initializers | M | Step 2 |
| 4 | Implement `input_workarounds::funcArgsToObject` (Gemma case) | S | Step 1 |
| 5 | Implement `input_workarounds::ensureNonNullContent` | S | Step 1 |
| 6 | Wire workarounds into `prepareInputs()` for both Jinja & GenAI paths | M | Steps 4-5 |
| 7 | Add unit tests for analyzer + workarounds | M | Steps 2-6 |
| 8 | (Future) Tier 2 dry-run probing | L | Step 6 |
| 9 | (Future) Auto-generate tool-call example for unsupported templates | L | Step 8 |

---

## Interaction with Existing Code

### What changes

| Component | Change |
|---|---|
| `GenAiServableProperties` | Add `ChatTemplateCaps caps`, `std::string detectedModelFamily` |
| `servable_initializer` (all 4 variants) | Call `ChatTemplateAnalyzer` after loading template; use auto-detected parser if none configured |
| `GenAiServable::prepareInputs()` | Call `input_workarounds::applyAll()` before template application |
| `VLM servable::prepareInputs()` | Same workaround call |
| `PyJinjaTemplateProcessor::applyChatTemplate()` | Receives already-transformed JSON (no change to Python code) |

### What does NOT change

- Output parsers (`OutputParser`, all model-specific parsers)
- `GenerationConfigBuilder` (stop strings, guided generation)
- Python Jinja template rendering logic
- GenAI tokenizer API usage
- Existing proto fields (`tool_parser`, `reasoning_parser`) — they become optional overrides

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Pattern matching gives false positive for custom/fine-tuned templates | Manual proto override always takes precedence; log detection result |
| Template source not available at initialization (e.g. embedded in tokenizer binary) | GenAI tokenizer exposes chat template string via `get_chat_template()`; use that |
| Workarounds break valid inputs | Apply workarounds only when `caps` indicates the template requires them; add tests for each workaround with real template examples |
| Performance overhead from JSON manipulation | Workarounds operate on already-parsed `rapidjson::Document`; negligible vs. LLM inference time |
| Two code paths (Jinja vs GenAI) need consistent workarounds | Shared `input_workarounds` module with path-specific entry points; test both paths |

---

## Open Questions

1. **Should auto-detection be opt-in or opt-out?** Proposed: opt-out (auto-detect by default, explicit proto value overrides). If a user sets `tool_parser: ""` (empty string), disable tool parsing entirely.

2. **Where to get template source for GenAI-only path (VLM)?** Use `tokenizer.get_chat_template()` or read `chat_template.jinja` / `tokenizer_config.json` directly from the model directory.

3. **Should workarounds also apply to the Responses API path?** Yes — same template, same requirements.

4. **Should we log a warning if auto-detection finds a parser but the user configured a different one?** Yes, at DEBUG level — the user's choice is intentional.

5. **Gemma4 `convert_tool_responses_gemma4` — is this needed for OVMS?** Depends on whether the Gemma4 template in OpenVINO GenAI handles tool responses natively. Needs testing.
