# OVMS family-port theory — LFM2.5-24B-A2B and experimental GLM

Status: **THEORY ONLY**. Checklist answers for a later porter. No parser port, no generation-policy code, no acceptance claim.

Companion to `OVMS-MODEL-FAMILY-ADAPTATION-PROTOCOL.md` (semantic-refit 2026.4, supplemented 2026-10-23). This note answers that protocol's seven-point pre-code list. It does not update A0 pins, `STATE.md`, or the acceptance ladder. Nothing here is `ACCEPTED`.

Sources read for this note: the official Hugging Face files named under each family, optimum-intel PR 1699 as fetched 2026-10-23 (draft, open, not merged), and the LFM / phase-owner code already in this tree. Local weight directories `C:\llm\models` and `C:\models` were listed. No LFM and no GLM checkpoint is present. **Weights are absent** for every checkpoint below.

## Gemma lesson (applies to both families)

`OutputParser` leaves `REASONING` only on the reasoning closer, except the Gemma4 handoff. That handoff, on `<|tool_call>` before `<channel|>`, closes the **current** reasoning segment and gives the tail to the tool parser. It does not open a new reasoning segment after the tool envelope. Ten alternating thought+tool cycles stay red (`Gemma4OutputParserTest.ParseTenAlternatingThoughtAndToolCalls`): calls parse, later thoughts leak, phase stays `TOOL_CALLS_WAITING_FOR_TOOL`.

`TOOL_CALLS_WAITING_FOR_TOOL` looks for another tool start, or for content-turn start tags. It does not look for a new reasoning open. A family that emits think, then a call, then think, then a call **inside one generation** needs a phase rule that re-enters `REASONING` after the envelope. Copying the Gemma handoff does not install that rule. A tool tag inside a still-open reasoning span stays reasoning, unless that family's own format starts the call before the closer.

## Family 1 — Liquid AI LFM2.5-24B-A2B

Checkpoint: `LiquidAI/LFM2.5-24B-A2B`. Architecture in `config.json`: `Lfm2MoeForCausalLM`, `model_type` `lfm2_moe`, vocab 128000, bos 124894 (`<|startoftext|>`), eos 124900 (`<|im_end|>`), pad 124893 (`<|pad|>`). Text-only MoE. This is not the small dense LFM2 / LFM2.5 checkpoint the in-tree tests name.

Official chat template: `chat_template.jinja` on that repo, 5443 bytes, md5 `db0096f3281573383decd027773bcb68`. Generation suffix, `add_generation_prompt` true, is exactly `<|im_start|>assistant\n<think>` (newline before `<think>`, nothing after it). There is no `enable_thinking` switch. The model card calls this checkpoint a pure reasoning model: the template always opens `<think>`.

The in-tree files `extras/chat_template_examples/chat_template_lfm25.jinja` and `src/test/llm/chat_templates/chat_template_lfm25.jinja` are the same as each other (5234 bytes, md5 `754acb987cfa2831bc6e3a2bb7a521b3`) and are **not** this template. Their generation suffix is `<|im_start|>assistant\n` with no `<think>`. They also accept a JSON string of arguments. The 24B template raises if arguments are a JSON string and requires a mapping. Pin the 5443-byte file. A visually similar Jinja file is not this contract.

### What `src/llm/io_processing/lfm2/` already covers

`Lfm2ToolParser` (tool parser name `lfm2`) parses one wire for LFM2 and LFM2.5:

- Open `<|tool_call_start|>`, close `<|tool_call_end|>`.
- Body is a Python list: `[name(arg=value, ...)]`. Several calls inside one pair are separated by `", "`.
- Empty call is `name()`. Strings may be single-quoted. `True` / `False` / `None`-style tokens are normalized. Objects and arrays are accepted.
- The parser also accepts a second `<|tool_call_start|>` envelope after the first. That is parser tolerance. The 24B template does not serialize that shape.
- `<|im_end|>` is cut from content. `DefaultContentParser` for tool parser `lfm2` erases the same token. It is an erase list, not a content-turn start tag.

`Lfm25ReasoningParser` (reasoning parser name `lfm2`, class `Lfm25ReasoningParser`) subclasses `Qwen3ReasoningParser` with start `<think>`, end `</think>`, `needsSpecialTokens` and `defaultDecodingWithSpecialTokens` set. The phase owner does not end reasoning on `<|tool_call_start|>`. There is no LFM Gemma-style handoff. `GenerationConfigBuilder` has no `lfm2` branch, so `enable_tool_guided_generation` does not install a grammar.

Tests in `lfm25_output_parser_test.cpp` cover a closed think, then a call, then optional content (`StreamingWithReasoningAndToolCall`, `StreamingWithReasoningAndToolCallAndContent`), one list with two calls, repeated envelopes, empty args, and `<|im_end|>`. They load tokenizer directory `src/test/llm_testing/LiquidAI/LFM2.5-8B-A1B`. That directory is not in this tree. Those tests are not a 24B run. There is no LFM test that a draft `<|tool_call_start|>` inside an open think stays in `reasoning_content`.

The header comment that LFM2.5 uses token id 124905 for `<|tool_call_start|>`, and the test constants 124901 / 124902 for `<think>` / `</think>`, match the **24B** `tokenizer.json` ids below. The comment also says both think delimiters are special tokens. On this checkpoint `special` is false. They are still one id each.

### Tokenizer ids (official `tokenizer.json`, not a local OV encode)

| Token | Id | `special` field |
| --- | --- | --- |
| `<|pad|>` | 124893 | true |
| `<|startoftext|>` | 124894 | true |
| `<|im_start|>` | 124899 | true |
| `<|im_end|>` | 124900 | true |
| `<think>` | 124901 | false |
| `</think>` | 124902 | false |
| `<|tool_list_start|>` | 124903 | true |
| `<|tool_list_end|>` | 124904 | true |
| `<|tool_call_start|>` | 124905 | false |
| `<|tool_call_end|>` | 124906 | false |

The model card calls `<|tool_call_start|>` and `<|tool_call_end|>` special tokens. The tokenizer file marks `special: false` and still assigns one id. They are not multi-token strings in this JSON. Whether the 2026.4 GenAI tokenizer skips them is **UNKNOWN** until a local encode. `<|tool_list_start|>` / `<|tool_list_end|>` are real ids. The 24B chat template never writes them. It writes ASCII `[` and `]`. Whether the model emits 124903 / 124904 is **UNKNOWN**.

### Checklist

1. Reasoning open `<think>` (id 124901). Reasoning close `</think>` (id 124902). Tool open `<|tool_call_start|>` (id 124905). Tool close `<|tool_call_end|>` (id 124906). Argument wire inside the pair: `[name(key=value, ...)]`, with `", "` between calls. String values use single quotes in the template. Mappings and lists use JSON. Empty arguments are `name()`.

2. A tool call is legal only after `</think>`. It is not legal before the closer. History renders optional `<think>…</think>`, then content, then one tool envelope. The README tool example omits the think span and shows the call directly under `<|im_start|>assistant`. That example is abbreviated. The template line that starts generation, and the model-card note that this checkpoint always thinks before it answers, are the contract. A `<|tool_call_start|>` while the think span is open stays reasoning. Do not add an LFM handoff that closes reasoning on that tag.

3. Several calls in one turn are one envelope. The template joins them with `", "` inside a single `[...]` pair. They are not separate `<|tool_call_start|>` envelopes. The existing parser's acceptance of repeated envelopes is tolerance, not the canonical wire.

4. Thinking on is the only generation mode. Suffix: `<|im_start|>assistant\n<think>`. The model writes the thought and the closer. There is no thinking-off suffix and no template-written `</think>`. `preserve_thinking` defaults false and only affects replay of past assistant `thinking` / `reasoning` / `reasoning_content`. It does not change the generation suffix. After the last user message, that past thinking is kept even when `preserve_thinking` is false.

5. Order the header can express, thinking always open, guided auto only: `Concat(Tag(begin empty, AnyText, end "</think>"), TriggeredTags)`. Trigger `"<|tool_call_start|>"`, `at_least_one` false, `stop_after_first` true, one tag whose begin is `<|tool_call_start|>[`, whose end is `]<|tool_call_end|>`. The trigger sits after the thought, so it cannot fire inside the thought. `stop_after_first` true matches one envelope. Parallel calls are inside the list, not repeated tags. Do not copy Qwen's `"</think>\n\n"` end string. This template does not write those two newlines. `TriggeredTags` can absorb whatever bytes the model emits between the closer and `<|tool_call_start|>`. Those bytes are **UNKNOWN**.

   Hard `required` / named cannot use that tag. The inner list is `name(key=value)`, not a JSON document and not Qwen XML. `JSONSchema` constrains JSON. `QwenXMLParametersFormat` constrains `<parameter=…>` inside `<function=name>`. Neither is this wire. The model card's option to ask for JSON calls in the system prompt is a different wire, not a substitute grammar. **API gap:** the 2026.4 header has no Pythonic / LFM parameter constructor. `TagsWithSeparator` of repeated `<|tool_call_start|>` blocks would describe envelopes the template does not emit. A hard mode that cannot compile this grammar must be an explicit API error. It must not downgrade to `AnyText` inside the tool tag.

6. Phase rule. With the 24B suffix, `detectAndSetImplicitReasoningStart` sees a prompt that ends in `<think>` and should enter `REASONING` without waiting for the model to emit the opener. The reasoning parser consumes the closer and returns to `UNKNOWN`. The next `<|tool_call_start|>` is a tool call. Text after `<|tool_call_end|>` is content. `<|im_end|>` is erased, not a new phase. Reasoning does not start again after the tool envelope in the same generation. The template opens exactly one `<think>` per assistant generation and does not open another after `render_tool_calls`. After a `tool` role result, the next request is a new generation and the suffix opens `<think>` again. That cross-turn cycle is the format. It is a new parse, not a phase re-entry. Intra-turn think/call/think/call is not part of this format. It is not a phase-machine requirement. The Gemma handoff must not be added.

7. Tests still required before any port claim. Draft `<|tool_call_start|>` inside an open `<think>` stays `reasoning_content`. A call after `</think>` becomes `tool_calls`. Two functions inside one `[...]` pair stay one envelope and two ids. A second envelope is not required by the template. Thinking-off is not a 24B case: there is no closer for the template to have written in advance. Repeated think/call cycles are the next request after a tool result, not one stream. Token-id checks must use this checkpoint's tokenizer, including the `special: false` added tokens, not only the 8B-A1B test path.

`lfm2_moe` export and `VLM_CB` / `LM_CB` for this graph are **UNKNOWN** in this tree. That is an inference gap, separate from the structural-tag gap. Weights are absent, so those checks cannot be run from the machines paths above.

**Verdict: blocked on an API gap.** The hard tool policy needs a Pythonic parameter constructor the 2026.4 header does not have. The parser wire is the existing `lfm2` / `lfm25` grammar plus the pins above. That is not a finished 24B port. Weights are absent, so runtime acceptance is also blocked. The template text and the Hugging Face token ids are not the missing piece.

## Family 2 — experimental GLM, two checkpoints

These are different models. Do not share a parser port, a token-id table, or an acceptance run between them.

| | GLM-4.7-Flash | GLM-Z1-32B-0414 int4 OV |
| --- | --- | --- |
| Repo | `zai-org/GLM-4.7-Flash` | `exzile/GLM-Z1-32B-0414-int4-ov`, base `zai-org/GLM-Z1-32B-0414` |
| Class | `Glm4MoeLiteForCausalLM` | `Glm4ForCausalLM` |
| `model_type` | `glm4_moe_lite` | `glm4` |
| Vocab | 154880 | 151552 |
| Local weights | absent | absent |

### Shared

Role strings, with different ids: `<|system|>`, `<|user|>`, `<|assistant|>`, `<|observation|>`, plus `[gMASK]` and `<sop>` as the bos prefix. Both templates contain the character sequence `</think>`. Both generation configs stop on end-of-text, `<|user|>`, and `<|observation|>` (4.7 ids 154820, 154827, 154829; Z1 ids 151329, 151336, 151338).

That is the overlap. The tool wire is not shared. The reasoning token ids are not shared. The OpenVINO graph is not shared.

### 2a. GLM-4.7-Flash

Official template: `zai-org/GLM-4.7-Flash` `chat_template.jinja`, 3120 bytes, md5 `ba2430d0ca7266e4ec29ddb19a4d7847`. The README does not paste a tool-call example. It points at vLLM / SGLang and at preserved thinking. The template is the tool contract. Tool parser name `glm` does not exist in this tree. No generation builder exists.

optimum-intel PR 1699 (fetched 2026-10-23): draft, open, not merged, `mergeable_state` dirty, updated 2026-04-23. Title: add OpenVINO export for `glm4_moe_lite`. It registers `Glm4MoeLiteOpenVINOConfig` for `text-generation` and `text-generation-with-past`, reuses the MiniCPM3 dummy past-key generator for MLA cache shapes (`qk_nope_head_dim` + `qk_rope_head_dim`, separate `v_head_dim`), and replaces `Glm4MoeLiteNaiveMoe.forward`'s Python expert loop with batched `bmm` so tracing does not depend on routed experts. `config.json` matches that description: `q_lora_rank` 768, `kv_lora_rank` 512, `qk_nope_head_dim` 192, `qk_rope_head_dim` 64, `v_head_dim` 256, 64 routed experts, 4 per token. The PR does not mention the chat template, tool calling, or reasoning tokens. It is not a graph in this tree. It is not the Z1 export.

Added tokens on the 4.7 tokenizer, one id each, `special: false` unless noted:

| Token | Id | `special` |
| --- | --- | --- |
| `<|endoftext|>` | 154820 | true |
| `[gMASK]` | 154822 | true |
| `<sop>` | 154824 | true |
| `<|system|>` | 154826 | true |
| `<|user|>` | 154827 | true |
| `<|assistant|>` | 154828 | true |
| `<|observation|>` | 154829 | true |
| `<think>` | 154841 | false |
| `</think>` | 154842 | false |
| `<tool_call>` | 154843 | false |
| `</tool_call>` | 154844 | false |
| `<tool_response>` | 154845 | false |
| `</tool_response>` | 154846 | false |
| `<arg_key>` | 154847 | false |
| `</arg_key>` | 154848 | false |
| `<arg_value>` | 154849 | false |
| `</arg_value>` | 154850 | false |

`skip_special_tokens` does not, by the Hugging Face flags, drop the think or tool tokens. They are still single ids. A local OV encode is **UNKNOWN** (weights absent).

#### Checklist (4.7-Flash)

1. Reasoning open `<think>` (154841). Reasoning close `</think>` (154842). Tool open `<tool_call>` (154843). Tool close `</tool_call>` (154844). Inside a call the template writes the function name, then zero or more `<arg_key>key</arg_key><arg_value>value</arg_value>` pairs. A string value is raw text. Any other JSON value is `tojson`. Empty arguments are `<tool_call>name</tool_call>`. Tool results use role `tool`: the first of a consecutive run is prefixed with `<|observation|>` (154829), then each result is `<tool_response>…</tool_response>`.

2. A tool call is legal only after `</think>`. History emits one think span, or a bare `</think>` when thinking is cleared, then content, then the tool envelopes. `<tool_call>` before the closer is still reasoning. This is the Qwen side of the channel rule, not the Gemma side.

3. Several calls are separate `<tool_call>…</tool_call>` envelopes. The template concatenates them with no separator character.

4. Generation suffix is `<|assistant|>` plus one token string. Thinking on (`enable_thinking` absent or true): `<think>`. The model must close it. Thinking explicitly false: the template itself writes `</think>` and generation starts after that closer. There is no empty `<think>…</think>` pair in the false branch. Replay flag `clear_thinking`: when it is defined and false, every assistant turn that has `reasoning_content` keeps `<think>…</think>`. Otherwise only turns after the last user keep it, and other turns emit a bare `</think>`.

5. Order the header can express when thinking is open: `Concat(Tag(begin empty, AnyText, end "</think>"), tool tags)`, with the tool trigger or the required tags placed after the closer so they cannot fire inside the thought. Thinking explicitly false: the same tool tags immediately, with no thought prefix and no second closer. Guided auto: `TriggeredTags` on `"<tool_call>"`, `at_least_one` false, `stop_after_first = !parallel_tool_calls`. Hard `required` / named: `at_least_one` true. Do not copy Qwen's `"</think>\n\n"`. This template glues `</think>` to the following text with no inserted blank line. Bytes the model emits between the closer and `<tool_call>` are **UNKNOWN**. `TriggeredTags` can absorb them. A hard glue of `"</think>"` directly to `"<tool_call>"` cannot, until a render pins those bytes.

   The tag body is the gap. `QwenXMLParametersFormat` is XGrammar's `qwen_xml_parameter` dialect: `<tool_call>\n<function=name>\n` and `<parameter=name>`. GLM-4.7 is `<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`. `JSONSchema` would emit a JSON document, not arg pairs, and string values in this template are not JSON strings. **API gap:** no `<arg_key>` / `<arg_value>` constructor. Hard `required` / named must fail closed. They must not pretend `QwenXMLParametersFormat` or one-argument `JSONSchema` is this wire.

   Canonical envelopes abut. `TagsWithSeparator` with separator `""` is the shape that matches the template. Whether the pinned XGrammar accepts an empty separator is **UNKNOWN**. If it rejects it, parallel hard calls are a second API gap. Do not insert a newline the template does not write just to make the separator compile. A single required call (`stop_after_first` true) does not need the separator. Its body still needs the missing constructor.

6. Phase rule for the shape the template actually serializes. One reasoning span, then content, then tool envelopes. The closer hands the tail back to `UNKNOWN`, and `<tool_call>` starts the tool parser. A later `<tool_call>` in `TOOL_CALLS_WAITING_FOR_TOOL` is another envelope. The template does not open `<think>` between those envelopes when it replays one assistant message.

   Cross-turn cycles are part of the format. After `<|observation|>` / `<tool_response>`, the next generation suffix opens `<think>` again, or writes `</think>` when thinking is off. That is a new request. The parser resets. It does not need the Gemma re-entry fix.

   Intra-turn think/call/think/call is a separate question. The Z.ai thinking-mode guide (GLM-4.7 series, fetched 2026-10-23) says interleaved thinking is on by default and that the model thinks between tool calls and again after tool results. The only concrete sample on that page is cross-request: one completion returns `reasoning_content` and `tool_calls`, the client sends them back with the tool result, and the next completion reasons again. The chat template cannot replay two think spans with a tool envelope between them inside one assistant message. The token sequence `<think>…</think><tool_call>…</tool_call><think>…</think><tool_call>…` in a single generation is **UNKNOWN**.

   If a trace shows that intra-turn sequence, it is a phase-machine requirement. `TOOL_CALLS_WAITING_FOR_TOOL` will not re-enter `REASONING` on a new `<think>`. The Gemma handoff does not fix that: it only closes the current reasoning segment, and 4.7 does not start the call before the closer. Do not record intra-turn cycles as supported behavior.

7. Tests, once a parser exists. None of these exist today. Draft `<tool_call>` inside an open `<think>` stays reasoning. Calls after `</think>` become separate `tool_calls`. Adjacent envelopes stay separate ids. Thinking off does not demand a closer: the prompt already ends in `</think>`. A second `<think>` after a tool envelope in the same generation is the unknown interleaved case and must be its own red test until the phase rule re-enters reasoning. Do not borrow a Qwen or Gemma green run.

**Verdict (4.7-Flash): blocked on an API gap.** The arg-pair body has no 2026.4 structural-tag constructor. PR 1699 does not remove that gap. Weights are absent. Intra-turn think/call/think/call stays UNKNOWN and, if real, is an additional phase-machine requirement.

### 2b. exzile/GLM-Z1-32B-0414-int4-ov

This is `zai-org/GLM-Z1-32B-0414` quantized to INT4 symmetric, group size 128, ratio 1.0, task `text-generation-with-past`, by the model card's own command (optimum-intel 2.0.0, OpenVINO 2026.2.1, NNCF 3.2.0, transformers 5.12.1). `model_type` is `glm4`, not `glm4_moe_lite`. It is not a product of PR 1699. The card says the tokenizer and chat template are the base model's and that strict `json_schema` constrained decoding was tried on an Arc Pro B70 under OVMS 2026.2. That is a JSON-schema runtime note. It is not a native tool grammar, and it is not a run on this 2026.4 tree.

Template pin: base `chat_template.jinja`, 1292 bytes, LF, md5 `29e67eda5786709350597558e5707b6f`. The exzile `chat_template.jinja` is the same text with CRLF (1336 bytes, md5 `a08741307e80d5988c4b5d03c1d70c48`). LF-normalized bytes match. Pin the logical template. Record the line-ending hash of whichever file the export copies.

The base `tokenizer_config.json` also embeds a `chat_template` string. That string is a different template. Its generation suffix is `<|assistant|>` with no `<think>`. Its metadata replay writes a space before the newline (`{{ meta }} \n`). The jinja file's generation suffix is:

```
<|assistant|>
<think>
```

which is the two characters newline and then `<think>`, with no newline after `<think>`. The Z1 model card says to add `<think>` via `chat_template.jinja`, and its prose writes `<think>\n` (a newline after the tag). The jinja does not write that trailing newline. For the OV repo, exzile's `tokenizer_config.json` has no `chat_template` field and no `added_tokens_decoder`. The on-disk contract for that artifact is the jinja file. Do not load the embedded tokenizer_config string and call it the same template.

Z1 `added_tokens_decoder` contains the role tokens and the image/video tokens. It does not contain `<think>`, `</think>`, `<tool_call>`, `<arg_key>`, or `<arg_value>`. Think markers are plain strings on this tokenizer. Whether each is one vocab id or several pieces is **UNKNOWN** until encoded. Do not copy 4.7 ids 154841 and 154842.

#### Checklist (Z1)

1. Reasoning open string `<think>`. Reasoning close string `</think>`. There is no tool-call tag. Replay of a call is an assistant message with `metadata` set to the function name and `content` set to the JSON arguments, rendered as `<|assistant|>{name}\n{json}` after the template has stripped through `</think>`. The Z1 jinja has no `tool_calls` loop and no `<tool_response>`. A tool result is role `observation`, rendered as `<|observation|>` plus the visible text. The 4.7 XML tokens are absent here.

2. A tool call is legal only after `</think>`. The generation prompt leaves `<think>` open. The metadata replay path does not put the function name inside the think span. A `name\n{json}` pattern before the closer is still reasoning text. The GLM-4-32B-0414 card's regex, which detects a call without looking at the think span, is peer server behavior. It is not authority to end Z1 reasoning on a JSON object. Do not add that handoff.

3. Separate tool envelopes are not part of the Z1 template. One metadata message is one call. The 0414 card for GLM-4-32B-0414 splits a completion on `<|assistant|>` and can see more than one call. The Z1 README does not restate that loop. Several calls in one Z1 generation are **UNKNOWN**. Do not describe them as 4.7 `<tool_call>` blocks.

4. Designated generation suffix, from the jinja the model card names: `<|assistant|>\n<think>`. There is no `enable_thinking` false branch in this file. History drops text through `</think>` (`content.split('</think>')[-1]`). The card says hidden thinking is not saved. The prose `<think>\n` and the embedded tokenizer_config suffix `<|assistant|>` disagree with the jinja bytes. Follow the jinja until a token-id render says otherwise.

5. One named call, once the join bytes are pinned, fits the header: `Concat(Tag(begin empty, AnyText, end "</think>"), ConstString(name + "\n"), JSONSchema(schema))`. `JSONSchema` here is the one-argument constructor. The argument object is JSON, so this is not the Gemma `JSONSchema(schema, 2)` gap, and it is not the 4.7 arg-pair gap. A union of those prefixes covers "one of the registered tools" for a single call. Guided auto has no distinct opener: the call begins with a function name, which is ordinary text. `TriggeredTags` has nothing safe to trigger on. Unguided auto is the protocol's unconstrained reference and does not require a trigger. The bytes between `</think>` and the function name are **UNKNOWN**. The jinja does not emit them. The model card does not show a generated call. A hard `Concat` that glues `"</think>"` to `ConstString(name)` rejects an unpinned newline. Until those bytes are pinned, the hard shape is not installable. Do not invent `"</think>\n\n"` from Qwen.

6. Phase rule for the known shape. Implicit reasoning start, because the jinja ends the prompt on `<think>`. The closer returns to `UNKNOWN`. The following `name\n{json}` is the call only after that closer. The template does not open a second `<think>` after the JSON in the same message. The next `<think>` is the next generation, after `<|observation|>`. Repeated think-then-tool cycles in the 0414 server loop are cross-turn. They are not a phase-machine requirement. Intra-turn think/call/think/call is not part of the known Z1 format.

7. Tests, none in this tree. Draft `name\n{json}` inside the open think stays reasoning. The same pattern after `</think>` becomes one tool call. A 4.7 `<tool_call>` string is not a Z1 call. Thinking-off is not a Z1 template mode. A second think after the JSON in the same generation is not a required cycle. Encode `<think>` and `</think>` on this tokenizer before claiming they are single tokens.

**Verdict (Z1-32B int4 OV): blocked on missing template facts.** Two official template strings disagree, the card's `<think>\n` prose disagrees with the jinja, and the bytes between `</think>` and the function name are unpinned, so the hard structural tag cannot be installed yet. The JSON object itself has a constructor once that join is pinned. Weights are absent. This checkpoint stays a different port from 4.7-Flash.

## What a porter does not get to claim

- No family above has a parser port in this tree. LFM has a shared LFM2 / LFM2.5 parser that still needs the 24B template pin, the adversarial in-think test, and a policy. GLM has no parser.
- No hard generation policy is installable for LFM2.5-24B-A2B or for GLM-4.7-Flash with the 2026.4 structural-tag types. Z1's hard policy waits on the unpinned join, not on a missing JSON constructor.
- Intra-turn think/call/think/call is a phase-machine requirement only where a family's format actually emits it. For these checkpoints that sequence is either absent (LFM, Z1 as specified) or UNKNOWN (GLM-4.7-Flash). The Gemma handoff is the wrong mechanism in every case.
- Local weights are absent under `C:\llm\models` and `C:\models`. No runtime lane was run.
