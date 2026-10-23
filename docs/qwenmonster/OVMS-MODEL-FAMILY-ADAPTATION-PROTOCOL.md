# OVMS Model-Family Adaptation Protocol — MFP-1

Status: **PROPOSED STANDARD**, distilled from Gemmamonster C5/C5fixed/C6 and current Qwenmonster work.
Scope: onboarding a model family to OVMS **without** confusing template, parser, generation policy, inference, and harness behavior.

Living copy for the 2026.4 semantic refit, supplemented 2026-10-23. The 2026-10-21 text at `C:\git\qwenmonster-qwm01\docs\qwenmonster\OVMS-MODEL-FAMILY-ADAPTATION-PROTOCOL.md` stays frozen. This file is the one to update.

## A0. Identity and isolation (must precede patches)

- Pin `BASE_SHA`, tree SHA, exact model checkpoint, official tokenizer revision, official chat-template hash, **public commit SHA for each transitive patched companion source** (GenAI, XGrammar, OpenVINO, Minja, Tokenizers), runtime DLLs/compiler/Bazel, OS/device/hardware and launch profile.
- Store a checked-in machine-readable cross-repository source lock and an offline verifier. **Verify semantic carry-forward after a divergent branch forward-port**, rather than conflating latest commit date with patch coverage.
- Record whether `VLM_CB` or `LM_CB` actually works. Build-success is not device/runtime acceptance.
- Freeze current known-good, use fresh worktree + feature branch, never refactor the working family incidentally.
- Keep C5, C5fixed and C6 **different identities**. A known-good snapshot is not automatically an ancestor or equivalent to a new release.
- Start and maintain `STATE.md` and evidence paths before code mutation.

## A1. Canonical protocol contract

- Official model card, canonical Jinja/template, tokenizer, prompt examples, assistant/tool-role serializer, generation suffix.
- Native role markers; reasoning channel; tool opener/closer; argument wire types; empty args; multiple calls; tool response replay.
- Render golden no-tool, one tool, N tools, tool-result loop, thought ON/OFF, preserve_thinking ON/OFF, partial assistant prefix.
- Compare rendered **token IDs** in addition to decoded text; do not treat visually equivalent Jinja as token-equivalent.
- Classify by capability fixture rather than name substring. Family variants are explicit exceptions.

## A2. Peer implementation dossier

- Sources: vLLM (parser state & streaming), llama.cpp (generation & grammar), SGLang (independent detector and failure cases), original model template/Transformers, current upstream OVMS/GenAI.
- For every mechanism record pinned upstream source, commit/PR, exact known fail/success, our corresponding code/test and **KEEP/REFINE/PORT SEMANTICS/TOLERANCE ONLY/DROP/UNKNOWN**.
- Peer server behavior is not authority to redefine canonical model protocol. Import **semantics**, not unrelated frameworks.
- Channel separation is by the family's own special tokens. A tool-call tag inside an unclosed reasoning span is still reasoning, unless that family's format itself starts the tool call before the reasoning closer. Gemma is that exception. **DROP** the vLLM heuristic that treats `<tool_call>` inside `REASONING` as an implicit end of thinking when the family is Qwen. The SGLang detector makes the same override. Our policy follows the chat template and the structural-tag grammar.
- Always include at least one adversarial test from a known upstream incident.
- Separate **reproduced failures**, **historically not reproduced anomalies**, **legitimate empty-schema behavior**, and **open upstream grammar bugs**. Never promote a control result from one layer as repair evidence for another.

## A3. OVMS architecture contract

- `ChatHistory/Jinja` owns request replay and rendered generation prefix.
- `GenerationConfigBuilder` owns tools policy and OpenVINO GenAI structured masks.
- `OutputParser` owns generic phase routing, flush/remainder and parser ownership.
- Family `BaseOutputParser` descendants own native FSM and argument normalization.
- `OVMSTextStreamer` owns token decode and ordered lifecycle flush/drain.
- `OpenAI API` owns tool_calls IDs/index, finish_reason and response contract.
- `Harness` owns *execution*, never the parser.
- Generation policies are our component. Upstream OVMS does not maintain them.
- A parser and a generation policy are different layers. The parser decides which span is reasoning, content, or a tool call. The generation policy constrains what the model is allowed to emit. A parser handoff is not a grammar, and a grammar is not a phase rule.
- No second parallel parser pipeline without evidence that the generic phase-owner path is insufficient.

## A4. Parser-first RED corpus and refit

- Streaming/unary and byte/token/chunk-split invariance.
- Canonical 1/N tool calls, repeated same function, 0 args, multiline and nested typed values.
- Reasoning→tool and content→tool, tool→tool, prompt already in reasoning.
- Malformed: bare start, missing close, dangling parameter, end-marker drift, quoted attributes, unknown tool, false-positive XML text.
- Preserve exact native text and emitted deltas before modifying code. `STOP`, `LENGTH`, cancellation, error must have different recovery obligations.
- No unknown tool or incomplete argument becomes executable. Reject invalid type instead of silently casting into an arbitrary string.
- Cross-request/choice reset and bounded buffer sizes; run anti-contamination tests.

## A5. Generation policy as independent component

Build a new family-specific builder only when its native wire differs. Explicit matrix:
- `none`: no guided tools; API must not publish executable tool call.
- `auto`: native unconstrained reference first; optional lazy constraints only after A/B and trigger analysis, never force a call.
- `required`: **at least one valid call**, from first constrained token, fail closed if unsupported.
- `named`: exactly specified registered schema, fail closed if unavailable.
- `parallel=false`: at most one call; `parallel=true`: multiple independent tool IDs permitted.
- Must pass reasoning-prefix compatibility for that family (see channel separation and the applied-family sections) and validate full JSON Schema including optional, required and nested object/array before claiming correctness.
- Compile/tag validation failure in a hard mode MUST be an explicit API error; never downgrade to `auto`.
- Preserve existing response_format when tools disabled; reject conflicting active constraints.
- Express the order with the structural-tag types the pinned GenAI header actually has. If the family's grammar needs a constructor that header does not have, name the API gap and do not pretend the hard policy is installed.

## A6. Runtime seam validation

- Check model and tokenizer actual token IDs for native markers (some are plain multi-token strings).
- Check GenAI/XGrammar version, structural-tag grammar compilation and runtime masks; raw serializer shape is not proof of successful compilation.
- Pin local patched dependency commits and loaded binaries as well as upstream versions. Building against mixed revisions is not acceptable.
- The official 2026.4 GenAI header has a one-argument `JSONSchema(const std::string& schema)` only. Gemma's `JSONSchema(schema, 2)` is a local API extension and is **not ported**. `gemma4` must not pretend to install a hard policy. `requiresValidStructuredOutput()` stays false. See the Gemma4 section.
- Diagnose whitespace faults by grammar location: bounded JSON object whitespace vs XML string-end marker prefix exclusion vs zero-parameter XML self-loop. Do not indiscriminately reuse one fix across formats.
- Record `VLM_CB/LM_CB` lane and GPU device, TTFT/TG, memory, faults and executor quarantine.
- Keep parser syntax and inference/kernel errors separate. Do not misdiagnose `CL_OUT_OF_RESOURCES` as grammar fault.

## A7. Behavioral/differential evaluation

- Run the same pinned native output corpus against vLLM, llama.cpp and OVMS: compare semantic JSON and reasoning, then streaming event reconstruction. Classify disagreements with proof.
- Run equal OpenAI request schemas and sampling controls against real model.
- Track phases independently: tool visible → model chose → model emitted → parser extracted → API returned → harness executed → subagent finished.
- Separate autonomous delegation rate from successful native tool delivery. Scale long prompts, multi-agent loops and 1/2/4 sessions only after short protocol gate.

## A8. Acceptance ladder (no fake GREEN)

1. `SOURCE VERIFIED`: source locations and pinned versions confirmed.
2. `RED REPRODUCED`: failing baseline test/log preserved.
3. `STATIC PATCH`: code committed and scoped.
4. `BUILD GREEN`: exact binaries/toolchain, test target resolves.
5. `UNIT GREEN`: deterministic parser, policy, template and negative controls pass.
6. `API GREEN`: unary + SSE + Responses if applicable, correct finish/IDs/deltas.
7. `RUNTIME GREEN`: actual model/device/lane on identical pinned candidate.
8. `AGENTIC GREEN`: repeated multi-turn tool loops, no silent drops, no cross-session contamination.
9. `ACCEPTED`: all gates and Gemma/no-change negative controls, evidence paths and owner signoff. A live model run of this candidate binary is required. Unit green, a successful build, and a release-candidate tag do not substitute for it.

A null/empty evidence file never counts as PASS. A patch from one model must not be promoted using another model's successful run. A red test that documents a known gap is not a pass, and the gap is not supported behavior. A live run counts only when the candidate binary actually generates and the saved response shows the behavior under acceptance: finish reason plus content, reasoning, or tool calls. An empty message, a unit test, or a tag by itself is not that run.

## A9. Release and upstream contribution

- One mechanism per atomic commit: parser → generation → reasoning/template → acceptance.
- Independently revertible, short upstream PRs: `Problem → Reproducer → Root cause → Fix → Negative control → Live evidence`.
- No model weights/runtime dumps/local paths/secret tokens in PR.
- Before every upstream version bump, conduct three-way rebase by **behavior**: official model contract + current upstream OVMS + pinned accepted local implementation.
- Drift watch for template, tokenizers, XGrammar masks, GenAI API and model revisions.
- After each stage update `ACTIVE_STAGE, BASE_SHA, CURRENT_HEAD, LAST_GREEN_TESTS, CURRENT_FAILURE, NEXT_ACTION, REVERT_POINT, EVIDENCE_PATH`.

## Channel separation

Separation is by the family's special tokens, not by a shared detector.

- A tool-call tag that appears while the reasoning span is still open stays inside reasoning.
- The exception is a family whose own format starts the tool call before the reasoning closer. Gemma does this: `<|tool_call>` is legal before `<channel|>`.
- Qwen does not. `<tool_call>` before `</think>` is draft thought text. Do not copy the vLLM heuristic that ends Qwen reasoning on `<tool_call>`.
- Do not add a Qwen tool-call-ends-reasoning handoff.

## What a family port must specify before code

1. Special tokens that open and close reasoning, and special tokens that open and close a tool call.
2. Whether a tool call is legal before the reasoning closer (Gemma) or only after it (Qwen).
3. Whether several tool calls in one turn are separate envelopes.
4. What the chat template appends as the generation suffix when thinking is on and when thinking is off.
5. Which 2026.4 GenAI structural-tag types can express that order: `Tag`, `AnyText`, `Concat`, `TriggeredTags`, `TagsWithSeparator`, `QwenXMLParametersFormat`. Name the API gap if the family's grammar needs a constructor this header does not have.
6. Parser phase rule: who consumes which token, and whether reasoning can start again after a tool envelope.
7. Tests: draft tool text inside reasoning stays reasoning; real calls after the closer become `tool_calls`; parallel calls stay separate; thinking off does not require a closer the template already wrote; repeated think/call cycles if the format allows them.

## Applied family: Qwen3.6 / qwen3coder + qwen3

Implemented in `src/llm/io_processing/qwen3coder/generation_config_builder.hpp`.

Selector: tool parser name `qwen3coder` constructs `Qwen3CoderGenerationConfigBuilder`. The reasoning parser paired with that policy is `qwen3`. Tool parser name `qwen3` still constructs `Hermes3GenerationConfigBuilder`.

Tokens:

- Reasoning open: `<think>`. Reasoning close: `</think>`.
- Tool-call open: `<tool_call>`. Tool-call close: `</tool_call>`.
- A tool call is legal only after `</think>`. Parallel calls are separate `<tool_call>` blocks, not one envelope with several functions.

Template suffix:

- Thinking open (`enable_thinking` absent or true): the template leaves `<think>\n` open. The model finishes the thought with `</think>`.
- Thinking explicitly false: the template itself writes `<think>\n\n</think>\n\n`. Generation starts after that closer.
- Official order: after `</think>`, zero or more adjacent `<tool_call>` blocks. History renders tool calls after `</think>`, not inside it. After a tool result, the next assistant turn opens a new think and may emit new calls.

Generation policy, thinking open:

- Hard `required` / named: `Concat(Tag(begin empty, AnyText, end "</think>\n\n"), TagsWithSeparator)`. `TagsWithSeparator` has `at_least_one`, separator newline, `stop_after_first = !parallel_tool_calls`, and each tag uses `QwenXMLParametersFormat`.
- Guided auto: the same thought prefix, then lazy `TriggeredTags` on `<tool_call>` with `at_least_one` false. The trigger is installed only after the thought, so it cannot fire inside the thought.
- Unguided auto and `none` install no tool grammar.

Generation policy, thinking explicitly false:

- Immediate `TagsWithSeparator` or `TriggeredTags`. No thought prefix. The template already wrote the closer, so the policy does not ask the model for another one.

Parser:

- A draft `<tool_call>` inside the open thought stays in `reasoning_content`. The real calls are only the blocks after `</think>`.
- Test: `Qwen3CoderOutputParserTest.OpenThinkDraftToolCallRemainsReasoning`.
- Do not add a Qwen handoff that ends reasoning on `<tool_call>`.

Grammar authority:

- xgrammar's official Qwen3.5/3.6 structural tag has the same shape: `sequence(think … </think>, "\n\n", TriggeredTags)`.
- vLLM and the SGLang detector override that shape by ending reasoning on `<tool_call>`. Our policy follows the template and the grammar.

## Applied family: Gemma4

Tokens:

- Reasoning open: `<|channel>`. Reasoning close: `<channel|>`.
- Tool-call open: `<|tool_call>`. Tool-call close: `<tool_call|>`.
- String marker inside a call: `<|"|>`.
- A tool call is born inside an open thought channel. `<|tool_call>` before `<channel|>` is legal. That is the opposite of Qwen.
- Private markers `<|channel>`, `<channel|>`, `<|tool_call>`, `<tool_call|>`, and `<|"|>` stay out of visible content.

Parser, `gemma4` + `gemma4` handoff in `OutputParser`:

- When `<|tool_call>` appears before `<channel|>`, the handoff closes the **current** reasoning segment and gives the tail to the tool parser.
- That handoff does not reopen reasoning after the tool envelope.

Known gap, not supported behavior:

- Ten alternating thought+tool cycles fail. The first thought is reasoning, all ten calls parse, and thoughts 1–9 leak into content with channel tags because the phase stays `TOOL_CALLS_WAITING_FOR_TOOL`.
- Reproducer, currently red: `Gemma4OutputParserTest.ParseTenAlternatingThoughtAndToolCalls`.
- Do not record this test as a passing gate and do not describe repeated think/call cycles as supported Gemma behavior until the phase rule actually re-enters reasoning.

Generation policy:

- Hard `required` and named choices install `TagsWithSeparator` with `at_least_one`. `parallel_tool_calls=false` sets `stop_after_first`.
- Tool arguments are an EBNF grammar with `[ \n\t]{0,2}` between JSON tokens. That is the G-JWS bound of `JSONSchema(schema, 2)`. The 2026.4 GenAI DLL does not read `max_whitespace_cnt` from a structural tag, so a one-argument `JSONSchema` or a raw field leaves the call open. Do not put Gemma arguments back on unbounded `JSONSchema`.
- `requiresValidStructuredOutput()` is true for that hard policy.
- Test: `QwenmonsterGenerationPolicy.ParserFamilyIsolation`.

## Chat templates

Use the canonical Qwen3.6 chat template.

- Source: `Qwen/Qwen3.6-35B-A3B`.
- md5 `52b6d51ae5b203cb67e64b648494dad2`, 7764 bytes.
- `preserve_thinking` is the 2026 addition on top of `last_query_index`.

Ornith's own template is not canonical. It dropped the `last_query_index` guard and re-injects `<think>` on every past assistant turn.

- Broken original md5 `fa5eb1eee497fc8f3bccc3fcf60d1a51`.
- Local Ornith file `C:\llm\models\ornith-35b-ov\chat_template.jinja` is already the canonical template.
- The broken original is `C:\llm\models\ornith-35b-ov\chat_template.jinja.backup-20261021-034305`.

A visually similar Jinja file is not token-equivalent. Pin the hash before calling a template port done.

## Current application

Snapshot at distillation (2026-10-21): A0 partially proven (C6 SHA), A1/official template token-level pin pending, A2 sourced and documented, A3 applied, A4 source patches only, A5 builder candidate, A6–A8 not yet run. No ACCEPTED status.

Semantic refit 2026.4 (2026-10-23): the Qwen3.6 policy above is implemented for tool parser `qwen3coder` with reasoning parser `qwen3`. Gemma4 hard policy uses an EBNF argument grammar with whitespace bound `{0,2}` because this GenAI DLL does not apply `max_whitespace_cnt`. A one-argument `JSONSchema` leaves the call open on newlines. Gemma4 reasoning may open again after a tool envelope. LFM2.5-24B-A2B and the experimental GLM notes are consciously deferred. `ACCEPTED` requires a live model run of the candidate binary. No ACCEPTED status.
