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

### A3.1. Anti-hallucination parsers-retainers

**The architectural target is an anti-hallucination parser-retainer (антигаллюцинационный парсер-ретейнер): preserve what the model actually emitted, establish its native protocol boundaries, and publish a tool call only after the applicable checks succeed.** The parser must not invent missing argument values, tool names, required fields, or a completed call from an unfinished candidate. Here, "hallucination-free" describes the parser's own extraction and publication contract; it is not a claim that the model's answer or chosen arguments are factually correct.

The layers cooperate in this order: canonical template and rendered prefix -> family generation policy and runtime grammar -> generic phase router -> native family retainer -> schema/publication checks -> API deltas -> harness execution. Generation constraints reduce malformed output; the retainer checks the actual generated output. Neither layer substitutes for the other, and parsing a call does not authorize or execute it.

| Boundary | Retainer contract |
| --- | --- |
| Channel ownership | Recognize reasoning, visible content and tool spans using the selected family's protocol. Draft tool text in Qwen reasoning cannot become a public call. Gemma's distinct handoff remains family-specific. |
| Argument fidelity | Keep literal string bytes and numeric spelling, including nested numbers. Permit only documented protocol transformations: clean wrapping quote removal, JSON encoding once, structural whitespace compaction and boolean normalization. Never globally unescape or silently coerce an invalid typed value. |
| Executable publication | Require a registered tool and valid arguments under the supported schema profile before publishing identity/arguments. Do not fill required fields from schema defaults. A low-level candidate or a syntactically complete JSON value alone is not an executable call. |
| End of generation | Distinguish STOP, LENGTH, cancellation and error. Any tolerated missing transport delimiter needs explicit evidence that the semantic payload was already complete; do not guess a missing value or semantic closing boundary. |
| Streaming lifecycle | Keep unary and reconstructed streaming arguments equivalent; preserve order and separate call identities; clear retained state between calls, choices and requests. Publish no speculative identity that later needs retraction. |

Qwen3Coder implements the retention/publication mechanisms in `qwen3coder_tool_parser.cpp`: `addParameterToCurrentFunctionDoc`, `finalizeOnGenerationEnd`, and `queueFullDeltas`. The latter checks registration and `validateQwenXmlToolArguments` before allocating a public streaming identity. The validator uses a restricted Draft 4 profile in `qwen3coder/tool_schema_profile.cpp`; "schema validation" does not mean arbitrary dialect/keyword support. Unsupported profiles must fail explicitly at the applicable policy boundary.

The specific Qwen STOP tolerance is limited to appending the outer `</tool_call>` after a completed `</function>`. An open parameter is dropped; LENGTH does not trigger that recovery. This is recognition of a complete payload with a missing outer delimiter, not permission to repair arbitrary model output.

A schema-valid call can still name the wrong file, contain an incorrect calculation, or choose an inappropriate action. The retainer preserves those emitted values; semantic correctness, execution authorization and tool results require separate checks and evidence. Lossless publication also does not establish arbitrary-precision numeric schema comparisons: the current schema validator still reads a separate RapidJSON DOM. Never serialize that validation DOM back over the retained argument text.

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
- The official 2026.4 GenAI header has a one-argument `JSONSchema(const std::string& schema)` only. The two-argument `JSONSchema(schema, 2)` extension is **not ported**. That API gap must not be mistaken for a disabled policy in this tree: the current Gemma builder installs a bounded EBNF argument grammar instead, and `requiresValidStructuredOutput()` reports the request's hard policy. A port without that equivalent implementation must fail closed in hard mode rather than claim the policy is installed. See the Gemma4 section; source inspection is not renewed Gemma acceptance.
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

### Carry-forward rules for family policies and retainers

Transfer a behavior contract with its tests and dependency requirements. A matching class name, copied builder, successful build or newer source revision is insufficient evidence of a successful port.

| Carry forward | Re-establish for the target family/runtime |
| --- | --- |
| Retention and publication invariants from A3.1 | Native delimiters, channel ownership, supported string quoting, typed argument representation, completion/recovery rules and state reset. Reuse a helper only where those contracts match; do not migrate a working family as an incidental refactor. |
| `none` / `auto` / `required` / named / parallel policy semantics | Actual builder selector, reasoning-parser pairing, rendered thinking prefix and grammar activation point. Keep a lazy auto trigger out of a reasoning span when the protocol requires it. |
| Schema and grammar requirements | Supported schema dialect/keywords, optional/required/nested fields, zero-argument tools, JSON/XML whitespace and termination rules. Pin the APIs and loaded DLLs that implement them; an API fallback needs equivalent tested semantics. |
| RED/negative controls and prior receipts | Run the target tuple's own checks and record its results separately. Prior family's PASS or historical source/DLL evidence cannot establish current runtime or agent acceptance. |

For this tree, `qwen3coder` selects `Qwen3CoderGenerationConfigBuilder` and pairs with `qwen3` reasoning; a tool parser selector of `qwen3` selects the Hermes builder instead. Ornith uses the Qwen3Coder path, with its separately pinned chat template. Do not infer either selector or template from a model name alone. Gemma keeps its own parser, policy and native handoff; the Qwen lossless change does not make their code shared.

Each port must record, per invariant: source implementation and commit, target implementation, dependency/API requirement, target test/evidence, and status (`NOT RUN`, `FAIL`, or `PASS` at the named acceptance layer). If a mechanism is deferred, state the gap; do not describe the architecture's target contract as implemented for every family.

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
- In the current source, the `TOOL_CALLS_WAITING_FOR_TOOL` branch separately recognizes a subsequent reasoning opener and re-enters reasoning. Do not transfer this Gemma transition to Qwen, whose tool opener does not end thinking.

Historical gap and current evidence boundary:

- The earlier failure leaked thoughts after the first tool envelope because the phase stayed `TOOL_CALLS_WAITING_FOR_TOOL`. Reproducer: `Gemma4OutputParserTest.ParseTenAlternatingThoughtAndToolCalls`.
- The current phase router contains the re-entry transition described above. The old statement "currently red" is not a current test receipt and must not be carried forward as one.
- This documentation update does not rerun that test, change Gemma code, or reopen Gemma acceptance. Require the separately pinned Gemma evidence before claiming its test/runtime status.

Generation policy:

- Hard `required` and named choices install `TagsWithSeparator` with `at_least_one`. `parallel_tool_calls=false` sets `stop_after_first`.
- Tool arguments are an EBNF grammar with `[ \n\t]{0,2}` between JSON tokens. That is the G-JWS bound of `JSONSchema(schema, 2)`. The 2026.4 GenAI DLL does not read `max_whitespace_cnt` from a structural tag, so a one-argument `JSONSchema` or a raw field leaves the call open. Do not put Gemma arguments back on unbounded `JSONSchema`.
- `requiresValidStructuredOutput()` is true for that hard policy.
- Test: `QwenmonsterGenerationPolicy.ParserFamilyIsolation`.

### How the Gemma4 retainer works

The reference is the existing native path in `src/llm/io_processing/gemma4/gemma4_tool_parser.cpp` and its header, read without modification. The production constructor receives tool schemas and enables `enforceToolRegistry` when the registry is nonempty. The description below concerns that registry-enabled path; the older constructor/compatibility path must not be assumed to have all of its checks.

1. **Route by the native channel.** The generic `OutputParser` recognizes Gemma's reasoning channel and hands a native tool span to `Gemma4ToolParser`. A call may start before the reasoning closer. The retainer recognizes `<|tool_call>`, `call:NAME`, the argument container and `<tool_call|>`; registered `call:NAME{` / `call:NAME(` preambles also have explicit boundary handling.
2. **Retain a candidate until its payload is complete.** The FSM records the name internally, then waits for the matching argument-container close. `findMatchingContainerEnd` tracks nested closers and skips both native `<|"|>`-delimited strings and JSON-quoted strings. A brace or marker occurring inside a string is not sufficient evidence of a container boundary. The registry-enabled scanner has a 64 KiB candidate-span limit and a container-depth limit of 64; rejected candidates are discarded/resynchronized rather than completed by guessing.
3. **Translate native values into JSON without losing their meaning.** `NativeValueParser` recursively handles objects, arrays and scalars. For a native string, `<|"|>` is the delimiter: its interior is literal model text, encoded once by `escapeAsJsonString`. A backslash followed by `t` stays those two characters. Ordinary JSON-quoted strings are a separate syntax branch and follow JSON escape semantics; do not confuse that branch with the literal native-string contract.
4. **Keep numeric tokens out of a floating-point round trip.** Gemma's anonymous-namespace `normalizeJsonLosslessly` uses `kParseNumbersAsStringsFlag`; its `NumberPreservingWriter::RawNumber` writes `RawValue(..., kNumberType)`. `writeJsonScalar` inserts that validated text into the output. Thus `1.10`, `1e3` and all digits of a long integer survive, including when the scalar is nested in a native container.
5. **Publish a completed call once and drain the rest.** The argument-container closer establishes payload completion; publication can occur before the outer `<tool_call|>` arrives. A name alone or an open value does not produce a call. `wrapCompletedToolCall` emits name/id/full arguments together, the stored call is cleared, and `drainPendingDelta` advances through any further buffered calls. Terminal handling may flush an already completed payload; it is not the Qwen-specific STOP repair described in A3.1.

The Gemma registry-enabled parser checks native structure, parseability and allowed tool names. It does **not** invoke Qwen's `validateQwenXmlToolArguments` full-argument schema validator. Do not turn the A3.1 target contract into a claim that every existing Gemma path already has Qwen's schema-publication checks. Gemma's hard generation grammar is a separate layer and must retain its own evidence. These are current-source descriptions, not newly executed Gemma tests or renewed acceptance.

### How the Gemma principle was adapted to Qwen3Coder

The transfer was semantic: **retain the model's argument text through validation and publication**. Gemma's parser, tokens, channel handoff and code were not transplanted into Qwen. The implementation checkpoint is `5af105a04904713b58b13bbd82a421c6f15844fa`.

| Mechanism | Gemma4 reference | Qwen3Coder adaptation |
| --- | --- | --- |
| Native envelope | `<\|tool_call>call:NAME{...}<tool_call\|>` and supported native variants | `<tool_call><function=NAME><parameter=KEY>VALUE</parameter>...</function></tool_call>`; keep the existing Qwen FSM |
| Reasoning boundary | Native tool opener can hand off from an open thought | Qwen tool text inside an open `<think>` remains reasoning; executable calls follow `</think>` |
| Literal strings | The `<\|"\|>` interior is encoded once | Schema `string` selects literal parameter bytes; remove only a clean wrapping quote pair, then `SetString` and JSON-encode once; never JSON-unescape that interior |
| Numeric/container retention | Local SAX number-preserving normalization inside the native value parser | A separate `utils.cpp` SAX helper uses the same number-preserving technique for the whole number/array/object value; retain the validated compact JSON in `Functool::exactJsonMembers` |
| Final argument writer | Write validated scalar text while traversing native containers | Keep DOM member placeholders for ordering/duplicate detection; in `State::AfterFunction`, write retained values with `RawValue`, and ordinary values with the usual writer; clear the map in `Functool::clear()` |
| Completeness and publication | Complete native argument container, then one full delta | Complete Qwen function/envelope, registered tool and supported-schema validation, then queued header and full-argument deltas; only STOP can recover the missing outer envelope after `</function>` |

For example, these native forms can represent the same retained arguments when Qwen's schema declares `arg1` as `string` and `amount` as `number`:

```text
Gemma4: <|tool_call>call:tool{arg1:<|"|>C:\test<|"|>,amount:1.10}<tool_call|>
Qwen:   <tool_call><function=tool><parameter=arg1>"C:\test"</parameter><parameter=amount>1.10</parameter></function></tool_call>
JSON:   {"arg1":"C:\\test","amount":1.10}
```

The example illustrates retention, not a new live-model receipt. The Qwen adaptation also preserves valid numeric/container text in the compatible `UNKNOWN` branch, but explicit schema strings are what give hashes, paths and other text their literal-string contract. Do not use that fallback to claim type inference from arbitrary model text. The Gemma scanner's limits, marker rules, constructor modes and completion tolerances were not added to Qwen by this lossless change.

## Argument retention across families

String tool arguments are the model's bytes, encoded once. Do not JSON-unescape them: `\t` stays a backslash and a `t`, and `\\` stays two backslashes. A clean wrapping quote pair is only a delimiter. Integers and floats keep the spelling the model wrote, including trailing zeros, exponent form, and every digit of an integer wider than 2^53. Numbers nested in an object or array argument keep that spelling too. Structural whitespace inside those containers may be compacted. `true`, `false` and `null` are JSON literals; Python `True` and `False` are normalized to them. Hashes, paths and snapshot tags are strings, so they follow the string rule.

Gemma already does this: delimited text goes through `escapeAsJsonString`, and JSON values go through its local `normalizeJsonLosslessly` (`kParseNumbersAsStringsFlag` plus a `RawNumber` that calls `RawValue(..., kNumberType)`). Qwen3Coder follows the same lossless approach for schema types `string`, `integer`, `number`, `array` and `object`: string bytes are encoded once, and validated JSON text is retained per parameter for publication. Gemma's implementation remains separate and is not edited or migrated to the Qwen helper. Hermes, Llama, Phi, Devstral, Onyx and MiniCPM still round-trip a parsed DOM and can change spelling. They need the same lossless JSON and literal string approach; implementing it for those families is outside this change.

## Chat templates

Use the canonical Qwen3.6 chat template.

- Source: `Qwen/Qwen3.6-35B-A3B`.
- md5 `52b6d51ae5b203cb67e64b648494dad2`, 7764 bytes.
- `preserve_thinking` is the 2026 addition on top of `last_query_index`.

Ornith's own template is not canonical. It dropped the `last_query_index` guard and re-injects `<think>` on every past assistant turn.

- Broken original md5 `fa5eb1eee497fc8f3bccc3fcf60d1a51`, sha256 `182e77dd83bd8e9ca818b240b82e28f243762cd5dda32e6eef327df7b1cd107e`, 7536 bytes.
- Live Ornith file `C:\llm\models\ornith-35b-ov\chat_template.jinja` starts from that canonical template and adds one boundary: a `role=tool` message sets `last_query_index`, so thoughts from assistant turns before the tool result are left out. `preserve_thinking` still keeps them. md5 `c9dd16f696ca20e66578498b5500c88e`, sha256 `173b4cb55af687f861b7ca0ea571805093876c4fb140e1f7d64dcc0b54f9ca0b`, 8147 bytes.
- The broken original stays only as `C:\llm\models\ornith-35b-ov\chat_template.jinja.backup-20261021-034305`. Do not copy it back over the live file.
- Repo pin: `docs/qwenmonster/chat-templates/ornith/chat_template.jinja` and `docs/qwenmonster/chat-templates/ornith/CHAT-TEMPLATE-PIN.json`. The installed copy is `C:\llm\models\ornith-35b-ov\chat_template.jinja`, and that file is read-only. Do not substitute either copy. A visually similar file is not token-equivalent.

## Current application

Snapshot at distillation (2026-10-21): A0 partially proven (C6 SHA), A1/official template token-level pin pending, A2 sourced and documented, A3 applied, A4 source patches only, A5 builder candidate, A6–A8 not yet run. No ACCEPTED status.

Semantic refit 2026.4 (2026-10-23): the Qwen3.6 policy above is implemented for tool parser `qwen3coder` with reasoning parser `qwen3`. Gemma4 hard policy uses an EBNF argument grammar with whitespace bound `{0,2}` because this GenAI DLL does not apply `max_whitespace_cnt`. A one-argument `JSONSchema` leaves the call open on newlines. Gemma4 reasoning may open again after a tool envelope. LFM2.5-24B-A2B and the experimental GLM notes are consciously deferred. `ACCEPTED` requires a live model run of the candidate binary. No ACCEPTED status.

Qwen3Coder argument-retention checkpoint: commit `5af105a04904713b58b13bbd82a421c6f15844fa` preserves literal strings and raw numeric/container JSON, including the compatible unclassified-parameter path. Its recorded checks are `*VariousArgumentTypes*` 34/34 (including 11 unchanged Onyx cases), `*Qwen3Coder*` 81/81, and successful test/server builds. The test filters overlap. See [testing handoff and evidence](QWEN3CODER-LOSSLESS-TESTING-HANDOFF.md) for commands, exact binaries and the NOT RUN boundaries. This is parser/build evidence; it does not establish live Qwen/Ornith, HTTP/SSE or agent acceptance. No Gemma implementation or acceptance was changed by that checkpoint or this documentation clarification.
