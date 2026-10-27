# LFM2.5 Hybrid C — thought / tool separation

Date: 2026-10-27  
Status: **APPROVED** (Design C / Hybrid) by user before product C++ changes  
Repo: `C:\git\gemmamonster-lfm-2026.4`  
Related: `docs/lfm/tool-use-20261027/` (retainer, policy, residuals), `docs/lfm/DESIGN.md` (family hardening)

## Problem

Under unconstrained `tool_choice=auto`, LFM2.5 sometimes emits free-form “agent plan” text instead of Liquid’s native tool envelope. Observed shapes include:

- Prose that narrates fake tools (`[tool:cwd]`, simulated results).
- Broken / pseudo JSON objects with keys such as `explanation`, `plan`, `commands` (often with bash/`/workspace` style steps).

Today the retainer correctly refuses to turn that text into executable `tool_calls` (no native `<|tool_call_start|>…<|tool_call_end|>`). The residual UX failure is that the same blobs leak into assistant **`content`**, looking like visible answers or fake actions, while real thoughts that use `<think>…</think>` already land in `reasoning_content`.

This is **not** Liquid’s official protocol. Official intent (Liquid Tool Use docs; vLLM LFM2.5 recipe):

| Channel | Manufacturer intent | Current OVMS |
| --- | --- | --- |
| Tools | Native envelope; parser `lfm2` | `Lfm2ToolParser` retainer (registry path) |
| Thinking | `<think>…</think>`; vLLM uses qwen3 reasoning parser | `reasoning_parser=lfm2` → `Lfm25ReasoningParser` (Qwen3 subclass) |

## Goals

1. Hide pseudo-plan / free-form thought blobs in `reasoning_content` (including malformed JSON with `explanation` / `plan` / `commands`).
2. **Never** promote non-native tool syntax into executable `tool_calls`.
3. Extract real tool calls **only** from the Liquid native envelope
   (`<|tool_call_start|>` + pythonic list + `<|tool_call_end|>`).
4. Keep manufacturer alignment: primary thought path remains `<think>` tags; demotion is a secondary safety net for untagged plan/prose.

## Non-goals

- Forced-auto grammar or heuristics that require a tool on every `auto` request (already rejected in policy design).
- Parsing `[tool:…]`, XML tools, OpenAI-style JSON tool lists, or plan-`commands` into `tool_calls`.
- Changing Gemma/Qwen/Onyx parsers, Pack/offload, or control OVMS on port 8137.
- Claiming that demotion makes unconstrained auto “agent-complete”; host loops still need native envelopes (prefer `native` + `action-required` for agent acceptance).
- Altering hard `required` / named EBNF (those modes already skip optional reasoning preamble).

## Approved approach: Hybrid C

**Primary path (unchanged):** `Lfm25ReasoningParser` strips `<think>` / `</think>` into `ReasoningDelta`. Tool-looking text inside an open think span stays reasoning (no Gemma-style tool handoff).

**Secondary path (new):** After the content/tool router owns a segment that would otherwise become visible `content`, classify completed non-envelope text. If it matches a **closed demotion profile** (pseudo-plan JSON or pseudo-tool prose), emit `ReasoningDelta` instead of `ContentDelta`. Ordinary assistant prose stays `content`.

**Tool path (unchanged contract):** Registry retainer publishes `ToolCallDelta` only after a complete native envelope passes registration + schema validation. Demotion never invents calls.

```text
stream tokens
    │
    ├─ <think> … </think>     → Lfm25ReasoningParser → reasoning_content
    │
    ├─ <|tool_call_start|>…   → Lfm2ToolParser retainer → tool_calls (or reject)
    │
    └─ other text segment
           │
           ├─ demotion match  → ReasoningDelta (hidden thought / plan)
           └─ otherwise       → ContentDelta (visible answer)
```

## Alternatives considered (not chosen)

| Option | Idea | Why rejected |
| --- | --- | --- |
| A — Think-only | Rely solely on `<think>`; leave untagged plans in content | Leaves the residual UX leak that motivated this work |
| B — Demote-everything-non-envelope | All text outside native tools → reasoning | Hides legitimate assistant answers; breaks chat UX |
| **C — Hybrid (approved)** | Keep `<think>` + demote only closed pseudo-plan / pseudo-tool profiles | Matches manufacturer primary path; fixes leak without silencing real content |

## Architecture

### Components

1. **`Lfm25ReasoningParser`** (`src/llm/io_processing/lfm2/lfm25_reasoning_parser.hpp`)  
   Unchanged. Continues to configure Qwen3-style tags with LFM2.5 special-token IDs.

2. **`Lfm2ToolParser` retainer** (`lfm2_tool_parser.*`)  
   Unchanged publication rules. Content emission sites that currently push `ContentDelta` for non-envelope text become the **integration point** that consults demotion when enabled.

3. **`lfm2/thought_demotion.hpp` (new)**  
   Pure, side-effect-free classifier + segment buffer helpers. No tokenizer dependency. Unit-tested in isolation.

4. **OutputParser wiring** (`output_parser.cpp`)  
   When `tool_parser=lfm2` **and** `reasoning_parser=lfm2`, construct the registry tool parser with demotion enabled. LFM2-24B tool-only deployments (`reasoning_parser` unset) leave demotion **off** so we do not invent a reasoning channel the serve config never asked for.

### Demotion profile (closed set)

A completed text segment is demoted iff **any** of the following holds after trimming ASCII whitespace. Segments are the contiguous non-envelope runs the retainer already isolates (text before an opener, between envelopes, or trailing text at finish).

**D1 — Pseudo-plan JSON (including malformed):**

- The segment is dominated by a JSON-like object: optional leading prose of at most 200 bytes, then `{…}` that accounts for the remainder (trailing whitespace allowed).
- Object key set intersects the fixed demotion key set:  
  `explanation`, `plan`, `commands`, `command`, `steps`.
- Detection must work when RapidJSON parse **fails** (broken JSON). Use a structural scan: find `{`, then scan for quoted keys from the demotion set at object-key positions (quote-aware, depth-aware). Do **not** require a successful parse to demote.
- Nested objects/arrays inside the plan do not expand the key set; only top-level keys of the outer plan object count.

**D2 — Pseudo-tool prose:**

- After trim, the segment matches at least one of:
  - A line (split on `\n`) whose trimmed form matches `\[tool:[^\]]+\]` (case-sensitive `tool:`), optionally followed by arguments/prose on the same line.
  - A whole-segment pattern that is only such lines plus blank lines (no ordinary paragraph before the first match).
- Native envelopes are **never** subject to D2; they are consumed by the retainer before demotion runs.

**Explicit non-matches (keep as content):**

- Ordinary prose without D1/D2 markers.
- Valid or invalid text that merely *mentions* the words plan/explanation without the JSON/object or `[tool:…]` shapes above.
- Native tool envelopes and their interiors.
- Content already inside `<think>…</think>` (handled by the reasoning parser; demotion does not re-process it).

### Streaming behavior

Demotion runs on **completed segments**, not on every token:

1. While the retainer is emitting proven non-envelope content and demotion is enabled, buffer the segment instead of immediately publishing `ContentDelta`.
2. Segment closes when a native start tag begins, or when `finishReason != NONE`.
3. On close: classify the buffer; publish one `ReasoningDelta` or one `ContentDelta` (or nothing if empty); clear the buffer.
4. Byte cap: demotion buffer shares the existing 1 MiB retained-output limit; overflow follows the retainer’s existing fail-closed path.
5. Optimistic streaming of “clearly ordinary” content is **out of scope for v1** to avoid retracting published content. Accept holding non-envelope text until segment close when demotion is on. Hard `required`/`named` generations that emit only envelopes are unaffected. Chat answers without tools flush at finish (one delayed content/reasoning burst) — documented trade-off; revisit only with evidence.

### Tool-call invariant

| Input | `tool_calls` | Visible `content` | `reasoning_content` |
| --- | --- | --- | --- |
| `<think>…</think>` only | none | empty | think body |
| Native envelope (valid) | published | prior demoted/kept segments only | think + demoted |
| Native envelope (invalid) | none (reject / terminal) | unchanged policy | demotion still applies to non-envelope text already segmented |
| D1 plan JSON / D2 `[tool:…]` | **none** | empty for that segment | full segment |
| Ordinary answer prose | none | prose | think only (if any) |

No code path may call `retainEnvelope` / publish `ToolCallDelta` from demotion output.

### Configuration

- Demotion **on** when OutputParser is constructed with `toolParserName == "lfm2"` and `reasoningParserName == "lfm2"`.
- Demotion **off** for tool-only `lfm2` (LFM2-24B staff configs) and for all other families.
- No new CLI flag in v1. If a flag is needed later, add it behind the same OutputParser construction site; do not invent per-request API fields in this design.

### Error handling

- Classifier never throws on malformed UTF-8/JSON; unmatched → keep as content.
- Retainer exceptions for bad envelopes remain unchanged (clear pending deltas, terminal lock).
- Demotion buffer is cleared on `resetState()` with the rest of the tool parser state.

## Testing strategy

Primary new suite: `src/test/llm/output_parsers/lfm_thought_demotion_test.cpp` (pure classifier) plus integrated cases in `lfm_strict_output_parser_test.cpp` / LFM25 fixture suite with real tokenizer IDs.

Minimum cases:

1. `<think>private</think>public` — unchanged reasoning/content split.
2. D1 well-formed `{"explanation":"…","plan":[…],"commands":[…]}` alone → all `reasoning_content`, zero tools.
3. D1 malformed JSON with same keys → demoted, zero tools.
4. D2 `[tool:cwd] /home/user` style segment → demoted, zero tools.
5. Ordinary prose → `content`.
6. Demoted prose then valid native envelope → reasoning + one tool call; no promotion of demoted text.
7. Valid native envelope only — still publishes; demotion idle.
8. LFM2 tool-only parser (no reasoning parser) — demotion off; D1 text stays `content` (documents gate).
9. Two-way splits across `{`, keys, and `[tool:` so buffering does not publish early content.
10. Regression: existing retainer green cases still pass with demotion on.

Evidence: focused gtest under existing Windows/standalone runners used for retainer/policy; no requirement to stop control OVMS 8137 for unit work. Live GPU re-smoke is optional follow-up after green unit/integration.

## Rollout and acceptance

1. Land classifier + unit tests.
2. Wire demotion into retainer content emission + OutputParser enablement.
3. Integration tests with LFM2.5 tokenizer fixture.
4. Optional: re-run deterministic+auto agent capture and confirm fake plans move to `reasoning_content` with still-empty `tool_calls` (behavior change is UX classification, not agent PASS unless native envelopes appear).

Acceptance for this design:

- Goals 1–4 met under tests above.
- No new path publishes non-native tools.
- LFM2-24B tool-only serve configs unchanged in behavior.

## Open questions (non-blocking)

Resolved defaults for implementation; escalate only if product owners disagree:

1. **Leading prose before plan JSON:** capped at 200 bytes then `{…}` (see D1). Longer preambles keep the whole segment as content unless D2 also matches.
2. **Key set:** fixed list above; do not grow without a design amend.
3. **v1 holds all demotion-enabled content until segment end** (no optimistic flush).
4. **No CLI flag in v1**; gated on `reasoning_parser=lfm2`.
