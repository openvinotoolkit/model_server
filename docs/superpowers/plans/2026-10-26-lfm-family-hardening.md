# LFM Family Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Strict literal/tool/reasoning support for the two approved LFM exports with measured CPU/GPU evidence.

**Architecture:** Keep LFM streaming phases; replace argument normalization with a bounded recursive literal reader. Patch only proven template defects. Keep compilation, protocol, GPU and benchmark acceptance separate.

**Tech Stack:** C++17, RapidJSON, gtest, minja, Python evidence runners, Windows Bazel, pinned OpenVINO/GenAI/Tokenizers 2026.4.

**Spec:** docs/lfm/DESIGN.md (approved in chat).

## Global Constraints

- Exactly OpenVINO/LFM2-24B-A2B-int4-ov and OpenVINO/LFM2.5-8B-A1B-int4-ov under C:\llm\models.
- No 2026.5 binaries, Gemma/Qwen implementation edits, protected RC3B/CIB2 changes or Pack V2 ABI changes.
- New branch dev/lfm-family-hardening-20261026 only after fresh baseline build/tests.
- Literal depth 64, byte limit 1 MiB and container limit 4096.
- No eval; no inferred GPU compatibility or acceptance without actual GPU tests.

## Review Focus

- Numeric spelling and nested string escapes survive serialization: Task 2 golden fixtures.
- Fragmentation at every quote/backslash/tag boundary emits each value once: Task 3 splits.
- Invalid syntax cannot become a successful tool call: Task 3 negative tests.
- Actual export history and reasoning fields differ from test templates: Task 4 real-template tests.
- UMA memory admission and loaded-DLL provenance affect valid comparisons: Task 5 runtime receipts.

### Task 1: Establish baseline and branch

Files: docs/lfm/evidence/*, docs/lfm/REPORT.md.
Interface: immutable source/runtime/model locks and fresh build-py_off/COMPLETE.json.

- [x] Fetch/check a5f60ccd; hash exports/runtime and record prior binary test provenance.
- [ ] Finish fresh Python Off build via scripts/windows/build-rc4.ps1 -Output C:\gm\lfm-2026.4\baseline-a5f60ccd -Jobs 4.
- [ ] Run new binary *LFM2*:*LFM25* with target tokenizer fixtures; require 90 passes.
- [ ] Restore version header, create named branch, commit design/plan/evidence inventory.

### Task 2: Strict literal reader

Create src/llm/io_processing/lfm2/python_literal.hpp and src/test/llm/output_parsers/lfm_literal_test.cpp.
Interface: std::string ovms::lfm2::parseArguments(std::string_view input), returning a complete JSON object or throwing std::invalid_argument.

- [ ] Add table-driven positive/negative C++ golden tests first; demonstrate old parser's lossy fallback with nested Python literals and escaped strings.
- [ ] Run focused red cases, preserving logs. Standalone direct-reader checks may run while the unmodified base builds.
- [ ] Implement bounded recursive descent; validate JSON-compatible number tokens without floating point conversion; JSON-encode decoded UTF-8 strings once.
- [ ] Verify all fixtures and fixed-seed mutation/depth/size fuzz; reject expression execution, duplicate names/keys and malformed escapes.
- [ ] Commit reader/tests after green validation.

### Task 3: Integrate argument reader and stream boundaries

Modify src/llm/io_processing/lfm2/lfm2_tool_parser.cpp; tests in the LFM2/LFM25 output-parser suites.
Consumes Task 2 parseArguments; produces existing ToolCall/Delta interfaces without shared parser changes.

- [ ] Add red tests for nested None/True/False, numeric spelling, malformed keyword/value, comma without a space and quoted delimiter text.
- [ ] Replace normalization/splitting with strict reader; maintain phases and error propagation via existing OutputParser exception path.
- [ ] Add exhaustive two-way and fixed-seed multipart splits, STOP/LENGTH/empty flush, multi-tool and invalid-completion tests; adjust only proven LFM boundary bugs.
- [ ] Build successor and run focused LFM then Gemma/Qwen regression suites with exact fixture provenance. Update legacy permissive LFM assertions to strict rejection.
- [ ] Commit with test receipts.

### Task 4: Template and reasoning round trip

Modify only extras/chat_template_examples/chat_template_lfm*.jinja and matching src/test/llm/chat_templates copies after red tests.
Create docs/lfm/templates per-model overrides only when required by real export behavior.

- [ ] Add minja end-to-end cases with O'Brien, literal backslashes, controls, Unicode, nested objects/lists and actual export templates; watch failures.
- [ ] Serialize argument literals safely; preserve LFM2.5 thinking/continuation conventions. Keep downloaded files immutable.
- [ ] Verify real-tokenizer reasoning/content separation, nested tool-result reinjection and unary/SSE reconstruction.
- [ ] Commit only changes proved by round-trip tests.

### Task 5: Model and performance evidence

Create scripts/windows/lfm-smoke.py, docs/lfm/evidence/runtime receipts and final REPORT.md.
Consumes successor build, locked runtime/model inputs and explicit template overrides.

- [ ] Start one owned loopback server at a time after host memory/service checks; pin loaded-module hashes.
- [ ] Test CPU and GPU separately for each model: load, generation, reasoning, tool result/continuation, unary/SSE and negative protocol cases. Save raw requests and responses.
- [ ] Warm up then measure 3 repetitions, output lengths 128/512, fixed short/long prompts, concurrency 1/2 and memory-admitted 4. Save precise scheduler/cache/token settings, TTFT, decode TPS, wall TPS, memory and failures.
- [ ] Stop only owned servers and verify cleanup. Failed admission stays separate from unsupported graph/kernel.

### Task 6: Pack V2 mapping and handoff

USER UPDATE: Pack work deferred. Preserve the existing static map and skip all further
Pack work below. Final source/test/runtime handoff remains in scope.

Files: docs/lfm/evidence/IR-PACK-CANDIDATES.json, mapping scripts and REPORT.md.
Consumes real IR edges/Const ranges; produces mapping to existing PackLayerSpec with no ABI/offload activation.

- [ ] Trace routing and gate/up/down computation; assign all nine tensors by graph semantics, not names. Verify axes, per-expert ranges and layout requirements against pinned CIB2 source.
- [ ] Separate static mapping from transformed MOECompressed compiled eligibility; report unresolved fields explicitly.
- [ ] Fresh whole-branch review, fix material findings with red/green checks, then final verification.
- [ ] Commit source/tests/report and return SHA, files, logs, metric table and PASS/FAIL/BLOCKED/NOT_RUN per gate. No acceptance unless actual GPU/tool/continuation passes for both models.
