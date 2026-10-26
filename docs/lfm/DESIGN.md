# LFM family hardening on RC4

Date: 2026-10-26. Status: APPROVED by user in this chat before product changes.

## Scope and source

Support exactly OpenVINO/LFM2-24B-A2B-int4-ov and OpenVINO/LFM2.5-8B-A1B-int4-ov.
Local exports remain immutable under C:\llm\models. Start at fetched release/gemmamonster-2026.4-rc4
a5f60ccd85d7b5f1ede3d14fd26a7e2c0b72de36. Its only changes from published 362b1d293
are README and Linux handoff documentation. Worktree: C:\git\gemmamonster-lfm-2026.4.
Create dev/lfm-family-hardening-20261026 after the source build and baseline tests pass.
RC3B, CIB2 builds, deployment files and running services must remain untouched.

## Approach

The handoff requires a bounded change to the existing LFM parser. Use a small recursive
literal reader inside the LFM implementation, retaining its streaming state machine.
Replacing shared parser infrastructure would broaden regression exposure; executing Python
expressions would violate the no-eval requirement. Neither alternative fits the requested scope.
Leave Gemma/Qwen implementations and Pack V2 ABI unchanged.

## Current evidence

PREFLIGHT.json records hashes of all 14 top-level files of each export and the download
metadata revision for each file. Revisions are f8feb127b6125c29ba4d75d6f10ab2564ecd695c
(LFM2) and 97164c6953b3291fcd58a877f0ed5dc3efa24135 (LFM2.5).
Model weight hashes match the respective LFS ETags. Export IR records OpenVINO 2026.2
and 2026.3 respectively; these are conversion provenance, not the execution runtime.

Runtime DLL hashes match the RC4 input lock. The runtime product version is
2026.4.0-22930-61afcb26271-releases/2026/4; GenAI 2026.4.0.0-3401-5f7f1278107;
Tokenizers 2026.4.0.0-736-a04accf6282. RC4's custom GPU input has a historical
source-to-binary attestation gap; this task must report rather than erase it.

Fresh execution of the existing RC4 Python Off test binary passed 90/90 LFM cases,
using the two target tokenizers. The legacy LFM2 suite hardcodes LFM2-2.6B: the
ignored fixture directory contains the 24B target tokenizer, explicitly an alias.
Four template tests use the original OPT fixture. This is existing-binary evidence,
not a fresh source build or model inference. Initial failed fixture discovery is retained.

## Literal grammar and errors

Accept quoted strings, decimal JSON-compatible numbers, lists, string-keyed dictionaries,
True/False/None and JSON true/false/null. Permit whitespace and trailing container commas.
Preserve numeric token spelling where valid JSON, including 1.10, 1e3 and large integers.
Reject bare identifiers, calls, comprehensions, attributes, tuples, bytes, string prefixes,
non-finite numbers, expressions, duplicate keys/keywords and missing keyword values.
The supported language is deliberately a literal subset, not all Python syntax.

Decode single/double quoted strings without executing code. Support escaped quotes,
backslash, standard controls, hexadecimal and Unicode escapes; reject malformed escapes,
unpaired surrogates and raw controls. Unknown escapes are rejected rather than silently
changing their value. Preserve decoded string values including Unicode, paths, apostrophes,
embedded JSON and delimiter text. Serialize decoded strings exactly once with a JSON writer.
Tests must include both escaped backslash paths and accidental escape sequences such as tab.

Builder choices: depth cap 64; input literal byte cap 1 MiB; container entry cap 4096.
Bounds apply to the new literal reader, without changing shared infrastructure.
Parse one completed argument list transactionally; publish no fabricated/partial arguments.
An incomplete streamed prefix waits for more bytes; invalid completed syntax uses the
existing parser error path, with explicit negative unary/SSE tests. A name delta already
emitted by the state machine must never become a successful completed tool on invalid arguments.
Determine the precise existing error propagation in the implementation plan and document
its observed HTTP/SSE outcome before acceptance. Do not silently turn invalid values into strings.

## Streaming and reasoning

Retain Lfm2ParseState and tag dispatch. Change only the argument reader and proven boundary
defects. Test every two-way split of golden messages and bounded randomized multipart splits,
split escapes/Unicode, multiple tools, empty final chunks, STOP/LENGTH and continuation.
Unary output and reconstructed SSE arguments must agree. Keep terminal behavior changes
inside LFM. Lfm25ReasoningParser delegates to Qwen3ReasoningParser: test the LFM configuration
with actual tokenizer IDs and content/reasoning separation, without changing the Qwen parser.

## Templates

Local LFM2.5 format_arg_value wraps raw strings in apostrophes without escaping.
The LFM2 export has no tool-call history renderer. First reproduce round-trip failures
with the real templates through the existing adapter/minja path. Only then change the
LFM template examples/test copies and provide explicit per-model override files for serving.
Do not edit downloaded model templates or globally substitute another model's template.
Use a JSON-quoted representation supported by the literal reader, including nested values.
Verify tool result reinjection, message ordering, reasoning history and continuation markers.
If the real LFM2 template cannot support the cycle, record a separate proven template adaptation.

## Tests, build and runtime gates

1. Fetched base identity, runtime/model locks, fresh Python Off build and baseline suites.
2. Red tests demonstrating literal/template defects; C++ positive/negative golden cases;
   deterministic bounded fuzz with fixed seeds, hostile nesting and no code execution.
3. Focused LFM plus existing Gemma/Qwen parser regression suites from the new binary.
4. Fresh successor OVMS build in a new output directory, never overwrite RC4 snapshots.
5. CPU then GPU load for each exact export, with source/EXE/DLL/model/config hashes and
   loaded-module receipts. Keep HTTP health, compile, load and successful generation separate.
6. Unary and SSE full tool cycle, host tool result reinjection and assistant continuation;
   LFM2.5 reasoning separation and cancellation/length handling. Deterministic fixture
   injection is protocol evidence; distinguish it from an actual generated tool invocation.
7. Arc 140V baseline: no offload, temperature 0, seed 1234; archive actual scheduler/cache,
   prompt, token counts and device settings. Three measured repetitions after a labeled warmup;
   128/512 output-token cases with fixed short/long inputs. Concurrency 1 then 2, and 4 only
   if memory permits. Report p50/p95 TTFT, decode TPS, end-to-end wall TPS, output length,
   process private/working memory and host available RAM; report GPU memory separately.
   Do not add UMA GPU budget to free RAM. Use owned processes, loopback ports, conservative
   memory admission and stop thresholds. Failure to fit is a separate memory gate.

Builder chooses the isolated loopback port, concrete prompts and safe batch/cache settings
from measured memory admission. Record them before comparisons. Never stop unrelated services.
If another model occupies the GPU or memory gate fails, keep measurements NOT_RUN with evidence.

## CIB2 mapping only

User update after design approval: defer Pack work to the future. Keep only the
already generated static source map. Do not pursue compiled eligibility or device
layout mapping in this task; offload remains disabled.

Read actual IR edges, shapes, element types and Const byte ranges, never infer compatibility
from config.json or tensor names. Produce per-layer gate/up/down, scales/zero-points,
expert-axis and contiguous-stride mapping to PackLayerSpec source_offsets/tensor_bytes,
num_expert and layout_params. Verify BIN bounds and weight fingerprints. Trace transformations
needed to reach the existing MoE primitive; distinguish source IR mapping from compiled-node
eligibility. Native Pack V2 tests are no proof of LFM GPU/OVMS support. No offload activation,
Pack ABI changes or edits to the CIB2 checkout are part of this task.

## Deliverable and acceptance

One durable report with changed files, commit SHA, exact commands/log paths, failed attempts,
comparative metric tables and each gate labeled PASS/FAIL/BLOCKED/NOT_RUN.
Use INCOMPLETE for exploratory artifacts, never acceptance. Accept only after both exact
exports pass real GPU model/tool/continuation gates. Build, tokenizer and parser passes alone
do not confer GPU, performance or release acceptance. Current acceptance: NO.
