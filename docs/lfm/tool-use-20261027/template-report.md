# LFM template refit

Status: TEMPLATE UNIT GREEN, 16/16 standalone real-tokenizer/Minja cases.
Live API and model acceptance: NOT_RUN by this worker; root owns those gates.

New suite: `src/test/llm/output_parsers/lfm_agent_template_test.cpp`.
Root should register it in the build if tests are not globbed. Tests use the exact
two existing target tokenizer fixtures (LFM2-2.6B directory is the documented 24B alias).

Reproduced source defects: LFM2 places assistant content after tool calls; both
overrides lack reasoning aliases and tool-definition unwrapping. JSON argument
string handling exposed a pinned-MinJa capability-detection limitation described below.

## Changes

- Wrapped OpenAI tools unwrap to native flat function definitions. Flat and wrapped
  forms produce identical token IDs with both exact target tokenizers.
- Replay `thinking`, `reasoning` and `reasoning_content` consistently, honoring each
  family's existing past-thinking retention rule and explicit preserve_thinking.
- LFM2 keeps assistant content before its tool calls and handles null/missing
  assistant content without emitting a null literal. Canonical tool role preserved.
- LFM2.5 handles mapping content and string items without losing their contents.
- Safe JSON string/container literals from the preceding hardening remain intact.
- The two model overrides and their example/test copies are synchronized.
- New script `scripts/windows/lfm-template-check.py` builds only a small standalone
  gtest target, retains compile/test logs, XML and input hashes, and refuses to
  overwrite an existing result label.

## Verification

Command: `rtk proxy python scripts/windows/lfm-template-check.py --label final-green`

Evidence root: `C:\gm\lfm-2026.4\tool-use-20261027\template`.
`red/RESULT.json`: compile 0, test 1, 7 failures of 12 cases before template edits.
`green/RESULT.json`: compile 0, test 1; detected Minja polyfill regression when
trying an upstream-style rejection of raw string arguments. This was reverted.
`green-compatible/RESULT.json`: compile 0, test 0, 12/12.
`final-green/RESULT.json`: compile 0, test 0, 16/16, including exact golden text/token
sequence, alias token equality, retention and canonical continuation suffix.
`reviewed-green/RESULT.json`: compile 0, test 0, 16/16 after final source review.
Input hashes include template, sources, SDK DLL files and target tokenizer fixtures.
No model weights were loaded and no serving files/server were modified.

## Pinned Minja seam and ruling

Do not delete the string-argument compatibility branch with the existing SDK DLL.
Minja `renders_tool_arguments` recognizes JSON/XML/Gemma argument representations
but omits LFM's `key=value` native syntax. Requiring object arguments at template
level consequently makes it infer no tool support and silently polyfill tool calls
as JSON content and tool results as user turns. This was reproduced in the failed
`green` run, not inferred from static source alone.

OVMS `probe.cpp::analyzeProbeToolArgumentResults` recognizes `key=value` and sets
requiresObjectArguments whenever the object probe renders natively. Its existing
adapter then converts API JSON argument strings to objects. The compatibility
branch remains for Minja detection; direct tokenizer callers must pass mapping
arguments for native keyword replay. The deleted string-rejection assertion was
an unsupported cross-runtime expectation, not a passed fix. Actual API string
argument replay still requires the root's live adapter test. No Minja/DLL changes.

Relevant read-only peer location:
`C:\gm\qwenmonster\minja\include\minja\chat-template.hpp` (lines 196-219),
checkout HEAD `83eb074e42140b0d1194d1dd61f29c31e4515ee9`,
and OVMS `src/llm/io_processing/chat_template/probe.cpp` (native args probe).

Official sources inspected on 2026-10-27 (exact revisions/hashes in
`template-official-sources.json`):
- https://huggingface.co/LiquidAI/LFM2.5-8B-A1B/raw/main/chat_template.jinja
- https://huggingface.co/LiquidAI/LFM2-24B-A2B/raw/main/chat_template.jinja
- https://huggingface.co/LiquidAI/LFM2.5-8B-A1B/raw/main/README.md

Current official LFM2 template explicitly rejects nonempty JSON argument strings,
supports reasoning aliases and content before calls. LFM2.5 upstream now supports
thinking/reasoning/reasoning_content. Both generation suffixes remain
`<|im_start|>assistant\n`; there is no canonical `enable_thinking` suffix branch.
No reasoning-off control is claimed. Downloads remain immutable.

## Full-suite legacy expectation migration

Root's candidate-tests.xml reproduced four LFM-specific Minja integration failures:
LFM2_ToolCallWithStringArgs, LFM25_ToolCallWithStringArgs,
LFM25_ToolCallWithStringArgsAndReasoning, LFM25_ReasoningAndEmptyToolCalls.
The exact-output assertions still expected apostrophe literals, and the capability
assertions expected the obsolete thinking-field workaround (or no reasoning parser
for LFM2). Current official LFM2 template at revision a3bbacd91a678b97712f0e323e52f8c24ba29542
replays explicit thinking/reasoning/reasoning_content fields. Its presence is a native
template capability, not a claim that all LFM2 variants generate thoughts.

Only these LFM blocks in src/test/llm/chat_template_end_to_end_minja_test.cpp were
migrated: assert detected lfm2 reasoning parser, no missnamed-field workaround,
and exact JSON-safe double-quoted replay. Existing native-call/reasoning/output
assertions were retained; LFM2 additionally asserts no application exception.
Root owns the final full-suite rebuild and GREEN evidence. Product templates are frozen.
