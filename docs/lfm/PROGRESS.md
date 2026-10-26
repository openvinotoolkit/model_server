# Execution ledger - docs/superpowers/plans/2026-10-26-lfm-family-hardening.md

Design approved by user in chat. Execution inline; fresh review at completion.
User steering: inspect our minja/GenAI forks for applicable existing repairs.
Read-only comparison saved in DEPENDENCY-FORKS.md. No dependency/DLL replacement.
User steering: Pack work deferred to the future. Retain existing static mapping only;
do not continue Pack/device-layout/compiled-eligibility work or activate offload.
Base: a5f60ccd85d7b5f1ede3d14fd26a7e2c0b72de36, detached until fresh build/test gate.
Task 1: complete. Fresh build succeeded (2814.689 seconds, 8561 actions); its new
test EXE passed 90/90. src/version.hpp restored without diff. Requested branch
dev/lfm-family-hardening-20261026 created only after these checks.
Deferred Pack artifacts moved out of the implementation tree to
C:\gm\lfm-2026.4\future-pack-notes. No further Pack work authorized in this task.
Task 1: in progress. Existing RC4 binary passed 90/90 with target tokenizer fixtures;
first fixture-discovery failure retained. Fresh source build active in C:\r20264\msnbck67.
Pre-flight: Task 2 parseArguments(string_view)->JSON object consumed only by Task 3;
Task 4 overrides consumed by Task 5; Task 6 mapping does not enable offload.
Ruling: user approval to continue implementation covers execution of this plan inline;
do not request a second process-only approval for the same authorized work.
Ruling: prepare isolated direct-reader red tests while the unmodified base compiles;
do not edit files consumed by that build until its baseline receipt is complete.
Task 2: direct reader RED observed (4 failing groups from extracted RC4 normalization),
then GREEN (4/4 including fixed-seed 1000-string round trips and hostile mutations).
Task 3: integrated RED linked against the freshly compiled, unchanged RC4 parser
libraries using evidence/parser-test-runner.py. Result: 6 failed, 3 passed, 1 expected
LFM2 reasoning skip. Failures cover chunk splitting, quoted markers/backslashes,
multiple calls and malformed literals. See parser-red.xml/log.
Task 4: actual export templates + real tokenizers in pinned GenAI/minja were RED 0/2;
explicit overrides are GREEN 2/2, including nested values, thinking and tool-result
reinjection. Originals remain immutable. Overrides currently only under docs/lfm/templates;
examples/test template copies are not changed yet (baseline gate must finish first).
Task 5: scripts/windows/lfm-smoke.py prepared; native process/host memory probes passed
against its own Python process. No model weights loaded and no GPU acceptance yet.

Next: wait for baseline build session 59729 / C:\r20264\msnbck67 to finish; preserve
src/version.hpp while its wrapper owns it. Then test its new EXE (90 LFM baseline
cases), create the requested branch, and integrate the reader. New source/test files
were created after baseline analysis and are not part of that baseline binary.

Task 3 Ruling: preserving literal delimiter text needs an opt-in, nonmutating
BaseOutputParser structural-end-tag predicate (default true) used by the generic
tool-end lookup; only LFM overrides it to scan buffered quotes with correct escape
parity. This is a bounded router extension, not a Gemma/Qwen implementation change
or Pack ABI change. Cover it with the already RED quoted-marker tests and regression
suites. LFM also needs its opt-in drain hook and a progress loop through the same
states; no new phases. Existing broad quote heuristic misreads an even run of
backslashes before a closing quote; replace only the LFM argument-end scan.

Legacy LFM fixtures contain deliberately non-literal raw newlines, unescaped quote
payloads and raw Windows paths. Convert positive fixtures to properly quoted literals
while preserving their intended values, and retain explicit negative coverage for the
old malformed spellings. Missing keyword/value cases must now assert strict rejection.
