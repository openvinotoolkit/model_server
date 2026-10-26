# Execution ledger - docs/superpowers/plans/2026-10-26-lfm-family-hardening.md

Design approved by user in chat. Execution inline; fresh review at completion.
User steering: inspect our minja/GenAI forks for applicable existing repairs.
Read-only comparison saved in DEPENDENCY-FORKS.md. No dependency/DLL replacement.
User steering: Pack work deferred to the future. Retain existing static mapping only;
do not continue Pack/device-layout/compiled-eligibility work or activate offload.
Base: a5f60ccd85d7b5f1ede3d14fd26a7e2c0b72de36.
Branch: dev/lfm-family-hardening-20261026.

Task 1: complete. Fresh Python Off build in C:\gm\lfm-2026.4\baseline-a5f60ccd
(2814.689s, 8561 actions, BUILD_PASS). ovms.exe sha256
ce6fd4eb550d6d29cea0ad052a4ea37a85493f873179b5779c6388a14eec72d7;
ovms_test.exe sha256 3e8b8bc79214b486871112da57b84fa730b3656d63b95b62e1d4aef5d78e0b14
at first receipt; later Task 3 relinks supersede the test EXE. Fresh LFM filter
90/90 in docs/lfm/evidence/fresh-baseline.xml. RC4 packaged binary also 90/90 with
target tokenizer fixtures (rc4-binary-target-tokenizers.xml).

Task 2: complete. Bounded literal reader committed (85389f3c0). literal-green 5/5.

Task 3: complete for focused LFM integration. Opt-in BaseOutputParser
isStructuralEndTag (default true) + drainPendingDelta; LFM quote/escape parity;
strict parseArguments; hold after "]" while end-tag prefix arrives; do not treat
nested a=[...] as the tool-list opener. Focused suite
docs/lfm/evidence/task3-focused-green.xml: 109 passed, 1 skipped (LFM2 reasoning).
Legacy fixtures updated to quoted literals + RejectsLegacyMalformedLiterals.

Task 4: overrides under docs/lfm/templates GREEN 2/2; copied into
extras/chat_template_examples/chat_template_lfm{2,25}.jinja and
src/test/llm/chat_templates/chat_template_lfm{2,25}.jinja. Downloaded
C:\llm\models templates remain immutable. Round-trip tests in focused green.

Gemma/Qwen parser regression after Task 3 relink: 201/201
(docs/lfm/evidence/task3-gemma-qwen-regression.xml).

Task 5: scripts/windows/lfm-smoke.py prepared. CPU/GPU model load, tool cycle,
and Arc 140V benchmarks NOT_RUN in this pass (no owned server started).

Task 6: Pack mapping deferred. Artifacts under C:\gm\lfm-2026.4\future-pack-notes.

Acceptance: NO until real GPU/tool/continuation gates for both exports.
