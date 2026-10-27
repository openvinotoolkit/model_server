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

Task 5 (GPU-first; CPU deferred by user):
- Rebuilt serving ovms.exe with Task 3 parser: sha256
  f484927cc5bacbb979f9edb11df85c494b6cdf369d7c179b29c48d6807dfc877
- LFM2.5-8B-A1B-int4-ov GPU protocol PASS (load/modules/unary/SSE/tool/
  continuation/tool-SSE/continuation-SSE). Evidence:
  C:\gm\lfm-2026.4\task5-gpu\gpu-LFM2.5-8B-A1B-int4-ov\RESULT.json and
  docs/lfm/evidence/task5-gpu-lfm25-RESULT.json. Peak WS ~4.9 GiB on Arc 140V.
- LFM2.5 GPU benchmark PASS (36 batches / 84 requests): TTFT p50/p95
  4.387s/29.048s, decode TPS p50/p95 51.73/55.04. Receipt:
  docs/lfm/evidence/task5-gpu-lfm25-bench-RESULT.json.
- LFM2-24B-A2B-int4-ov GPU was RAM-admission blocked (~14.1 GiB free vs
  14436 MiB floor; docs/lfm/evidence/task5-gpu-lfm2-24b-admission.json);
  after host memory freed, GPU protocol PASS. Evidence:
  C:\gm\lfm-2026.4\task5-gpu\gpu-LFM2-24B-A2B-int4-ov\RESULT.json and
  docs/lfm/evidence/task5-gpu-lfm2-24b-RESULT.json. Peak WS ~13.1 GiB.
- LFM2-24B GPU benchmark PASS (36 batches / 84 requests): TTFT p50/p95
  4.892s/38.069s, decode TPS p50/p95 40.20/42.09. Receipt:
  docs/lfm/evidence/task5-gpu-lfm2-24b-bench-RESULT.json.
- CPU gates: NOT_RUN (user deferred).

Task 6: Pack mapping deferred. Artifacts under C:\gm\lfm-2026.4\future-pack-notes.

Acceptance (GPU-first scope): YES — both exports cleared protocol + benchmark
on GPU (tool/continuation/SSE included). CPU and Pack remain deferred.

Launch scripts (OMP pattern from start-qwen.ps1): scripts/windows/start-lfm25.ps1
and start-lfm2-24b.ps1; mirrored to C:\Users\testc\.omp\agent\. Serving EXE
C:\gm\lfm-2026.4\serving\ovms.exe (sha256 f484927c…), RC4 DLLs, LM pipeline,
template overrides, OFFLOAD_RATIO=0, port 8137.
