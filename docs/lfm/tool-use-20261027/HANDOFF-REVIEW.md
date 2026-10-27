# Handoff for review — LFM family hardening + tool-use refit

Branch: `dev/lfm-family-hardening-20261026`  
PR: https://github.com/openvinotoolkit/model_server/pull/4697  
Base (pre-refit): `a13dff479`  
Head: see latest commit on the branch.

**Cloud-oriented full problem set:** [`HANDOFF-CLOUD-REVIEW.md`](HANDOFF-CLOUD-REVIEW.md)  
**Design C:** APPROVED, implementation **NOT started** —  
[`docs/superpowers/specs/2026-10-27-lfm25-hybrid-thought-parser-design.md`](../../superpowers/specs/2026-10-27-lfm25-hybrid-thought-parser-design.md),  
[`docs/superpowers/plans/2026-10-27-lfm25-hybrid-thought-parser.md`](../../superpowers/plans/2026-10-27-lfm25-hybrid-thought-parser.md)

## Scope to review

1. **LFM tool retainer / router** — `src/llm/io_processing/lfm2/lfm2_tool_parser.*`, `output_parser.*`
2. **Generation policy** — `lfm2/generation_config_builder.hpp`, `tool_grammar.hpp`, selector in `generation_config_builder.hpp`
3. **Templates** — `docs/lfm/templates/LFM2*.jinja` (+ extras/test copies)
4. **Tests** — `lfm_retainer_test`, `lfm_generation_config_builder_test`, `lfm_agent_template_test`, fixture migrations
5. **Harnesses** — `lfm-smoke.py`, `lfm-agent-loop.py` (staff `parallel_tool_calls=true`), `lfm-tool-use-probe.py`
6. **OMP start** — `start-lfm2-24b.ps1` long profile; `start-lfm25.ps1` candidate-final pin
7. **Design C docs** — Hybrid thought demotion (review before C++)

## Open problems (see cloud handoff for detail)

1. Design C approved / **not implemented**
2. Free-form thought / pseudo-plan JSON leaks into `content` (needs Design C)
3. Offload speed cliff: r0≈55 → r20≈31.5 → r30≈24 tok/s
4. Memory/u4 DONE: r0+u4≈58.5 (+3.5); r20+u4≈32.4 (cliff remains); warm/long/seqs2/dynKV no decode recovery
5. `tool_choice=none` final prose hang after write+run_python (harness skip workaround)
6. A9 residuals (deterministic+auto prose; historical 24B parallel=false)
7. Staff `parallel_tool_calls=true`; `false` fail-closed by design
8. Windows `plugin_config` ProcessStartInfo JSON escaping pitfalls
9. Peer differential NOT_RUN

## Evidence (outside git, absolute)

- Suites: `C:\gm\lfm-2026.4\tool-use-20261027\final-tests.xml` (319 PASS / 1 SKIP)
- GPU: `runtime-native-required2-lfm25|lfm2-24b` PASS under native+action-required
- Residual: `...\residual-rootcause\REPORT.md`
- Speed: `...\speed-offload-20261027\SUMMARY-v4.json`
- Memory/u4: `...\speed-offload-20261027\MEMORY-PRESSURE-REPORT.md`
- In-repo: `FINAL-REPORT.md`, `parser-report.md`, `policy-report.md`, `PEER-DOSSIER.md`, `HANDOFF-CLOUD-REVIEW.md`

## Review focus / non-goals

- Confirm auto stays unconstrained (no forced-auto heuristic).
- Confirm `parallel=false` still fail-closes; staff default is `parallel_tool_calls=true`.
- Prefer r0+u4 for speed; treat r20 as VRAM trade, not a speed mode.
- Do not treat peer-server differential runs as done (dossier: NOT_RUN).
- Do not expand Design C into Pack/offload engine or Gemma/Qwen rewrites.

## Suggested PR checks

- [ ] Source review of retainer + grammar admission vs `isSupportedQwenXmlToolSchema`
- [ ] Template golden / Minja path for string tool args
- [ ] Long start script: dynamic KV + 2-slot headroom on Arc 140V UMA is intentional
- [ ] Design C demotion profile review (before product C++)
- [ ] Accept residuals: deterministic+auto prose; historical 24B under parallel=false
