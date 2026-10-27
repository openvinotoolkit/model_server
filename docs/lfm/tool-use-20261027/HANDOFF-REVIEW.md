# Handoff for review — LFM family hardening + tool-use refit

Branch: `dev/lfm-family-hardening-20261026`  
Base (pre-refit): `a13dff479`  
Head (after this push): see latest commit on the branch.

## Scope to review

1. **LFM tool retainer / router** — `src/llm/io_processing/lfm2/lfm2_tool_parser.*`, `output_parser.*`
2. **Generation policy** — `lfm2/generation_config_builder.hpp`, `tool_grammar.hpp`, selector in `generation_config_builder.hpp`
3. **Templates** — `docs/lfm/templates/LFM2*.jinja` (+ extras/test copies)
4. **Tests** — `lfm_retainer_test`, `lfm_generation_config_builder_test`, `lfm_agent_template_test`, fixture migrations
5. **Harnesses** — `lfm-smoke.py`, `lfm-agent-loop.py` (staff `parallel_tool_calls=true`), `lfm-tool-use-probe.py`
6. **OMP start** — `scripts/windows/start-lfm2-24b.ps1` **long** profile (seqs=2, batch=4096, `cache_size=0`, prefix on, interval 256, candidate-final EXE)

## Evidence (outside git, absolute)

- Suites: `C:\gm\lfm-2026.4\tool-use-20261027\final-tests.xml` (319 PASS / 1 SKIP)
- GPU: `runtime-native-required2-lfm25|lfm2-24b` PASS under native+action-required
- Residual root-cause: `...\residual-rootcause\REPORT.md` (free `auto` delegate)
- Reports in-repo: `FINAL-REPORT.md`, `parser-report.md`, `policy-report.md`, `PEER-DOSSIER.md`

## Review focus / non-goals

- Confirm auto stays unconstrained (no forced-auto heuristic).
- Confirm `parallel=false` still fail-closes; staff default is `parallel_tool_calls=true`.
- Do not treat peer-server differential runs as done (dossier: NOT_RUN).
- Do not expand into Pack/offload or Gemma/Qwen rewrites.

## Suggested PR checks

- [ ] Source review of retainer + grammar admission vs `isSupportedQwenXmlToolSchema`
- [ ] Template golden / Minja path for string tool args
- [ ] Long start script: dynamic KV + 2-slot headroom on Arc 140V UMA is intentional, not smoke profile
- [ ] Accept residuals: deterministic+auto prose; historical 24B native+auto under parallel=false
